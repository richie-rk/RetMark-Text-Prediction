import json
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from pk import redact

# Not tunables: how long to wait on a process that has stopped answering.
STARTUP_TIMEOUT_S = 30.0
REQUEST_TIMEOUT_S = 10.0


class HarperUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Lint:
    start: int
    end: int
    rule: str
    kind: str
    message: str
    suggestions: tuple[str, ...]


def _utf16_position(text, index):
    line = text.count("\n", 0, index)
    line_start = text.rfind("\n", 0, index) + 1
    col = sum(2 if ord(c) > 0xFFFF else 1 for c in text[line_start:index])
    return {"line": line, "character": col}


def _char_index(text, position):
    """Python string index of an LSP (line, UTF-16 column) position."""
    line_start = 0
    for _ in range(position["line"]):
        line_start = text.index("\n", line_start) + 1
    units = 0
    i = line_start
    while units < position["character"] and i < len(text):
        units += 2 if ord(text[i]) > 0xFFFF else 1
        i += 1
    return i


def _suggestion_text(flagged, suggestion):
    if "ReplaceWith" in suggestion:
        return "".join(suggestion["ReplaceWith"])
    if "InsertAfter" in suggestion:
        return flagged + "".join(suggestion["InsertAfter"])
    if suggestion == "Remove" or "Remove" in suggestion:
        return ""
    return None


class HarperLS:
    """A persistent `harper-ls --stdio` process, driven as a minimal LSP client.

    One process serves every dialect: harper-ls builds a document's linter
    once, with the dialect configured when the document is opened, so each
    dialect gets its own document and checks are serialised.
    """

    def __init__(self, path=""):
        exe = path or shutil.which("harper-ls")
        if not exe or not Path(exe).is_file():
            raise HarperUnavailable(f"harper-ls not found ({path or 'not on PATH'})")
        # Harper's own dictionary and ignore files live under a directory we
        # never write to, so Harper never consults the user's real ones (§8).
        self._tmp = Path(tempfile.mkdtemp(prefix="pk-harper-"))
        self._absent = self._tmp / "absent"
        self._dialect = "American"
        self._docs = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._inbox = queue.Queue()
        try:
            self._proc = subprocess.Popen(
                [exe, "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            shutil.rmtree(self._tmp, ignore_errors=True)
            raise HarperUnavailable(f"cannot start harper-ls: {e}") from None
        threading.Thread(target=self._read_loop, daemon=True).start()
        try:
            self._request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": self._tmp.as_uri(),
                    "capabilities": {"workspace": {"configuration": True}},
                },
                timeout=STARTUP_TIMEOUT_S,
            )
            self._notify("initialized", {})
        except Exception as e:
            self.close()
            raise HarperUnavailable(f"harper-ls did not start: {e}") from None

    def _settings(self):
        return {
            "harper-ls": {
                "dialect": self._dialect,
                "userDictPath": str(self._absent / "user.txt"),
                "workspaceDictPath": str(self._absent / "workspace.txt"),
                "fileDictPath": str(self._absent / "files"),
                "ignoredLintsPath": str(self._absent / "ignored"),
                "statsPath": str(self._absent / "stats.txt"),
                "diagnosticDelayMs": 0,
            }
        }

    def settings_paths_exist(self):
        return self._absent.exists()

    def _read_loop(self):
        out = self._proc.stdout
        while True:
            length = None
            while True:
                line = out.readline()
                if not line:
                    self._inbox.put(None)
                    return
                line = line.strip()
                if not line:
                    break
                name, _, value = line.partition(b":")
                if name.strip().lower() == b"content-length":
                    length = int(value)
            if length is None:
                continue
            self._inbox.put(json.loads(out.read(length)))

    def _send(self, message):
        body = json.dumps(message).encode("utf-8")
        try:
            self._proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
            self._proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise HarperUnavailable(f"harper-ls pipe closed: {e}") from None

    def _notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _wait(self, predicate, timeout):
        while True:
            try:
                message = self._inbox.get(timeout=timeout)
            except queue.Empty:
                raise HarperUnavailable("harper-ls stopped responding") from None
            if message is None:
                raise HarperUnavailable("harper-ls exited")
            if "method" in message and "id" in message:
                # harper-ls pulls its configuration before every document update.
                result = [self._settings()] if message["method"] == "workspace/configuration" else None
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})
                continue
            if predicate(message):
                return message

    def _request(self, method, params, timeout=REQUEST_TIMEOUT_S):
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        reply = self._wait(lambda m: m.get("id") == rid and "method" not in m, timeout)
        if "error" in reply:
            raise HarperUnavailable(f"harper-ls error on {method}: {reply['error']}")
        return reply.get("result")

    def _drain(self):
        """Discard replies left over from a request that timed out, so a late
        publishDiagnostics can never be read as the answer to a new check."""
        while True:
            try:
                message = self._inbox.get_nowait()
            except queue.Empty:
                return
            if message is None:
                raise HarperUnavailable("harper-ls exited")
            if "method" in message and "id" in message:
                result = [self._settings()] if message["method"] == "workspace/configuration" else None
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def lint(self, text, dialect):
        with self._lock:
            self._drain()
            self._dialect = dialect
            uri, version = self._docs.get(dialect, (None, 0))
            if uri is None:
                uri = (self._tmp / f"{dialect.lower()}.txt").as_uri()
                self._notify(
                    "textDocument/didOpen",
                    {"textDocument": {"uri": uri, "languageId": "plaintext", "version": 1, "text": text}},
                )
                version = 1
            else:
                version += 1
                self._notify(
                    "textDocument/didChange",
                    {"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]},
                )
            self._docs[dialect] = (uri, version)
            published = self._wait(
                lambda m: m.get("method") == "textDocument/publishDiagnostics" and m["params"]["uri"] == uri,
                REQUEST_TIMEOUT_S,
            )
            return [self._resolve(uri, text, d) for d in published["params"]["diagnostics"]]

    def _resolve(self, uri, text, diagnostic):
        # Diagnostics carry no suggestions and a UTF-16 range; the serialised
        # lint inside the "ignore" code action has both, in Python indices.
        actions = self._request(
            "textDocument/codeAction",
            {"textDocument": {"uri": uri}, "range": diagnostic["range"], "context": {"diagnostics": []}},
        ) or []
        for action in actions:
            if action.get("command") != "HarperIgnoreLint":
                continue
            lint = action["arguments"][1]
            start, end = lint["span"]["start"], lint["span"]["end"]
            if lint["message"] != diagnostic["message"]:
                continue
            if _utf16_position(text, start) != diagnostic["range"]["start"]:
                continue
            if _utf16_position(text, end) != diagnostic["range"]["end"]:
                continue
            flagged = text[start:end]
            suggestions = (_suggestion_text(flagged, s) for s in lint["suggestions"])
            return Lint(
                start=start,
                end=end,
                rule=str(diagnostic.get("code", "")),
                kind=lint["lint_kind"],
                message=lint["message"],
                suggestions=tuple(s for s in suggestions if s is not None),
            )
        # harper-ls sometimes returns no code actions for a valid range (seen
        # after some emoji sequences); the diagnostic alone still says what
        # and where, just without suggestions.
        start = _char_index(text, diagnostic["range"]["start"])
        end = _char_index(text, diagnostic["range"]["end"])
        code = str(diagnostic.get("code", ""))
        return Lint(start, end, code, "Spelling" if code == "SpellCheck" else "", diagnostic["message"], ())

    def close(self):
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                with self._lock:
                    self._request("shutdown", None, timeout=2.0)
                    self._notify("exit", None)
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
                proc.wait()
        shutil.rmtree(self._tmp, ignore_errors=True)


DIALECT_NAMES = {"american": ("American",), "british": ("British",), "both": ("American", "British")}
SPELL_RULE = "SpellCheck"


@dataclass(frozen=True)
class Issue:
    start: int
    end: int
    kind: str
    rule: str
    message: str
    suggestions: tuple[str, ...]
    word: str


def _merge_suggestions(lists):
    out = []
    seen = set()
    for suggestions in lists:
        for s in suggestions:
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return tuple(out)


class Corrector:
    """Harper decides what is wrong. "spelling" means Harper's SpellCheck
    rule, which flags one word missing from the dictionary; every other lint
    is "grammar". With dialect "both" an issue must appear in both runs, so
    a word correct in either spelling is accepted."""

    def __init__(self, harper, dialect, dictionary):
        self.harper = harper
        self.dialects = DIALECT_NAMES[dialect]
        self.dictionary = dictionary

    def check(self, text):
        runs = [self.harper.lint(text, d) for d in self.dialects]
        keyed = [{(l.start, l.end, l.rule): l for l in run} for run in runs]
        issues = []
        for key, first in keyed[0].items():
            if not all(key in k for k in keyed[1:]):
                continue
            word = text[first.start:first.end]
            kind = "spelling" if first.rule == SPELL_RULE else "grammar"
            if kind == "spelling" and word in self.dictionary:
                continue
            suggestions = _merge_suggestions(k[key].suggestions for k in keyed)
            issues.append(Issue(first.start, first.end, kind, first.rule, first.message, suggestions, word))
        issues.sort(key=lambda i: (i.start, i.end))
        return issues

    def warm(self):
        """harper-ls builds each dialect's linter on first use (~150 ms each);
        doing that at startup keeps it off the first completed sentence."""
        self.check("Warm up.")

    def is_valid(self, word):
        if word in self.dictionary:
            return True
        return not any(i.kind == "spelling" for i in self.check(word))


def damerau_levenshtein(a, b):
    """Unrestricted Damerau-Levenshtein distance, case-insensitive."""
    a, b = a.lower(), b.lower()
    inf = len(a) + len(b)
    last_row = {}
    d = [[inf] * (len(b) + 2) for _ in range(len(a) + 2)]
    for i in range(len(a) + 1):
        d[i + 1][0] = inf
        d[i + 1][1] = i
    for j in range(len(b) + 1):
        d[0][j + 1] = inf
        d[1][j + 1] = j
    for i in range(1, len(a) + 1):
        last_match_col = 0
        for j in range(1, len(b) + 1):
            i1 = last_row.get(b[j - 1], 0)
            j1 = last_match_col
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                last_match_col = j
            d[i + 1][j + 1] = min(
                d[i][j] + cost,
                d[i + 1][j] + 1,
                d[i][j + 1] + 1,
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        last_row[a[i - 1]] = i
    return d[len(a) + 1][len(b) + 1]


def clean_sentence(text, corrector, tally, candidate_min_count, max_edit_distance, ts):
    """Count this sentence's flagged words, then return its cleaned and
    redacted text (§8, §9). Redaction spans come from the raw sentence; an
    issue touching one is neither counted nor corrected, and placeholders go
    in last so a correction can never resurrect what was redacted."""
    spans = redact.find(text)

    def redacted(i):
        return any(i.start < e and s < i.end for s, e, _ in spans)

    issues = [i for i in corrector.check(text) if i.kind == "spelling" and not redacted(i)]
    for i in issues:
        tally.increment(i.word, ts)

    edits = [(s, e, redact.placeholder(label)) for s, e, label in spans]
    for i in issues:
        if i.word in tally.dictionary:
            continue
        if tally.count(i.word) >= candidate_min_count:
            continue
        if i.suggestions and damerau_levenshtein(i.word, i.suggestions[0]) <= max_edit_distance:
            edits.append((i.start, i.end, i.suggestions[0]))
    edits.sort()
    out = []
    pos = 0
    for s, e, replacement in edits:
        out.append(text[pos:s])
        out.append(replacement)
        pos = e
    out.append(text[pos:])
    return "".join(out)
