import json
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

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

    def lint(self, text, dialect):
        with self._lock:
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
        raise HarperUnavailable(f"no lint details for diagnostic {diagnostic}")

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
