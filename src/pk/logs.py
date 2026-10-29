import json
import os
import threading
from datetime import datetime
from pathlib import Path

from pk import redact

SOURCE = "cli"
STREAMS = ("raw", "clean")


def local_now():
    return datetime.now().astimezone()


def timestamp(dt):
    return dt.isoformat(timespec="milliseconds")


class LogWriter:
    """Buffered JSONL writer: lines queue in memory and a background thread
    appends them every `flush_interval_s`, and once more on close."""

    def __init__(self, data_dir, flush_interval_s, session, clock=local_now, background=True):
        self.root = Path(data_dir) / "logs"
        self.session = session
        self.clock = clock
        self.flush_interval_s = flush_interval_s
        self._pending = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        if background:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def record(self, type, **fields):
        return {"ts": timestamp(self.clock()), "session": self.session, "source": SOURCE, "type": type, **fields}

    def clean_record(self, text):
        return {"ts": timestamp(self.clock()), "session": self.session, "source": SOURCE, "text": text}

    def queue(self, stream, record):
        assert stream in STREAMS
        with self._lock:
            self._pending.append((stream, record))

    def flush(self):
        with self._lock:
            pending, self._pending = self._pending, []
        by_file = {}
        for stream, record in pending:
            # The ts carries its own local date, so a line lands in the file
            # of the day it happened even if it is flushed after midnight.
            path = self.root / stream / f"{record['ts'][:10]}.jsonl"
            by_file.setdefault(path, []).append((stream, record))
        done = set()
        try:
            for path, items in by_file.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                # A crash can leave a torn last line; end it first so it cannot
                # swallow the first line written now.
                torn = path.exists() and path.stat().st_size > 0 and not _ends_with_newline(path)
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(("\n" if torn else "") + "".join(json.dumps(r, ensure_ascii=False) + "\n" for _, r in items))
                    f.flush()
                    os.fsync(f.fileno())
                done.add(path)
        except Exception:
            # Keep unwritten lines for the next flush rather than lose them.
            with self._lock:
                self._pending[:0] = [item for path, items in by_file.items() if path not in done for item in items]
            raise

    def _run(self):
        while not self._stop.wait(self.flush_interval_s):
            try:
                self.flush()
            except Exception as e:  # retried next interval; shown in the toolbar
                self.error = e

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.flush()


def _ends_with_newline(path):
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def parse_line(line):
    """A clean-log record, or None for a line a crash left unreadable."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ("ts", "session", "text")):
        return None
    return record


class SentenceEvents:
    """Raw events held back until their sentence completes, so nothing is
    written before redaction. Redaction runs on every edit: PII typed and
    then deleted still marks the sentence, because its keystrokes would
    otherwise reach the log one `key` event at a time."""

    def __init__(self):
        self.events = []
        self.labels_seen = set()

    def add(self, record, tag=None):
        # `tag` names the character a key event typed, so the event can follow
        # that character if an edit moves it into a later sentence.
        self.events.append((record, tag))

    def split(self, later_tags):
        """Move out the events whose characters now sit after this sentence."""
        moved = SentenceEvents()
        moved.events = [(r, t) for r, t in self.events if t is not None and t in later_tags]
        self.events = [(r, t) for r, t in self.events if t is None or t not in later_tags]
        return moved

    def observe(self, text):
        for _, _, label in redact.find(text):
            self.labels_seen.add(label)

    def finish(self, final_text, writer):
        spans = redact.find(final_text)
        labels = self.labels_seen | {label for _, _, label in spans}
        if labels:
            ordered = [l for l in redact.LABELS if l in labels]
            writer.queue("raw", writer.record("redacted_sentence", labels=ordered, text=redact.apply(final_text, spans)))
        else:
            for record, _ in self.events:
                writer.queue("raw", record)
        self.events = []
        self.labels_seen = set()


def clean_dir(data_dir):
    return Path(data_dir) / "logs" / "clean"


def pending_clean(data_dir, offsets):
    """Complete clean-log lines past each file's consumed byte offset, today's
    file included: [(file name, start offset, end offset, record)]."""
    out = []
    folder = clean_dir(data_dir)
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.jsonl")):
        offset = offsets.get(path.name, 0)
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
        pos = offset
        # A line without its newline is still being written: leave it for later.
        for line in data.split(b"\n")[:-1]:
            end = pos + len(line) + 1
            record = parse_line(line) if line.strip() else None
            if record is not None:
                out.append((path.name, pos, end, record))
            pos = end
    return out
