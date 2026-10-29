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
            by_file.setdefault(path, []).append(json.dumps(record, ensure_ascii=False))
        for path, lines in by_file.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write("".join(line + "\n" for line in lines))
                f.flush()
                os.fsync(f.fileno())

    def _run(self):
        while not self._stop.wait(self.flush_interval_s):
            self.flush()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.flush()


class SentenceEvents:
    """Raw events held back until their sentence completes, so nothing is
    written before redaction. Redaction runs on every edit: PII typed and
    then deleted still marks the sentence, because its keystrokes would
    otherwise reach the log one `key` event at a time."""

    def __init__(self):
        self.events = []
        self.labels_seen = set()

    def add(self, record):
        self.events.append(record)

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
            for record in self.events:
                writer.queue("raw", record)
        self.events = []
        self.labels_seen = set()
