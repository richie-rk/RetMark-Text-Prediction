import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from pk.logs import LogWriter, SentenceEvents, timestamp
from pk.persistence import load_state, save_state
from pk.session import Recorder

IST = timezone(timedelta(hours=5, minutes=30))


class Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock(datetime(2026, 10, 5, 23, 59, 59, 500000, tzinfo=IST))


@pytest.fixture
def writer(tmp_path, clock):
    w = LogWriter(tmp_path, flush_interval_s=3600, session="sess-1", clock=clock, background=False)
    yield w
    w.close()


def lines(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def raw_file(tmp_path, day="2026-10-05"):
    return tmp_path / "logs" / "raw" / f"{day}.jsonl"


def clean_file(tmp_path, day="2026-10-05"):
    return tmp_path / "logs" / "clean" / f"{day}.jsonl"


def test_timestamp_format(clock):
    assert timestamp(clock()) == "2026-10-05T23:59:59.500+05:30"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d", timestamp(clock()))


def test_nothing_reaches_disk_before_flush(tmp_path, writer):
    writer.queue("raw", writer.record("key", key="a"))
    assert not raw_file(tmp_path).exists()
    writer.flush()
    [line] = lines(raw_file(tmp_path))
    assert line == {"ts": "2026-10-05T23:59:59.500+05:30", "session": "sess-1", "source": "cli", "type": "key", "key": "a"}


def test_flush_appends_and_fsyncs(tmp_path, writer, monkeypatch):
    synced = []
    import pk.logs

    monkeypatch.setattr(pk.logs.os, "fsync", lambda fd: synced.append(fd))
    writer.queue("raw", writer.record("key", key="a"))
    writer.flush()
    writer.queue("raw", writer.record("key", key="b"))
    writer.flush()
    assert [l["key"] for l in lines(raw_file(tmp_path))] == ["a", "b"]
    assert len(synced) == 2


def test_one_file_per_local_day(tmp_path, writer, clock):
    writer.queue("clean", writer.clean_record("first."))
    clock.now += timedelta(seconds=1)
    writer.queue("clean", writer.clean_record(" second."))
    writer.flush()
    assert [l["text"] for l in lines(clean_file(tmp_path))] == ["first."]
    assert [l["text"] for l in lines(clean_file(tmp_path, "2026-10-06"))] == [" second."]
    assert set(lines(clean_file(tmp_path))[0]) == {"ts", "session", "source", "text"}


def test_background_thread_flushes_on_interval_and_on_close(tmp_path, clock):
    w = LogWriter(tmp_path, flush_interval_s=0.05, session="s", clock=clock)
    w.queue("raw", w.record("key", key="a"))
    import time

    deadline = time.time() + 5
    while not raw_file(tmp_path).exists() and time.time() < deadline:
        time.sleep(0.01)
    assert len(lines(raw_file(tmp_path))) == 1
    w.queue("raw", w.record("key", key="b"))
    w.close()
    assert len(lines(raw_file(tmp_path))) == 2


def test_clean_sentence_events_are_queued(tmp_path, writer):
    events = SentenceEvents()
    for c in "hi.":
        events.add(writer.record("key", key=c))
        events.observe("hi."[: "hi.".index(c) + 1])
    events.finish("hi.", writer)
    writer.flush()
    assert [l["key"] for l in lines(raw_file(tmp_path))] == ["h", "i", "."]


def test_redacted_sentence_replaces_its_events(tmp_path, writer):
    events = SentenceEvents()
    text = "pan abcde1234f ok."
    for i, c in enumerate(text):
        events.add(writer.record("key", key=c))
        events.observe(text[: i + 1])
    events.finish(text, writer)
    writer.flush()
    [line] = lines(raw_file(tmp_path))
    assert line["type"] == "redacted_sentence"
    assert line["labels"] == ["PAN"]
    assert line["text"] == "pan [REDACTED-PAN] ok."


def test_pii_typed_then_deleted_still_redacts(tmp_path, writer):
    events = SentenceEvents()
    events.observe("card 4111111111111111")
    events.observe("card ")
    events.observe("card gone.")
    events.add(writer.record("key", key="4"))
    events.finish("card gone.", writer)
    writer.flush()
    [line] = lines(raw_file(tmp_path))
    assert line["type"] == "redacted_sentence"
    assert line["labels"] == ["CARD"]
    assert line["text"] == "card gone."


def test_recorder_writes_nothing_while_paused_except_marker(tmp_path, writer):
    rec = Recorder(writer, paused=False)
    rec.key("a")
    rec.observe("a")
    rec.toggle_pause("a")
    rec.key("b")
    rec.special("backspace")
    rec.observe("ab")
    assert rec.complete_sentence("ab.", "ab.") is False
    writer.flush()
    types = [(l["type"], l.get("key")) for l in lines(raw_file(tmp_path))]
    assert types == [("key", "a"), ("pause", None)]
    assert lines(clean_file(tmp_path)) == []


def test_recorder_resume_marker_and_learning(tmp_path, writer):
    rec = Recorder(writer, paused=True)
    rec.key("x")
    rec.toggle_pause("x")
    rec.key("h")
    rec.observe("h")
    assert rec.complete_sentence("h.", "h.") is True
    writer.flush()
    assert [l["type"] for l in lines(raw_file(tmp_path))] == ["resume", "key"]
    assert [l["text"] for l in lines(clean_file(tmp_path))] == ["h."]


def test_toggle_ends_the_sentence_unlearned(tmp_path, writer):
    rec = Recorder(writer, paused=False)
    rec.key("h")
    rec.observe("h")
    rec.toggle_pause("h")
    rec.toggle_pause("")
    rec.key("i")
    rec.observe("i")
    rec.complete_sentence("i.", "i.")
    writer.flush()
    assert [l["type"] for l in lines(raw_file(tmp_path))] == ["key", "pause", "resume", "key"]
    assert [l["text"] for l in lines(clean_file(tmp_path))] == ["i."]


def test_pause_survives_restart(tmp_path):
    state = load_state(tmp_path)
    assert state.paused is False
    state.paused = True
    save_state(tmp_path, state)
    assert load_state(tmp_path).paused is True
    raw = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert raw["paused"] is True
    assert "format_version" in raw
