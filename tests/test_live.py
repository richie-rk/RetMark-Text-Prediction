import json
from dataclasses import replace
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pk.cli.live import render_inspect, run_live, toolbar_fragments
from pk.config import load_config
from pk.persistence import InstanceLock, load_state

ROOT = Path(__file__).resolve().parent.parent


def cfg_for(tmp_path):
    cfg = load_config(ROOT / "config.toml")
    return replace(cfg, paths=replace(cfg.paths, data_dir=tmp_path))


def drive(cfg, keys, inspect=False):
    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return run_live(cfg, inspect=inspect)


def clean_lines(tmp_path):
    folder = tmp_path / "logs" / "clean"
    return [json.loads(l)["text"] for p in sorted(folder.glob("*.jsonl")) for l in p.read_text(encoding="utf-8").splitlines()]


def test_harness_types_learns_checkpoints_and_releases_the_lock(tmp_path):
    cfg = cfg_for(tmp_path)
    assert drive(cfg, "Hello there. How are\x04", inspect=True) == 0
    assert clean_lines(tmp_path) == ["Hello there."]
    for name in ("model.safetensors", "graph.json", "state.json", "candidates.json"):
        assert (tmp_path / name).exists(), name
    raw = [json.loads(l) for p in (tmp_path / "logs" / "raw").glob("*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()]
    assert [r["key"] for r in raw if r["type"] == "key"] == list("Hello there. How are")
    assert len({r["session"] for r in raw}) == 1
    lock = InstanceLock(tmp_path)
    lock.acquire()
    lock.release()


def test_harness_enter_and_ctrl_c(tmp_path):
    cfg = cfg_for(tmp_path)
    assert drive(cfg, "First line\rsecond\x03") == 0
    assert clean_lines(tmp_path) == ["First line\n"]


def test_harness_pause_persists(tmp_path):
    cfg = cfg_for(tmp_path)
    assert drive(cfg, "secret words. \x10hidden. \x04") == 0
    assert load_state(tmp_path).paused is True
    assert clean_lines(tmp_path) == ["secret words."]
    drive(cfg, "more hidden. \x04")
    assert clean_lines(tmp_path) == ["secret words."]


def test_harness_refuses_when_locked(tmp_path):
    from pk.persistence import LockHeld

    import pytest

    cfg = cfg_for(tmp_path)
    holder = InstanceLock(tmp_path).acquire()
    try:
        with pytest.raises(LockHeld):
            drive(cfg, "\x04")
    finally:
        holder.release()


def test_toolbar_and_inspect_render(tmp_path):
    from pk.logs import LogWriter
    from pk.persistence import Workspace
    from pk.session import LiveSession
    from pk.tokenizer import Tokenizer

    cfg = cfg_for(tmp_path)
    tok = Tokenizer(cfg.model.boundary_chars)
    ws = Workspace.load(cfg, tok.vocab_size)

    class NoCorrector:
        def check(self, text):
            return []

    s = LiveSession(cfg, tok, ws, NoCorrector(), None, None, learn=False)
    s.type("Hello there. How")
    text = "".join(t for _, t in toolbar_fragments(s))
    assert "GRAPH" in text and "● RECORDING" in text and "graph:" in text
    out = render_inspect(s.inspect(), 100)
    assert "committed tokens" in out and "‖S slow‖" in out
