import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pk.engine.model import PARAM_NAMES, init_params
from pk.engine.optim import Adam
from pk.graph import WordGraph
from pk.persistence import (
    InstanceLock,
    LockHeld,
    ModelMismatch,
    atomic_write_bytes,
    load_graph,
    load_model,
    load_state,
    save_graph,
    save_model,
    save_state,
)

D, H, V = 8, 16, 50


def trained():
    params = init_params(D, H, V, std=0.02, seed=1)
    adam = Adam(params, beta1=0.9, half_life_tokens=1000, eps=1e-8)
    rng = np.random.default_rng(0)
    for n in (5, 17):
        grads = {k: rng.normal(size=p.shape).astype(np.float32) for k, p in params.items()}
        params = adam.step(params, grads, lr=1e-3, tokens_in_step=n)
    return params, adam


def test_model_save_then_load_is_bitwise_equal(tmp_path):
    params, adam = trained()
    save_model(tmp_path, params, adam)
    fresh = Adam(params, beta1=0.9, half_life_tokens=1000, eps=1e-8)
    loaded = load_model(tmp_path, D, H, fresh)
    for name in PARAM_NAMES:
        assert loaded[name].dtype == params[name].dtype
        assert loaded[name].tobytes() == params[name].tobytes()
        assert fresh.m[name].tobytes() == adam.m[name].tobytes()
        assert fresh.v[name].tobytes() == adam.v[name].tobytes()
    assert fresh.step_count == 2
    assert fresh.tokens == 22
    assert fresh.half_lives == adam.half_lives


def test_model_metadata(tmp_path):
    from safetensors import safe_open

    params, adam = trained()
    save_model(tmp_path, params, adam)
    with safe_open(str(tmp_path / "model.safetensors"), framework="np") as f:
        meta = f.metadata()
        keys = set(f.keys())
    assert meta == {"format_version": "1", "d_model": str(D), "mlp_hidden": str(H), "tokenizer": "gpt2"}
    assert set(PARAM_NAMES) <= keys
    assert {"adam_step", "adam_half_lives", "adam_tokens"} <= keys


@pytest.mark.parametrize("d,h", [(16, 16), (8, 32)])
def test_config_mismatch_refuses_to_load(tmp_path, d, h):
    params, adam = trained()
    save_model(tmp_path, params, adam)
    with pytest.raises(ModelMismatch) as e:
        load_model(tmp_path, d, h, Adam(init_params(d, h, V, 0.02, 1), 0.9, 1000, 1e-8))
    assert str(D) in str(e.value) and str(d) in str(e.value)


def test_missing_model_returns_none(tmp_path):
    assert load_model(tmp_path, D, H, None) is None


def test_interrupted_write_leaves_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_bytes(path, b"old contents")

    import pk.persistence

    def crash(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(pk.persistence.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(path, b"new contents")
    assert path.read_bytes() == b"old contents"


def test_interrupted_model_save_leaves_previous_model_loadable(tmp_path, monkeypatch):
    params, adam = trained()
    save_model(tmp_path, params, adam)
    before = (tmp_path / "model.safetensors").read_bytes()
    import pk.persistence

    real_fsync = os.fsync

    def die(fd):
        real_fsync(fd)
        raise OSError("power cut")

    monkeypatch.setattr(pk.persistence.os, "fsync", die)
    with pytest.raises(OSError):
        save_model(tmp_path, init_params(D, H, V, 0.02, 9), adam)
    assert (tmp_path / "model.safetensors").read_bytes() == before


def test_graph_round_trip(tmp_path):
    g = WordGraph((".",))
    g.update("Hello world. hello there.")
    save_graph(tmp_path, g)
    raw = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert raw["format_version"] == 1
    loaded = load_graph(tmp_path, (".",))
    assert (loaded.edges, loaded.totals, loaded.surface) == (g.edges, g.totals, g.surface)
    assert load_graph(tmp_path / "nope", (".",)).edges == {}


def test_state_round_trip(tmp_path):
    s = load_state(tmp_path)
    s.paused = True
    s.offsets = {"2026-10-05.jsonl": 123}
    s.handover = [[True, False], [False, True]]
    s.last_checkpoint = "2026-10-05T10:00:00.000+05:30"
    save_state(tmp_path, s)
    assert load_state(tmp_path) == s


LOCK_HOLDER = """
import sys, time
from pk.persistence import InstanceLock
lock = InstanceLock(sys.argv[1])
lock.acquire()
print("held", flush=True)
time.sleep(30)
"""


def test_lock_is_exclusive_across_processes(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        with pytest.raises(LockHeld):
            InstanceLock(tmp_path).acquire()
    finally:
        proc.kill()
        proc.wait()
    lock = InstanceLock(tmp_path)
    lock.acquire()
    lock.release()
    assert (tmp_path / ".lock").exists()
    assert (tmp_path / ".lock").stat().st_size == 0


def test_lock_released_by_holder_can_be_retaken(tmp_path):
    a = InstanceLock(tmp_path)
    a.acquire()
    b = InstanceLock(tmp_path)
    with pytest.raises(LockHeld):
        b.acquire()
    a.release()
    b.acquire()
    b.release()


def test_workspace_first_launch_then_checkpoint_round_trip(tmp_path):
    from dataclasses import replace

    from pk.config import load_config
    from pk.persistence import Workspace

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config.toml")
    cfg = replace(cfg, paths=replace(cfg.paths, data_dir=tmp_path))
    ws = Workspace.load(cfg, vocab_size=V)
    # First launch: fresh seeded weights, empty graph, default state; nothing
    # is written until a checkpoint.
    assert ws.params["E"].shape == (V, cfg.model.d_model)
    assert ws.graph.edges == {} and ws.state.paused is False
    assert not (tmp_path / "model.safetensors").exists()
    ws.graph.update("hello world.")
    ws.dictionary.add("Koshy")
    ws.tally.increment("richy", "ts")
    ws.state.paused = True
    ws.checkpoint("2026-10-05T10:00:00.000+05:30")
    for name in ("model.safetensors", "graph.json", "state.json", "dictionary.txt", "rejected.txt", "candidates.json"):
        assert (tmp_path / name).exists(), name
    again = Workspace.load(cfg, vocab_size=V)
    assert again.params["E"].tobytes() == ws.params["E"].tobytes()
    assert again.graph.edges == ws.graph.edges
    assert again.state.paused is True
    assert again.state.last_checkpoint == "2026-10-05T10:00:00.000+05:30"
    assert "koshy" in again.dictionary
    assert again.tally.count("richy") == 1
