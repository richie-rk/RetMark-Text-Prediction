import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pk import consolidate
from pk.config import load_config
from pk.persistence import InstanceLock, LockHeld, load_state, save_state

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 10, 7)


def cfg_for(tmp_path, **train):
    cfg = load_config(ROOT / "config.toml")
    train.setdefault("batch_size", 2)
    train.setdefault("bptt_window", 16)
    train.setdefault("consolidation_epochs", 1)
    return replace(cfg, paths=replace(cfg.paths, data_dir=tmp_path), train=replace(cfg.train, **train))


def write_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def clean(day, session, text, i=0):
    return {"ts": f"{day}T10:00:{i:02d}.000+05:30", "session": session, "source": "cli", "text": text}


def seed_logs(tmp_path):
    clean_dir = tmp_path / "logs" / "clean"
    raw_dir = tmp_path / "logs" / "raw"
    write_lines(clean_dir / "2026-10-05.jsonl", [clean("2026-10-05", "a", "The cat sat.", 0), clean("2026-10-05", "a", " The dog ran.", 1)])
    write_lines(clean_dir / "2026-10-06.jsonl", [clean("2026-10-06", "b", "Hello there.", 0)])
    write_lines(clean_dir / "2026-10-07.jsonl", [clean("2026-10-07", "c", "Today is here.", 0)])
    for day in ("2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07"):
        write_lines(raw_dir / f"{day}.jsonl", [{"ts": f"{day}T10:00:00.000+05:30", "session": "x", "source": "cli", "type": "pause"}])


def test_consolidation_trains_reports_and_second_run_is_empty(tmp_path):
    cfg = cfg_for(tmp_path)
    seed_logs(tmp_path)
    summary = consolidate.run(cfg, today=TODAY)
    assert summary["sentences"] == 4
    assert summary["tokens"] > 0
    assert summary["device"] in ("cpu", "cuda")
    assert summary["loss_after"] < summary["loss_before"]
    assert summary["duration_s"] >= 0
    assert (tmp_path / "model.safetensors").exists()
    again = consolidate.run(cfg, today=TODAY)
    assert again["sentences"] == 0
    assert again["nothing_pending"] is True


def test_rollover_moves_only_consumed_past_days(tmp_path):
    cfg = cfg_for(tmp_path)
    seed_logs(tmp_path)
    consolidate.run(cfg, today=TODAY)
    clean_left = sorted(p.name for p in (tmp_path / "logs" / "clean").glob("*.jsonl"))
    raw_left = sorted(p.name for p in (tmp_path / "logs" / "raw").glob("*.jsonl"))
    assert clean_left == ["2026-10-07.jsonl"]
    assert raw_left == ["2026-10-07.jsonl"]
    assert sorted(p.name for p in (tmp_path / "archive" / "clean").glob("*")) == ["2026-10-05.jsonl", "2026-10-06.jsonl"]
    assert sorted(p.name for p in (tmp_path / "archive" / "raw").glob("*")) == ["2026-10-04.jsonl", "2026-10-05.jsonl", "2026-10-06.jsonl"]
    state = load_state(tmp_path)
    assert set(state.offsets) == {"2026-10-07.jsonl"}
    # Today's file keeps its offset; lines appended later become pending.
    write_lines(tmp_path / "logs" / "clean" / "2026-10-07.jsonl", [clean("2026-10-07", "c", " More text.", 1)])
    assert consolidate.run(cfg, today=TODAY)["sentences"] == 1


def test_partially_consumed_past_day_stays(tmp_path):
    seed_logs(tmp_path)
    data = tmp_path
    size = (data / "logs" / "clean" / "2026-10-05.jsonl").stat().st_size
    offsets = {"2026-10-05.jsonl": size - 5, "2026-10-06.jsonl": (data / "logs" / "clean" / "2026-10-06.jsonl").stat().st_size}
    offsets = consolidate.rollover(data, offsets, TODAY)
    assert (data / "logs" / "clean" / "2026-10-05.jsonl").exists()
    assert (data / "logs" / "raw" / "2026-10-05.jsonl").exists()
    assert not (data / "logs" / "clean" / "2026-10-06.jsonl").exists()
    assert "2026-10-06.jsonl" not in offsets and "2026-10-05.jsonl" in offsets


def test_replay_sample_is_seeded_uniform_and_bounded(tmp_path):
    archive = [consolidate.Sentence(f"d{i}.jsonl", 0, f"s{i}", f"t{i}.", f"x{i}") for i in range(10)]
    a = consolidate.replay_sample(archive, 1.0, 3, np.random.default_rng(1234))
    b = consolidate.replay_sample(archive, 1.0, 3, np.random.default_rng(1234))
    assert a == b and len(a) == 3 and len(set(a)) == 3
    assert len(consolidate.replay_sample(archive, 1.0, 50, np.random.default_rng(0))) == 10
    assert consolidate.replay_sample([], 1.0, 5, np.random.default_rng(0)) == []
    assert consolidate.replay_sample(archive, 0.0, 5, np.random.default_rng(0)) == []


def test_replay_draws_from_archive(tmp_path):
    cfg = cfg_for(tmp_path)
    seed_logs(tmp_path)
    consolidate.run(cfg, today=TODAY)
    write_lines(tmp_path / "logs" / "clean" / "2026-10-07.jsonl", [clean("2026-10-07", "c", " More text.", 1)])
    summary = consolidate.run(cfg, today=TODAY)
    assert summary["sentences"] == 1
    assert summary["replay_sentences"] == 1


def test_sessions_group_in_order_and_concatenate_as_typed():
    s = consolidate.Sentence
    sentences = [s("2026-10-05.jsonl", 0, "a", "t", "One."), s("2026-10-05.jsonl", 1, "b", "t", "Other."), s("2026-10-06.jsonl", 0, "a", "t", " Two.")]
    assert consolidate.session_texts(sentences) == [("a", "One. Two."), ("b", "Other.")]


def test_windows_carry_targets_across_edges():
    ids = list(range(10))
    wins = consolidate.windows(ids, 4)
    # The last token has no successor, so it is never an input.
    assert [w[0] for w in wins] == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    assert [w[1] for w in wins] == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]


def test_refuses_while_harness_holds_the_lock(tmp_path):
    cfg = cfg_for(tmp_path)
    seed_logs(tmp_path)
    holder = InstanceLock(tmp_path).acquire()
    try:
        with pytest.raises(LockHeld):
            consolidate.run(cfg, today=TODAY)
    finally:
        holder.release()
    assert load_state(tmp_path).offsets == {}


def test_graph_is_not_touched(tmp_path):
    cfg = cfg_for(tmp_path)
    seed_logs(tmp_path)
    consolidate.run(cfg, today=TODAY)
    assert not (tmp_path / "graph.json").exists()


def test_numpy_and_cupy_agree(tmp_path):
    from pk.xp import consolidation_backend

    xp, device = consolidation_backend()
    if device != "cuda":
        pytest.skip("no CUDA device")
    cfg = cfg_for(tmp_path)
    from pk.engine.model import init_params
    from pk.tokenizer import Tokenizer

    tok = Tokenizer(cfg.model.boundary_chars)
    m = cfg.model
    params = init_params(m.d_model, m.mlp_hidden, tok.vocab_size, m.init_std, m.seed)
    text = "The cat sat on the mat. The dog ran far away! Did it?"
    ids = tok.encode(text)
    flags = tok.token_boundary_flags(text, ids, at_end=True)
    cpu = consolidate.loss_and_grads(params, ids[:-1], ids[1:], flags[:-1], cfg)
    gpu_params = {k: xp.asarray(v) for k, v in params.items()}
    gpu = consolidate.loss_and_grads(gpu_params, ids[:-1], ids[1:], flags[:-1], cfg)
    assert gpu[0] == pytest.approx(cpu[0], rel=1e-4)
    for k in params:
        np.testing.assert_allclose(xp.asnumpy(gpu[2][k]), cpu[2][k], rtol=1e-4, atol=1e-6)
    # A short job: the before and after losses agree as well.
    from pk.engine.optim import Adam

    t = cfg.train
    streams = [(ids, flags)]
    results = []
    for to in (lambda a: a, xp.asarray):
        p = {k: to(v) for k, v in params.items()}
        adam = Adam(p, t.adam_beta1, t.adam_beta2_half_life_tokens, t.adam_eps)
        before = consolidate.evaluate(p, streams, cfg)[0]
        p = consolidate.train(p, adam, streams, cfg, t.consolidation_lr, 1, np.random.default_rng(0))
        results.append((before, consolidate.evaluate(p, streams, cfg)[0]))
    assert results[1][0] == pytest.approx(results[0][0], rel=1e-4)
    assert results[1][1] == pytest.approx(results[0][1], rel=1e-4)
