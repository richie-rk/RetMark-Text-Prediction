from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pk.config import load_config
from pk.engine.model import sequence_loss, step
from pk.engine.state import zero_state
from pk.graph import WordGraph
from pk.persistence import Workspace
from pk.predictor import Predictor
from pk.session import OnlineTrainer
from pk.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "my name is richy koshy."


@pytest.fixture(scope="module")
def tok():
    cfg = load_config(ROOT / "config.toml")
    return Tokenizer(cfg.model.boundary_chars)


def make(tmp_path, tok, **train):
    cfg = load_config(ROOT / "config.toml")
    cfg = replace(cfg, paths=replace(cfg.paths, data_dir=tmp_path), train=replace(cfg.train, **train))
    ws = Workspace.load(cfg, tok.vocab_size)
    return cfg, ws, OnlineTrainer(ws, tok, cfg, background=False)


def sentence_loss(cfg, tok, params, text):
    ids = tok.encode(text)
    flags = tok.token_boundary_flags(text, ids, at_end=True)
    m = cfg.model
    loss, _, _ = sequence_loss(params, ids, flags, zero_state(m.d_model), m.fast_decay, m.boundary_decay)
    return loss


def engine_primary(cfg, tok, params, text):
    chunks = tok.chunks(text)
    committed, partial = chunks[:-1], chunks[-1]
    m = cfg.model
    state = zero_state(m.d_model)
    for chunk in committed:
        for t in tok.encode(chunk):
            state = step(params, state, t, False, m.fast_decay, m.boundary_decay)
    words = Predictor(tok, WordGraph(m.boundary_chars), cfg).engine_words(params, state, partial, "")
    return words[0] if words else None


def test_online_training_learns_a_repeated_sentence(tmp_path, tok):
    # Q3 (B): one sentence per update gives 200 Adam steps at the shipped
    # online_lr; measured straight through the online path, no Harper.
    cfg, ws, trainer = make(tmp_path, tok, online_update_sentences=1)
    before = sentence_loss(cfg, tok, ws.params, SENTENCE)
    assert engine_primary(cfg, tok, ws.params, "my name is richy ") != "koshy"
    for _ in range(200):
        trainer.add_sentence(SENTENCE)
    assert ws.adam.step_count == 200
    after = sentence_loss(cfg, tok, ws.params, SENTENCE)
    assert after <= before / 2, (before, after)
    assert engine_primary(cfg, tok, ws.params, "my name is richy ") == "koshy"


def test_update_triggers_on_sentence_count(tmp_path, tok):
    cfg, ws, trainer = make(tmp_path, tok)
    for _ in range(3):
        trainer.add_sentence("one two three.")
    assert ws.adam.step_count == 0
    trainer.add_sentence("one two three.")
    assert ws.adam.step_count == 1
    assert trainer.pending == []


def test_update_triggers_on_token_count(tmp_path, tok):
    cfg, ws, trainer = make(tmp_path, tok, online_update_max_tokens=10)
    trainer.add_sentence("one two three four five.")
    assert ws.adam.step_count == 0
    trainer.add_sentence(" six seven eight nine ten.")
    assert ws.adam.step_count == 1


def test_update_swaps_params_without_touching_old_arrays(tmp_path, tok):
    cfg, ws, trainer = make(tmp_path, tok, online_update_sentences=1)
    old = ws.params
    snapshot = {k: v.copy() for k, v in old.items()}
    trainer.add_sentence("hello there.")
    assert ws.params is not old
    for k in old:
        np.testing.assert_array_equal(old[k], snapshot[k])
    assert any(not np.array_equal(ws.params[k], old[k]) for k in old)
    # Adam counts loss targets: 3 tokens give 2 next-token targets.
    assert ws.adam.tokens == len(tok.encode("hello there.")) - 1


def test_training_state_carries_forward_with_boundary_decay(tmp_path, tok):
    cfg, ws, trainer = make(tmp_path, tok, online_update_sentences=1)
    assert float(np.abs(trainer.train_state.S_s).sum()) == 0.0
    trainer.add_sentence("hello there.")
    assert float(np.abs(trainer.train_state.S_s).sum()) > 0.0


def test_background_update_runs_off_the_calling_thread(tmp_path, tok):
    cfg = load_config(ROOT / "config.toml")
    cfg = replace(cfg, paths=replace(cfg.paths, data_dir=tmp_path), train=replace(cfg.train, online_update_sentences=1))
    ws = Workspace.load(cfg, tok.vocab_size)
    trainer = OnlineTrainer(ws, tok, cfg, background=True)
    import threading

    seen = []
    original = trainer._update

    def spy(batch):
        seen.append(threading.current_thread() is threading.main_thread())
        return original(batch)

    trainer._update = spy
    trainer.add_sentence("hello there.")
    trainer.close()
    assert seen == [False]
    assert ws.adam.step_count == 1
