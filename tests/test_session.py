import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import regex

from pk.config import load_config
from pk.corrector import Issue
from pk.engine.model import step
from pk.engine.state import zero_state
from pk.logs import LogWriter
from pk.persistence import Workspace
from pk.predictor import ENGINE, GRAPH
from pk.session import LiveSession, OnlineTrainer
from pk.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture(scope="module")
def tok():
    return Tokenizer(load_config(ROOT / "config.toml").model.boundary_chars)


class FakeCorrector:
    def __init__(self, misspelt=None):
        self.misspelt = misspelt or {}
        self.dictionary = None

    def check(self, text):
        issues = []
        for m in regex.finditer(r"[\p{L}']+", text):
            w = m.group()
            if w.lower() in self.misspelt and w not in self.dictionary:
                issues.append(Issue(m.start(), m.end(), "spelling", "SpellCheck", "m", tuple(self.misspelt[w.lower()]), w))
        return issues


class Env:
    def __init__(self, tmp_path, tok, misspelt=None, paused=False, **train):
        cfg = load_config(ROOT / "config.toml")
        train.setdefault("online_update_sentences", 1000)
        self.cfg = replace(cfg, paths=replace(cfg.paths, data_dir=tmp_path), train=replace(cfg.train, **train))
        self.tmp = tmp_path
        self.tok = tok
        self.ws = Workspace.load(self.cfg, tok.vocab_size)
        self.ws.state.paused = paused
        self.clock = lambda: datetime(2026, 10, 5, 12, 0, 0, tzinfo=IST)
        self.writer = LogWriter(tmp_path, 3600, "sess", clock=self.clock, background=False)
        self.trainer = OnlineTrainer(self.ws, tok, self.cfg, background=False)
        self.learned = []
        original = self.trainer.add_sentence
        self.trainer.add_sentence = lambda text: (self.learned.append(text), original(text))
        self.corrector = FakeCorrector(misspelt)
        self.corrector.dictionary = self.ws.dictionary
        self.restart()

    def restart(self):
        self.s = LiveSession(self.cfg, self.tok, self.ws, self.corrector, self.writer, self.trainer, clock=self.clock)

    def raw(self):
        self.writer.flush()
        p = self.tmp / "logs" / "raw" / "2026-10-05.jsonl"
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def clean(self):
        self.writer.flush()
        p = self.tmp / "logs" / "clean" / "2026-10-05.jsonl"
        return [json.loads(l)["text"] for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


@pytest.fixture
def env(tmp_path, tok):
    return Env(tmp_path, tok)


def fresh_state(env, text):
    """State after the committed chunks of `text`, computed from scratch."""
    m = env.cfg.model
    chunks = env.tok.chunks(text)
    committed = "".join(chunks[:-1])
    ids = env.tok.encode(committed)
    flags = env.tok.token_boundary_flags(text, ids, at_end=False)
    s = zero_state(m.d_model)
    for t, f in zip(ids, flags):
        s = step(env.ws.params, s, t, f, m.fast_decay, m.boundary_decay)
    return s


def assert_same_state(a, b):
    for name in ("S_s", "z_s", "S_f", "z_f"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-5, atol=1e-7)


def test_sentence_completes_when_its_boundary_chunk_commits(env):
    env.s.type("Hello there.")
    assert env.learned == []
    env.s.type(" ")
    assert env.learned == ["Hello there."]
    assert env.ws.graph.edges["hello"] == {"there": 1}
    assert env.clean() == ["Hello there."]


def test_sentences_are_stored_as_typed(env):
    env.s.type("Hello there. How are you? Fine")
    assert env.learned == ["Hello there.", " How are you?"]


def test_mid_word_marks_do_not_end_sentences(env):
    env.s.type("pi is 3.5 and see claude.ai today ")
    assert env.learned == []
    assert not env.s.boundary_fired


def test_enter_commits_the_newline_and_freezes_the_line(env):
    env.s.type("Hi there")
    env.s.enter()
    assert env.learned == ["Hi there\n"]
    assert env.s.boundary_fired
    assert env.s.partial == ""
    env.s.backspace()
    assert env.s.text == "Hi there\n"
    env.s.left()
    assert env.s.cursor == len("Hi there\n")


def test_wordless_pieces_carry_into_the_next_sentence(env):
    env.s.type("Hi.")
    env.s.enter()
    env.s.enter()
    env.s.type("Next. ")
    assert env.learned == ["Hi.", "\n\nNext."]


def test_state_follows_backspace_and_rechunking(env):
    env.s.type("hello wrold")
    for _ in range(5):
        env.s.backspace()
    env.s.type("world we'l")
    env.s.type("l be ")
    assert env.s.text == "hello world we'll be "
    assert_same_state(env.s.live_state(), fresh_state(env, env.s.text))


def test_backspace_stops_at_the_last_completed_sentence(env):
    env.s.type("One. Two")
    for _ in range(10):
        env.s.backspace()
    assert env.s.text == "One."
    assert env.learned == ["One."]


def test_backspace_uncommits_and_recommits_cleanly(env):
    env.s.type("hello ")
    env.s.backspace()
    env.s.type(" ")
    assert_same_state(env.s.live_state(), fresh_state(env, "hello "))
    assert len(env.s.handover.records) == 1


def test_left_right_move_within_the_editable_region(env):
    env.s.type("ab")
    env.s.left()
    env.s.type("x")
    assert env.s.text == "axb"
    env.s.right()
    assert env.s.cursor == 3
    env.s.right()
    assert env.s.cursor == 3


def test_correction_slot_shows_and_clears_on_next_word(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"thier": ["their", "there"]})
    env.s.type("I saw thier")
    assert env.s.correction is None
    env.s.type(" ")
    assert (env.s.correction.typed, env.s.correction.suggestion) == ("thier", "their")
    env.s.type("car ")
    assert env.s.correction is None


def test_insert_applies_the_correction(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"thier": ["their"]})
    env.s.type("I saw thier ")
    env.s.insert()
    assert env.s.text == "I saw their "
    assert env.s.correction is None
    env.s.type("car. ")
    raw = env.raw()
    assert {"type": "correct", "typed": "thier", "fixed": "their"}.items() <= next(r for r in raw if r["type"] == "correct").items()
    assert env.learned == ["I saw their car."]
    assert_same_state(env.s.live_state(), fresh_state(env, env.s.text))


def test_f2_claims_the_word(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"koshy": ["kohl"]})
    env.s.type("hi koshy ")
    env.s.f2()
    assert "koshy" in env.ws.dictionary
    assert env.s.correction is None
    env.s.type("ok. ")
    assert any(r["type"] == "dict_add" and r["word"] == "koshy" for r in env.raw())
    assert env.learned == ["hi koshy ok."]


def test_insert_and_f2_do_nothing_without_a_slot(env):
    env.s.type("hello ")
    env.s.insert()
    env.s.f2()
    assert env.s.text == "hello "
    assert len(env.ws.dictionary) == 0


def test_keep_typing_never_approves_and_counts_at_completion(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"thier": ["their"]})
    env.s.type("thier car. ")
    assert "thier" not in env.ws.dictionary
    assert env.ws.tally.count("thier") == 1
    assert env.learned == ["their car."]


def test_sentence_waits_while_the_slot_points_into_it(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr. ")
    assert env.s.correction is not None
    assert env.learned == []
    env.s.insert()
    assert env.s.text == "I like color. "
    assert env.learned == ["I like color."]


def test_waiting_sentence_completes_on_next_word(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr. Next ")
    assert env.s.correction is None
    assert env.learned == ["I like color."]
    # Backspace cannot reach into the waiting sentence.
    env2 = Env(tmp_path / "b", tok, misspelt={"colr": ["color"]})
    env2.s.type("I like colr. ")
    for _ in range(5):
        env2.s.backspace()
    assert env2.s.text == "I like colr."


def test_tab_accepts_primary_with_a_space(env):
    env.ws.graph.update("hello world.")
    env.s.type("hello ")
    assert env.s.prediction.slots.primary.word == "world"
    env.s.tab()
    assert env.s.text == "hello world "
    assert env.raw() == []  # nothing written before the sentence completes
    env.s.type("again. ")
    accept = [r for r in env.raw() if r["type"] == "accept"]
    assert accept and accept[0]["slot"] == "primary" and accept[0]["origin"] == "graph" and accept[0]["word"] == "world"


def test_tab_replaces_the_typed_partial(env):
    env.ws.graph.update("hello world.")
    env.s.type("hello Wo")
    env.s.tab()
    assert env.s.text == "hello World "


def test_tab_without_primary_inserts_a_tab(env):
    env.s.type("zzz.")
    env.s.tab()
    assert env.s.text == "zzz.\t"


def test_handover_records_word_commits(env):
    env.ws.graph.update("hello world.")
    env.restart()
    env.s.type("hello world ")
    # Both words were in the graph's top 3 when they started.
    assert [g for _, g in env.s.handover.records] == [True, True]
    assert env.s.mode == GRAPH


def test_nothing_written_or_learned_while_paused(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"thier": ["their"]}, paused=True)
    env.s.type("I saw thier car. Then more. ")
    env.s.close()
    assert env.learned == []
    assert env.ws.graph.edges == {}
    assert env.ws.tally.count("thier") == 0
    assert len(env.s.handover.records) == 0
    assert env.raw() == []
    assert env.clean() == []
    # Predictions and corrections keep working.
    assert env.s.prediction is not None


def test_pause_toggle_ends_the_sentence_and_resume_learns_again(env):
    env.s.type("half a sent")
    env.s.toggle_pause()
    assert env.ws.state.paused is True
    env.s.type("ence typed while paused. ")
    env.s.toggle_pause()
    env.s.type("Fresh start. ")
    assert env.learned == ["Fresh start."]
    types = [r["type"] for r in env.raw()]
    assert types.count("pause") == 1 and types.count("resume") == 1
    assert "redacted_sentence" not in types


def test_close_completes_a_trailing_sentence(env):
    env.s.type("Hello there.")
    env.s.close()
    assert env.learned == ["Hello there."]


def test_close_logs_unfinished_sentence_without_learning(env):
    env.s.type("unfinished thought")
    env.s.close()
    assert env.learned == []
    assert [r["key"] for r in env.raw() if r["type"] == "key"] == list("unfinished thought")


def test_redacted_sentence_never_reaches_learners_unredacted(env):
    env.s.type("my pan is abcde1234f ok. ")
    assert env.learned == ["my pan is [REDACTED-PAN] ok."]
    [r] = [r for r in env.raw() if r["type"] == "redacted_sentence"]
    assert r["labels"] == ["PAN"]
    assert not any(r["type"] == "key" for r in env.raw())
    assert all("abcde" not in w for w in env.ws.graph.surface)


def test_inspect_reports_engine_internals(env):
    env.s.type("Hello there. How")
    info = env.s.inspect()
    assert [t for t, _ in info["tokens"]][-3:] == ["Hello", " there", "."]
    assert all(isinstance(i, int) for _, i in info["tokens"])
    assert info["S_s_norm"] >= 0 and info["S_f_norm"] >= 0
    assert len(info["top"]) == 10
    assert abs(sum(p for _, p in info["top"])) <= 1.0
    assert 0.0 <= info["mlp_frac_pos"] <= 1.0
    assert isinstance(info["boundary_fired"], bool)


def test_engine_mode_after_window(tmp_path, tok):
    env = Env(tmp_path, tok)
    from pk.predictor import Handover

    env.s.handover = Handover(env.cfg.handover.window, [(True, False)] * env.cfg.handover.window)
    env.s.type("hello ")
    assert env.s.mode == ENGINE
    assert env.s.prediction.slots.primary.origin == "engine"


# --- regressions from review -------------------------------------------------


def keys_typed(env):
    return "".join(r["key"] for r in env.raw() if r["type"] == "key")


def test_typing_at_a_waiting_sentence_end_starts_a_new_sentence(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr. ")
    env.s.backspace()
    env.s.type("Then we left. ")
    assert env.learned == ["I like color.", "Then we left."]
    env2 = Env(tmp_path / "b", tok, misspelt={"colr": ["color"]})
    env2.s.type("I like colr. ")
    env2.s.left()
    env2.s.type("Ok")
    env2.s.close()
    assert "I like color.Ok" not in env2.learned
    assert env2.learned[0] == "I like color."


def test_enter_with_a_waiting_sentence_never_leaks_pii(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr.")
    env.s.enter()
    env.s.type("98765 ")
    env.s.type("43210 ok. ")
    env.s.close()
    assert not any(c.isdigit() for c in keys_typed(env))
    assert any(r["type"] == "redacted_sentence" for r in env.raw())


def test_enter_mid_line_never_leaks_pii(env):
    env.s.type("my pan isabcde1234f")
    for _ in range(10):
        env.s.left()
    env.s.enter()
    env.s.close()
    assert "abcde1234f" not in keys_typed(env)
    assert "1234" not in keys_typed(env)


def test_pause_while_a_sentence_waits_finishes_it_first(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr. Mo")
    env.s.toggle_pause()
    assert env.learned == ["I like color."]
    types = [r["type"] for r in env.raw()]
    assert types[-1] == "pause"


def test_waiting_sentence_from_pause_is_never_learned(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]}, paused=True)
    env.s.type("I like colr. secr")
    env.s.toggle_pause()
    env.s.type("et and more. Second one. Third. ")
    env.s.close()
    assert not any("colr" in t or "color" in t or "secr" in t for t in env.learned)
    assert env.learned == ["et and more.", " Second one.", " Third."]
    # Nothing moved final_pos backwards into the paused text.
    assert env.s.final_pos >= len("I like colr. secr")


def test_paused_text_after_the_cursor_is_never_learned(tmp_path, tok):
    env = Env(tmp_path, tok, paused=True)
    env.s.type("private notes here")
    for _ in range(10):
        env.s.left()
    env.s.toggle_pause()
    for _ in range(10):
        env.s.right()
    env.s.type(". Ok. ")
    env.s.close()
    assert not any("notes" in t or "here" in t for t in env.learned)
    assert "notes" not in env.ws.graph.surface


def test_waiting_sentence_keeps_its_handover_records(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr. ")
    assert len(env.s.handover.records) == 3


@pytest.mark.parametrize("typed,fixed", [("colr", "color"), ("thiss", "this")])
def test_insert_keeps_the_finished_line_read_only(tmp_path, tok, typed, fixed):
    env = Env(tmp_path, tok, misspelt={typed: [fixed]})
    env.s.type(f"I like {typed}.")
    env.s.enter()
    env.s.insert()
    assert env.s.text == f"I like {fixed}.\n"
    env.s.backspace()
    assert env.s.text == f"I like {fixed}.\n"
    env.s.type("ab")
    env.s.backspace()
    env.s.backspace()
    assert env.s.text == f"I like {fixed}.\n"


def test_last_word_before_enter_commits(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s.type("I like colr")
    env.s.enter()
    assert env.s.correction is not None and env.s.correction.typed == "colr"
    assert len(env.s.handover.records) == 3


def test_tab_after_a_typed_partial_records_handover(env):
    env.ws.graph.update("hello world.")
    env.restart()
    env.s.type("hello wo")
    env.s.tab()
    assert env.s.text == "hello world "
    assert len(env.s.handover.records) == 2
    assert env.s.handover.records[-1] == (False, True)


def test_sentence_waits_for_an_outstanding_async_check(tmp_path, tok):
    import time

    env = Env(tmp_path, tok, misspelt={"colr": ["color"]})
    env.s = LiveSession(env.cfg, tok, env.ws, env.corrector, env.writer, env.trainer, clock=env.clock, async_checks=True)
    env.s.type("I like colr.")
    env.s.type(" ")
    assert env.learned == []
    deadline = time.time() + 5
    while env.s.correction is None and time.time() < deadline:
        env.s.poll()
        time.sleep(0.01)
    assert env.s.correction is not None and env.s.correction.typed == "colr"
    assert env.learned == []
    env.s.insert()
    # The corrected word is re-checked like any commit; once that check
    # comes back clean the sentence completes.
    deadline = time.time() + 5
    while not env.learned and time.time() < deadline:
        env.s.poll()
        time.sleep(0.01)
    assert env.learned == ["I like color."]


def test_async_clean_word_releases_the_sentence(tmp_path, tok):
    import time

    env = Env(tmp_path, tok)
    env.s = LiveSession(env.cfg, tok, env.ws, env.corrector, env.writer, env.trainer, clock=env.clock, async_checks=True)
    env.s.type("I like it. ")
    deadline = time.time() + 5
    while not env.learned and time.time() < deadline:
        env.s.poll()
        time.sleep(0.01)
    assert env.learned == ["I like it."]


def test_harper_failure_during_cleaning_does_not_crash_typing(tmp_path, tok):
    from pk.corrector import HarperUnavailable

    env = Env(tmp_path, tok)

    def broken(text):
        raise HarperUnavailable("gone")

    env.s.type("hello there")
    env.corrector.check = broken
    env.s.type(". Next ")
    assert env.learned == ["hello there."]
    assert env.s.error is not None


def test_tab_never_rewrites_text_before_the_floor(env):
    env.ws.graph.update("It was a wonderful day.")
    env.restart()
    env.s.type("It was a wo")
    env.s.toggle_pause()
    env.s.type("n")
    floor_text = env.s.text[: env.s.floor]
    env.s.tab()
    assert env.s.text.startswith(floor_text)
    assert env.s.text == "It was a wonderful "
    assert env.s.floor == len("It was a wo")
    env.s.backspace()
    assert env.s.text == "It was a wonderful"


def test_paused_sentence_with_pii_writes_nothing(tmp_path, tok):
    env = Env(tmp_path, tok, paused=True)
    env.s.type("call me on 9876543210. ok")
    env.s.close()
    assert env.raw() == []


def test_edit_that_splits_a_sentence_never_leaks_pii_keystrokes(env):
    env.s.type("x 98765 4321Z0")
    for _ in range(13):
        env.s.left()
    env.s.type(".")
    env.s.right()
    for _ in range(11):
        env.s.right()
    assert env.s.text[env.s.cursor - 1] == "Z"
    env.s.backspace()
    env.s.close()
    assert env.s.text == "x. 98765 43210"
    assert not any(c.isdigit() for c in keys_typed(env))


def test_f2_removes_the_word_from_the_tally(tmp_path, tok):
    env = Env(tmp_path, tok, misspelt={"koshy": ["kohl"]})
    for _ in range(3):
        env.s.type("koshy is here. ")
    assert env.ws.tally.count("koshy") == 3
    env.s.type("koshy ")
    env.s.f2()
    assert env.ws.tally.count("koshy") == 0
    assert env.ws.tally.review(3) == []


def test_prediction_cost_does_not_grow_with_the_session(env):
    seen = []
    original = env.s.predictor.predict

    def spy(params, state, text, partial, mode):
        seen.append(len(text))
        return original(params, state, text, partial, mode)

    env.s.predictor.predict = spy
    env.s.type("one two three. " * 20)
    assert max(seen[-15:]) < 40
