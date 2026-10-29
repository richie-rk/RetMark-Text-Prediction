import pytest

from pk.graph import START, WordGraph
from pk.predictor import ENGINE, GRAPH, Handover, Suggestion, allocate, analyse

BOUNDARY = (".", "?", "!", "\n")


def words(slots):
    return [None if s is None else s.word for s in (slots.second, slots.primary, slots.third, slots.side)]


def g(*items):
    return [(w, 1, 1.0) for w in items]


# Slot order in these assertions: second, primary, third, side.


def test_graph_mode_uses_graph_top_three_and_no_side():
    slots = allocate(GRAPH, ["eng1", "eng2", "eng3"], g("a", "b", "c", "d"), spike=False)
    assert words(slots) == ["b", "a", "c", None]
    assert slots.primary.origin == "graph"


def test_graph_mode_never_pads_with_engine():
    slots = allocate(GRAPH, ["eng1", "eng2", "eng3"], g("a"), spike=False)
    assert words(slots) == [None, "a", None, None]


def test_engine_mode_main_slots_and_side():
    slots = allocate(ENGINE, ["x", "y", "z"], g("y", "q", "r"), spike=False)
    assert words(slots) == ["y", "x", "z", "q"]
    assert slots.primary.origin == "engine"
    assert slots.side.origin == "graph"


def test_engine_mode_spike_takes_primary():
    slots = allocate(ENGINE, ["x", "koshy", "z"], g("koshy", "kumar", "k"), spike=True)
    # Engine's top two distinct words other than the spike fill second/third.
    assert words(slots) == ["x", "koshy", "z", "kumar"]
    assert slots.primary.origin == "graph"
    assert slots.second.origin == "engine"


def test_spike_rule_only_applies_in_engine_mode():
    slots = allocate(GRAPH, ["x", "y", "z"], g("koshy", "kumar"), spike=True)
    assert words(slots) == ["kumar", "koshy", None, None]


def test_duplicates_removed_case_insensitively():
    slots = allocate(ENGINE, ["The", "a", "b"], g("the", "THE", "c"), spike=False)
    assert words(slots) == ["a", "The", "b", "c"]
    spiked = allocate(ENGINE, ["Koshy", "x", "y"], g("koshy", "z"), spike=True)
    assert words(spiked) == ["x", "koshy", "y", "z"]


def test_no_redacted_suggestions():
    slots = allocate(GRAPH, [], g("[REDACTED-CARD]", "ok"), spike=False)
    assert "[REDACTED-CARD]" not in words(slots)
    slots = allocate(ENGINE, ["REDACTED-CARD", "fine", "x"], g(), spike=False, after_bracket=True)
    assert words(slots)[1] == "fine"


def test_empty_sources_leave_slots_empty():
    assert words(allocate(ENGINE, [], [], spike=False)) == [None, None, None, None]


def test_handover_starts_in_graph_mode_until_window_fills():
    h = Handover(window=4)
    for _ in range(3):
        h.record(True, False)
    assert h.mode() == GRAPH
    h.record(True, False)
    assert h.mode() == ENGINE


def test_handover_ties_go_to_engine_and_losses_to_graph():
    h = Handover(window=2)
    h.record(True, True)
    h.record(False, False)
    assert h.rates() == (0.5, 0.5)
    assert h.mode() == ENGINE
    h.record(False, True)
    assert h.rates() == (0.0, 0.5)
    assert h.mode() == GRAPH


def test_handover_keeps_only_the_window_and_round_trips():
    h = Handover(window=3)
    for bits in [(True, False)] * 5 + [(False, True)]:
        h.record(*bits)
    assert len(h.records) == 3
    restored = Handover.from_list(3, h.to_list())
    assert restored.records == h.records


@pytest.mark.parametrize(
    "text,partial,prev,word,prefix",
    [
        ("", "", START, "", ""),
        ("hello wor", " wor", "hello", "wor", ""),
        ("Hello ", " ", "hello", "", ""),
        ("I don'", "'", "i", "don'", "don"),
        ("well-kno", "kno", START, "well-kno", "well-"),
        ("one. two", " two", START, "two", ""),
        ("pi is 3.5 or", " or", "5", "or", ""),
        ("line\nnext", "next", START, "next", ""),
        ("pay [REDACTED-CARD] now", " now", None, "now", ""),
    ],
)
def test_analyse(text, partial, prev, word, prefix):
    ctx = analyse(text, partial, BOUNDARY)
    assert (ctx.prev, ctx.word, ctx.engine_prefix) == (prev, word, prefix)


def test_graph_suggestion_keeps_typed_case():
    graph = WordGraph(BOUNDARY)
    graph.update("hello world.")
    from pk.predictor import graph_words

    assert graph_words(graph, "hello", "Wo", 3) == [("World", 1, 1.0)]
    assert graph_words(graph, "hello", "", 3) == [("world", 1, 1.0)]


def test_suggestion_is_hashable_value():
    assert Suggestion("a", "engine") == Suggestion("a", "engine")


def test_predictor_runs_both_sources_on_the_real_engine():
    from pathlib import Path

    from pk.config import load_config
    from pk.engine.model import init_params, step
    from pk.engine.state import zero_state
    from pk.predictor import Predictor
    from pk.tokenizer import Tokenizer

    cfg = load_config(Path(__file__).resolve().parent.parent / "config.toml")
    tok = Tokenizer(cfg.model.boundary_chars)
    m = cfg.model
    params = init_params(m.d_model, m.mlp_hidden, tok.vocab_size, m.init_std, m.seed)
    graph = WordGraph(BOUNDARY)
    graph.update("hello world.")
    state = zero_state(m.d_model)
    for t in tok.encode("hello"):
        state = step(params, state, t, False, m.fast_decay, m.boundary_decay)
    pred = Predictor(tok, graph, cfg).predict(params, state, "hello wor", " wor", GRAPH)
    assert pred.slots.primary == Suggestion("world", "graph")
    assert len(pred.engine_words) == 3
    assert all(w.startswith("wor") for w in pred.engine_words)
    engine_mode = Predictor(tok, graph, cfg).predict(params, state, "hello wor", " wor", ENGINE)
    assert engine_mode.slots.primary.origin == "engine"


def test_side_slot_uses_followers_beyond_the_top_three(monkeypatch):
    from pathlib import Path

    from pk.config import load_config
    from pk.engine.state import zero_state
    from pk.predictor import Predictor
    from pk.tokenizer import Tokenizer

    cfg = load_config(Path(__file__).resolve().parent.parent / "config.toml")
    graph = WordGraph(BOUNDARY)
    for w, n in (("cat", 4), ("dog", 3), ("fish", 2), ("bird", 1)):
        for _ in range(n):
            graph.update(f"the {w}.")
    p = Predictor(Tokenizer(cfg.model.boundary_chars), graph, cfg)
    monkeypatch.setattr(p, "engine_words", lambda *a: ["cat", "dog", "fish"])
    pred = p.predict(None, zero_state(8), "the ", " ", ENGINE)
    assert pred.slots.side == Suggestion("bird", "graph")
    assert len(pred.graph_words) >= 4
