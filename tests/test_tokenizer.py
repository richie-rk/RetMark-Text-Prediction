from dataclasses import dataclass

import numpy as np
import pytest

from pk.tokenizer import Tokenizer, current_word, predict_words, split_partial

BOUNDARY = (".", "?", "!", "\n")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer(BOUNDARY)


def test_vocabulary(tok):
    assert tok.vocab_size == 50257
    assert len(tok.token_str) == 50257
    assert tok.token_str[tok.encode(" hello")[0]] == " hello"


@pytest.mark.parametrize(
    "text,committed,partial",
    [
        ("", [], ""),
        ("hello", [], "hello"),
        ("hello wor", ["hello"], " wor"),
        ("hello ", ["hello"], " "),
        ("I don'", ["I", " don"], "'"),
        ("we'll", ["we"], "'ll"),
        ("hello.", ["hello"], "."),
        ("Done. (see", ["Done", ".", " ("], "see"),
        ("hello\n", ["hello"], "\n"),
    ],
)
def test_committed_partial_split(tok, text, committed, partial):
    assert split_partial(tok.chunks(text)) == (committed, partial)


def test_chunks_reassemble_text(tok):
    text = "Hello there.  It's 3.5 o'clock\n(see claude.ai)!"
    assert "".join(tok.chunks(text)) == text


def test_prefix_search_matches_linear_scan(tok):
    for prefix in [b"ri", b" wor", b"'", b"\n", b""]:
        expected = sorted(i for i, b in enumerate(tok.token_bytes) if b.startswith(prefix))
        assert sorted(tok.ids_with_prefix(prefix).tolist()) == expected


@pytest.mark.parametrize(
    "text,at_end,expected",
    [
        ("Hi. There", False, [2]),
        ("3.5 and v2.1", False, []),
        ("see claude.ai now", False, []),
        ("config.toml", False, []),
        ("Wait... ok", False, [4, 5, 6]),
        ("Really?!", True, [6, 7]),
        ('He said "go." Then', False, [11]),
        ("(done.) next", False, [5]),
        ("end.", False, []),
        ("end.", True, [3]),
        ("a\nb", False, [1]),
        ("e.g. this", False, [3]),
    ],
)
def test_boundary_positions(tok, text, at_end, expected):
    assert tok.boundary_positions(text, at_end=at_end) == expected


def test_token_boundary_flags(tok):
    text = "Pi is 3.14. Yes\n"
    ids = tok.encode(text)
    flags = tok.token_boundary_flags(text, ids, at_end=True)
    fired = [tok.token_str[i] for i, f in zip(ids, flags) if f]
    assert fired == [".", "\n"]
    assert flags.count(True) == 2


@pytest.mark.parametrize(
    "text,word,before",
    [
        ("hello wor", "wor", "hello "),
        ("hello ", "", "hello "),
        ("I don'", "don'", "I "),
        ("well-kno", "well-kno", ""),
        ("hello.", "", "hello."),
        ("naïve caf", "caf", "naïve "),
    ],
)
def test_current_word(text, word, before):
    assert current_word(text) == (before, word)


# A fake engine: each token's next-token distribution is a fixed table, so
# healing and extension can be tested without the real model.
@dataclass(frozen=True)
class FakeState:
    history: tuple
    logits: np.ndarray | None


class FakeModel:
    def __init__(self, tok, table, default=None):
        self.tok = tok
        self.table = {tok.encode(k)[0] if isinstance(k, str) else k: v for k, v in table.items()}
        self.default = default or {}
        self.steps = 0

    def logits_for(self, token):
        logits = np.full(self.tok.vocab_size, -np.inf, dtype=np.float32)
        for i, (s, score) in enumerate(self.table.get(token, self.default).items()):
            ids = self.tok.encode(s)
            assert len(ids) == 1, s
            logits[ids[0]] = score
        return logits

    def step(self, state, token):
        self.steps += 1
        return FakeState(state.history + (token,), self.logits_for(token))


def zero(model, after=None):
    if after is None:
        return FakeState((), None)
    return FakeState((), model.logits_for(model.tok.encode(after)[0]))


def test_no_engine_suggestions_before_any_token(tok):
    model = FakeModel(tok, {})
    assert predict_words(tok, model.step, zero(model), "", "", 3, 4) == []


def test_healing_unfri_keeps_only_tokens_starting_with_remainder(tok):
    # "unfri" encodes as "un" "f" "ri": state advances through "un" and "f",
    # and only tokens starting with "ri" survive.
    assert [tok.token_str[i] for i in tok.encode("unfri")] == ["un", "f", "ri"]
    model = FakeModel(
        tok,
        {
            "f": {"riend": 5.0, " the": 9.0, "ruit": 8.0, "ri": 1.0},
            "riend": {"ing": 5.0},
            "ing": {" now": 5.0},
            "ri": {"end": 5.0},
            "end": {"ing": 5.0},
        },
    )
    state = zero(model, after="the")
    words = predict_words(tok, model.step, state, "unfri", "", 3, 4)
    assert words == ["unfriending"]
    assert all(w.startswith("unfri") for w in words)


def test_extension_stops_at_whitespace(tok):
    model = FakeModel(tok, {"x": {" hel": 5.0}, " hel": {"lo": 5.0}, "lo": {" world": 5.0}})
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 1, 4) == ["hello"]


def test_extension_stops_at_boundary(tok):
    model = FakeModel(tok, {"x": {" hel": 5.0}, " hel": {"lo": 5.0}, "lo": {".": 5.0}})
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 1, 4) == ["hello"]


def test_extension_stops_at_non_word_character(tok):
    model = FakeModel(tok, {"x": {" hel": 5.0}, " hel": {"lo": 5.0}, "lo": {",": 5.0}})
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 1, 4) == ["hello"]


def test_extension_stops_at_token_limit(tok):
    # The seed counts toward the limit, so a word has at most 4 tokens.
    chain = {"x": {" a": 5.0}, " a": {"b": 5.0}, "b": {"c": 5.0}, "c": {"d": 5.0}, "d": {"e": 5.0}}
    model = FakeModel(tok, chain)
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 1, 4) == ["abcd"]
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 1, 2) == ["ab"]


def test_apostrophes_and_hyphens_extend(tok):
    model = FakeModel(tok, {"x": {" don": 5.0}, " don": {"'t": 5.0}, "'t": {" go": 5.0}})
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 1, 4) == ["don't"]


def test_suggestion_includes_committed_part_of_current_word(tok):
    # "I don'" -> partial chunk "'", committed part of the word "don".
    model = FakeModel(tok, {"x": {"'t": 5.0, "'s": 3.0}, "'t": {" go": 1.0}, "'s": {" go": 1.0}})
    words = predict_words(tok, model.step, zero(model, after="x"), "'", "don", 3, 4)
    assert words == ["don't", "don's"]


def test_invalid_seeds_are_skipped_and_list_is_refilled(tok):
    model = FakeModel(
        tok,
        {"x": {" (": 9.0, " ...": 8.0, ' "': 7.5, " cat": 7.0, " dog": 6.0, " Cat": 5.5, " cow": 5.0}},
        default={" end": 1.0},
    )
    words = predict_words(tok, model.step, zero(model, after="x"), " ", "", 3, 4)
    assert words == ["cat", "dog", "cow"]


def test_seeds_collapsing_to_the_same_word_are_deduplicated(tok):
    model = FakeModel(
        tok,
        {"x": {" hel": 9.0, " hell": 8.0, " help": 7.0}, " hel": {"lo": 5.0}, " hell": {"o": 5.0}, "lo": {" a": 1.0}, "o": {" a": 1.0}},
    )
    assert predict_words(tok, model.step, zero(model, after="x"), " ", "", 3, 4) == ["hello", "help"]


def test_no_suggestions_after_punctuation_only_partial(tok):
    model = FakeModel(tok, {".": {" The": 5.0}})
    assert predict_words(tok, model.step, zero(model, after="."), ".", "", 3, 4) == []


def test_partial_whitespace_predicts_from_state(tok):
    model = FakeModel(tok, {"hello": {" world": 5.0, " there": 4.0}})
    state = zero(model, after="hello")
    assert predict_words(tok, model.step, state, " ", "", 3, 4) == ["world", "there"]
    assert model.steps == 2


def test_empty_partial_predicts_unconstrained(tok):
    model = FakeModel(tok, {"\n": {"The": 5.0, " A": 4.0, ",": 6.0}})
    state = zero(model, after="\n")
    assert predict_words(tok, model.step, state, "", "", 3, 1) == ["The", "A"]


def test_typed_case_is_kept(tok):
    model = FakeModel(tok, {"x": {" Wor": 5.0}, " Wor": {"ld": 5.0}, "ld": {" x": 1.0}})
    assert predict_words(tok, model.step, zero(model, after="x"), " Wor", "", 3, 4) == ["World"]
