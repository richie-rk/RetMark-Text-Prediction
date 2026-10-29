import json
from pathlib import Path

import pytest

from pk.corrector import Corrector, HarperLS, Issue, Lint, clean_sentence, damerau_levenshtein
from pk.dictionary import CandidateTally, PersonalDictionary

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def harper():
    client = HarperLS()
    yield client
    client.close()


def corrector(harper, tmp_path, dialect="both"):
    return Corrector(harper, dialect, PersonalDictionary(tmp_path / "dictionary.txt"))


def spelling(issues):
    return [(i.start, i.end) for i in issues if i.kind == "spelling"]


# --- Harper-backed checks -------------------------------------------------


def test_both_dialects_accepts_either_spelling(harper, tmp_path):
    c = corrector(harper, tmp_path)
    assert spelling(c.check("The colour and the color.")) == []
    assert spelling(c.check("I saw thier car.")) == [(6, 11)]


def test_single_dialect_flags_the_other_spelling(harper, tmp_path):
    assert spelling(corrector(harper, tmp_path, "american").check("The colour.")) == [(4, 10)]
    assert spelling(corrector(harper, tmp_path, "british").check("The color.")) == [(4, 9)]


def test_issue_fields(harper, tmp_path):
    [issue] = [i for i in corrector(harper, tmp_path).check("I saw thier car.") if i.kind == "spelling"]
    assert issue.word == "thier"
    assert issue.suggestions[0] == "their"
    assert issue.message


def test_grammar_issues_are_reported(harper, tmp_path):
    issues = corrector(harper, tmp_path).check("He go to to school.")
    assert any(i.kind == "grammar" for i in issues)


def test_personal_dictionary_suppresses_spelling_case_insensitively(harper, tmp_path):
    c = corrector(harper, tmp_path)
    assert spelling(c.check("my name is Koshy.")) != []
    c.dictionary.add("koshy")
    assert spelling(c.check("my name is Koshy.")) == []
    assert c.is_valid("KOSHY")
    assert not c.is_valid("thier")


def test_both_merges_suggestions_american_first():
    us = [Lint(4, 9, "SpellCheck", "Spelling", "m", ("color", "cloud", "club"))]
    gb = [Lint(4, 9, "SpellCheck", "Spelling", "m", ("colour", "Cloud", "club"))]

    class Fake:
        def lint(self, text, dialect):
            return us if dialect == "American" else gb

    c = Corrector(Fake(), "both", PersonalDictionary(None))
    [issue] = c.check("the colur")
    assert issue.suggestions == ("color", "cloud", "club", "colour")


def test_both_requires_a_double_miss():
    us = [Lint(0, 3, "SpellCheck", "Spelling", "m", ("a",)), Lint(4, 7, "SpellCheck", "Spelling", "m", ("b",))]
    gb = [Lint(4, 7, "SpellCheck", "Spelling", "m", ("b",)), Lint(8, 9, "Typo", "Grammar", "g", ())]

    class Fake:
        def lint(self, text, dialect):
            return us if dialect == "American" else gb

    issues = Corrector(Fake(), "both", PersonalDictionary(None)).check("abc def g")
    assert [(i.start, i.end, i.kind) for i in issues] == [(4, 7, "spelling")]


# --- dictionary, rejects and tally ----------------------------------------


def test_personal_dictionary_round_trip(tmp_path):
    d = PersonalDictionary(tmp_path / "dictionary.txt")
    d.add("Koshy")
    d.add("koshy")
    d.save()
    assert (tmp_path / "dictionary.txt").read_text(encoding="utf-8") == "Koshy\n"
    assert "KOSHY" in PersonalDictionary(tmp_path / "dictionary.txt")


def test_tally_counts_and_persists(tmp_path):
    d = PersonalDictionary(tmp_path / "dictionary.txt")
    r = PersonalDictionary(tmp_path / "rejected.txt")
    t = CandidateTally(tmp_path / "candidates.json", d, r)
    t.increment("Richy", "2026-10-05T10:00:00.000+05:30")
    t.increment("richy", "2026-10-06T10:00:00.000+05:30")
    t.increment("Richy", "2026-10-07T10:00:00.000+05:30")
    assert t.count("RICHY") == 3
    t.save()
    data = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    entry = data["richy"]
    assert entry["count"] == 3
    assert entry["first_seen"] == "2026-10-05T10:00:00.000+05:30"
    assert entry["last_seen"] == "2026-10-07T10:00:00.000+05:30"
    t2 = CandidateTally(tmp_path / "candidates.json", d, r)
    assert t2.review(3) == [("Richy", 3)]
    assert t2.review(4) == []


def test_approve_and_reject(tmp_path):
    d = PersonalDictionary(tmp_path / "dictionary.txt")
    r = PersonalDictionary(tmp_path / "rejected.txt")
    t = CandidateTally(tmp_path / "candidates.json", d, r)
    for _ in range(3):
        t.increment("richy", "ts")
        t.increment("teh", "ts")
    t.approve(["richy"])
    t.reject(["teh"])
    assert "richy" in d and t.count("richy") == 0
    assert "teh" in r and t.count("teh") == 0
    t.increment("teh", "ts")
    t.increment("richy", "ts")
    assert t.count("teh") == 0 and t.count("richy") == 0
    t.approve(["teh"])
    assert "teh" in d and "teh" not in r


# --- cleaning ---------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,dist",
    [("thier", "their", 1), ("ca", "abc", 2), ("kitten", "sitting", 3), ("Same", "same", 0), ("", "ab", 2)],
)
def test_damerau_levenshtein(a, b, dist):
    assert damerau_levenshtein(a, b) == dist


class FakeCorrector:
    def __init__(self, issues):
        self.issues = issues

    def check(self, text):
        return [i for i in self.issues if i.end <= len(text)]


def make_tally(tmp_path):
    d = PersonalDictionary(tmp_path / "dictionary.txt")
    return d, CandidateTally(tmp_path / "candidates.json", d, PersonalDictionary(tmp_path / "rejected.txt"))


def issue(text, word, suggestions):
    start = text.index(word)
    return Issue(start, start + len(word), "spelling", "SpellCheck", "m", tuple(suggestions), word)


def test_cleaning_replaces_close_misspelling_and_counts_it(tmp_path):
    text = "I saw thier car."
    d, t = make_tally(tmp_path)
    clean = clean_sentence(text, FakeCorrector([issue(text, "thier", ["their"])]), t, 3, 2, "ts")
    assert clean == "I saw their car."
    assert t.count("thier") == 1


def test_cleaning_keeps_far_suggestion_word(tmp_path):
    text = "my name is koshy."
    d, t = make_tally(tmp_path)
    clean = clean_sentence(text, FakeCorrector([issue(text, "koshy", ["kitchen"])]), t, 3, 2, "ts")
    assert clean == text


def test_cleaning_keeps_consistent_spellings_once_counted(tmp_path):
    text = "hi richy."
    d, t = make_tally(tmp_path)
    fake = FakeCorrector([issue(text, "richy", ["rich"])])
    assert clean_sentence(text, fake, t, 3, 2, "ts") == "hi rich."
    assert clean_sentence(text, fake, t, 3, 2, "ts") == "hi rich."
    # Third occurrence reaches candidate_min_count: increment happens first.
    assert clean_sentence(text, fake, t, 3, 2, "ts") == "hi richy."


def test_cleaning_keeps_dictionary_words(tmp_path):
    text = "hi richy."
    d, t = make_tally(tmp_path)
    d.add("Richy")
    assert clean_sentence(text, FakeCorrector([issue(text, "richy", ["rich"])]), t, 3, 2, "ts") == text
    assert t.count("richy") == 0


def test_cleaning_redacts_and_skips_issues_inside_redactions(tmp_path):
    text = "pan abcde1234f thier."
    d, t = make_tally(tmp_path)
    fake = FakeCorrector([issue(text, "abcde1234f", ["abide"]), issue(text, "thier", ["their"])])
    assert clean_sentence(text, fake, t, 3, 2, "ts") == "pan [REDACTED-PAN] their."
    assert t.count("abcde1234f") == 0
    assert t.count("thier") == 1


def test_cleaning_with_real_harper(harper, tmp_path):
    d, t = make_tally(tmp_path)
    c = Corrector(harper, "both", d)
    assert clean_sentence("I saw thier car.", c, t, 3, 2, "ts") == "I saw their car."


def test_emoji_sequences_never_break_checking(harper, tmp_path):
    c = corrector(harper, tmp_path)
    text = "😀😀 x 🇮🇳 I havv a cat."
    issues = [i for i in c.check(text) if i.kind == "spelling"]
    assert [text[i.start:i.end] for i in issues] == ["havv"]
