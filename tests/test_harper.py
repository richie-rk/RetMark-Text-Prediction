from pathlib import Path

import pytest

from pk.config import load_config
from pk.corrector import HarperLS, HarperUnavailable

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def harper():
    cfg = load_config(ROOT / "config.toml")
    client = HarperLS(cfg.correct.harper_ls_path)
    yield client
    client.close()


def _spelling(lints):
    return [(l.start, l.end, l.suggestions) for l in lints if l.rule == "SpellCheck"]


@pytest.mark.parametrize("dialect", ["American", "British"])
def test_harper_ls_flags_a_misspelling(harper, dialect):
    lints = harper.lint("I saw thier car.", dialect)
    spelling = _spelling(lints)
    assert len(spelling) == 1
    start, end, suggestions = spelling[0]
    assert (start, end) == (6, 11)
    assert suggestions[0] == "their"


def test_one_process_serves_both_dialects(harper):
    # Alternate many times: a stale linter would leak one dialect's verdict
    # into the other's check.
    for _ in range(10):
        assert _spelling(harper.lint("The colour is nice.", "American"))
        assert not _spelling(harper.lint("The colour is nice.", "British"))
        assert _spelling(harper.lint("The color is nice.", "British"))
        assert not _spelling(harper.lint("The color is nice.", "American"))


def test_spans_are_python_string_indices(harper):
    text = "😀 naïve thier"
    lints = _spelling(harper.lint(text, "American"))
    assert [text[s:e] for s, e, _ in lints] == ["thier"]


def test_ignores_harper_user_dictionaries(harper):
    # Harper's own dictionary files are pointed at paths that do not exist,
    # so a fresh client always flags the same words.
    assert harper.settings_paths_exist() is False


def test_missing_binary_is_reported(tmp_path):
    with pytest.raises(HarperUnavailable):
        HarperLS(str(tmp_path / "no-such-harper-ls"))
