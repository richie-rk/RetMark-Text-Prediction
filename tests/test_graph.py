import pytest

from pk.graph import START, WordGraph

BOUNDARY = (".", "?", "!", "\n")


def graph(*sentences):
    g = WordGraph(BOUNDARY)
    for s in sentences:
        g.update(s)
    return g


def test_counts_and_totals():
    g = graph("the cat sat.", " the cat ran.", " the dog sat.")
    assert g.edges["the"] == {"cat": 2, "dog": 1}
    assert g.totals["the"] == 3
    assert g.edges["cat"] == {"sat": 1, "ran": 1}
    assert g.totals["cat"] == 2


def test_sentence_start_edges():
    g = graph("The cat sat.", " A dog ran.", " the end.")
    assert g.edges[START] == {"the": 2, "a": 1}
    assert g.totals[START] == 3


def test_no_edge_across_a_boundary():
    g = graph("one two. three four! five\nsix")
    assert "three" not in g.edges.get("two", {})
    assert "five" not in g.edges.get("four", {})
    assert "six" not in g.edges.get("five", {})
    assert g.edges[START] == {"one": 1, "three": 1, "five": 1, "six": 1}


def test_mid_word_marks_do_not_split():
    g = graph("pi is 3.5 and see claude.ai today.")
    assert g.edges["is"] == {"3": 1}
    assert g.edges["3"] == {"5": 1}
    assert "claude" in g.edges["see"]
    assert g.edges[START] == {"pi": 1}


def test_keys_are_lowercase_and_surface_forms_are_counted():
    g = graph("Hello World.", " hello World.", " HELLO world.")
    assert g.edges["hello"] == {"world": 3}
    assert g.surface["hello"] == {"Hello": 1, "hello": 1, "HELLO": 1}
    assert g.display("world") == "World"


def test_display_tie_breaks_lexicographically():
    g = graph("Zed zed.")
    assert g.display("zed") == "Zed"


def test_words_keep_apostrophes_and_hyphens():
    g = graph("don't over-think it.")
    assert g.edges["don't"] == {"over-think": 1}


def test_placeholders_never_become_nodes():
    g = graph("[REDACTED-CARD] paid for it.", " call [REDACTED-PHONE] now.", " my pan [REDACTED-PAN].")
    nodes = set(g.edges) | {w for f in g.edges.values() for w in f} | set(g.surface)
    assert not any("redacted" in n.lower() for n in nodes)
    assert "card" not in nodes
    # A placeholder takes a word position: no edges through it, and the word
    # after a leading placeholder is not a sentence start.
    assert "now" not in g.edges.get("call", {})
    assert "paid" not in g.edges[START]
    assert g.edges["paid"] == {"for": 1}


def test_followers_sorted_with_probabilities():
    g = graph("a b.", " a b.", " a c.", " a d.")
    assert g.followers("a", "") == [("b", 2, 0.5), ("c", 1, 0.25), ("d", 1, 0.25)]


def test_followers_with_prefix_renormalise_over_filtered():
    g = graph("a bat.", " a bat.", " a bee.", " a cow.")
    assert g.followers("a", "b") == [("bat", 2, pytest.approx(2 / 3)), ("bee", 1, pytest.approx(1 / 3))]
    assert g.followers("a", "B") == g.followers("a", "b")
    assert g.followers("zzz", "") == []


def test_spike_at_and_below_thresholds():
    # spike_prob 0.8, spike_min_count 5
    at = graph(*(["richy koshy."] * 4 + ["richy k."]))
    assert at.followers("richy", "")[0][2] == pytest.approx(0.8)
    assert at.is_spike("richy", "koshy", "", 0.8, 5)
    below_prob = graph(*(["richy koshy."] * 3 + ["richy k.", "richy x."]))
    assert not below_prob.is_spike("richy", "koshy", "", 0.8, 5)
    below_count = graph(*(["richy koshy."] * 4))
    assert below_count.followers("richy", "")[0][2] == pytest.approx(1.0)
    assert not below_count.is_spike("richy", "koshy", "", 0.8, 5)


def test_stats():
    g = graph(*(["richy koshy."] * 5 + ["a b.", " a c."]))
    stats = g.stats(spike_min_count=5, top=20)
    assert stats["nodes"] == 6  # <s>, richy, koshy, a, b, c
    assert stats["edges"] == 5  # <s>->richy, <s>->a, richy->koshy, a->b, a->c
    assert stats["strongest"][0][:2] == ("richy", "koshy")
    assert all(edge[0] != "a" for edge in stats["strongest"])
