import regex

from pk.tokenizer import WORD_CHAR, boundary_positions

START = "<s>"
PLACEHOLDER = r"\[REDACTED-[A-Z]+\]"
ITEM_RE = regex.compile(f"({PLACEHOLDER})|{WORD_CHAR}+")


class WordGraph:
    """First-order word transitions, held as plain dictionaries."""

    def __init__(self, boundary_chars):
        self.boundary_chars = frozenset(boundary_chars)
        self.edges = {}
        self.totals = {}
        self.surface = {}

    def _segments(self, text):
        start = 0
        for i in boundary_positions(text, self.boundary_chars, at_end=True):
            yield text[start:i]
            start = i + 1
        yield text[start:]

    def _add_edge(self, prev, word):
        followers = self.edges.setdefault(prev, {})
        followers[word] = followers.get(word, 0) + 1
        self.totals[prev] = self.totals.get(prev, 0) + 1

    def update(self, sentence):
        for segment in self._segments(sentence):
            # A placeholder holds its word position (None) but is never a node.
            items = [None if m.group(1) else m.group() for m in ITEM_RE.finditer(segment)]
            for form in items:
                if form is not None:
                    forms = self.surface.setdefault(form.lower(), {})
                    forms[form] = forms.get(form, 0) + 1
            if items and items[0] is not None:
                self._add_edge(START, items[0].lower())
            for a, b in zip(items, items[1:]):
                if a is not None and b is not None:
                    self._add_edge(a.lower(), b.lower())

    def display(self, key):
        forms = self.surface.get(key)
        if not forms:
            return key
        return min(forms, key=lambda f: (-forms[f], f))

    def followers(self, prev, prefix):
        """[(key, count, probability)], most frequent first. With a prefix the
        probabilities are renormalised over the matching followers only."""
        prefix = prefix.lower()
        found = [(k, c) for k, c in self.edges.get(prev, {}).items() if k.startswith(prefix)]
        if not found:
            return []
        total = self.totals[prev] if not prefix else sum(c for _, c in found)
        found.sort(key=lambda kc: (-kc[1], kc[0]))
        return [(k, c, c / total) for k, c in found]

    def is_spike(self, prev, key, prefix, spike_prob, spike_min_count):
        if self.totals.get(prev, 0) < spike_min_count:
            return False
        for k, _, p in self.followers(prev, prefix):
            if k == key:
                return p >= spike_prob
        return False

    def stats(self, spike_min_count, top):
        nodes = set(self.edges)
        for followers in self.edges.values():
            nodes.update(followers)
        strongest = [
            (prev, word, count, count / self.totals[prev])
            for prev, followers in self.edges.items()
            if self.totals[prev] >= spike_min_count
            for word, count in followers.items()
        ]
        strongest.sort(key=lambda e: (-e[3], -e[2], e[0], e[1]))
        return {
            "nodes": len(nodes),
            "edges": sum(len(f) for f in self.edges.values()),
            "strongest": strongest[:top],
        }
