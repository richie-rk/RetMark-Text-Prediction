import json
from pathlib import Path

from pk.persistence import atomic_write_bytes, atomic_write_json


class PersonalDictionary:
    """A word list matched case-insensitively, one word per line on disk.
    Also used for rejected.txt, which has the same shape."""

    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self._words = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._words.setdefault(line.strip().lower(), line.strip())

    def __contains__(self, word):
        return word.lower() in self._words

    def __iter__(self):
        return iter(self._words.values())

    def __len__(self):
        return len(self._words)

    def add(self, word):
        self._words.setdefault(word.lower(), word)

    def remove(self, word):
        self._words.pop(word.lower(), None)

    def save(self):
        if self.path is not None:
            atomic_write_bytes(self.path, "".join(w + "\n" for w in self._words.values()).encode("utf-8"))


class CandidateTally:
    """Flagged words not in the personal dictionary, counted per occurrence.

    A rejected word leaves the tally for good, so it is never counted again
    and cleaning goes back to correcting it.
    """

    def __init__(self, path, dictionary, rejected):
        self.path = Path(path)
        self.dictionary = dictionary
        self.rejected = rejected
        self.entries = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def increment(self, word, ts):
        if word in self.dictionary or word in self.rejected:
            return
        entry = self.entries.setdefault(word.lower(), {"count": 0, "first_seen": ts, "last_seen": ts, "forms": {}})
        entry["count"] += 1
        entry["last_seen"] = ts
        entry["forms"][word] = entry["forms"].get(word, 0) + 1

    def count(self, word):
        entry = self.entries.get(word.lower())
        return 0 if entry is None else entry["count"]

    def display(self, key):
        forms = self.entries[key]["forms"]
        return min(forms, key=lambda f: (-forms[f], f)) if forms else key

    def listing(self):
        rows = [(self.display(k), e["count"], e["first_seen"], e["last_seen"]) for k, e in self.entries.items()]
        rows.sort(key=lambda r: (-r[1], r[0].lower()))
        return rows

    def review(self, min_count):
        return [(word, count) for word, count, _, _ in self.listing() if count >= min_count]

    def approve(self, words):
        for w in words:
            self.dictionary.add(w)
            self.rejected.remove(w)
            self.entries.pop(w.lower(), None)

    def reject(self, words):
        for w in words:
            self.rejected.add(w)
            self.dictionary.remove(w)
            self.entries.pop(w.lower(), None)

    def save(self):
        atomic_write_json(self.path, self.entries)
