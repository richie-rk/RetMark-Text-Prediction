import bisect

import numpy as np
import regex
import tiktoken

ENCODING = "gpt2"

# §6 word characters: letters (with combining marks), decimal digits,
# apostrophes (ASCII and U+2019) and hyphens.
WORD_CHAR = r"[\p{L}\p{M}\p{Nd}'’\-]"
WORD_RE = regex.compile(WORD_CHAR + "+")
_TRAILING_WORD_RE = regex.compile(WORD_CHAR + "+$")
_WS_THEN_WORD_RE = regex.compile(r"\s+" + WORD_CHAR + "+")

# A sentence-ending mark only ends a sentence when what follows it says so:
# "3.5", "claude.ai" and "v2.1" must not fire decay or end a sentence.
CLOSERS = frozenset("\"')]}’”»")


def is_boundary_at(text, i, boundary_chars, at_end):
    c = text[i]
    if c not in boundary_chars:
        return False
    if c.isspace():
        return True
    if i + 1 == len(text):
        return at_end
    nxt = text[i + 1]
    return nxt.isspace() or nxt in boundary_chars or nxt in CLOSERS


def boundary_positions(text, boundary_chars, at_end):
    """Indices of characters in `text` that end a sentence.

    `at_end` says whether `text` is complete; when it is not, a mark at the
    very end stays undecided (False) until the next character arrives.
    """
    return [i for i in range(len(text)) if is_boundary_at(text, i, boundary_chars, at_end)]


def split_partial(chunks):
    if not chunks:
        return [], ""
    return list(chunks[:-1]), chunks[-1]


def current_word(text):
    """Split text into (everything before the word at the end, that word)."""
    m = _TRAILING_WORD_RE.search(text)
    if m is None:
        return text, ""
    return text[: m.start()], m.group()


class Tokenizer:
    def __init__(self, boundary_chars):
        self.boundary_chars = frozenset(boundary_chars)
        self.enc = tiktoken.get_encoding(ENCODING)
        self.vocab_size = self.enc.n_vocab
        self._pattern = regex.compile(self.enc._pat_str)

        # Prefix search runs on bytes: a GPT-2 token can be a fragment of a
        # multi-byte character, which has no faithful str form.
        self.token_bytes = [self.enc.decode_single_token_bytes(i) for i in range(self.vocab_size)]
        self.token_str = [b.decode("utf-8", errors="replace") for b in self.token_bytes]
        order = sorted(range(self.vocab_size), key=lambda i: self.token_bytes[i])
        self._sorted_bytes = [self.token_bytes[i] for i in order]
        self._sorted_ids = np.array(order, dtype=np.int64)

        all_word = np.zeros(self.vocab_size, dtype=bool)
        ws_then_word = np.zeros(self.vocab_size, dtype=bool)
        for i, b in enumerate(self.token_bytes):
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                continue
            all_word[i] = WORD_RE.fullmatch(s) is not None
            ws_then_word[i] = _WS_THEN_WORD_RE.fullmatch(s) is not None
        self.all_word = all_word
        self.ws_then_word = ws_then_word

    def chunks(self, text):
        return self._pattern.findall(text)

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(list(ids))

    def ids_with_prefix(self, prefix):
        lo = bisect.bisect_left(self._sorted_bytes, prefix)
        # The first byte string past every extension of `prefix`: drop trailing
        # 0xff bytes, then bump the last remaining byte.
        upper = prefix.rstrip(b"\xff")
        if upper:
            hi = bisect.bisect_left(self._sorted_bytes, upper[:-1] + bytes([upper[-1] + 1]))
        else:
            hi = len(self._sorted_bytes)
        return self._sorted_ids[lo:hi]

    def boundary_positions(self, text, at_end):
        return boundary_positions(text, self.boundary_chars, at_end)

    def token_boundary_flags(self, text, ids, at_end):
        """Per token: does it contain a sentence boundary of `text`?"""
        return self.flags_for_marks(text, ids, self.boundary_positions(text, at_end))

    def flags_for_marks(self, text, ids, char_marks):
        """Per token of `text`: does it cover one of the character indices in
        `char_marks`? Works on bytes, since a token can split a character."""
        byte_pos = []
        offset = 0
        for c in text:
            byte_pos.append(offset)
            offset += len(c.encode("utf-8"))
        marks = {byte_pos[i] for i in char_marks}
        flags = []
        start = 0
        for t in ids:
            end = start + len(self.token_bytes[t])
            flags.append(any(start <= m < end for m in marks))
            start = end
        return flags

def predict_words(tok, step, state, partial, word_prefix, top_k, max_tokens):
    """Engine word suggestions for the partial chunk (§4, token healing).

    `step(state, token) -> state` advances a copy; `state.logits` is the
    next-token distribution after the last processed token, or None before
    any. `word_prefix` is the part of the current word already committed in
    earlier chunks, so "don" + "'t" becomes "don't".
    """
    if partial and partial.strip() and not regex.search(WORD_CHAR, partial):
        return []
    base = state
    if partial:
        ids = tok.encode(partial)
        for t in ids[:-1]:
            base = step(base, t)
        head = tok.decode(ids[:-1])
        remainder = tok.token_bytes[ids[-1]]
    else:
        head = ""
        remainder = b""
    if base.logits is None:
        return []

    candidates = tok.ids_with_prefix(remainder)
    # A seed must keep the suggestion a single word: once the head already
    # holds word characters, a seed with whitespace in it would start a second.
    if head.strip():
        candidates = candidates[tok.all_word[candidates]]
    else:
        candidates = candidates[tok.all_word[candidates] | tok.ws_then_word[candidates]]
    logits = np.asarray(base.logits)
    scores = logits[candidates]
    order = np.argsort(-scores, kind="stable")

    words = []
    seen = set()
    for idx in order:
        if not np.isfinite(scores[idx]):
            break
        seed = int(candidates[idx])
        pieces = [tok.token_str[seed]]
        scratch = base
        last = seed
        while len(pieces) < max_tokens:
            scratch = step(scratch, last)
            nxt = int(np.argmax(np.asarray(scratch.logits)))
            if not tok.all_word[nxt]:
                break
            pieces.append(tok.token_str[nxt])
            last = nxt
        word = word_prefix + (head + "".join(pieces)).lstrip()
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(word)
        if len(words) == top_k:
            break
    return words
