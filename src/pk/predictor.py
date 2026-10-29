from collections import deque
from dataclasses import dataclass
from functools import partial as bind

from pk.engine.model import step
from pk.graph import ITEM_RE, START
from pk.tokenizer import boundary_positions, current_word, predict_words

ENGINE = "ENGINE"
GRAPH = "GRAPH"
REDACTED = "[REDACTED"


@dataclass(frozen=True)
class Suggestion:
    word: str
    origin: str


@dataclass(frozen=True)
class Slots:
    primary: Suggestion | None = None
    second: Suggestion | None = None
    third: Suggestion | None = None
    side: Suggestion | None = None

    def main(self):
        return [s for s in (self.primary, self.second, self.third) if s is not None]


class Handover:
    def __init__(self, window, records=()):
        self.window = window
        self.records = deque(records, maxlen=window)

    def record(self, engine_hit, graph_hit):
        self.records.append((bool(engine_hit), bool(graph_hit)))

    def rates(self):
        if not self.records:
            return 0.0, 0.0
        n = len(self.records)
        return sum(e for e, _ in self.records) / n, sum(g for _, g in self.records) / n

    def mode(self):
        engine, graph = self.rates()
        return ENGINE if len(self.records) >= self.window and engine >= graph else GRAPH

    def to_list(self):
        return [list(r) for r in self.records]

    @classmethod
    def from_list(cls, window, items):
        return cls(window, (tuple(map(bool, r)) for r in items))


def _allowed(word, after_bracket):
    w = word.lstrip()
    if w.startswith(REDACTED):
        return False
    # "[" is not a word character, so a typed "[RED" reaches the sources as
    # the word "RED" with the bracket just before it.
    return not (after_bracket and w.upper().startswith(REDACTED[1:]))


def allocate(mode, engine_words, graph_words, spike, after_bracket=False):
    """Give whole slots to one source or the other; scores are never mixed."""
    engine = [w for w in engine_words if _allowed(w, after_bracket)]
    graph = [w for w, _, _ in graph_words if _allowed(w, after_bracket)]
    taken = set()

    def pick(candidates, origin, n):
        out = []
        for w in candidates:
            if len(out) == n:
                break
            if w.lower() in taken:
                continue
            taken.add(w.lower())
            out.append(Suggestion(w, origin))
        return out

    if mode == GRAPH:
        main = pick(graph, "graph", 3)
        side = []
    elif spike and graph:
        main = pick(graph[:1], "graph", 1) + pick(engine, "engine", 2)
        side = pick(graph[1:], "graph", 1)
    else:
        main = pick(engine, "engine", 3)
        side = pick(graph, "graph", 1)
    main += [None] * (3 - len(main))
    return Slots(primary=main[0], second=main[1], third=main[2], side=side[0] if side else None)


@dataclass(frozen=True)
class Context:
    prev: str | None
    word: str
    engine_prefix: str
    after_bracket: bool


def analyse(text, partial_chunk, boundary_chars):
    """Locate the word being typed and the word before it (§6 words).

    prev is None when the previous item is a redaction placeholder: the graph
    has no edges out of placeholders.
    """
    before, word = current_word(text)
    _, partial_word = current_word(partial_chunk)
    engine_prefix = word[: len(word) - len(partial_word)]
    items = list(ITEM_RE.finditer(before))
    if not items:
        prev = START
    else:
        last = items[-1]
        gap_start = last.end()
        marks = boundary_positions(text, boundary_chars, at_end=False)
        if any(gap_start <= m < len(before) for m in marks):
            prev = START
        elif last.group(1):
            prev = None
        else:
            prev = last.group().lower()
    return Context(prev=prev, word=word, engine_prefix=engine_prefix, after_bracket=before.endswith("["))


def graph_words(graph, prev, typed, n):
    """Top graph followers as display words, with the typed partial kept as typed."""
    if prev is None:
        return []
    out = []
    for key, count, prob in graph.followers(prev, typed)[:n]:
        out.append((typed + graph.display(key)[len(typed):], count, prob))
    return out


@dataclass(frozen=True)
class Prediction:
    context: Context
    engine_words: list
    graph_words: list
    spike: bool
    mode: str
    slots: Slots


class Predictor:
    def __init__(self, tokenizer, graph, cfg):
        self.tok = tokenizer
        self.graph = graph
        self.cfg = cfg

    def engine_words(self, params, state, partial_chunk, engine_prefix):
        m = self.cfg.model
        stepper = bind(_step_word, params, m.fast_decay, m.boundary_decay)
        p = self.cfg.predict
        return predict_words(self.tok, stepper, state, partial_chunk, engine_prefix, p.top_k, p.word_extend_max_tokens)

    def predict(self, params, state, text, partial_chunk, mode):
        """Both sources always run, whichever owns the main slots (§7)."""
        ctx = analyse(text, partial_chunk, self.cfg.model.boundary_chars)
        engine = self.engine_words(params, state, partial_chunk, ctx.engine_prefix)
        graph = graph_words(self.graph, ctx.prev, ctx.word, 3)
        g = self.cfg.graph
        spike = bool(graph) and self.graph.is_spike(ctx.prev, graph[0][0].lower(), ctx.word, g.spike_prob, g.spike_min_count)
        slots = allocate(mode, engine, graph, spike, ctx.after_bracket)
        return Prediction(context=ctx, engine_words=engine, graph_words=graph, spike=spike, mode=mode, slots=slots)


def _step_word(params, fast_decay, boundary_decay, state, token):
    # Healing and extension only ever step through word and whitespace
    # tokens, which never carry a sentence boundary.
    return step(params, state, token, False, fast_decay, boundary_decay)
