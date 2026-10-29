import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy

from pk import redact
from pk.corrector import HarperUnavailable, clean_sentence
from pk.engine import layers
from pk.engine.model import backward_sequence, forward_sequence, step
from pk.engine.optim import clip_global_norm
from pk.engine.state import zero_state
from pk.logs import SentenceEvents, local_now, timestamp
from pk.persistence import save_state
from pk.predictor import Handover, Predictor
from pk.tokenizer import WORD_RE, boundary_positions, current_word

# Not tunables: how much history the --inspect panel shows.
INSPECT_TOKENS = 20
INSPECT_TOP = 10


class Recorder:
    """Raw and clean log streams for one harness run, gated by pause.

    A pause toggle ends the current sentence: whatever was recorded before
    the toggle is redacted and queued but never learned, and nothing typed
    while paused reaches a log or a learner. With no writer (pk predict) it
    records nothing.
    """

    def __init__(self, writer, paused):
        self.writer = writer
        self.paused = paused
        self._events = SentenceEvents()

    def event(self, type, events=None, tag=None, **fields):
        if self.writer is not None and not self.paused:
            (events or self._events).add(self.writer.record(type, **fields), tag)

    def key(self, char, tag=None):
        self.event("key", tag=tag, key=char)

    def special(self, name, events=None):
        self.event("special", events=events, key=name)

    def observe(self, sentence_text):
        if not self.paused:
            self._events.observe(sentence_text)

    def begin_next(self, later_tags=frozenset()):
        """Detach the current sentence's events and start a fresh buffer. Key
        events for characters that now sit after the sentence (an edit moved
        them) carry over, so the sentence they ended up in is the one whose
        redaction check covers them."""
        events = self._events
        self._events = events.split(later_tags)
        return events

    def finish(self, events, raw_text, clean_text):
        """Close a sentence. Returns True if it is to be learned."""
        if self.paused:
            return False  # nothing typed while paused reaches a log
        if self.writer is None:
            return True
        events.finish(raw_text, self.writer)
        if clean_text is not None:
            self.writer.queue("clean", self.writer.clean_record(clean_text))
        return True

    def complete_sentence(self, raw_text, clean_text):
        return self.finish(self.begin_next(), raw_text, clean_text)

    def toggle_pause(self, unfinished_text):
        if not self.paused:
            if self.writer is not None:
                self._events.finish(unfinished_text, self.writer)
                self.writer.queue("raw", self.writer.record("pause"))
            self._events = SentenceEvents()
            self.paused = True
        else:
            self._events = SentenceEvents()
            self.paused = False
            if self.writer is not None:
                self.writer.queue("raw", self.writer.record("resume"))
        return self.paused

    def close(self, unfinished_text):
        if self.writer is not None and not self.paused:
            self._events.finish(unfinished_text, self.writer)


class OnlineTrainer:
    """The small online update (§10). Sentences queue until a batch is due;
    the update runs on a worker thread, so typing never waits on it.

    The Adam step writes new parameter arrays and swaps the workspace's
    reference in one assignment, so a prediction that took the old reference
    keeps a consistent model and never waits on a lock. `lock` only keeps a
    checkpoint from copying Adam moments mid-step.
    """

    def __init__(self, ws, tok, cfg, background=True):
        self.ws = ws
        self.tok = tok
        self.cfg = cfg
        self.train_state = zero_state(cfg.model.d_model)
        self.pending = []
        self._pending_tokens = 0
        self.lock = threading.Lock()
        self.last_loss = None
        self.last_duration_s = None
        self.error = None
        self._queue = None
        self._thread = None
        if background:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def add_sentence(self, text):
        ids = self.tok.encode(text)
        if not ids:
            return
        flags = self.tok.token_boundary_flags(text, ids, at_end=True)
        self.pending.append((ids, flags))
        self._pending_tokens += len(ids)
        t = self.cfg.train
        if len(self.pending) >= t.online_update_sentences or self._pending_tokens >= t.online_update_max_tokens:
            batch, self.pending, self._pending_tokens = self.pending, [], 0
            if self._queue is None:
                self._update(batch)
            else:
                self._queue.put(batch)

    def _run(self):
        while True:
            batch = self._queue.get()
            try:
                if batch is None:
                    return
                self._update(batch)
            except Exception as e:  # surfaced in the harness; typing goes on
                self.error = e
            finally:
                self._queue.task_done()

    def _update(self, batch):
        m, t = self.cfg.model, self.cfg.train
        ids, flags, cuts = [], [], []
        for sentence_ids, sentence_flags in batch:
            cuts += [True] + [False] * (len(sentence_ids) - 1)
            ids += sentence_ids
            flags += sentence_flags
        started = time.perf_counter()
        params = self.ws.params
        n = len(ids)
        # Only this thread changes params, Adam and the training state, so the
        # forward and backward passes need no lock; the lock covers the Adam
        # step and the swap, which a checkpoint must never see half done.
        logits, final, cache = forward_sequence(params, numpy.asarray(ids), flags, self.train_state, m.fast_decay, m.boundary_decay)
        if n > 1:
            # One stream from the snapshot: every position with a successor
            # in the batch has a target; gradients stop at sentence starts.
            targets = numpy.asarray(ids[1:] + [0])
            mask = numpy.arange(n) < n - 1
            loss_sum, ce = layers.softmax_ce_forward(logits, targets, mask)
            grads = backward_sequence(layers.softmax_ce_backward(ce) / (n - 1), cache, cuts)
            grads, _ = clip_global_norm(grads, t.grad_clip)
            with self.lock:
                self.ws.params = self.ws.adam.step(params, grads, t.online_lr, n - 1)
            self.last_loss = float(loss_sum) / (n - 1)
        self.train_state = replace(final, logits=None)
        self.last_duration_s = time.perf_counter() - started

    def wait(self):
        if self._queue is not None:
            self._queue.join()

    def close(self):
        """Finish any queued update and stop. A batch still filling is dropped:
        consolidation replays those sentences from the clean log."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None


@dataclass(frozen=True)
class Correction:
    start: int
    end: int
    typed: str
    suggestion: str


@dataclass(frozen=True)
class _Snap:
    start: int
    chunk: str
    state: object
    ids: tuple
    flags: tuple
    hidden: object

    @property
    def end(self):
        return self.start + len(self.chunk)


class LiveSession:
    """The live-typing model behind `pk live`, with no UI in it.

    Everything is computed from the text before the cursor. The region after
    the last finalised sentence is re-chunked on every change; the engine
    state keeps one snapshot per committed chunk there and, on any edit,
    restores the longest still-matching prefix and re-advances. So the state
    always equals the tokenisation of the visible committed text, and it is
    never reset. Text before the floor (finalised sentences, a sentence
    waiting on its correction, finished lines) is read-only.
    """

    def __init__(self, cfg, tok, ws, corrector, writer, trainer, clock=local_now, async_checks=False, learn=True):
        self.cfg = cfg
        self.tok = tok
        self.ws = ws
        self.corrector = corrector
        self.trainer = trainer
        self.clock = clock
        self.learn = learn
        self.predictor = Predictor(tok, ws.graph, cfg)
        self.recorder = Recorder(writer, ws.state.paused)
        self.handover = Handover.from_list(cfg.handover.window, ws.state.handover)

        self.text = ""
        self.tags = []
        self._next_tag = 0
        self.cursor = 0
        self.final_pos = 0
        self.line_start = 0
        self.breaks = []
        self.base_state = zero_state(cfg.model.d_model)
        self.base_hidden = None
        self.base_flag = False
        self.tail = deque(maxlen=INSPECT_TOKENS)
        self.snaps = []
        # Sentences whose boundary has committed but which wait on their
        # correction slot (or a spell check still in flight): [(end, events)].
        self.waiting = []
        self.carry = ""
        self.correction = None
        self.word_starts = {}
        self.committed = []
        self.recorded = set()
        self.outstanding = []
        self.prediction = None
        self.partial = ""
        self.error = None
        self._pred_len = 0
        self._live = self.base_state

        self.async_checks = async_checks
        self._check_jobs = queue.Queue()
        self._check_results = queue.Queue()
        if async_checks:
            threading.Thread(target=self._check_worker, daemon=True).start()
        self._refresh()

    # --- properties -------------------------------------------------------

    @property
    def paused(self):
        return self.recorder.paused

    @property
    def mode(self):
        return self.handover.mode()

    @property
    def floor(self):
        return max(self.final_pos, self.sentence_start, self.line_start)

    @property
    def sentence_start(self):
        return self.waiting[-1][0] if self.waiting else self.final_pos

    @property
    def boundary_fired(self):
        return self.snaps[-1].flags[-1] if self.snaps and self.snaps[-1].flags else self.base_flag

    def live_state(self):
        return self._live

    # --- keys -------------------------------------------------------------

    def type(self, chars):
        for c in chars:
            if c == "\n":
                self.enter()
                continue
            self._next_tag += 1
            self._replace(self.cursor, self.cursor, c, tag=self._next_tag)
            self.recorder.key(c, tag=self._next_tag)

    def backspace(self):
        self.recorder.special("backspace")
        if self.cursor > self.floor:
            self._replace(self.cursor - 1, self.cursor, "")

    def left(self):
        self.recorder.special("left")
        if self.cursor > self.floor:
            self.cursor -= 1
            self._refresh()

    def right(self):
        self.recorder.special("right")
        if self.cursor < len(self.text):
            self.cursor += 1
            self._refresh()

    def enter(self):
        self.recorder.special("enter")
        # The newline commits at once and the finished line becomes read-only;
        # chunking restarts after it, so a following space cannot merge in.
        self._replace(self.cursor, self.cursor, "\n", newline=True)

    def tab(self):
        primary = self.prediction.slots.primary if self.prediction else None
        self.recorder.special("tab")
        if primary is None:
            self._replace(self.cursor, self.cursor, "\t")
            return
        word = self.prediction.context.word
        self.recorder.event("accept", slot="primary", origin=primary.origin, word=primary.word)
        start = self.cursor - len(word)
        # A word can straddle the floor (a pause toggle mid-word): text before
        # the floor is final, so only the rest of the suggestion goes in.
        cut = max(0, self.floor - start)
        self._replace(start + cut, self.cursor, primary.word[cut:] + " ")

    def _events_for(self, pos):
        """The held events of the waiting sentence containing `pos`, if any."""
        start = self.final_pos
        for end, events in self.waiting:
            if start <= pos < end:
                return events
            start = end
        return None

    def insert(self):
        c = self.correction
        events = self._events_for(c.start) if c else None
        self.recorder.special("insert", events=events)
        if c is None:
            return
        self.recorder.event("correct", events=events, typed=c.typed, fixed=c.suggestion)
        self.correction = None
        self._replace(c.start, c.end, c.suggestion)

    def f2(self):
        c = self.correction
        events = self._events_for(c.start) if c else None
        self.recorder.special("f2", events=events)
        if c is None:
            return
        # Claiming a word also takes it out of the candidate tally.
        self.ws.tally.approve([c.typed])
        self.recorder.event("dict_add", events=events, word=c.typed)
        self.correction = None
        self._refresh()

    def toggle_pause(self):
        """End everything typed so far, as one unit, before the flag flips:
        waiting sentences finish in the state they were typed in, and the
        rest (wherever the cursor is) is closed unlearned."""
        self.correction = None
        self.outstanding = []
        self._release()
        unfinished = self.text[self.final_pos:]
        self.cursor = len(self.text)
        self._force_commit()
        self.recorder.toggle_pause(unfinished)
        self.ws.state.paused = self.recorder.paused
        self.ws.state.handover = self.handover.to_list()
        if self.learn:
            save_state(self.cfg.paths.data_dir, self.ws.state)
        self._refresh()

    def close(self):
        """Exit: commit the trailing chunk, finish what completed, and log the
        unfinished remainder (redacted, never learned)."""
        self.cursor = len(self.text)
        self._refresh(at_end=True)
        self.correction = None
        self.outstanding = []
        self._refresh(at_end=True)
        self.recorder.close(self.text[self.final_pos:])
        self.ws.state.handover = self.handover.to_list()

    # --- edits ------------------------------------------------------------

    def _replace(self, start, end, new, newline=False, tag=None):
        delta = len(new) - (end - start)
        self.text = self.text[:start] + new + self.text[end:]
        self.tags[start:end] = [tag] * len(new)
        if self.cursor >= end:
            self.cursor += delta
        elif self.cursor > start:
            self.cursor = start + len(new)
        if delta:
            self._shift(start, end, delta)
        if newline:
            self.breaks.append(self.cursor)
            self.line_start = self.cursor
        self._refresh()

    def _shift(self, start, end, delta):
        """Move remembered positions after an edit of text[start:end].

        Markers that end something (a waiting sentence, a line, a chunk
        break) move only when the edit lands strictly before them: text typed
        exactly at a marker belongs after it. Items that start at an edited
        replacement's start (a word being completed by Tab) keep their place.
        """

        def item(p):
            return p + delta if p >= end and not (p == start and end > start) else p

        def keep(p):
            return p <= start or p >= end

        def marker(p):
            return p + delta if p > start else p

        self.committed = [(item(s), w) for s, w in self.committed if keep(s)]
        self.outstanding = [(item(s), w) for s, w in self.outstanding if keep(s)]
        self.word_starts = {item(s): v for s, v in self.word_starts.items() if keep(s)}
        self.recorded = {item(s) for s in self.recorded if keep(s)}
        self.breaks = [marker(b) for b in self.breaks]
        self.waiting = [(marker(e), ev) for e, ev in self.waiting]
        self.line_start = marker(self.line_start)
        if self.correction and self.correction.start >= end:
            c = self.correction
            self.correction = replace(c, start=c.start + delta, end=c.end + delta)

    # --- the core recomputation ------------------------------------------

    def _chunks(self, upto, at_end):
        """[(start, chunk, committed, boundary marks)] for the editable region."""
        bc = self.cfg.model.boundary_chars
        cuts = [self.final_pos] + sorted(b for b in set(self.breaks) if self.final_pos < b < upto) + [upto]
        if upto in self.breaks and upto > self.final_pos:
            cuts.append(upto)
        out = []
        for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
            last = i == len(cuts) - 2
            segment = self.text[a:b]
            marks = {a + m for m in boundary_positions(segment, bc, at_end=at_end or not last)}
            pieces = self.tok.chunks(segment)
            pos = a
            for j, piece in enumerate(pieces):
                committed = at_end or not (last and j == len(pieces) - 1)
                out.append((pos, piece, committed, {m for m in marks if pos <= m < pos + len(piece)}))
                pos += len(piece)
        return out

    def _advance(self, chunks):
        params = self.ws.params
        m = self.cfg.model
        committed = [c for c in chunks if c[2]]
        keep = 0
        while keep < len(self.snaps) and keep < len(committed) and (self.snaps[keep].start, self.snaps[keep].chunk) == committed[keep][:2]:
            keep += 1
        snaps = self.snaps[:keep]
        state = snaps[-1].state if snaps else self.base_state
        for start, piece, _, marks in committed[keep:]:
            ids = self.tok.encode(piece)
            flags = self.tok.flags_for_marks(piece, ids, [mk - start for mk in marks])
            hidden = None
            for t, f in zip(ids, flags):
                state, hidden = step(params, state, t, f, m.fast_decay, m.boundary_decay, return_hidden=True)
            snaps.append(_Snap(start, piece, state, tuple(ids), tuple(flags), hidden))
        self.snaps = snaps
        partial = chunks[-1][1] if chunks and not chunks[-1][2] else ""
        return state, partial

    def _refresh(self, at_end=False):
        # Redaction is checked on every edit against all unfinished text,
        # after the cursor too: an edit can move typed characters into a
        # different sentence than the one whose events hold their keystrokes.
        self.recorder.observe(self.text[self.sentence_start:])
        chunks = self._chunks(self.cursor, at_end)
        self._live, self.partial = self._advance(chunks)
        self._track_words()
        moved = self.final_pos
        self._complete_sentences(chunks)
        if self.correction and self.text[self.correction.start:self.correction.end] != self.correction.typed:
            self.correction = None
        self._release()
        if self.final_pos != moved:
            self._live, self.partial = self._advance(self._chunks(self.cursor, at_end))
        self._predict()

    def _blocked(self, start, end):
        c = self.correction
        if c is not None and start <= c.start < end:
            return True
        return any(start <= s < end for s, _ in self.outstanding)

    def _complete_sentences(self, chunks):
        for start, piece, committed, marks in chunks:
            end = start + len(piece)
            if not committed or not marks or end <= self.sentence_start:
                continue
            later = {t for t in self.tags[end:] if t is not None}
            if self.waiting or self._blocked(self.sentence_start, end):
                self.waiting.append((end, self.recorder.begin_next(later)))
                # Later text must never re-chunk into a waiting sentence.
                self.breaks.append(end)
            else:
                self._finalise(end, self.recorder.begin_next(later))

    def _release(self):
        while self.waiting and not self._blocked(self.final_pos, self.waiting[0][0]):
            end, events = self.waiting.pop(0)
            self._finalise(end, events)

    def _finalise(self, end, events):
        raw = self.text[self.final_pos:end]
        has_words = WORD_RE.search(raw) is not None
        if self.learn and has_words and not self.paused:
            try:
                clean = clean_sentence(
                    self.carry + raw,
                    self.corrector,
                    self.ws.tally,
                    self.cfg.dictionary.candidate_min_count,
                    self.cfg.correct.max_edit_distance,
                    timestamp(self.clock()),
                )
            except HarperUnavailable as e:
                # Never lose typing to a checker failure: learn the sentence
                # redacted but uncorrected, and say so in the toolbar.
                self.error = f"spell check unavailable: {e}"
                clean = redact.redact(self.carry + raw)[0]
            self.recorder.finish(events, raw, clean)
            self.ws.graph.update(clean)
            if self.trainer is not None:
                self.trainer.add_sentence(clean)
            self.carry = ""
        else:
            self.recorder.finish(events, raw, None)
            # A piece with no words (a blank line) joins the next sentence's
            # clean text, so the clean stream concatenates to what was typed.
            self.carry = self.carry + raw if not has_words and not self.paused else ""
        self._move_base(end)

    def _move_base(self, end):
        done = [s for s in self.snaps if s.end <= end]
        if done:
            self.base_state = done[-1].state
            self.base_hidden = done[-1].hidden
            self.base_flag = done[-1].flags[-1] if done[-1].flags else self.base_flag
            for s in done:
                self.tail.extend((self.tok.token_str[t], t) for t in s.ids)
        self.snaps = [s for s in self.snaps if s.end > end]
        self.final_pos = end
        self.breaks = [b for b in self.breaks if b > end]
        self.committed = [(s, w) for s, w in self.committed if s >= end]
        self.outstanding = [(s, w) for s, w in self.outstanding if s >= end]
        self.word_starts = {s: v for s, v in self.word_starts.items() if s >= end}
        self.recorded = {s for s in self.recorded if s >= end}

    def _force_commit(self):
        """End the current region at the cursor as if the text ended there."""
        chunks = self._chunks(self.cursor, at_end=True)
        self._advance(chunks)
        self.correction = None
        self._move_base(self.cursor)
        self.breaks.append(self.cursor)
        self.line_start = self.cursor
        self.carry = ""
        self.committed = []
        self.word_starts = {}

    def _track_words(self):
        T = self.text[: self.cursor]
        old = self.prediction
        if old is not None and old.context.word == "" and self._pred_len < len(T):
            m = WORD_RE.match(T, self._pred_len)
            if m is not None and self._pred_len not in self.word_starts:
                self.word_starts[self._pred_len] = (
                    [w.lower() for w in old.engine_words[:3]],
                    [w.lower() for w, _, _ in old.graph_words[:3]],
                )
        # A word is committed once a non-word character follows it.
        now = [(m.start(), m.group()) for m in WORD_RE.finditer(T, self.final_pos) if m.end() < len(T)]
        new = [w for w in now if w not in self.committed]
        gone = [w for w in self.committed if w not in now]
        for s, _ in gone:
            if s in self.recorded and self.handover.records:
                self.handover.records.pop()
                self.recorded.discard(s)
        self.committed = now
        for s, word in new:
            self._on_word_commit(s, s + len(word), word)

    def _on_word_commit(self, start, end, word):
        # The correction slot clears when the next word is committed.
        self.correction = None
        if self.learn and not self.paused and start in self.word_starts and start not in self.recorded:
            engine, graph = self.word_starts[start]
            self.handover.record(word.lower() in engine, word.lower() in graph)
            self.recorded.add(start)
        sentence_start = self.sentence_start
        job = (start, end, word, sentence_start, self.text[sentence_start:end])
        if self.async_checks:
            self.outstanding.append((start, word))
            self._check_jobs.put(job)
        else:
            try:
                issues = self.corrector.check(job[4])
            except HarperUnavailable as e:
                self.error = f"spell check unavailable: {e}"
                issues = []
            self._apply_check(job, issues)

    def _check_worker(self):
        while True:
            job = self._check_jobs.get()
            try:
                issues = self.corrector.check(job[4])
            except Exception:
                issues = []
            self._check_results.put((job, issues))

    def poll(self):
        """Apply finished background spell checks. Returns True if anything changed."""
        changed = False
        while True:
            try:
                job, issues = self._check_results.get_nowait()
            except queue.Empty:
                break
            start, _, word = job[:3]
            if (start, word) in self.outstanding:
                self.outstanding.remove((start, word))
            changed |= self._apply_check(job, issues)
            changed = True
        if changed:
            self._refresh()
        return changed

    def _apply_check(self, job, issues):
        start, end, word, sentence_start, _ = job
        if self.text[start:end] != word or (start, word) not in self.committed:
            return False
        if self.committed and self.committed[-1] != (start, word):
            return False  # a later word has been committed since
        for i in issues:
            if i.kind == "spelling" and i.suggestions and (sentence_start + i.start, sentence_start + i.end) == (start, end):
                self.correction = Correction(start, end, word, i.suggestions[0])
                return True
        return False

    def _predict(self):
        # Only the editable region matters (the previous word is <s> across a
        # finalised boundary), so the cost stays flat however long the session.
        # A word straddling the region start is kept whole.
        # The window reaches back over a word straddling the region start, the
        # gap before it and the word before that, so prev is still right
        # after a mid-sentence pause toggle.
        start = self.final_pos
        for word_part in (True, False, True):
            while start > 0 and (WORD_RE.match(self.text, start - 1, start) is not None) == word_part:
                start -= 1
        T = self.text[start : self.cursor]
        self.prediction = self.predictor.predict(self.ws.params, self._live, T, self.partial, self.mode)
        self._pred_len = self.cursor

    # --- inspection -------------------------------------------------------

    def inspect(self):
        state = self._live
        tokens = list(self.tail) + [(self.tok.token_str[t], t) for s in self.snaps for t in s.ids]
        top = []
        if state.logits is not None:
            logits = numpy.asarray(state.logits, dtype=numpy.float64)
            p = numpy.exp(logits - logits.max())
            p /= p.sum()
            idx = numpy.argsort(-p)[:INSPECT_TOP]
            top = [(self.tok.token_str[i], float(p[i])) for i in idx]
        hidden = self.snaps[-1].hidden if self.snaps else self.base_hidden
        return {
            "tokens": tokens[-INSPECT_TOKENS:],
            "S_s_norm": float(numpy.linalg.norm(state.S_s)),
            "S_f_norm": float(numpy.linalg.norm(state.S_f)),
            "z_s_norm": float(numpy.linalg.norm(state.z_s)),
            "z_f_norm": float(numpy.linalg.norm(state.z_f)),
            "top": top,
            "mlp_mean": None if hidden is None else float(hidden.mean()),
            "mlp_frac_pos": 0.0 if hidden is None else float((hidden > 0).mean()),
            "boundary_fired": bool(self.boundary_fired),
        }
