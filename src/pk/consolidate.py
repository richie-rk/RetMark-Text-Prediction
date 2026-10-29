import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy

from pk.engine import layers
from pk.engine.model import backward_sequence, forward_sequence
from pk.engine.optim import clip_global_norm
from pk.engine.state import zero_state
from pk.logs import local_now, parse_line, pending_clean
from pk.persistence import InstanceLock, Workspace, save_model, save_state
from pk.tokenizer import Tokenizer
from pk.xp import consolidation_backend


@dataclass(frozen=True)
class Sentence:
    file: str
    offset: int
    session: str
    ts: str
    text: str


def load_archive(data_dir):
    out = []
    folder = Path(data_dir) / "archive" / "clean"
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.jsonl")):
        offset = 0
        with open(path, "rb") as f:
            for line in f:
                r = parse_line(line) if line.strip() else None
                if r is not None:
                    out.append(Sentence(path.name, offset, r["session"], r["ts"], r["text"]))
                offset += len(line)
    return out


def replay_sample(archive, ratio, pending_count, rng):
    n = min(int(round(ratio * pending_count)), len(archive))
    if n <= 0:
        return []
    picks = rng.choice(len(archive), size=n, replace=False)
    return [archive[i] for i in sorted(picks)]


def session_texts(sentences):
    """Group by session in original order and concatenate each session's
    sentences exactly as stored: each clean entry already holds its leading
    whitespace and terminator, so the result is the stream that was typed."""
    ordered = sorted(sentences, key=lambda s: (s.file, s.offset))
    groups = {}
    for s in ordered:
        groups.setdefault(s.session, []).append(s.text)
    return [(session, "".join(texts)) for session, texts in groups.items()]


def token_streams(tok, sentences):
    streams = []
    for _, text in session_texts(sentences):
        ids = tok.encode(text)
        if len(ids) > 1:
            streams.append((ids, tok.token_boundary_flags(text, ids, at_end=True)))
    return streams


def windows(ids, window, flags=None):
    """Cut a stream into (inputs, targets[, flags]) windows. Position i
    predicts i+1, so targets run across window edges and the final token
    is never an input."""
    out = []
    for start in range(0, len(ids) - 1, window):
        end = min(start + window, len(ids) - 1)
        item = (list(ids[start:end]), list(ids[start + 1 : end + 1]))
        if flags is not None:
            item += (list(flags[start:end]),)
        out.append(item)
    return out


def loss_and_grads(params, inputs, targets, flags, cfg, state0=None, with_grads=True):
    """Summed cross-entropy over a window and its gradients (of the sum)."""
    m = cfg.model
    xp = layers.array_module(params["E"])
    if state0 is None:
        state0 = zero_state(m.d_model, xp, params["E"].dtype)
    logits, final, cache = forward_sequence(params, inputs, flags, state0, m.fast_decay, m.boundary_decay)
    targets = xp.asarray(targets)
    mask = xp.ones(len(inputs), dtype=bool)
    loss_sum, ce = layers.softmax_ce_forward(logits, targets, mask)
    grads = backward_sequence(layers.softmax_ce_backward(ce), cache) if with_grads else None
    return float(loss_sum), len(inputs), grads, replace(final, logits=None), logits


def evaluate(params, streams, cfg, top3=True):
    """(summed CE, target count, top-3 token hits) over whole streams, each
    from zero state with the state carried across windows."""
    m = cfg.model
    xp = layers.array_module(params["E"])
    total, count, hits = 0.0, 0, 0
    for ids, flags in streams:
        state = zero_state(m.d_model, xp, params["E"].dtype)
        for inputs, targets, wflags in windows(ids, cfg.train.bptt_window, flags):
            loss, n, _, state, logits = loss_and_grads(params, inputs, targets, wflags, cfg, state, with_grads=False)
            total += loss
            count += n
            if top3:
                # A target is in the top 3 when fewer than 3 logits beat it.
                t = xp.asarray(targets)
                target_logit = xp.take_along_axis(logits, t[:, None], axis=-1)
                hits += int(((logits > target_logit).sum(axis=-1) < 3).sum())
    return total, count, hits


def train(params, adam, streams, cfg, lr, epochs, rng):
    """Truncated BPTT: each session starts from zero state and carries it,
    without gradient, across its windows. Gradients accumulate over
    batch_size windows per Adam step; the loss is the token-weighted mean."""
    t, m = cfg.train, cfg.model
    xp = layers.array_module(params["E"])
    order = list(range(len(streams)))
    acc, acc_n, acc_windows = None, 0, 0

    def apply(params, acc, acc_n):
        grads = {k: g / acc_n for k, g in acc.items()}
        grads, _ = clip_global_norm(grads, t.grad_clip)
        return adam.step(params, grads, lr, acc_n)

    for _ in range(epochs):
        rng.shuffle(order)
        for si in order:
            ids, flags = streams[si]
            state = zero_state(m.d_model, xp, params["E"].dtype)
            for inputs, targets, wflags in windows(ids, t.bptt_window, flags):
                _, n, grads, state, _ = loss_and_grads(params, inputs, targets, wflags, cfg, state)
                acc = grads if acc is None else {k: acc[k] + grads[k] for k in acc}
                acc_n += n
                acc_windows += 1
                if acc_windows == t.batch_size:
                    params = apply(params, acc, acc_n)
                    acc, acc_n, acc_windows = None, 0, 0
    if acc is not None:
        params = apply(params, acc, acc_n)
    return params


def _to(xp, tree):
    return {k: xp.asarray(v) for k, v in tree.items()}


def _host(tree):
    return {k: v if isinstance(v, numpy.ndarray) else v.get() for k, v in tree.items()}


def _move(src, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        # Never overwrite archived lines: append instead.
        with open(dst, "ab") as out, open(src, "rb") as inp:
            out.write(inp.read())
            out.flush()
            os.fsync(out.fileno())
        src.unlink()
    else:
        os.replace(src, dst)


def rollover(data_dir, offsets, today):
    """Archive past-day clean files that are fully consumed, with their raw
    files; past-day raw files with no clean file go too. Returns the
    offsets without the archived files."""
    data = Path(data_dir)
    clean, raw = data / "logs" / "clean", data / "logs" / "raw"
    cutoff = today.isoformat()
    offsets = dict(offsets)
    if clean.exists():
        for path in sorted(clean.glob("*.jsonl")):
            if path.stem < cutoff and offsets.get(path.name, 0) >= path.stat().st_size:
                _move(path, data / "archive" / "clean")
                offsets.pop(path.name, None)
                if (raw / path.name).exists():
                    _move(raw / path.name, data / "archive" / "raw")
    if raw.exists():
        for path in sorted(raw.glob("*.jsonl")):
            if path.stem < cutoff and not (clean / path.name).exists():
                _move(path, data / "archive" / "raw")
    return offsets


def run(cfg, today=None, clock=local_now):
    """The consolidation job (§10). Raises LockHeld if the harness is running."""
    data = cfg.paths.data_dir
    lock = InstanceLock(data).acquire()
    try:
        started = time.perf_counter()
        today = today or clock().date()
        tok = Tokenizer(cfg.model.boundary_chars)
        ws = Workspace.load(cfg, tok.vocab_size)
        lines = pending_clean(data, ws.state.offsets)
        xp, device = consolidation_backend()
        summary = {"sentences": len(lines), "replay_sentences": 0, "tokens": 0, "loss_before": None, "loss_after": None, "device": device}
        if not lines:
            summary.update(nothing_pending=True, duration_s=time.perf_counter() - started)
            ws.state.offsets = rollover(data, ws.state.offsets, today)
            save_state(data, ws.state)
            return summary

        pending = [Sentence(name, start, r["session"], r["ts"], r["text"]) for name, start, _, r in lines]
        rng = numpy.random.default_rng(cfg.model.seed)
        replay = replay_sample(load_archive(data), cfg.train.replay_ratio, len(pending), rng)
        pending_streams = token_streams(tok, pending)
        train_streams = token_streams(tok, pending + replay)

        params = _to(xp, ws.params)
        adam = ws.adam
        adam.m, adam.v = _to(xp, adam.m), _to(xp, adam.v)
        before, count, _ = evaluate(params, pending_streams, cfg, top3=False)
        params = train(params, adam, train_streams, cfg, cfg.train.consolidation_lr, cfg.train.consolidation_epochs, rng)
        after, _, _ = evaluate(params, pending_streams, cfg, top3=False)

        ws.params = _host(params)
        adam.m, adam.v = _host(adam.m), _host(adam.v)
        save_model(data, ws.params, adam)
        for name, _, end, _ in lines:
            ws.state.offsets[name] = max(ws.state.offsets.get(name, 0), end)
        ws.state.offsets = rollover(data, ws.state.offsets, today)
        save_state(data, ws.state)

        summary.update(
            nothing_pending=False,
            replay_sentences=len(replay),
            tokens=sum(len(ids) for ids, _ in pending_streams),
            loss_before=before / count if count else None,
            loss_after=after / count if count else None,
            duration_s=time.perf_counter() - started,
        )
        return summary
    finally:
        lock.release()
