"""Offline evaluation and parameter sweeps (§13). Nothing here writes to disk."""
import functools
import json
import math
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy

from pk import consolidate
from pk.config import ConfigError, _validate
from pk.engine.model import init_params
from pk.engine.optim import Adam
from pk.session import OnlineTrainer
from pk.tokenizer import Tokenizer


@functools.lru_cache(maxsize=4)
def tokenizer(boundary_chars):
    return Tokenizer(boundary_chars)


def load_sentences(path):
    path = Path(path)
    out = []
    offset = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append(consolidate.Sentence(path.name, offset, r["session"], r["ts"], r["text"]))
            offset += len(line)
    return out


def _key(s):
    return (s.session, s.ts, s.text)


def history(cfg, exclude=()):
    """Every clean sentence on record, archived or not, minus held-out lines."""
    data = cfg.paths.data_dir
    skip = {_key(s) for s in exclude}
    out = []
    for folder in (data / "archive" / "clean", data / "logs" / "clean"):
        if folder.exists():
            for path in sorted(folder.glob("*.jsonl")):
                out.extend(s for s in load_sentences(path) if _key(s) not in skip)
    return out


def chronological(sentences):
    return sorted(sentences, key=lambda s: (datetime.fromisoformat(s.ts), s.file, s.offset))


def fresh_params(cfg):
    m = cfg.model
    return init_params(m.d_model, m.mlp_hidden, tokenizer(m.boundary_chars).vocab_size, m.init_std, m.seed)


def trial(cfg, sentences):
    """Train a fresh model the way the live system would have: the online
    path in chronological order (training state reset per session, as each
    harness run starts from zero), then one consolidation over the same
    sentences with replay off. Same seed, same starting weights, in memory."""
    m, t = cfg.model, cfg.train
    tok = tokenizer(m.boundary_chars)
    params = fresh_params(cfg)
    ws = SimpleNamespace(params=params, adam=Adam(params, t.adam_beta1, t.adam_beta2_half_life_tokens, t.adam_eps))
    ordered = chronological(sentences)
    trainer, session = None, None
    for s in ordered:
        if s.session != session:
            trainer, session = OnlineTrainer(ws, tok, cfg, background=False), s.session
        trainer.add_sentence(s.text)
    rng = numpy.random.default_rng(m.seed)
    streams = consolidate.token_streams(tok, ordered)
    return consolidate.train(ws.params, ws.adam, streams, cfg, t.consolidation_lr, t.consolidation_epochs, rng)


def evaluate_sentences(params, cfg, sentences):
    tok = tokenizer(cfg.model.boundary_chars)
    total, count, hits = consolidate.evaluate(params, consolidate.token_streams(tok, sentences), cfg)
    mean = total / count if count else float("nan")
    return {
        "sentences": len(sentences),
        "tokens": count,
        "mean_ce": mean,
        "perplexity": math.exp(mean) if count else float("nan"),
        "top3_hit_rate": hits / count if count else float("nan"),
    }


def resolve_param(cfg, name):
    """'fast_decay' or 'model.fast_decay' -> ('model', 'fast_decay', type)."""
    matches = []
    for section in fields(cfg):
        if "." in name and not name.startswith(section.name + "."):
            continue
        key = name.split(".", 1)[1] if "." in name else name
        for f in fields(section.type):
            if f.name == key:
                matches.append((section.name, f.name, f.type))
    if len(matches) != 1:
        raise ConfigError(f"unknown or ambiguous parameter {name!r}")
    return matches[0]


def with_param(cfg, section, key, kind, raw):
    if kind is int:
        value = int(raw)
    elif kind is float:
        value = float(raw)
    elif kind is str:
        value = raw
    else:
        raise ConfigError(f"{section}.{key} cannot be swept")
    new = replace(cfg, **{section: replace(getattr(cfg, section), **{key: value})})
    _validate(new)
    return new, value


def sweep(cfg, name, values, holdout):
    section, key, kind = resolve_param(cfg, name)
    trials = [with_param(cfg, section, key, kind, raw) for raw in values]
    rows = []
    for trial_cfg, value in trials:
        params = trial(trial_cfg, history(trial_cfg, exclude=holdout))
        result = evaluate_sentences(params, trial_cfg, holdout)
        rows.append({"value": value, "perplexity": result["perplexity"], "mean_ce": result["mean_ce"], "top3_hit_rate": result["top3_hit_rate"]})
    return rows
