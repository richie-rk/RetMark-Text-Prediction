from pathlib import Path

import pytest

from pk import evaluation
from pk.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_top3_hit_rate_stays_above_floor():
    """Retrain from the seed with the current config.toml and score the frozen
    held-out day, so a bad tuning change fails here. Dormant until enough
    real typing has been frozen into data/holdout.jsonl."""
    cfg = load_config(ROOT / "config.toml")
    path = cfg.paths.data_dir / "holdout.jsonl"
    holdout = evaluation.load_sentences(path) if path.exists() else []
    if len(holdout) < cfg.eval.holdout_min_sentences:
        pytest.skip(f"held-out set has {len(holdout)} sentences, fewer than {cfg.eval.holdout_min_sentences}")
    params = evaluation.trial(cfg, evaluation.history(cfg, exclude=holdout))
    result = evaluation.evaluate_sentences(params, cfg, holdout)
    assert result["top3_hit_rate"] > cfg.eval.top3_floor
