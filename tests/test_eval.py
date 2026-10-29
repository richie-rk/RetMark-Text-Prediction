import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pk import evaluation
from pk.cli.main import app
from pk.config import load_config

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copy(ROOT / "config.toml", tmp_path / "config.toml")
    archive = tmp_path / "data" / "archive" / "clean"
    archive.mkdir(parents=True)
    shutil.copy(FIXTURES / "train.jsonl", archive / "2026-09-01.jsonl")
    shutil.copy(FIXTURES / "holdout.jsonl", tmp_path / "holdout.jsonl")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def one_json(result):
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 1, result.stdout
    return json.loads(lines[0])


def test_eval_reports_metrics_deterministically_and_writes_nothing(project):
    before = snapshot(project)
    a = runner.invoke(app, ["eval", "--holdout", "holdout.jsonl", "--json"])
    b = runner.invoke(app, ["eval", "--holdout", "holdout.jsonl", "--json"])
    assert a.exit_code == 0, a.stdout
    ra, rb = one_json(a), one_json(b)
    assert ra == rb
    assert set(ra) >= {"sentences", "tokens", "mean_ce", "perplexity", "top3_hit_rate"}
    assert ra["sentences"] == 3
    assert ra["perplexity"] == pytest.approx(math.exp(ra["mean_ce"]), rel=1e-12)
    assert 0.0 <= ra["top3_hit_rate"] <= 1.0
    assert snapshot(project) == before


def test_trained_toy_model_beats_untrained(project):
    cfg = load_config(project / "config.toml")
    holdout = evaluation.load_sentences(project / "holdout.jsonl")
    untrained = evaluation.evaluate_sentences(evaluation.fresh_params(cfg), cfg, holdout)
    trained = evaluation.trial(cfg, evaluation.history(cfg, exclude=holdout))
    result = evaluation.evaluate_sentences(trained, cfg, holdout)
    assert result["perplexity"] < untrained["perplexity"]


def test_sweep_runs_each_value_and_mutates_nothing(project):
    before = snapshot(project)
    result = runner.invoke(app, ["sweep", "fast_decay", "0.9,1.0", "--holdout", "holdout.jsonl", "--json"])
    assert result.exit_code == 0, result.stdout
    out = one_json(result)
    assert out["param"] == "model.fast_decay"
    assert [row["value"] for row in out["rows"]] == [0.9, 1.0]
    assert all(row["perplexity"] > 0 and 0 <= row["top3_hit_rate"] <= 1 for row in out["rows"])
    assert snapshot(project) == before
    human = runner.invoke(app, ["sweep", "train.online_lr", "3e-4", "--holdout", "holdout.jsonl"])
    assert human.exit_code == 0 and "perplexity" in human.stdout


def test_sweep_trials_are_reproducible(project):
    cfg = load_config(project / "config.toml")
    holdout = evaluation.load_sentences(project / "holdout.jsonl")
    rows_a = evaluation.sweep(cfg, "online_update_sentences", ["1", "4"], holdout)
    rows_b = evaluation.sweep(cfg, "online_update_sentences", ["1", "4"], holdout)
    assert rows_a == rows_b
    assert rows_a[0]["perplexity"] != rows_a[1]["perplexity"]


def test_sweep_rejects_unknown_or_invalid_values(project):
    assert runner.invoke(app, ["sweep", "nonsense", "1", "--holdout", "holdout.jsonl", "--json"]).exit_code == 1
    assert runner.invoke(app, ["sweep", "fast_decay", "2.0", "--holdout", "holdout.jsonl", "--json"]).exit_code == 1


def test_history_excludes_held_out_lines(project):
    cfg = load_config(project / "config.toml")
    holdout = evaluation.load_sentences(project / "holdout.jsonl")
    archive = project / "data" / "archive" / "clean"
    shutil.copy(FIXTURES / "holdout.jsonl", archive / "2026-09-02.jsonl")
    hist = evaluation.history(cfg, exclude=holdout)
    keys = {(s.session, s.ts, s.text) for s in hist}
    assert not keys & {(s.session, s.ts, s.text) for s in holdout}
    assert len(hist) == 14
