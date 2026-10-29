import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pk.cli.main import app
from pk.persistence import InstanceLock, load_state

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copy(ROOT / "config.toml", tmp_path / "config.toml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def pk(*args):
    return runner.invoke(app, list(args), catch_exceptions=False)


def one_json(result):
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 1, result.stdout
    return json.loads(lines[0])


def seed_graph(project):
    from pk.graph import WordGraph
    from pk.persistence import save_graph

    g = WordGraph((".", "?", "!", "\n"))
    for _ in range(5):
        g.update("richy koshy.")
    g.update("hello world.")
    save_graph(project / "data", g)


@pytest.mark.parametrize(
    "args",
    [
        ["status"],
        ["pause"],
        ["resume"],
        ["words", "list"],
        ["words", "approve", "koshy"],
        ["words", "reject", "teh"],
        ["graph", "top", "richy"],
        ["graph", "stats"],
        ["consolidate"],
        ["generate", "--prompt", "Hello", "--tokens", "3", "--temperature", "0"],
        ["predict", "hello wor"],
    ],
)
def test_every_command_prints_exactly_one_json_object(project, args):
    result = pk(*args, "--json")
    assert result.exit_code == 0, result.stdout
    assert isinstance(one_json(result), dict)


@pytest.mark.parametrize(
    "args",
    [["consolidate"], ["pause"], ["resume"], ["words", "approve", "x"], ["words", "reject", "x"]],
)
def test_writing_commands_exit_2_while_the_harness_holds_the_lock(project, args):
    lock = InstanceLock(project / "data").acquire()
    try:
        result = pk(*args, "--json")
        assert result.exit_code == 2
        assert one_json(result)["ok"] is False
        assert pk(*args).exit_code == 2
    finally:
        lock.release()


@pytest.mark.parametrize("args", [["status"], ["graph", "stats"], ["words", "list"], ["predict", "hi"]])
def test_read_only_commands_run_while_locked(project, args):
    lock = InstanceLock(project / "data").acquire()
    try:
        assert pk(*args, "--json").exit_code == 0
    finally:
        lock.release()


def test_words_review_exits_2_while_locked(project):
    lock = InstanceLock(project / "data").acquire()
    try:
        assert pk("words", "review").exit_code == 2
    finally:
        lock.release()


def test_pause_and_resume_set_the_flag(project):
    pk("pause")
    assert load_state(project / "data").paused is True
    pk("pause")
    assert load_state(project / "data").paused is True
    pk("resume")
    assert load_state(project / "data").paused is False
    assert one_json(pk("status", "--json"))["paused"] is False


def test_config_error_exits_1(project):
    (project / "config.toml").write_text("[model]\nd_model = 'x'\n", encoding="utf-8")
    result = pk("status", "--json")
    assert result.exit_code == 1
    assert "config error" in one_json(result)["error"]


def test_model_mismatch_exits_1(project):
    from pk.engine.model import init_params
    from pk.engine.optim import Adam
    from pk.persistence import save_model

    params = init_params(8, 16, 50257, 0.02, 1)
    save_model(project / "data", params, Adam(params, 0.9, 1000, 1e-8))
    result = pk("generate", "--prompt", "hi", "--tokens", "1", "--temperature", "0", "--json")
    assert result.exit_code == 1
    assert "d_model=8" in one_json(result)["error"]


def test_predict_reports_slots_mode_and_correction(project):
    seed_graph(project)
    out = one_json(pk("predict", "richy ", "--json"))
    assert out["mode"] == "GRAPH"
    assert out["slots"]["primary"] == {"word": "koshy", "origin": "graph"}
    assert out["slots"]["side"] is None
    out = one_json(pk("predict", "I saw thier ", "--json"))
    assert out["correction"] == {"typed": "thier", "suggestion": "their"}
    # predict writes nothing.
    assert not (project / "data" / "model.safetensors").exists()
    assert not (project / "data" / "logs").exists()


def test_generate_greedy_is_deterministic_and_sampling_is_seeded(project):
    a = one_json(pk("generate", "--prompt", "The cat", "--tokens", "5", "--temperature", "0", "--json"))
    b = one_json(pk("generate", "--prompt", "The cat", "--tokens", "5", "--temperature", "0", "--json"))
    assert a == b and len(a["tokens"]) == 5
    c = one_json(pk("generate", "--prompt", "The cat", "--tokens", "5", "--temperature", "1.5", "--json"))
    d = one_json(pk("generate", "--prompt", "The cat", "--tokens", "5", "--temperature", "1.5", "--json"))
    assert c == d
    assert pk("generate", "--prompt", "x", "--tokens", "2", "--temperature", "0").stdout.startswith("x")


def test_graph_top_and_stats(project):
    seed_graph(project)
    top = one_json(pk("graph", "top", "Richy", "--json"))
    assert top["followers"] == [{"word": "koshy", "count": 5, "probability": 1.0}]
    assert one_json(pk("graph", "top", "<s>", "--n", "1", "--json"))["followers"][0]["word"] == "richy"
    stats = one_json(pk("graph", "stats", "--json"))
    assert stats["nodes"] == 5 and stats["edges"] == 4
    assert stats["strongest"][0] == {"from": "richy", "to": "koshy", "count": 5, "probability": 1.0}


def test_words_flow(project):
    from pk.dictionary import CandidateTally, PersonalDictionary

    data = project / "data"
    t = CandidateTally(data / "candidates.json", PersonalDictionary(data / "dictionary.txt"), PersonalDictionary(data / "rejected.txt"))
    for _ in range(3):
        t.increment("Richy", "ts")
    t.increment("teh", "ts")
    t.save()
    listed = one_json(pk("words", "list", "--json"))["candidates"]
    assert [(r["word"], r["count"]) for r in listed] == [("Richy", 3), ("teh", 1)]
    pk("words", "approve", "Richy")
    pk("words", "reject", "teh")
    assert (data / "dictionary.txt").read_text(encoding="utf-8") == "Richy\n"
    assert (data / "rejected.txt").read_text(encoding="utf-8") == "teh\n"
    assert one_json(pk("words", "list", "--json"))["candidates"] == []


def test_words_review_ticks_and_rejects(project, monkeypatch):
    from pk.cli import main
    from pk.dictionary import CandidateTally, PersonalDictionary

    data = project / "data"
    t = CandidateTally(data / "candidates.json", PersonalDictionary(data / "dictionary.txt"), PersonalDictionary(data / "rejected.txt"))
    for w in ("richy", "koshy", "teh"):
        for _ in range(3):
            t.increment(w, "ts")
    t.increment("rare", "ts")
    t.save()
    monkeypatch.setattr(main, "review_dialog", lambda candidates: ["koshy", "richy"])
    assert pk("words", "review").exit_code == 0
    assert sorted((data / "dictionary.txt").read_text(encoding="utf-8").split()) == ["koshy", "richy"]
    assert (data / "rejected.txt").read_text(encoding="utf-8") == "teh\n"
    # Below candidate_min_count: not offered, left alone.
    assert [r["word"] for r in one_json(pk("words", "list", "--json"))["candidates"]] == ["rare"]


def test_words_review_cancel_changes_nothing(project, monkeypatch):
    from pk.cli import main
    from pk.dictionary import CandidateTally, PersonalDictionary

    data = project / "data"
    t = CandidateTally(data / "candidates.json", PersonalDictionary(data / "dictionary.txt"), PersonalDictionary(data / "rejected.txt"))
    for _ in range(3):
        t.increment("richy", "ts")
    t.save()
    monkeypatch.setattr(main, "review_dialog", lambda candidates: None)
    assert pk("words", "review").exit_code == 0
    assert not (data / "dictionary.txt").exists() and not (data / "rejected.txt").exists()


def test_review_dialog_drives_with_keys():
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from pk.cli.main import review_dialog

    with create_pipe_input() as inp:
        # Tick the first entry, then Tab to OK and press Enter.
        inp.send_text(" \t\r")
        with create_app_session(input=inp, output=DummyOutput()):
            assert review_dialog([("richy", 3), ("teh", 3)]) == ["richy"]


def run_main(monkeypatch, capsys, *args):
    import sys

    from pk.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["pk", *args])
    with pytest.raises(SystemExit) as e:
        cli_main.main()
    return e.value.code, capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [["generate", "--json"], ["graph", "top", "--json"], ["generate", "--prompt", "x", "--tokens", "-1", "--temperature", "0", "--json"]],
)
def test_usage_errors_exit_1_with_one_json_object(project, monkeypatch, capsys, args):
    code, out = run_main(monkeypatch, capsys, *args)
    assert code == 1
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 1 and json.loads(lines[0])["ok"] is False


def test_unexpected_errors_still_print_one_json_object(project):
    (project / "data").mkdir()
    (project / "data" / "state.json").write_text("{not json", encoding="utf-8")
    result = pk("status", "--json")
    assert result.exit_code == 1
    assert one_json(result)["ok"] is False


def test_eval_on_empty_or_malformed_holdout_is_an_error(project):
    (project / "empty.jsonl").write_text("", encoding="utf-8")
    result = pk("eval", "--holdout", "empty.jsonl", "--json")
    assert result.exit_code == 1 and one_json(result)["ok"] is False
    (project / "bad.jsonl").write_text('{"ts": "x"}\n', encoding="utf-8")
    result = pk("eval", "--holdout", "bad.jsonl", "--json")
    assert result.exit_code == 1 and one_json(result)["ok"] is False
