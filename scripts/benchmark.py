"""Phase 1 acceptance numbers (SPEC §13). Run on the target Windows machine:

    python scripts/benchmark.py            # all measurements
    python scripts/benchmark.py --quick    # skip the consolidation run

Works in a temporary data directory with fresh weights; never touches data/.
Keystroke times are the compute from keystroke to slots ready, excluding
the toolbar debounce.
"""
import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEXT = (
    "The morning was quiet and the coffee was still too hot to drink. I opened the laptop and read through "
    "the notes from yesterday. There were three things left to finish before the review. First, the parser "
    "still dropped trailing spaces. Second, the tests for the new endpoint were flaky on slow machines! "
    "Third, nobody had written the release notes yet. I started with the parser because it was the smallest "
    "problem. It took about an hour, which was longer than I had hoped. After lunch I looked at the flaky "
    "tests again? They were waiting on a timer that was far too short. Doubling it fixed every run.\n"
    "Tomorrow I will write the release notes and send them to the team for comments. "
)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summary_ms(values):
    return {"n": len(values), "p50_ms": pct(values, 0.5) * 1e3, "p95_ms": pct(values, 0.95) * 1e3, "max_ms": max(values) * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the consolidation measurement")
    ap.add_argument("--consolidation-tokens", type=int, default=100_000)
    args = ap.parse_args()

    from pk.config import load_config
    from pk.corrector import Corrector, HarperLS
    from pk.logs import LogWriter
    from pk.persistence import Workspace
    from pk.session import LiveSession, OnlineTrainer
    from pk.tokenizer import Tokenizer

    data = Path(tempfile.mkdtemp(prefix="pk-bench-"))
    base = load_config(ROOT / "config.toml")
    cfg = replace(base, paths=replace(base.paths, data_dir=data))
    tok = Tokenizer(cfg.model.boundary_chars)
    results = {"platform": platform.platform(), "python": sys.version.split()[0]}
    harper = HarperLS(cfg.correct.harper_ls_path)
    try:
        # 1. Keystrokes with the online trainer idle, then with updates running.
        for label, train in (("keystroke_idle", {"online_update_sentences": 10**9}), ("keystroke_during_updates", {"online_update_sentences": 1})):
            c = replace(cfg, train=replace(cfg.train, **train))
            ws = Workspace.load(c, tok.vocab_size)
            writer = LogWriter(data, c.logs.flush_interval_s, label)
            trainer = OnlineTrainer(ws, tok, c)
            corrector = Corrector(harper, c.correct.dialect, ws.dictionary)
            corrector.warm()
            session = LiveSession(c, tok, ws, corrector, writer, trainer, async_checks=True)
            times = []
            busy = []
            for ch in TEXT * 2:
                updating = not trainer._queue.empty() if trainer._queue is not None else False
                t0 = time.perf_counter()
                if ch == "\n":
                    session.enter()
                else:
                    session.type(ch)
                session.poll()
                dt = time.perf_counter() - t0
                times.append(dt)
                if updating or trainer._queue.unfinished_tasks:
                    busy.append(dt)
            session.close()
            trainer.close()
            writer.close()
            results[label] = summary_ms(times)
            if label == "keystroke_during_updates":
                results[label]["keystrokes_while_update_running"] = summary_ms(busy) if busy else None
                results[label]["online_updates"] = ws.adam.step_count
    finally:
        harper.close()

    # 2. Online update for a 30-token sentence, and a default 4-sentence batch.
    ws = Workspace.load(cfg, tok.vocab_size)
    words = "the quick brown fox jumps over the lazy dog while the farmer watches from the old wooden porch and the sun sets slowly behind the distant hills every evening in late summer".split()
    sentence = "The"
    for w in words[1:]:
        if len(tok.encode(sentence + " " + w + ".")) <= 30:
            sentence += " " + w
    sentence += "."
    one = replace(cfg, train=replace(cfg.train, online_update_sentences=1))
    trainer = OnlineTrainer(ws, tok, one, background=False)
    durations = []
    for _ in range(30):
        trainer.add_sentence(sentence)
        durations.append(trainer.last_duration_s)
    results["online_update_30_tokens"] = {"tokens": len(tok.encode(sentence)), **summary_ms(durations)}
    trainer = OnlineTrainer(ws, tok, cfg, background=False)
    batch = []
    for _ in range(10):
        for _ in range(cfg.train.online_update_sentences):
            trainer.add_sentence(" " + sentence)
        batch.append(trainer.last_duration_s)
    results["online_update_default_batch"] = {"sentences": cfg.train.online_update_sentences, **summary_ms(batch)}

    # 3. Consolidation throughput.
    if not args.quick:
        from pk import consolidate
        from pk.xp import consolidation_backend

        _, device = consolidation_backend()
        cdata = Path(tempfile.mkdtemp(prefix="pk-bench-cons-"))
        ccfg = replace(cfg, paths=replace(cfg.paths, data_dir=cdata), train=replace(cfg.train, replay_ratio=0.0))
        per = len(tok.encode(TEXT))
        target = args.consolidation_tokens
        lines = []
        session_text = TEXT
        reps = max(1, target // per)
        for i in range(reps):
            for j, s in enumerate(session_text.split(". ")):
                text = s if j == 0 and i == 0 else " " + s
                lines.append({"ts": "2000-01-01T00:00:00.000+00:00", "session": f"s{i % 20}", "source": "cli", "text": text + "."})
        (cdata / "logs" / "clean").mkdir(parents=True)
        with open(cdata / "logs" / "clean" / "2000-01-01.jsonl", "w", encoding="utf-8") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        t0 = time.perf_counter()
        s = consolidate.run(ccfg)
        results["consolidation"] = {**s, "wall_s": time.perf_counter() - t0, "target_tokens": target}
        shutil.rmtree(cdata, ignore_errors=True)

    shutil.rmtree(data, ignore_errors=True)
    print(json.dumps(results, indent=1))
    verdicts = [
        ("keystroke p95 < 30 ms (idle)", results["keystroke_idle"]["p95_ms"] < 30),
        ("keystroke p95 < 30 ms (with online updates)", results["keystroke_during_updates"]["p95_ms"] < 30),
        ("online update, 30-token sentence < 200 ms (p95)", results["online_update_30_tokens"]["p95_ms"] < 200),
    ]
    if "consolidation" in results:
        c = results["consolidation"]
        if c["device"] == "cuda":
            verdicts.append(("consolidation of 100k tokens < 30 min on CUDA", c["wall_s"] < 1800))
        else:
            print("consolidation ran on CPU: the 30-minute target applies to the CUDA path only")
    for name, ok in verdicts:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
