import json
import sys
from pathlib import Path

import typer

from pk.config import ConfigError, load_config
from pk.corrector import HarperUnavailable
from pk.persistence import LockHeld, ModelMismatch

app = typer.Typer(add_completion=False, no_args_is_help=True)
words_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Candidate words and the personal dictionary.")
graph_app = typer.Typer(add_completion=False, no_args_is_help=True, help="The word transition graph.")
app.add_typer(words_app, name="words")
app.add_typer(graph_app, name="graph")

EXIT_OK, EXIT_ERROR, EXIT_LOCKED = 0, 1, 2
CONFIG_FILE = "config.toml"
JSON = typer.Option(False, "--json", help="Print exactly one JSON object.")


class Fail(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@app.callback()
def root():
    """Predictive keyboard, Phase 1."""


def config():
    try:
        return load_config(Path.cwd() / CONFIG_FILE)
    except ConfigError as e:
        raise Fail(EXIT_ERROR, f"config error: {e}") from None


def emit(as_json, payload, human):
    """With --json print exactly one JSON object and nothing else."""
    if as_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        typer.echo(human)


def run(as_json, fn):
    try:
        payload, human = fn()
        emit(as_json, payload, human)
        code = EXIT_OK
    except Fail as e:
        code, message = e.code, str(e)
    except LockHeld as e:
        code, message = EXIT_LOCKED, f"{e}. Close pk live first."
    except (ModelMismatch, HarperUnavailable) as e:
        code, message = EXIT_ERROR, str(e)
    if code != EXIT_OK:
        if as_json:
            sys.stdout.write(json.dumps({"ok": False, "error": message}, ensure_ascii=False) + "\n")
        else:
            typer.echo(message, err=True)
    raise typer.Exit(code)


def _tokenizer(cfg):
    from pk.tokenizer import Tokenizer

    return Tokenizer(cfg.model.boundary_chars)


def _workspace(cfg, tok):
    from pk.persistence import Workspace

    return Workspace.load(cfg, tok.vocab_size)


def _locked(cfg):
    from pk.persistence import InstanceLock

    return InstanceLock(cfg.paths.data_dir)


# --- live ---------------------------------------------------------------------


@app.command()
def live(inspect: bool = typer.Option(False, "--inspect", help="Show engine internals.")):
    """Interactive live-typing harness."""

    def go():
        from pk.cli.live import run_live

        run_live(config(), inspect=inspect)
        return None, ""

    try:
        go()
        code = EXIT_OK
    except Fail as e:
        typer.echo(str(e), err=True)
        code = e.code
    except LockHeld as e:
        typer.echo(f"{e}.", err=True)
        code = EXIT_LOCKED
    except (ModelMismatch, HarperUnavailable) as e:
        typer.echo(str(e), err=True)
        code = EXIT_ERROR
    raise typer.Exit(code)


# --- predict / generate -------------------------------------------------------


def _slot(s):
    return None if s is None else {"word": s.word, "origin": s.origin}


@app.command()
def predict(text: str, as_json: bool = JSON):
    """The four slots, mode and correction for TEXT, as if typed."""

    def go():
        from pk.corrector import Corrector, HarperLS
        from pk.session import LiveSession

        cfg = config()
        tok = _tokenizer(cfg)
        ws = _workspace(cfg, tok)
        harper = HarperLS(cfg.correct.harper_ls_path)
        try:
            s = LiveSession(cfg, tok, ws, Corrector(harper, cfg.correct.dialect, ws.dictionary), None, None, learn=False)
            s.type(text)
        finally:
            harper.close()
        slots = s.prediction.slots
        engine, graph = s.handover.rates()
        c = s.correction
        payload = {
            "text": text,
            "slots": {"second": _slot(slots.second), "primary": _slot(slots.primary), "third": _slot(slots.third), "side": _slot(slots.side)},
            "mode": s.mode,
            "engine_hit_rate": engine,
            "graph_hit_rate": graph,
            "correction": None if c is None else {"typed": c.typed, "suggestion": c.suggestion},
        }
        w = lambda x: x.word if x else "—"  # noqa: E731
        human = f"{w(slots.second)}  [{w(slots.primary)}]  {w(slots.third)}   |  graph: {w(slots.side)}\n"
        human += f"mode {s.mode} (engine {engine:.0%}, graph {graph:.0%})"
        if c is not None:
            human += f"\ncorrection: {c.typed} → {c.suggestion}"
        return payload, human

    run(as_json, go)


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt"),
    tokens: int = typer.Option(..., "--tokens", min=0),
    temperature: float = typer.Option(..., "--temperature", min=0.0),
    as_json: bool = JSON,
):
    """Free-running generation from the engine. Temperature 0 is greedy."""

    def go():
        import numpy

        from pk.engine.model import init_params, step
        from pk.engine.state import boundary_decay, zero_state
        from pk.persistence import load_model
        from pk.tokenizer import is_boundary_at

        cfg = config()
        tok = _tokenizer(cfg)
        m = cfg.model
        params = load_model(cfg.paths.data_dir, m.d_model, m.mlp_hidden, None)
        if params is None:
            params = init_params(m.d_model, m.mlp_hidden, tok.vocab_size, m.init_std, m.seed)
        rng = numpy.random.default_rng(m.seed)
        text = prompt
        ids = tok.encode(prompt)
        state = zero_state(m.d_model)
        # A mark's boundary status depends on the next character, which for
        # the last token is only known once the following token is chosen;
        # decay comes after the read, so it can be applied then.
        undecided = None
        flags = tok.token_boundary_flags(prompt, ids, at_end=False)
        for t, f in zip(ids, flags):
            state = step(params, state, t, f, m.fast_decay, m.boundary_decay)
        if ids and text[-1] in m.boundary_chars and not flags[-1]:
            undecided = len(text) - 1
        generated = []
        for _ in range(tokens):
            if state.logits is None:
                break
            logits = numpy.asarray(state.logits, dtype=numpy.float64)
            if temperature == 0:
                nxt = int(numpy.argmax(logits))
            else:
                p = numpy.exp((logits - logits.max()) / temperature)
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            piece = tok.token_str[nxt]
            if undecided is not None and is_boundary_at(text + piece, undecided, m.boundary_chars, at_end=False):
                state = boundary_decay(state, m.boundary_decay)
            text += piece
            generated.append(nxt)
            marks = [i for i in range(len(text) - len(piece), len(text)) if text[i] in m.boundary_chars]
            fired = any(is_boundary_at(text, i, m.boundary_chars, at_end=False) for i in marks)
            undecided = marks[-1] if marks and not fired else None
            state = step(params, state, nxt, fired, m.fast_decay, m.boundary_decay)
        out = tok.decode(generated)
        return {"prompt": prompt, "text": out, "tokens": generated, "temperature": temperature}, prompt + out

    run(as_json, go)


# --- consolidate / status / pause ---------------------------------------------


@app.command()
def consolidate(as_json: bool = JSON):
    """Run the consolidation job."""

    def go():
        from pk import consolidate as job

        s = job.run(config())
        if s.get("nothing_pending"):
            return s, "Nothing pending: no new sentences to consolidate."
        human = (
            f"sentences {s['sentences']} (+{s['replay_sentences']} replayed), tokens {s['tokens']}\n"
            f"mean loss {s['loss_before']:.4f} → {s['loss_after']:.4f}\n"
            f"duration {s['duration_s']:.1f}s on {s['device']}"
        )
        return s, human

    run(as_json, go)


@app.command()
def status(as_json: bool = JSON):
    """Pause flag, pending sentences, Adam steps, mode and file sizes."""

    def go():
        from safetensors import safe_open

        from pk.logs import pending_clean
        from pk.persistence import load_state
        from pk.predictor import Handover

        cfg = config()
        data = cfg.paths.data_dir
        state = load_state(data)
        handover = Handover.from_list(cfg.handover.window, state.handover)
        steps = tokens = 0
        model = data / "model.safetensors"
        if model.exists():
            with safe_open(str(model), framework="np") as f:
                steps = int(f.get_tensor("adam_step"))
                tokens = int(f.get_tensor("adam_tokens"))
        files = {}
        if data.exists():
            for p in sorted(data.rglob("*")):
                if p.is_file():
                    files[p.relative_to(data).as_posix()] = p.stat().st_size
        engine, graph = handover.rates()
        payload = {
            "paused": state.paused,
            "pending_sentences": len(pending_clean(data, state.offsets)),
            "adam_steps": steps,
            "adam_tokens": tokens,
            "mode": handover.mode(),
            "engine_hit_rate": engine,
            "graph_hit_rate": graph,
            "handover_records": len(handover.records),
            "last_checkpoint": state.last_checkpoint,
            "files": files,
        }
        human = "\n".join(
            [
                f"{'PAUSED' if state.paused else 'recording'}",
                f"pending consolidation: {payload['pending_sentences']} sentences",
                f"adam steps: {steps} ({tokens} tokens)",
                f"mode {payload['mode']}: engine {engine:.0%}, graph {graph:.0%} over {len(handover.records)}/{cfg.handover.window} words",
                "files:",
            ]
            + [f"  {name:40s} {size:>12,d} B" for name, size in files.items()]
        )
        return payload, human

    run(as_json, go)


def _set_paused(flag, as_json):
    def go():
        from pk.persistence import load_state, save_state

        cfg = config()
        with _locked(cfg):
            state = load_state(cfg.paths.data_dir)
            state.paused = flag
            save_state(cfg.paths.data_dir, state)
        return {"paused": flag}, "paused: not learning" if flag else "recording"

    run(as_json, go)


@app.command()
def pause(as_json: bool = JSON):
    """Stop recording and learning until resumed."""
    _set_paused(True, as_json)


@app.command()
def resume(as_json: bool = JSON):
    """Resume recording and learning."""
    _set_paused(False, as_json)


# --- words ----------------------------------------------------------------------


def _tally(cfg):
    from pk.dictionary import CandidateTally, PersonalDictionary

    data = cfg.paths.data_dir
    d = PersonalDictionary(data / "dictionary.txt")
    r = PersonalDictionary(data / "rejected.txt")
    return CandidateTally(data / "candidates.json", d, r)


@words_app.command("list")
def words_list(as_json: bool = JSON):
    """Candidate words with their counts."""

    def go():
        rows = _tally(config()).listing()
        payload = {"candidates": [{"word": w, "count": c, "first_seen": f, "last_seen": l} for w, c, f, l in rows]}
        human = "\n".join(f"{c:6d}  {w}" for w, c, _, _ in rows) or "No candidate words."
        return payload, human

    run(as_json, go)


def _decide(approve, words, as_json):
    def go():
        cfg = config()
        with _locked(cfg):
            tally = _tally(cfg)
            (tally.approve if approve else tally.reject)(words)
            tally.save()
            tally.dictionary.save()
            tally.rejected.save()
        verb = "approved" if approve else "rejected"
        return {verb: list(words)}, f"{verb}: {', '.join(words)}"

    run(as_json, go)


@words_app.command("approve")
def words_approve(words: list[str], as_json: bool = JSON):
    """Add WORDS to the personal dictionary."""
    _decide(True, words, as_json)


@words_app.command("reject")
def words_reject(words: list[str], as_json: bool = JSON):
    """Reject WORDS: they never reappear as candidates."""
    _decide(False, words, as_json)


def review_dialog(candidates):
    """Checklist of (word, count). Returns the ticked words, or None if cancelled."""
    from prompt_toolkit.shortcuts import checkboxlist_dialog

    return checkboxlist_dialog(
        title="Review candidate words",
        text="Tick words to add to your dictionary. Unticked words are rejected for good.",
        values=[(w, f"{w}  ({c})") for w, c in candidates],
    ).run()


@words_app.command("review")
def words_review():
    """Interactive checklist: ticked words join the dictionary, the rest are rejected."""
    try:
        cfg = config()
        with _locked(cfg):
            tally = _tally(cfg)
            candidates = tally.review(cfg.dictionary.candidate_min_count)
            if not candidates:
                typer.echo("No candidates to review.")
                raise typer.Exit(EXIT_OK)
            ticked = review_dialog(candidates)
            if ticked is None:
                typer.echo("Cancelled: nothing changed.")
                raise typer.Exit(EXIT_OK)
            words = [w for w, _ in candidates]
            tally.approve([w for w in words if w in ticked])
            tally.reject([w for w in words if w not in ticked])
            tally.save()
            tally.dictionary.save()
            tally.rejected.save()
        typer.echo(f"approved {len(ticked)}, rejected {len(words) - len(ticked)}")
    except Fail as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(e.code)
    except LockHeld as e:
        typer.echo(f"{e}. Close pk live first.", err=True)
        raise typer.Exit(EXIT_LOCKED)


# --- graph ----------------------------------------------------------------------


@graph_app.command("top")
def graph_top(word: str, n: int = typer.Option(10, "--n", min=1), as_json: bool = JSON):
    """Followers of WORD with counts and probabilities."""

    def go():
        from pk.persistence import load_graph

        cfg = config()
        g = load_graph(cfg.paths.data_dir, cfg.model.boundary_chars)
        key = word if word == "<s>" else word.lower()
        rows = g.followers(key, "")[:n]
        payload = {
            "word": word,
            "total": g.totals.get(key, 0),
            "followers": [{"word": g.display(k), "count": c, "probability": p} for k, c, p in rows],
        }
        human = "\n".join(f"{c:6d}  {p:6.1%}  {g.display(k)}" for k, c, p in rows) or f"No followers for {word!r}."
        return payload, human

    run(as_json, go)


@graph_app.command("stats")
def graph_stats(as_json: bool = JSON):
    """Node and edge counts and the strongest edges."""

    def go():
        from pk.persistence import load_graph

        cfg = config()
        g = load_graph(cfg.paths.data_dir, cfg.model.boundary_chars)
        s = g.stats(cfg.graph.spike_min_count, 20)
        payload = {
            "nodes": s["nodes"],
            "edges": s["edges"],
            "strongest": [{"from": a, "to": b, "count": c, "probability": p} for a, b, c, p in s["strongest"]],
        }
        lines = [f"nodes {s['nodes']}, edges {s['edges']}", f"strongest edges (from words with ≥ {cfg.graph.spike_min_count} uses):"]
        lines += [f"  {p:6.1%}  {c:6d}  {a} → {b}" for a, b, c, p in s["strongest"]]
        return payload, "\n".join(lines)

    run(as_json, go)


def main():
    app()
