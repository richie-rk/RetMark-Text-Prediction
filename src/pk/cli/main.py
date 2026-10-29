import json
import sys
from pathlib import Path

import typer

from pk.config import ConfigError, load_config
from pk.corrector import HarperUnavailable
from pk.persistence import LockHeld, ModelMismatch

app = typer.Typer(add_completion=False, no_args_is_help=True)

EXIT_OK, EXIT_ERROR, EXIT_LOCKED = 0, 1, 2
CONFIG_FILE = "config.toml"


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


def finish(as_json, payload, human, code=EXIT_OK):
    """With --json print exactly one JSON object and nothing else."""
    if as_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        typer.echo(human)
    raise typer.Exit(code)


def guarded(as_json, fn):
    try:
        return fn()
    except typer.Exit:
        raise
    except Fail as e:
        code, message = e.code, str(e)
    except LockHeld as e:
        code, message = EXIT_LOCKED, str(e)
    except (ModelMismatch, HarperUnavailable, ConfigError) as e:
        code, message = EXIT_ERROR, str(e)
    if as_json:
        sys.stdout.write(json.dumps({"ok": False, "error": message}, ensure_ascii=False) + "\n")
    else:
        typer.echo(message, err=True)
    raise typer.Exit(code)


@app.command()
def live(inspect: bool = typer.Option(False, "--inspect", help="Show engine internals.")):
    """Interactive live-typing harness."""
    from pk.cli.live import run_live

    raise typer.Exit(guarded(False, lambda: run_live(config(), inspect=inspect)))


def main():
    app()
