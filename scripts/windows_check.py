"""Step-1 platform check: run on the target Windows machine.

    python scripts/windows_check.py

Confirms the dependencies import, the gpt2 encoding loads, and harper-ls
starts and flags a known misspelling in both dialects from one process.
"""
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    failures = []
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 expected")

    import numpy, prompt_toolkit, rich, safetensors, tiktoken, typer  # noqa: F401

    from pk.config import load_config
    from pk.corrector import HarperLS
    from pk.xp import consolidation_backend

    cfg = load_config(ROOT / "config.toml")
    enc = tiktoken.get_encoding("gpt2")
    if enc.n_vocab != 50257:
        failures.append(f"gpt2 vocabulary is {enc.n_vocab}, expected 50257")
    print(f"consolidation backend: {consolidation_backend()[1]}")

    harper = HarperLS(cfg.correct.harper_ls_path)
    try:
        for dialect, text, flagged in [
            ("American", "I saw thier colour.", {"thier", "colour"}),
            ("British", "I saw thier color.", {"thier", "color"}),
        ]:
            lints = [l for l in harper.lint(text, dialect) if l.rule == "SpellCheck"]
            got = {text[l.start:l.end] for l in lints}
            print(f"harper-ls {dialect}: flagged {sorted(got)}")
            if got != flagged:
                failures.append(f"{dialect}: expected {sorted(flagged)}, got {sorted(got)}")
    finally:
        harper.close()

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
