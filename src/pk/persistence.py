import json
import os
from dataclasses import dataclass, field
from pathlib import Path

STATE_FORMAT_VERSION = 1


def atomic_write_bytes(path, data):
    """Write via a temporary file in the same directory, fsync, then replace,
    so an interrupted write always leaves the previous file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


@dataclass
class State:
    paused: bool = False
    offsets: dict = field(default_factory=dict)
    handover: list = field(default_factory=list)
    last_checkpoint: str | None = None


def load_state(data_dir):
    path = Path(data_dir) / "state.json"
    if not path.exists():
        return State()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return State(
        paused=bool(raw.get("paused", False)),
        offsets=dict(raw.get("offsets", {})),
        handover=list(raw.get("handover", [])),
        last_checkpoint=raw.get("last_checkpoint"),
    )


def save_state(data_dir, state):
    atomic_write_json(
        Path(data_dir) / "state.json",
        {
            "format_version": STATE_FORMAT_VERSION,
            "paused": state.paused,
            "offsets": state.offsets,
            "handover": state.handover,
            "last_checkpoint": state.last_checkpoint,
        },
    )
