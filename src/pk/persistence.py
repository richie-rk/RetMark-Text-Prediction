import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy
from safetensors import safe_open
from safetensors.numpy import save as safetensors_bytes

from pk.engine.model import PARAM_NAMES
from pk.graph import WordGraph
from pk.tokenizer import ENCODING

STATE_FORMAT_VERSION = 1
GRAPH_FORMAT_VERSION = 1
MODEL_FORMAT_VERSION = 1
# Windows refuses os.replace while another process has the target open
# (a reader, an indexer); a short retry rides that out.
REPLACE_RETRIES = 20
REPLACE_RETRY_S = 0.05


class ModelMismatch(RuntimeError):
    pass


class LockHeld(RuntimeError):
    pass


def atomic_write_bytes(path, data):
    """Write via a temporary file in the same directory, fsync, then replace,
    so an interrupted write always leaves the previous file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    done = False
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == REPLACE_RETRIES - 1:
                    raise
                time.sleep(REPLACE_RETRY_S)
        done = True
    finally:
        if not done:
            tmp.unlink(missing_ok=True)


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


def save_model(data_dir, params, adam):
    tensors = {name: numpy.ascontiguousarray(params[name]) for name in PARAM_NAMES}
    for name in PARAM_NAMES:
        tensors[f"adam_m.{name}"] = numpy.ascontiguousarray(adam.m[name])
        tensors[f"adam_v.{name}"] = numpy.ascontiguousarray(adam.v[name])
    tensors["adam_step"] = numpy.array(adam.step_count, dtype=numpy.int64)
    tensors["adam_half_lives"] = numpy.array(adam.half_lives, dtype=numpy.float64)
    tensors["adam_tokens"] = numpy.array(adam.tokens, dtype=numpy.int64)
    d_model, mlp_hidden = params["W1"].shape
    metadata = {
        "format_version": str(MODEL_FORMAT_VERSION),
        "d_model": str(d_model),
        "mlp_hidden": str(mlp_hidden),
        "tokenizer": ENCODING,
    }
    atomic_write_bytes(Path(data_dir) / "model.safetensors", safetensors_bytes(tensors, metadata=metadata))


def load_model(data_dir, d_model, mlp_hidden, adam):
    """Return the saved parameters (and fill `adam`), or None if no model exists."""
    path = Path(data_dir) / "model.safetensors"
    if not path.exists():
        return None
    with safe_open(str(path), framework="np") as f:
        meta = f.metadata() or {}
        saved = (meta.get("d_model"), meta.get("mlp_hidden"))
        if saved != (str(d_model), str(mlp_hidden)):
            raise ModelMismatch(
                f"{path} has d_model={saved[0]}, mlp_hidden={saved[1]} "
                f"but config.toml has d_model={d_model}, mlp_hidden={mlp_hidden}"
            )
        if meta.get("tokenizer") != ENCODING:
            raise ModelMismatch(f"{path} was trained with tokenizer {meta.get('tokenizer')!r}, not {ENCODING!r}")
        params = {name: f.get_tensor(name) for name in PARAM_NAMES}
        if adam is not None:
            for name in PARAM_NAMES:
                adam.m[name] = f.get_tensor(f"adam_m.{name}")
                adam.v[name] = f.get_tensor(f"adam_v.{name}")
            adam.step_count = int(f.get_tensor("adam_step"))
            adam.half_lives = float(f.get_tensor("adam_half_lives"))
            adam.tokens = int(f.get_tensor("adam_tokens"))
    return params


def save_graph(data_dir, graph):
    atomic_write_json(
        Path(data_dir) / "graph.json",
        {"format_version": GRAPH_FORMAT_VERSION, "edges": graph.edges, "totals": graph.totals, "surface": graph.surface},
    )


def load_graph(data_dir, boundary_chars):
    graph = WordGraph(boundary_chars)
    path = Path(data_dir) / "graph.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        graph.edges = raw["edges"]
        graph.totals = raw["totals"]
        graph.surface = raw["surface"]
    return graph


class InstanceLock:
    """An OS lock on data/.lock. The OS drops it when the holder exits, so a
    crashed harness never leaves a stale lock behind."""

    def __init__(self, data_dir):
        self.path = Path(data_dir) / ".lock"
        self._fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise LockHeld(f"{self.path} is held by another pk process (is pk live running?)") from None
        self._fd = fd
        return self

    def release(self):
        if self._fd is None:
            return
        if sys.platform == "win32":
            import msvcrt

            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


class Workspace:
    """Everything under data_dir, loaded once into RAM. Missing files start
    fresh in memory; only checkpoint() writes."""

    def __init__(self, cfg, params, adam, graph, state, dictionary, rejected, tally):
        self.cfg = cfg
        self.params = params
        self.adam = adam
        self.graph = graph
        self.state = state
        self.dictionary = dictionary
        self.rejected = rejected
        self.tally = tally

    @classmethod
    def load(cls, cfg, vocab_size):
        from pk.dictionary import CandidateTally, PersonalDictionary
        from pk.engine.model import init_params
        from pk.engine.optim import Adam

        data = cfg.paths.data_dir
        m, t = cfg.model, cfg.train
        fresh = init_params(m.d_model, m.mlp_hidden, vocab_size, m.init_std, m.seed)
        adam = Adam(fresh, t.adam_beta1, t.adam_beta2_half_life_tokens, t.adam_eps)
        params = load_model(data, m.d_model, m.mlp_hidden, adam) or fresh
        dictionary = PersonalDictionary(data / "dictionary.txt")
        rejected = PersonalDictionary(data / "rejected.txt")
        tally = CandidateTally(data / "candidates.json", dictionary, rejected)
        return cls(cfg, params, adam, load_graph(data, m.boundary_chars), load_state(data), dictionary, rejected, tally)

    def checkpoint(self, ts):
        data = self.cfg.paths.data_dir
        self.state.last_checkpoint = ts
        save_model(data, self.params, self.adam)
        save_graph(data, self.graph)
        save_state(data, self.state)
        self.dictionary.save()
        self.rejected.save()
        self.tally.save()
