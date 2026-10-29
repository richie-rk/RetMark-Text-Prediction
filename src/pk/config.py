import tomllib
import types
from dataclasses import dataclass, fields, replace
from pathlib import Path


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    mlp_hidden: int
    boundary_chars: tuple[str, ...]
    boundary_decay: float
    fast_decay: float
    init_std: float
    seed: int


@dataclass(frozen=True)
class PredictConfig:
    top_k: int
    word_extend_max_tokens: int
    debounce_ms: int


@dataclass(frozen=True)
class GraphConfig:
    spike_prob: float
    spike_min_count: int


@dataclass(frozen=True)
class HandoverConfig:
    window: int


@dataclass(frozen=True)
class TrainConfig:
    online_lr: float
    online_update_sentences: int
    online_update_max_tokens: int
    consolidation_lr: float
    consolidation_epochs: int
    replay_ratio: float
    batch_size: int
    bptt_window: int
    grad_clip: float
    adam_beta1: float
    adam_beta2_half_life_tokens: int
    adam_eps: float


@dataclass(frozen=True)
class CorrectConfig:
    dialect: str
    max_edit_distance: int
    harper_ls_path: str


@dataclass(frozen=True)
class PersistConfig:
    checkpoint_interval_s: float


@dataclass(frozen=True)
class LogsConfig:
    flush_interval_s: float


@dataclass(frozen=True)
class DictionaryConfig:
    candidate_min_count: int


@dataclass(frozen=True)
class EvalConfig:
    holdout_min_sentences: int
    top3_floor: float


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    model: ModelConfig
    predict: PredictConfig
    graph: GraphConfig
    handover: HandoverConfig
    train: TrainConfig
    correct: CorrectConfig
    persist: PersistConfig
    logs: LogsConfig
    dictionary: DictionaryConfig
    eval: EvalConfig


DIALECTS = ("american", "british", "both")


def _coerce(where, kind, value):
    # bool is a subclass of int, so `true` would otherwise pass as 1.
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected {kind.__name__}, got a boolean")
    if kind is int and isinstance(value, int):
        return value
    if kind is float and isinstance(value, (int, float)):
        return float(value)
    if kind is str and isinstance(value, str):
        return value
    if kind is Path and isinstance(value, str):
        return Path(value)
    if isinstance(kind, types.GenericAlias) and kind.__origin__ is tuple:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return tuple(value)
        raise ConfigError(f"{where}: expected a list of strings")
    raise ConfigError(f"{where}: expected {kind.__name__}, got {type(value).__name__}")


def _section(name, cls, raw):
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}]: missing section")
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ConfigError(f"[{name}]: unknown keys {sorted(unknown)}")
    values = {}
    for f in fields(cls):
        if f.name not in raw:
            raise ConfigError(f"[{name}]: missing key {f.name!r}")
        values[f.name] = _coerce(f"{name}.{f.name}", f.type, raw[f.name])
    return cls(**values)


def _validate(cfg):
    checks = [
        ("model.d_model", cfg.model.d_model > 0 and cfg.model.d_model % 2 == 0, "must be a positive even number (two heads split it in half)"),
        ("model.mlp_hidden", cfg.model.mlp_hidden > 0, "must be positive"),
        ("model.boundary_chars", len(cfg.model.boundary_chars) > 0 and all(len(c) == 1 for c in cfg.model.boundary_chars), "must be a non-empty list of single characters"),
        ("model.boundary_decay", 0.0 < cfg.model.boundary_decay <= 1.0, "must be in (0, 1]"),
        ("model.fast_decay", 0.0 < cfg.model.fast_decay <= 1.0, "must be in (0, 1]"),
        ("model.init_std", cfg.model.init_std > 0.0, "must be positive"),
        ("predict.top_k", cfg.predict.top_k > 0, "must be positive"),
        ("predict.word_extend_max_tokens", cfg.predict.word_extend_max_tokens > 0, "must be positive"),
        ("predict.debounce_ms", cfg.predict.debounce_ms >= 0, "must not be negative"),
        ("graph.spike_prob", 0.0 < cfg.graph.spike_prob <= 1.0, "must be in (0, 1]"),
        ("graph.spike_min_count", cfg.graph.spike_min_count > 0, "must be positive"),
        ("handover.window", cfg.handover.window > 0, "must be positive"),
        ("train.online_lr", cfg.train.online_lr > 0.0, "must be positive"),
        ("train.online_update_sentences", cfg.train.online_update_sentences > 0, "must be positive"),
        ("train.online_update_max_tokens", cfg.train.online_update_max_tokens > 0, "must be positive"),
        ("train.consolidation_lr", cfg.train.consolidation_lr > 0.0, "must be positive"),
        ("train.consolidation_epochs", cfg.train.consolidation_epochs > 0, "must be positive"),
        ("train.replay_ratio", cfg.train.replay_ratio >= 0.0, "must not be negative"),
        ("train.batch_size", cfg.train.batch_size > 0, "must be positive"),
        ("train.bptt_window", cfg.train.bptt_window > 0, "must be positive"),
        ("train.grad_clip", cfg.train.grad_clip > 0.0, "must be positive"),
        ("train.adam_beta1", 0.0 <= cfg.train.adam_beta1 < 1.0, "must be in [0, 1)"),
        ("train.adam_beta2_half_life_tokens", cfg.train.adam_beta2_half_life_tokens > 0, "must be positive"),
        ("train.adam_eps", cfg.train.adam_eps > 0.0, "must be positive"),
        ("correct.dialect", cfg.correct.dialect in DIALECTS, f"must be one of {DIALECTS}"),
        ("correct.max_edit_distance", cfg.correct.max_edit_distance >= 0, "must not be negative"),
        ("persist.checkpoint_interval_s", cfg.persist.checkpoint_interval_s > 0.0, "must be positive"),
        ("logs.flush_interval_s", cfg.logs.flush_interval_s > 0.0, "must be positive"),
        ("dictionary.candidate_min_count", cfg.dictionary.candidate_min_count > 0, "must be positive"),
        ("eval.holdout_min_sentences", cfg.eval.holdout_min_sentences > 0, "must be positive"),
        ("eval.top3_floor", 0.0 <= cfg.eval.top3_floor <= 1.0, "must be in [0, 1]"),
    ]
    for where, ok, message in checks:
        if not ok:
            raise ConfigError(f"{where}: {message}")


def load_config(path):
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None

    sections = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"unknown sections {sorted(unknown)}")
    cfg = Config(**{name: _section(name, cls, raw.get(name)) for name, cls in sections.items()})
    # A relative data_dir means "next to config.toml", not "wherever pk was launched from".
    cfg = replace(cfg, paths=PathsConfig(data_dir=path.resolve().parent / cfg.paths.data_dir))
    _validate(cfg)
    return cfg
