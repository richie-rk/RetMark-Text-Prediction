from pathlib import Path

import pytest

from pk.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_defaults():
    cfg = load_config(ROOT / "config.toml")
    assert cfg.paths.data_dir == ROOT / "data"
    assert cfg.model.d_model == 64
    assert cfg.model.mlp_hidden == 256
    assert cfg.model.boundary_chars == (".", "?", "!", "\n")
    assert cfg.model.boundary_decay == 0.7
    assert cfg.model.fast_decay == 0.9
    assert cfg.model.init_std == 0.02
    assert cfg.model.seed == 1234
    assert cfg.predict.top_k == 3
    assert cfg.predict.word_extend_max_tokens == 4
    assert cfg.predict.debounce_ms == 50
    assert cfg.graph.spike_prob == 0.8
    assert cfg.graph.spike_min_count == 5
    assert cfg.handover.window == 500
    assert cfg.train.online_lr == 3e-4
    assert cfg.train.online_update_sentences == 4
    assert cfg.train.online_update_max_tokens == 256
    assert cfg.train.consolidation_lr == 1e-3
    assert cfg.train.consolidation_epochs == 2
    assert cfg.train.replay_ratio == 1.0
    assert cfg.train.batch_size == 32
    assert cfg.train.bptt_window == 128
    assert cfg.train.grad_clip == 1.0
    assert cfg.train.adam_beta1 == 0.9
    assert cfg.train.adam_beta2_half_life_tokens == 10_000_000
    assert cfg.train.adam_eps == 1e-8
    assert cfg.correct.dialect == "both"
    assert cfg.correct.max_edit_distance == 2
    assert cfg.persist.checkpoint_interval_s == 300
    assert cfg.logs.flush_interval_s == 5
    assert cfg.dictionary.candidate_min_count == 3
    assert cfg.eval.holdout_min_sentences == 500
    assert cfg.eval.top3_floor == 0.25


def test_config_is_immutable():
    cfg = load_config(ROOT / "config.toml")
    with pytest.raises(AttributeError):
        cfg.model.d_model = 8


def _write_variant(tmp_path, old, new):
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_relative_data_dir_resolves_against_config_location(tmp_path):
    cfg = load_config(_write_variant(tmp_path, 'data_dir = "data"', 'data_dir = "elsewhere"'))
    assert cfg.paths.data_dir == tmp_path / "elsewhere"


@pytest.mark.parametrize(
    "old,new",
    [
        ("d_model = 64\n", ""),
        ("[eval]\nholdout_min_sentences = 500\ntop3_floor = 0.25\n", ""),
        ("d_model = 64", 'd_model = "64"'),
        ("d_model = 64", "d_model = 64.0"),
        ("seed = 1234", "seed = true"),
        ("top_k = 3", "top_k = 0"),
        ("d_model = 64", "d_model = 63"),
        ("boundary_decay = 0.7", "boundary_decay = 1.5"),
        ("fast_decay = 0.9", "fast_decay = 0.0"),
        ('dialect = "both"', 'dialect = "klingon"'),
        ('boundary_chars = [".", "?", "!", "\\n"]', "boundary_chars = []"),
        ("top_k = 3", "top_k = 3\nsurprise = 1"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, old, new):
    with pytest.raises(ConfigError):
        load_config(_write_variant(tmp_path, old, new))


def test_float_field_accepts_integer_literal(tmp_path):
    cfg = load_config(_write_variant(tmp_path, "grad_clip = 1.0", "grad_clip = 1"))
    assert cfg.train.grad_clip == 1.0
    assert isinstance(cfg.train.grad_clip, float)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")
