import numpy as np
import pytest

from pk.engine.optim import Adam, clip_global_norm


def test_clip_global_norm_scales_all_gradients_together():
    grads = {"a": np.array([3.0, 0.0]), "b": np.array([[4.0]])}
    clipped, norm = clip_global_norm(grads, 1.0)
    assert norm == pytest.approx(5.0)
    np.testing.assert_allclose(clipped["a"], [0.6, 0.0])
    np.testing.assert_allclose(clipped["b"], [[0.8]])


def test_clip_leaves_small_gradients_alone():
    grads = {"a": np.array([0.3, 0.4])}
    clipped, norm = clip_global_norm(grads, 1.0)
    assert norm == pytest.approx(0.5)
    np.testing.assert_array_equal(clipped["a"], grads["a"])


def test_beta2_derives_from_token_half_life():
    adam = Adam({"w": np.zeros(1)}, beta1=0.9, half_life_tokens=1000, eps=1e-8)
    assert adam.beta2_for(1000) == pytest.approx(0.5)
    assert adam.beta2_for(500) == pytest.approx(2 ** -0.5)


def test_step_returns_new_arrays_and_leaves_old_params_untouched():
    params = {"w": np.array([1.0, -1.0])}
    adam = Adam(params, beta1=0.9, half_life_tokens=1000, eps=1e-8)
    new = adam.step(params, {"w": np.array([0.5, -0.5])}, lr=0.1, tokens_in_step=10)
    np.testing.assert_array_equal(params["w"], [1.0, -1.0])
    assert new["w"] is not params["w"]
    # First bias-corrected Adam step moves each weight by lr against the sign.
    np.testing.assert_allclose(new["w"], [0.9, -0.9], rtol=1e-6)


def test_bias_correction_tracks_half_lives_across_varying_step_sizes():
    params = {"w": np.array([0.0])}
    adam = Adam(params, beta1=0.9, half_life_tokens=100, eps=0.0)
    g = {"w": np.array([2.0])}
    p = params
    for tokens in (7, 300, 1, 50):
        p = adam.step(p, g, lr=0.01, tokens_in_step=tokens)
    # A constant gradient keeps every bias-corrected step exactly lr in size,
    # whatever beta2 each step used.
    np.testing.assert_allclose(p["w"], [-0.04], rtol=1e-9)
    assert adam.step_count == 4
    assert adam.tokens == 358
    assert adam.half_lives == pytest.approx(3.58)


def test_half_life_change_keeps_correction_exact():
    params = {"w": np.array([0.0])}
    adam = Adam(params, beta1=0.9, half_life_tokens=100, eps=0.0)
    g = {"w": np.array([1.0])}
    p = adam.step(params, g, lr=0.01, tokens_in_step=50)
    adam.half_life_tokens = 1000
    p = adam.step(p, g, lr=0.01, tokens_in_step=50)
    np.testing.assert_allclose(p["w"], [-0.02], rtol=1e-9)
    assert adam.half_lives == pytest.approx(0.5 + 0.05)
