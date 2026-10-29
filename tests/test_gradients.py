import numpy as np
import pytest

from pk.engine import layers
from pk.engine.model import PARAM_NAMES, forward_sequence, backward_sequence, init_params, step
from pk.engine.state import RecurrentState, zero_state

D, H, V, T = 8, 16, 23, 12
FAST, LAM = 0.9, 0.7
STEPS = (1e-3, 1e-4, 1e-5, 1e-6)


def well_conditioned(seed=0):
    """Spec-shaped parameters away from the degenerate initial point.

    At the spec's init every LayerNorm gain is exactly 1 and many gradients
    sit below float64 finite-difference noise, so the check would neither
    catch a missing gain nor resolve tiny entries reliably.
    """
    rng = np.random.default_rng(seed)
    p = init_params(D, H, V, std=0.4, seed=seed, dtype=np.float64)
    for name in ("g1", "g2", "gf"):
        p[name] = 1.0 + rng.normal(0.0, 0.3, size=p[name].shape)
    for name in ("be1", "be2", "bef", "b1", "b2", "bo"):
        p[name] = rng.normal(0.0, 0.1, size=p[name].shape)
    half = D // 2
    state = RecurrentState(
        S_s=rng.normal(0.0, 0.3, (half, half)),
        z_s=rng.uniform(0.5, 1.5, half),
        S_f=rng.normal(0.0, 0.3, (half, half)),
        z_f=rng.uniform(0.5, 1.5, half),
        logits=None,
    )
    return p, state


def sequence(seed=0):
    rng = np.random.default_rng(seed + 100)
    tokens = rng.integers(0, V, size=T + 1)
    boundaries = np.zeros(T, dtype=bool)
    boundaries[4] = True  # a boundary token inside the sequence
    return tokens[:-1], tokens[1:], boundaries


def loss_and_grads(params, state, inputs, targets, boundaries, cuts=None):
    logits, _, cache = forward_sequence(params, inputs, boundaries, state, FAST, LAM)
    mask = np.ones(len(inputs), dtype=bool)
    loss_sum, ce_cache = layers.softmax_ce_forward(logits, targets, mask)
    n = mask.sum()
    dlogits = layers.softmax_ce_backward(ce_cache) / n
    grads = backward_sequence(dlogits, cache, cuts)
    return loss_sum / n, grads


def loss_only(params, state, inputs, targets, boundaries):
    logits, _, _ = forward_sequence(params, inputs, boundaries, state, FAST, LAM)
    mask = np.ones(len(inputs), dtype=bool)
    loss_sum, _ = layers.softmax_ce_forward(logits, targets, mask)
    return loss_sum / mask.sum()


def test_finite_difference_gradient_check():
    params, state = well_conditioned()
    inputs, targets, boundaries = sequence()
    _, grads = loss_and_grads(params, state, inputs, targets, boundaries)
    rng = np.random.default_rng(7)
    worst = {}
    for name in PARAM_NAMES:
        p = params[name]
        flat = p.reshape(-1)
        count = min(20, flat.size)
        picks = rng.choice(flat.size, size=count, replace=False)
        errors = []
        for j in picks:
            analytic = grads[name].reshape(-1)[j]
            best = np.inf
            for h in STEPS:
                saved = flat[j]
                flat[j] = saved + h
                up = loss_only(params, state, inputs, targets, boundaries)
                flat[j] = saved - h
                down = loss_only(params, state, inputs, targets, boundaries)
                flat[j] = saved
                numeric = (up - down) / (2 * h)
                scale = max(abs(analytic), abs(numeric))
                err = 0.0 if scale == 0.0 else abs(analytic - numeric) / scale
                best = min(best, err)
            errors.append(best)
        worst[name] = max(errors)
    failing = {k: v for k, v in worst.items() if not v < 1e-5}
    assert not failing, failing


def test_every_parameter_gets_a_gradient():
    params, state = well_conditioned()
    inputs, targets, boundaries = sequence()
    _, grads = loss_and_grads(params, state, inputs, targets, boundaries)
    assert set(grads) == set(PARAM_NAMES)
    for name in PARAM_NAMES:
        assert grads[name].shape == params[name].shape
        assert np.any(grads[name] != 0.0), name


def test_cut_stops_gradient_into_earlier_state():
    """A cut at position j equals training the two halves separately, the
    second half starting from the first half's final state as a constant."""
    params, state = well_conditioned(seed=3)
    inputs, targets, boundaries = sequence(seed=3)
    j = 6
    cuts = np.zeros(T, dtype=bool)
    cuts[j] = True

    logits, _, cache = forward_sequence(params, inputs, boundaries, state, FAST, LAM)
    mask = np.ones(T, dtype=bool)
    _, ce = layers.softmax_ce_forward(logits, targets, mask)
    joint = backward_sequence(layers.softmax_ce_backward(ce), cache, cuts)

    logits_a, mid, cache_a = forward_sequence(params, inputs[:j], boundaries[:j], state, FAST, LAM)
    _, ce_a = layers.softmax_ce_forward(logits_a, targets[:j], mask[:j])
    ga = backward_sequence(layers.softmax_ce_backward(ce_a), cache_a, None)
    logits_b, _, cache_b = forward_sequence(params, inputs[j:], boundaries[j:], mid, FAST, LAM)
    _, ce_b = layers.softmax_ce_forward(logits_b, targets[j:], mask[j:])
    gb = backward_sequence(layers.softmax_ce_backward(ce_b), cache_b, None)

    np.testing.assert_allclose(np.concatenate([logits_a, logits_b]), logits, rtol=1e-12)
    for name in PARAM_NAMES:
        np.testing.assert_allclose(joint[name], ga[name] + gb[name], rtol=1e-9, atol=1e-12)


def test_step_matches_forward_sequence():
    params, state = well_conditioned(seed=5)
    inputs, _, boundaries = sequence(seed=5)
    logits, final, _ = forward_sequence(params, inputs, boundaries, state, FAST, LAM)
    s = state
    for i, t in enumerate(inputs):
        s = step(params, s, int(t), bool(boundaries[i]), FAST, LAM)
        np.testing.assert_allclose(s.logits, logits[i], rtol=1e-10, atol=1e-12)
    for name in ("S_s", "z_s", "S_f", "z_f"):
        np.testing.assert_allclose(getattr(s, name), getattr(final, name), rtol=1e-10)


def test_boundary_decay_scales_both_heads_after_the_read():
    params, _ = well_conditioned(seed=2)
    s0 = zero_state(D, dtype=np.float64)
    plain = step(params, s0, 3, False, FAST, LAM)
    decayed = step(params, s0, 3, True, FAST, LAM)
    # Same read (logits), state scaled by lambda afterwards.
    np.testing.assert_allclose(plain.logits, decayed.logits, rtol=1e-12)
    for name in ("S_s", "z_s", "S_f", "z_f"):
        np.testing.assert_allclose(getattr(decayed, name), LAM * getattr(plain, name), rtol=1e-12)


def test_fast_head_decays_every_token_and_slow_head_does_not():
    params, _ = well_conditioned(seed=4)
    s = zero_state(D, dtype=np.float64)
    s1 = step(params, s, 1, False, FAST, LAM)
    s2 = step(params, s1, 2, False, FAST, LAM)
    s2_alone = step(params, zero_state(D, dtype=np.float64), 2, False, FAST, LAM)
    np.testing.assert_allclose(s2.S_s, s1.S_s + s2_alone.S_s, rtol=1e-12)
    np.testing.assert_allclose(s2.S_f, FAST * s1.S_f + s2_alone.S_f, rtol=1e-12)
    np.testing.assert_allclose(s2.z_f, FAST * s1.z_f + s2_alone.z_f, rtol=1e-12)


def test_fast_decay_one_makes_heads_identical_in_behaviour():
    params, _ = well_conditioned(seed=6)
    inputs, _, boundaries = sequence(seed=6)
    _, final, _ = forward_sequence(params, inputs, boundaries, zero_state(D, dtype=np.float64), 1.0, LAM)
    # With fast_decay = 1 both heads follow the same recurrence; their states
    # differ only through their (different) projections, never in decay.
    s = zero_state(D, dtype=np.float64)
    for i, t in enumerate(inputs):
        s = step(params, s, int(t), bool(boundaries[i]), 1.0, LAM)
    np.testing.assert_allclose(s.S_f, final.S_f, rtol=1e-10)


def test_spec_init_is_seeded_and_float32():
    a = init_params(64, 256, 50257, std=0.02, seed=1234)
    b = init_params(64, 256, 50257, std=0.02, seed=1234)
    assert a["E"].dtype == np.float32
    assert a["E"].shape == (50257, 64)
    assert a["Wq"].shape == (64, 64)
    assert a["W1"].shape == (64, 256)
    assert a["W2"].shape == (256, 64)
    assert a["bo"].shape == (50257,)
    for name in PARAM_NAMES:
        np.testing.assert_array_equal(a[name], b[name])
    assert np.all(a["g1"] == 1.0) and np.all(a["be1"] == 0.0)
    assert np.all(a["b1"] == 0.0) and np.all(a["bo"] == 0.0)
    assert abs(float(a["E"].std()) - 0.02) < 1e-3


@pytest.mark.parametrize("fn", ["phi", "gelu"])
def test_elementwise_layers_backward(fn):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, 50)
    fwd = getattr(layers, f"{fn}_forward")
    bwd = getattr(layers, f"{fn}_backward")
    y, cache = fwd(x)
    dy = rng.normal(size=50)
    dx = bwd(dy, cache)
    h = 1e-6
    numeric = (fwd(x + h)[0] - fwd(x - h)[0]) / (2 * h) * dy
    np.testing.assert_allclose(dx, numeric, rtol=1e-6, atol=1e-8)


def test_phi_is_elu_plus_one():
    y, _ = layers.phi_forward(np.array([-1.0, 0.0, 2.0]))
    np.testing.assert_allclose(y, [np.exp(-1.0), 1.0, 3.0])


def test_float32_survives_step_forward_and_backward():
    params = init_params(D, H, V, std=0.02, seed=1)
    s = step(params, zero_state(D), 3, True, FAST, LAM)
    assert s.logits.dtype == np.float32 and s.S_s.dtype == np.float32
    inputs, targets, boundaries = sequence()
    logits, final, cache = forward_sequence(params, inputs, boundaries, zero_state(D), FAST, LAM)
    assert logits.dtype == np.float32 and final.S_f.dtype == np.float32
    _, ce = layers.softmax_ce_forward(logits, targets, np.ones(T, dtype=bool))
    grads = backward_sequence(layers.softmax_ce_backward(ce), cache)
    assert all(g.dtype == np.float32 for g in grads.values())
