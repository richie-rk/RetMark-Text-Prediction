import numpy

from pk.engine import layers
from pk.engine.state import RecurrentState

PARAM_NAMES = ("E", "g1", "be1", "Wq", "Wk", "Wv", "Wo", "g2", "be2", "W1", "b1", "W2", "b2", "gf", "bef", "bo")


def init_params(d_model, mlp_hidden, vocab_size, std, seed, dtype=numpy.float32):
    rng = numpy.random.default_rng(seed)

    def normal(*shape):
        return (rng.standard_normal(shape) * std).astype(dtype)

    d, h, V = d_model, mlp_hidden, vocab_size
    return {
        "E": normal(V, d),
        "g1": numpy.ones(d, dtype),
        "be1": numpy.zeros(d, dtype),
        "Wq": normal(d, d),
        "Wk": normal(d, d),
        "Wv": normal(d, d),
        "Wo": normal(d, d),
        "g2": numpy.ones(d, dtype),
        "be2": numpy.zeros(d, dtype),
        "W1": normal(d, h),
        "b1": numpy.zeros(h, dtype),
        "W2": normal(h, d),
        "b2": numpy.zeros(d, dtype),
        "gf": numpy.ones(d, dtype),
        "bef": numpy.zeros(d, dtype),
        "bo": numpy.zeros(V, dtype),
    }


def _heads(x):
    # (..., d) -> (..., 2, d/2): slow head is the first half, fast the second,
    # so concat(a_s, a_f) is a plain reshape back.
    return x.reshape(x.shape[:-1] + (2, x.shape[-1] // 2))


def step(params, state, token, boundary, fast_decay, boundary_decay):
    """Advance one token for inference. Returns a new state carrying the logits."""
    p = params
    xp = layers.array_module(p["E"])
    x = p["E"][token]
    u, _ = layers.layernorm_forward(x, p["g1"], p["be1"])
    q = _heads(layers.phi_forward(u @ p["Wq"])[0])
    k = _heads(layers.phi_forward(u @ p["Wk"])[0])
    v = _heads(u @ p["Wv"])

    S_s = state.S_s + xp.outer(k[0], v[0])
    z_s = state.z_s + k[0]
    S_f = fast_decay * state.S_f + xp.outer(k[1], v[1])
    z_f = fast_decay * state.z_f + k[1]
    a_s = (q[0] @ S_s) / (q[0] @ z_s + layers.ATTN_EPS)
    a_f = (q[1] @ S_f) / (q[1] @ z_f + layers.ATTN_EPS)
    a = xp.concatenate([a_s, a_f])

    r = x + a @ p["Wo"]
    n2, _ = layers.layernorm_forward(r, p["g2"], p["be2"])
    g, _ = layers.gelu_forward(n2 @ p["W1"] + p["b1"])
    m = r + g @ p["W2"] + p["b2"]
    nf, _ = layers.layernorm_forward(m, p["gf"], p["bef"])
    logits = nf @ p["E"].T + p["bo"]
    if boundary:
        S_s, z_s, S_f, z_f = (boundary_decay * t for t in (S_s, z_s, S_f, z_f))
    return RecurrentState(S_s=S_s, z_s=z_s, S_f=S_f, z_f=z_f, logits=logits)


def forward_sequence(params, tokens, boundaries, state0, fast_decay, boundary_decay):
    """Run a token sequence from `state0`. The recurrence runs step by step;
    everything that does not depend on the state runs for all positions at
    once, including the logits in one matrix multiply.

    Returns (logits (T, V), final state, cache for backward_sequence).
    """
    p = params
    xp = layers.array_module(p["E"])
    tokens = xp.asarray(tokens)
    x, c_emb = layers.embedding_forward(p["E"], tokens)
    u, c_ln1 = layers.layernorm_forward(x, p["g1"], p["be1"])
    qp, c_q = layers.linear_forward(u, p["Wq"])
    kp, c_k = layers.linear_forward(u, p["Wk"])
    v, c_v = layers.linear_forward(u, p["Wv"])
    q, c_phq = layers.phi_forward(qp)
    k, c_phk = layers.phi_forward(kp)

    gammas = xp.asarray([1.0, fast_decay], dtype=x.dtype)
    S0 = xp.stack([state0.S_s, state0.S_f])
    z0 = xp.stack([state0.z_s, state0.z_f])
    a, (S, z), c_att = layers.linear_attention_forward(
        _heads(q), _heads(k), _heads(v), [bool(b) for b in boundaries], S0, z0, gammas, boundary_decay
    )
    a = a.reshape(x.shape)

    ao, c_o = layers.linear_forward(a, p["Wo"])
    r = x + ao
    n2, c_ln2 = layers.layernorm_forward(r, p["g2"], p["be2"])
    hp, c_1 = layers.linear_forward(n2, p["W1"], p["b1"])
    g, c_gelu = layers.gelu_forward(hp)
    mo, c_2 = layers.linear_forward(g, p["W2"], p["b2"])
    m = r + mo
    nf, c_lnf = layers.layernorm_forward(m, p["gf"], p["bef"])
    logits, c_out = layers.output_forward(nf, p["E"], p["bo"])

    final = RecurrentState(S_s=S[0], z_s=z[0], S_f=S[1], z_f=z[1], logits=logits[-1])
    cache = (c_emb, c_ln1, c_q, c_k, c_v, c_phq, c_phk, c_att, c_o, c_ln2, c_1, c_gelu, c_2, c_lnf, c_out, p["E"].shape[0])
    return logits, final, cache


def backward_sequence(dlogits, cache, cuts=None):
    (c_emb, c_ln1, c_q, c_k, c_v, c_phq, c_phk, c_att, c_o, c_ln2, c_1, c_gelu, c_2, c_lnf, c_out, V) = cache
    grads = {}
    dnf, dE_out, grads["bo"] = layers.output_backward(dlogits, c_out)
    dm, grads["gf"], grads["bef"] = layers.layernorm_backward(dnf, c_lnf)
    dr = dm
    dg, grads["W2"], grads["b2"] = layers.linear_backward(dm, c_2)
    dhp = layers.gelu_backward(dg, c_gelu)
    dn2, grads["W1"], grads["b1"] = layers.linear_backward(dhp, c_1)
    dr_ln, grads["g2"], grads["be2"] = layers.layernorm_backward(dn2, c_ln2)
    dr = dr + dr_ln
    da, grads["Wo"], _ = layers.linear_backward(dr, c_o)
    dq, dk, dv = layers.linear_attention_backward(_heads(da), c_att, cuts)
    dq = layers.phi_backward(dq.reshape(da.shape), c_phq)
    dk = layers.phi_backward(dk.reshape(da.shape), c_phk)
    dv = dv.reshape(da.shape)
    du_q, grads["Wq"], _ = layers.linear_backward(dq, c_q)
    du_k, grads["Wk"], _ = layers.linear_backward(dk, c_k)
    du_v, grads["Wv"], _ = layers.linear_backward(dv, c_v)
    dx_ln, grads["g1"], grads["be1"] = layers.layernorm_backward(du_q + du_k + du_v, c_ln1)
    dx = dr + dx_ln
    # The tied embedding collects gradient from both the lookup and the output.
    grads["E"] = dE_out + layers.embedding_backward(dx, c_emb, V)
    return grads
