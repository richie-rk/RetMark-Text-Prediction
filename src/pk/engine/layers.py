"""Engine layers. Each forward() returns its output and an explicit cache;
each backward() takes the upstream gradient and that cache.

Every function derives its array module from its inputs, so the same code
runs on NumPy and CuPy.
"""
import math

import numpy

LN_EPS = 1e-5
ATTN_EPS = 1e-6
# A plain float on purpose: under NumPy 2 promotion a numpy.float64 scalar
# turns float32 activations into float64, and the logits matmul then misses
# the float32 BLAS path entirely (about 40x slower).
_GELU_C = math.sqrt(2.0 / math.pi)


def array_module(a):
    if isinstance(a, numpy.ndarray):
        return numpy
    import cupy

    return cupy.get_array_module(a)


def _scatter_add(target, idx, values):
    xp = array_module(target)
    if xp is numpy:
        numpy.add.at(target, idx, values)
    else:
        import cupyx

        cupyx.scatter_add(target, idx, values)


def embedding_forward(E, ids):
    return E[ids], ids


def embedding_backward(dx, cache, vocab_size):
    ids = cache
    xp = array_module(dx)
    dE = xp.zeros((vocab_size, dx.shape[-1]), dtype=dx.dtype)
    _scatter_add(dE, ids, dx)
    return dE


def layernorm_forward(x, g, b):
    xp = array_module(x)
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / xp.sqrt(var + LN_EPS)
    xhat = xc * inv
    return xhat * g + b, (xhat, inv, g)


def layernorm_backward(dy, cache):
    xhat, inv, g = cache
    lead = tuple(range(dy.ndim - 1))
    dg = (dy * xhat).sum(axis=lead)
    db = dy.sum(axis=lead)
    dxhat = dy * g
    n = dy.shape[-1]
    dx = inv * (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True) / n)
    return dx, dg, db


def linear_forward(x, W, b=None):
    y = x @ W
    if b is not None:
        y = y + b
    return y, (x, W, b is not None)


def linear_backward(dy, cache):
    x, W, has_bias = cache
    dx = dy @ W.T
    dW = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    db = dy.reshape(-1, dy.shape[-1]).sum(axis=0) if has_bias else None
    return dx, dW, db


def phi_forward(x):
    xp = array_module(x)
    e = xp.exp(xp.minimum(x, 0.0))
    return xp.where(x > 0, x + 1.0, e), (x, e)


def phi_backward(dy, cache):
    x, e = cache
    xp = array_module(x)
    return dy * xp.where(x > 0, 1.0, e)


def gelu_forward(x):
    xp = array_module(x)
    t = xp.tanh(_GELU_C * (x + 0.044715 * x**3))
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_backward(dy, cache):
    x, t = cache
    dinner = _GELU_C * (1.0 + 3 * 0.044715 * x**2)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner)


def linear_attention_forward(q, k, v, boundaries, S0, z0, gammas, lam):
    """Two-head causal linear attention, recurrence run step by step.

    q, k, v: (T, heads, dh). S0: (heads, dh, dh), z0: (heads, dh).
    gammas: per-head per-token decay, shape (heads,). The state update
    comes before the read; boundary decay comes after it.
    """
    xp = array_module(q)
    T = q.shape[0]
    g3 = gammas[:, None, None]
    g2 = gammas[:, None]
    S, z = S0, z0
    Ss = xp.empty((T,) + S0.shape, dtype=q.dtype)
    zs = xp.empty((T,) + z0.shape, dtype=q.dtype)
    a = xp.empty_like(v)
    dens = xp.empty(q.shape[:2], dtype=q.dtype)
    for i in range(T):
        S = g3 * S + k[i][:, :, None] * v[i][:, None, :]
        z = g2 * z + k[i]
        num = xp.einsum("hd,hde->he", q[i], S)
        den = (q[i] * z).sum(axis=-1) + ATTN_EPS
        a[i] = num / den[:, None]
        Ss[i], zs[i], dens[i] = S, z, den
        if boundaries[i]:
            S, z = lam * S, lam * z
    cache = (q, k, v, boundaries, Ss, zs, a, dens, gammas, lam)
    return a, (S, z), cache


def linear_attention_backward(da, cache, cuts=None):
    """Backpropagation through time over both heads' S and z.

    `cuts[i]` treats the state entering position i as a constant, which
    confines backpropagation to a span (a sentence) without changing the
    forward values.
    """
    q, k, v, boundaries, Ss, zs, a, dens, gammas, lam = cache
    xp = array_module(q)
    T = q.shape[0]
    g3 = gammas[:, None, None]
    g2 = gammas[:, None]
    dq = xp.empty_like(q)
    dk = xp.empty_like(k)
    dv = xp.empty_like(v)
    dS_next = xp.zeros_like(Ss[0])
    dz_next = xp.zeros_like(zs[0])
    for i in range(T - 1, -1, -1):
        f = lam if boundaries[i] else 1.0
        dS = f * dS_next
        dz = f * dz_next
        dnum = da[i] / dens[i][:, None]
        dden = -(da[i] * a[i]).sum(axis=-1) / dens[i]
        dq[i] = xp.einsum("hde,he->hd", Ss[i], dnum) + dden[:, None] * zs[i]
        dS = dS + q[i][:, :, None] * dnum[:, None, :]
        dz = dz + dden[:, None] * q[i]
        dk[i] = xp.einsum("hde,he->hd", dS, v[i]) + dz
        dv[i] = xp.einsum("hde,hd->he", dS, k[i])
        if cuts is not None and cuts[i]:
            dS_next = xp.zeros_like(dS)
            dz_next = xp.zeros_like(dz)
        else:
            dS_next = g3 * dS
            dz_next = g2 * dz
    return dq, dk, dv


def output_forward(h, E, bo):
    return h @ E.T + bo, (h, E)


def output_backward(dlogits, cache):
    h, E = cache
    return dlogits @ E, dlogits.T @ h, dlogits.sum(axis=0)


def softmax_ce_forward(logits, targets, mask):
    """Summed cross-entropy over masked positions; callers divide by the count."""
    xp = array_module(logits)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsum = xp.log(xp.exp(shifted).sum(axis=-1))
    safe = xp.where(mask, targets, 0)
    picked = xp.take_along_axis(shifted, safe[:, None], axis=-1)[:, 0]
    nll = logsum - picked
    loss_sum = (nll * mask).sum()
    return loss_sum, (shifted, logsum, safe, mask)


def softmax_ce_backward(cache):
    """Gradient of the summed loss with respect to the logits."""
    shifted, logsum, safe, mask = cache
    xp = array_module(shifted)
    p = xp.exp(shifted - logsum[:, None])
    rows = xp.arange(p.shape[0])
    p[rows, safe] -= 1.0
    return p * mask[:, None].astype(p.dtype)
