from dataclasses import dataclass, replace
from typing import Any

import numpy


@dataclass(frozen=True)
class RecurrentState:
    """Retention state. Arrays are never mutated in place, so a copy is free.

    `logits` is the next-token distribution after the last processed token,
    or None before any token: predictions read it without stepping again.
    """

    S_s: Any
    z_s: Any
    S_f: Any
    z_f: Any
    logits: Any = None


def zero_state(d_model, xp=numpy, dtype=numpy.float32):
    half = d_model // 2
    return RecurrentState(
        S_s=xp.zeros((half, half), dtype=dtype),
        z_s=xp.zeros(half, dtype=dtype),
        S_f=xp.zeros((half, half), dtype=dtype),
        z_f=xp.zeros(half, dtype=dtype),
        logits=None,
    )


def boundary_decay(state, lam):
    return replace(state, S_s=lam * state.S_s, z_s=lam * state.z_s, S_f=lam * state.S_f, z_f=lam * state.z_f)

