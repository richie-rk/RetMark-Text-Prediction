import math

from pk.engine import layers


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total <= max_norm:
        return grads, total
    scale = max_norm / total
    return {k: g * scale for k, g in grads.items()}, total


class Adam:
    """Adam whose beta2 is derived per step from a token half-life, so the
    second-moment memory spans a fixed number of tokens whatever the batch.

    With beta2 varying per step, v's bias correction is 1 - prod(beta2) =
    1 - 2**-half_lives, which needs the half-lives elapsed rather than a step
    count. Accumulating half-lives (not tokens) keeps it exact if the
    configured half-life changes between runs.
    """

    def __init__(self, params, beta1, half_life_tokens, eps):
        self.beta1 = beta1
        self.half_life_tokens = half_life_tokens
        self.eps = eps
        self.m = {k: layers.array_module(p).zeros_like(p) for k, p in params.items()}
        self.v = {k: layers.array_module(p).zeros_like(p) for k, p in params.items()}
        self.step_count = 0
        self.half_lives = 0.0
        self.tokens = 0

    def beta2_for(self, tokens_in_step):
        return 2.0 ** (-tokens_in_step / self.half_life_tokens)

    def step(self, params, grads, lr, tokens_in_step):
        """Return new parameter arrays; `params` itself is left untouched so a
        reader holding the old dict never sees a half-applied update."""
        beta2 = self.beta2_for(tokens_in_step)
        self.step_count += 1
        self.half_lives += tokens_in_step / self.half_life_tokens
        self.tokens += tokens_in_step
        c1 = 1.0 - self.beta1**self.step_count
        c2 = 1.0 - 2.0 ** (-self.half_lives)
        new = {}
        for k, p in params.items():
            g = grads[k]
            xp = layers.array_module(p)
            self.m[k] *= self.beta1
            self.m[k] += (1.0 - self.beta1) * g
            self.v[k] *= beta2
            self.v[k] += (1.0 - beta2) * (g * g)
            update = (self.m[k] / c1) / (xp.sqrt(self.v[k] / c2) + self.eps)
            new[k] = (p - lr * update).astype(p.dtype)
        return new
