"""Exact literal hand beliefs using inclusion-exclusion over five slots."""
import math
import jax
import jax.numpy as jnp


def _partitions(items):
    if not items:
        return [()]
    result = []
    bit = 1 << items[0]
    for rest in _partitions(items[1:]):
        result.append((bit,) + rest)
        for i in range(len(rest)):
            result.append(rest[:i] + (rest[i] | bit,) + rest[i+1:])
    return result


def exact_belief(mask, counts):
    # Dummy physical cards fill empty slots without conditioning occupied slots.
    missing = ~jnp.any(mask, axis=-1)
    weight = jnp.concatenate([mask.astype(jnp.float32), missing[:, None]], -1)
    count = jnp.concatenate([counts, jnp.array([5.0])])
    partitions = _partitions(list(range(5)))
    indices = jnp.array([list(p)+[0]*(5-len(p)) for p in partitions])
    coeff = jnp.array([math.prod((-1)**(s.bit_count()-1)*math.factorial(s.bit_count()-1) for s in p) for p in partitions], dtype=jnp.float32)
    membership = (jnp.arange(32)[:, None] & (1 << jnp.arange(5))[None, :]) != 0
    def partition(w):
        intersections = jnp.prod(jnp.where(membership[:, :, None], w[None, :, :], 1.0), axis=1)
        block = (intersections @ count).at[0].set(1.0)
        return jnp.sum(coeff*jnp.prod(block[indices], axis=-1))
    z, grad = jax.value_and_grad(partition)(weight)
    posterior = jnp.maximum(weight*grad / jnp.maximum(z, 1e-10), 0)[:, :25]
    return posterior / jnp.maximum(posterior.sum(-1, keepdims=True), 1e-10)
