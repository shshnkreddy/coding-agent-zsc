"""Fixed-size exact without-replacement card marginals."""
import math
import jax.numpy as jnp


def partitions(items):
    if not items:
        return [()]
    head, *tail = items
    out = []
    for partition in partitions(tail):
        out.append(((head,),) + partition)
        for i in range(len(partition)):
            out.append(partition[:i] + ((head,) + partition[i],) + partition[i+1:])
    return out


def exact_beliefs(mask, counts):
    # Empty slots receive distinct artificial cards, making every hand size
    # share the same fixed five-slot marginalization.
    empty = ~jnp.any(mask, -1)
    m = jnp.concatenate([mask, jnp.eye(5, dtype=jnp.bool_) & empty[:, None]], -1)
    n = jnp.concatenate([counts, jnp.ones(5)])
    bits = (jnp.arange(32)[:, None] & (1 << jnp.arange(5))[None, :]) > 0
    intersection = jnp.all(jnp.where(bits[:, :, None], m[None, :, :], True), axis=1).astype(jnp.float32)
    intersection = intersection.at[0].set(0.)
    moment = (intersection @ n).at[0].set(1.)
    blocks, coefs = [], []
    for slot in range(5):
        slot_blocks, slot_coefs = [], []
        for partition in partitions([s for s in range(5) if s != slot]):
            slot_blocks.append([sum(1 << s for s in group) for group in partition] + [0] * (4-len(partition)))
            slot_coefs.append(math.prod((-1)**(len(group)-1) * math.factorial(len(group)-1) for group in partition))
        blocks.append(slot_blocks)
        coefs.append(slot_coefs)
    b = jnp.asarray(blocks, dtype=jnp.int32)
    c = jnp.asarray(coefs, dtype=jnp.float32)
    # Condition each slot on each identity. The four other cards must be
    # assigned to distinct physical copies, using inclusion-exclusion.
    conditioned = moment[b, None] - intersection[b]
    z = jnp.sum(jnp.prod(conditioned, axis=-2) * c[:, :, None], axis=1)
    w = jnp.maximum(z, 0.) * m * n
    p = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-12)
    return p[:, :25]
