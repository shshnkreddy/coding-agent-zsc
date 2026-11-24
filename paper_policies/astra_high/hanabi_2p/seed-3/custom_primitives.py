"""Fixed-size exact without-replacement inference for a five-card hand."""
import math
import jax.numpy as jnp


def _partitions(xs):
    if not xs:
        return [()]
    first, *rest = xs
    result = []
    for p in _partitions(rest):
        result.append(((first,),) + p)
        for i in range(len(p)):
            result.append(p[:i] + ((first,) + p[i],) + p[i+1:])
    return result


def exact_marginals(mask, counts):
    # The partition lattice performs inclusion-exclusion over physical-card
    # collisions. A block requires its slots to use the same physical card.
    ps = _partitions(list(range(5)))
    blocks = [[sum(1 << s for s in b) for b in p] for p in ps]
    padded = [b + [0] * (5-len(b)) for b in blocks]
    coefficients = [math.prod((-1)**(len(b)-1)*math.factorial(len(b)-1) for b in p) for p in ps]
    slot_block = [[next(b for b in bs if b & (1<<s)) for s in range(5)] for bs in blocks]
    other_blocks = [[[b for b in bs if not b & (1<<s)] + [0] * (5-len(bs)) for s in range(5)] for bs in blocks]
    # Empty slots have a separate dummy identity, avoiding a special dynamic
    # hand size and preserving the distribution of occupied slots.
    occupied = jnp.any(mask,axis=-1)
    m = jnp.concatenate([mask,~occupied[:,None]],axis=-1)
    n = jnp.concatenate([counts,jnp.array([5.],jnp.float32)])
    membership = (jnp.arange(32)[:,None] & (1 << jnp.arange(5)[None,:])) > 0
    intersection = jnp.all(jnp.where(membership[:,:,None],m[None,:,:],True),axis=1)
    block_weight = (intersection*n).sum(-1).at[0].set(1.)
    others = jnp.prod(block_weight[jnp.array(other_blocks)],axis=-1)
    terms = jnp.array(coefficients,jnp.float32)[:,None]*others
    numerator = jnp.einsum('ps,pst->st',terms,intersection[jnp.array(slot_block)].astype(jnp.float32))*n
    numerator = jnp.maximum(numerator[:,:25],0.)*mask
    fallback = mask*counts
    numerator = jnp.where(numerator.sum(-1,keepdims=True)>0,numerator,fallback)
    return numerator/jnp.maximum(numerator.sum(-1,keepdims=True),1e-10)


def feasible_support(mask, counts):
    occupied = jnp.any(mask,axis=-1)
    m = jnp.concatenate([mask,~occupied[:,None]],axis=-1)
    n = jnp.concatenate([counts.astype(jnp.int32),jnp.array([5],jnp.int32)])
    member = (jnp.arange(32)[:,None] & (1 << jnp.arange(5)[None,:])) > 0
    unions = jnp.any(member[:,:,None]&m[None,:,:],axis=1)
    sizes = member.sum(-1)
    capacity = unions.astype(jnp.int32)@n
    # Fixing one slot to type t uses one physical copy. For every subset of
    # the other slots, the union of allowed cards must still cover the subset.
    feasible_subset = capacity[:,None]-unions >= sizes[:,None]
    possible = jnp.all(jnp.where(~member[:,:,None],feasible_subset[:,None,:],True),axis=0)
    return (possible & m & (n>0))[:,:25]
