"""Shared, locally observable four-turn clustering protocol."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'row_signal_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.concatenate((jnp.full((9,), -1, dtype=jnp.int32), jnp.zeros((2,), dtype=jnp.int32)))


def _first_legal(mask, lo=0, hi=1068):
    ids = jnp.arange(1068, dtype=jnp.int32)
    ok = mask & (ids >= lo) & (ids < hi)
    return jnp.argmax(ok.astype(jnp.int32)).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    nf = obs.node_features
    mask = obs.legal_actions
    sub = nf[0, 11].astype(jnp.int32)
    active = nf[0, 10] > 0
    base_memory = jnp.where(reset, init_agent(), memory)
    known = base_memory[:9]
    turn = base_memory[9]
    role = jnp.where(reset & active, jnp.asarray(1, dtype=jnp.int32), base_memory[10])

    visible = nf[:9, :3]
    has_color = jnp.sum(visible, axis=1) > 0
    seen_color = jnp.argmax(visible, axis=1).astype(jnp.int32)
    known = jnp.where(has_color, seen_color, known)
    turn_now = jnp.where((active & (sub == 0)), turn + 1, turn)
    next_memory = jnp.concatenate((known, turn_now[None], role[None]))

    public_unseen = nf[:9, 9] == 0
    unlocked_now = nf[:9, 8] == 0
    fresh = public_unseen & unlocked_now & (known < 0)
    unknown = unlocked_now & (known < 0)
    any_fresh = jnp.any(fresh)
    any_unknown = jnp.any(unknown)
    observe_card = jnp.where(any_fresh, jnp.argmax(fresh.astype(jnp.int32)),
                             jnp.where(any_unknown, jnp.argmax(unknown.astype(jnp.int32)),
                                       jnp.argmax(unlocked_now.astype(jnp.int32))))
    observe_action = 1 + observe_card.astype(jnp.int32)

    pos = obs.positions
    color_ok = known >= 0
    target_y = 3 + jnp.maximum(known, 0)
    unlocked = unlocked_now
    wrong = color_ok & unlocked & (pos[:, 1] != target_y)
    cards = jnp.arange(9, dtype=jnp.int32)
    xs = jnp.arange(9, dtype=jnp.int32)
    ys = jnp.arange(9, dtype=jnp.int32)
    cand = 10 + cards[:, None, None] * 81 + xs[None, :, None] * 9 + ys[None, None, :]
    cand_flat = cand.reshape(-1)
    row_ok = ys[None, None, :] == target_y[:, None, None]
    legal_cand = mask[cand_flat].reshape((9, 9, 9))
    preferred = wrong[:, None, None] & row_ok & legal_cand
    preserve = color_ok & unlocked & (pos[:, 1] == target_y)
    preserve = preserve[:, None, None] & row_ok & legal_cand
    pref_flat = preferred.reshape(-1)
    preserve_flat = preserve.reshape(-1)
    pref_action = cand_flat[jnp.argmax(pref_flat.astype(jnp.int32))]
    preserve_action = cand_flat[jnp.argmax(preserve_flat.astype(jnp.int32))]
    move_fallback = _first_legal(mask, 10, 739)
    move_action = jnp.where(jnp.any(pref_flat), pref_action,
                             jnp.where(jnp.any(preserve_flat), preserve_action, move_fallback))

    reveal = _first_legal(mask, 739, 743)
    hint_sets = nf[9:13, 3:6] > 0
    hint_revealed = jnp.sum(nf[9:13, 3:6], axis=1) > 0
    hint_placed = nf[9:13, 8] > 0
    known_idx = jnp.clip(known, 0, 2)
    contains = jnp.take(hint_sets, known_idx, axis=1)
    place_cand = 743 + jnp.arange(4, dtype=jnp.int32)[:, None] * 81
    place_cand = place_cand + pos[None, :, 0] * 9 + pos[None, :, 1]
    place_legal = mask[place_cand]
    correct_place = (hint_revealed & ~hint_placed)[:, None] & color_ok[None, :] & contains & unlocked[None, :]
    correct_place = correct_place & place_legal
    correct_flat = correct_place.reshape(-1)
    place_flat = place_cand.reshape(-1)
    correct_action = place_flat[jnp.argmax(correct_flat.astype(jnp.int32))]
    place_fallback = _first_legal(mask, 743, 1067)
    place_action = jnp.where(jnp.any(correct_flat), correct_action, place_fallback)
    has_reveal = jnp.any(mask[739:743])
    end_action = jnp.asarray(1067, dtype=jnp.int32)
    base_action = jnp.where(sub == 0, observe_action, move_action)
    base_action = jnp.where(sub == 1, observe_action, base_action)
    base_action = jnp.where(sub == 3, jnp.where(has_reveal, reveal, place_action), base_action)
    threshold = jnp.where(role > 0, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(2, dtype=jnp.int32))
    base_action = jnp.where((sub == 0) & (turn >= threshold), end_action, base_action)
    action = jnp.where(active, base_action, jnp.asarray(0, dtype=jnp.int32))
    action = jnp.where(mask[action], action, _first_legal(mask))
    return action.astype(jnp.int32), next_memory, {}
