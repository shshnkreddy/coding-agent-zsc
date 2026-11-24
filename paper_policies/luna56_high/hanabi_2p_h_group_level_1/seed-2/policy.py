"""A conservative two-player H-group Level 1 Hanabi policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "hgroup_level1_conservative",
    "adapter": "hanabi",
    "adapter_version": 1,
    "num_agents": 2,
    "num_colors": 5,
    "num_ranks": 5,
    "hand_size": 5,
    "observation_size": 658,
    "num_actions": 21,
    "policy_structure": "shared_local",
}


def init_agent():
    return jnp.zeros((5, 35), dtype=jnp.float32)


def _high(mask):
    ix = jnp.arange(5, dtype=jnp.int32)
    return jnp.argmax(jnp.where(mask, ix + 1, 0)).astype(jnp.int32)


def _low(mask):
    ix = jnp.arange(5, dtype=jnp.int32)
    return jnp.argmin(jnp.where(mask, ix, 99)).astype(jnp.int32)


def _shift5(x, removed):
    ext = jnp.concatenate([x, jnp.zeros((1, x.shape[1]), dtype=x.dtype)], axis=0)
    ix = jnp.arange(5, dtype=jnp.int32)
    src = ix + (ix >= removed).astype(jnp.int32)
    return ext[src]


def _stack_heights(f):
    vals = jnp.arange(1, 6, dtype=jnp.float32)
    return jnp.asarray([jnp.sum(f[167 + 5 * c : 172 + 5 * c] * vals) for c in range(5)])


def _discard_counts(f):
    starts = (0, 3, 5, 7, 9)
    lens = (3, 2, 2, 2, 1)
    rows = []
    for c in range(5):
        rows.append(jnp.asarray([jnp.sum(f[203 + 10 * c + starts[r] : 203 + 10 * c + starts[r] + lens[r]]) for r in range(5)]))
    return jnp.stack(rows)


def _card_tensor(f, block):
    base = 308 + 175 * block
    return f[base : base + 125].reshape((5, 5, 5))


def _knowledge(f, block):
    base = 308 + 175 * block
    return f[base : base + 175].reshape((5, 35))


def _clue_focus(touched, clued, occupied):
    chop = _low(occupied & ~clued)
    new = touched & ~clued
    any_new = jnp.any(new)
    chop_touched = touched[chop] & (~clued[chop]) & occupied[chop]
    high_new = _high(new)
    high_retouch = _high(touched & clued)
    focus = jnp.where(any_new, jnp.where(chop_touched, chop, high_new), high_retouch)
    valid = jnp.any(touched) & (jnp.sum(occupied) > 0)
    return focus, valid, chop, any_new


def _update_notes(f, memory):
    own_k = _knowledge(f, 0)
    current_flags = own_k[:, 25:35]
    old_notes = memory[:, :25]
    old_flags = memory[:, 25:35]
    la = f[253:308]
    actor_self = la[0] > 0.5
    own_removal = actor_self & ((la[2] > 0.5) | (la[3] > 0.5))
    removed = jnp.argmax(la[23:28]).astype(jnp.int32)
    shifted_notes = jnp.where(own_removal, _shift5(old_notes, removed), old_notes)
    shifted_flags = jnp.where(own_removal, _shift5(old_flags, removed), old_flags)
    clue_from_partner = (la[1] > 0.5) & ((la[4] > 0.5) | (la[5] > 0.5))
    touched = la[18:23] > 0.5
    clued_before = jnp.sum(shifted_flags, axis=1) > 0.5
    occupied = jnp.sum(own_k[:, :25], axis=1) > 0.0
    focus, valid, chop, _ = _clue_focus(touched, clued_before, occupied)
    heights = _stack_heights(f)
    colors = jnp.arange(5, dtype=jnp.int32)[:, None]
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    rr = ranks.astype(jnp.float32)
    hh = heights[:, None]
    literal = own_k[:, :25].reshape((5, 5, 5)) > 0.0
    playable_now = literal & (rr == hh)
    color_clue = la[4] > 0.5
    named_color = jnp.argmax(la[8:13]).astype(jnp.int32)
    named_rank = jnp.argmax(la[13:18]).astype(jnp.int32)
    color_lit = literal & (colors == named_color)
    rank_lit = literal & (ranks == named_rank)
    clue_lit = jnp.where(color_clue, color_lit, rank_lit)
    clue_play = clue_lit & playable_now
    future = clue_lit & (rr > (hh - 1.0))
    save_lit = clue_lit & ((rr == 4.0) | (rr == 1.0))
    focus_note = jnp.where(focus == chop, clue_play | save_lit, clue_play)
    good_note = clue_play | future
    row_focus = focus_note.reshape((5, 25))[focus]
    new_notes = jnp.where(jnp.arange(5)[:, None] == focus, row_focus[None, :], good_note.reshape((5, 25)))
    should_write = clue_from_partner & valid
    notes = jnp.where(should_write, jnp.where(touched[:, None], new_notes, shifted_notes), shifted_notes)
    return jnp.concatenate([notes, current_flags], axis=1)


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    memory0 = jnp.zeros((5, 35), dtype=jnp.float32)
    state = jnp.where(reset, memory0, _update_notes(f, memory))
    notes = state[:, :25]
    heights = _stack_heights(f)
    discards = _discard_counts(f)
    own_k = _knowledge(f, 0)
    own_lit = own_k[:, :25].reshape((5, 5, 5)) > 0.0
    own_flags = own_k[:, 25:35]
    own_clued = jnp.sum(own_flags, axis=1) > 0.5
    own_occ = jnp.sum(own_k[:, :25], axis=1) > 0.0
    own_note = notes.reshape((5, 5, 5)) > 0.5
    note_count = jnp.sum(own_note, axis=(1, 2))
    rr = jnp.arange(5, dtype=jnp.float32)[None, :]
    cc = jnp.arange(5, dtype=jnp.float32)[:, None]
    current_play = rr == (heights[:, None] - 1.0)
    note_play = own_note & current_play[None, :, :]
    safe = (note_count > 0) & (jnp.sum(note_play, axis=(1, 2)) == note_count)

    la = f[253:308]
    actor_partner = la[1] > 0.5
    clue_event = actor_partner & ((la[4] > 0.5) | (la[5] > 0.5))
    touched = la[18:23] > 0.5
    focus, focus_valid, chop, any_new = _clue_focus(touched, own_clued, own_occ)
    color_clue = la[4] > 0.5
    named_color = jnp.argmax(la[8:13]).astype(jnp.int32)
    named_rank = jnp.argmax(la[13:18]).astype(jnp.int32)
    future_color = (cc == named_color) & (rr > heights[:, None])
    future_rank = (rr == named_rank) & (cc < 5)
    future_identity = jnp.where(color_clue, future_color, future_rank)
    connector = future_identity & (rr == heights[:, None])
    connector_any = jnp.any(own_lit & connector, axis=(1, 2))
    finesse_slot = _high(own_occ & ~own_clued)
    prompt_ok = clue_event & focus_valid & any_new & (~(focus == chop)) & (~jnp.any(safe))
    prompt_ok = prompt_ok & jnp.any(connector_any)

    # The visible partner hand is the dedicated first observation block;
    # only the partner's clue flags live in native-knowledge block 1.
    partner_cards = f[:125].reshape((5, 5, 5))
    partner_k = _knowledge(f, 1)
    partner_flags = partner_k[:, 25:35]
    partner_clued = jnp.sum(partner_flags, axis=1) > 0.5
    partner_occ = jnp.sum(partner_cards, axis=(1, 2)) > 0.0
    pcolors = jnp.argmax(jnp.sum(partner_cards, axis=2), axis=1)
    pranks = jnp.argmax(jnp.sum(partner_cards, axis=1), axis=1)
    pplay = partner_occ & (pranks.astype(jnp.float32) == heights[pcolors.astype(jnp.int32)])
    pchop = _low(partner_occ & ~partner_clued)
    disc_flat = discards[pcolors.astype(jnp.int32), pranks.astype(jnp.int32)]
    multiplicity = jnp.where(pranks == 0, 3.0, jnp.where(pranks == 4, 1.0, 2.0))
    pcritical = disc_flat >= (multiplicity - 1.0)
    p_unique_two = (pranks == 1) & (disc_flat == 0.0)
    psave = (pranks == 4) | p_unique_two | pcritical

    # Neutral scores ensure that legal play actions remain a valid fallback
    # even when no clue or discard heuristic applies.
    action_scores = jnp.zeros((21,), dtype=jnp.float32)
    for a in range(10, 20):
        is_color = a < 15
        val = a - 10 if is_color else a - 15
        touched_a = jnp.where(is_color, partner_occ & (pcolors == val), partner_occ & (pranks == val))
        focus_a, valid_a, chop_a, new_a = _clue_focus(touched_a, partner_clued, partner_occ)
        fplay = pplay[focus_a]
        fsave = (focus_a == chop_a) & psave[focus_a]
        useful = (fplay | fsave) & valid_a & new_a
        score = jnp.where(useful, jnp.where(fplay, 500.0, 400.0), -1000000.0)
        score = score + jnp.where(useful & is_color, 8.0, 0.0)
        score = score + jnp.where(useful, jnp.sum(touched_a.astype(jnp.float32)), 0.0)
        action_scores = action_scores.at[a].set(score)

    discardable = own_occ & ~own_clued & (note_count == 0)
    chop_self = _low(discardable)
    fallback_discard = _low(own_occ)
    discard_slot = jnp.where(jnp.any(discardable), chop_self, fallback_discard)
    action_scores = action_scores.at[discard_slot].set(120.0)
    safe_slot = _high(safe)
    safe_scores = action_scores.at[5 + safe_slot].set(900.0)
    action_scores = jnp.where(jnp.any(safe), safe_scores, action_scores)
    prompt_scores = action_scores.at[5 + finesse_slot].set(880.0)
    action_scores = jnp.where(prompt_ok, prompt_scores, action_scores)
    action_scores = jnp.where(legal, action_scores, -1000000.0)
    chosen = jnp.argmax(action_scores).astype(jnp.int32)
    chosen = jnp.where(legal[20] & ~jnp.any(legal[:20]), 20, chosen)
    return chosen, state, {}
