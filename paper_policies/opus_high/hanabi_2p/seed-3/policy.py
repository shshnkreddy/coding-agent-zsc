"""Convention-light Hanabi agent.

Every decision is justified by publicly observable information: plays require a
high literal playability probability derived from the native knowledge block,
hints are chosen so that the partner's own public knowledge makes a playable
card nearly certain, and discards target the slot most likely to be useless.  No
private signalling scheme is used, so the behaviour stays meaningful when the
partner was produced by an independent search.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

# ---------------------------------------------------------------- constants --
COPIES = (3.0, 2.0, 2.0, 2.0, 1.0)  # copies per rank id

# play gate: required playability probability by remaining lives (3, 2, 1)
PLAY_THRESHOLD_3 = 0.72
PLAY_THRESHOLD_2 = 0.85
PLAY_THRESHOLD_1 = 0.95
PLAY_FLOOR = 0.34          # never play below this literal probability
FOCUS_BONUS = 0.70         # credit for a card the partner pointed at alone
WEAK_BONUS = 0.10          # credit for a card touched inside a wider hint
FOCUS_FLOOR = 0.25         # relaxed literal floor for a pointed-at card

HINT_CONFIDENT = 0.80      # partner considers a card playable above this
SAVE_BONUS = 30.0          # touching a fresh critical card protects it
TOUCH_PLAY_BONUS = 8.0     # touching a fresh playable card is always welcome
TOUCH_JUNK_COST = 25.0     # a fresh non-playable touch may mislead the partner
TOUCH_JUNK_QUADRATIC = 15.0
BOMB_INVITE_COST = 60.0    # never leave an unplayable card looking playable
TRUST_FLOOR = 0.25         # apparent playability a trusting reader may act on
ENDGAME_RELAX = 0.6        # play threshold multiplier in the last few draws
ENDGAME_FLOOR = 0.15       # relaxed literal floor in the final round


def _shift_left(vec, slot):
    """Remove `slot` from a 5-vector, shifting later entries left."""
    idx = jnp.arange(5)
    src = jnp.where(idx < slot, idx, idx + 1)
    return jnp.where(src < 5, vec[jnp.clip(src, 0, 4)], 0.0)


def _normalize(weights, axes, fallback):
    total = weights.sum(axes, keepdims=True)
    return jnp.where(total > 1e-6, weights / jnp.maximum(total, 1e-6), fallback)


def init_agent():
    return {
        'flags': jnp.zeros((5,), dtype=jnp.float32),
        'weak_flags': jnp.zeros((5,), dtype=jnp.float32),
        'prev_touch': jnp.zeros((5,), dtype=jnp.float32),
    }


def agent_act(obs, reset, memory, rng):
    features = obs.features.astype(jnp.float32)
    legal = obs.legal_actions
    IDX = jnp.arange(5)
    copies = jnp.asarray(COPIES, dtype=jnp.float32)

    # ------------------------------------------------------------- decoding --
    partner_hand = features[0:125].reshape(5, 5, 5)          # [slot, color, rank]
    deck_size = features[127:167].sum()
    fireworks = features[167:192].reshape(5, 5)              # [color, rank]
    height = (fireworks * (IDX[None, :] + 1.0)).sum(-1)      # [color]
    tokens = features[192:200].sum()
    lives = features[200:203].sum()
    discard_block = features[203:253].reshape(5, 10)
    discarded = jnp.stack([
        discard_block[:, 0:3].sum(-1),
        discard_block[:, 3:5].sum(-1),
        discard_block[:, 5:7].sum(-1),
        discard_block[:, 7:9].sum(-1),
        discard_block[:, 9:10].sum(-1),
    ], axis=-1)                                              # [color, rank]
    last = features[253:308]
    knowledge = features[308:658].reshape(2, 5, 35)
    v0_self = knowledge[0, :, 0:25].reshape(5, 5, 5)         # [slot, color, rank]
    touch_self = (knowledge[0, :, 25:35].sum(-1) > 0.5).astype(jnp.float32)
    v0_partner = knowledge[1, :, 0:25].reshape(5, 5, 5)
    touch_partner = (knowledge[1, :, 25:35].sum(-1) > 0.5).astype(jnp.float32)

    # ------------------------------------------------- public card accounting --
    ranks = jnp.broadcast_to(IDX[None, :], (5, 5)).astype(jnp.float32)
    total = jnp.broadcast_to(copies[None, :], (5, 5))
    on_firework = (ranks < height[:, None]).astype(jnp.float32)
    remaining_public = jnp.clip(total - discarded - on_firework, 0.0, None)
    partner_counts = partner_hand.sum(0)
    remaining_mine = jnp.clip(remaining_public - partner_counts, 0.0, None)

    playable = (ranks == height[:, None]).astype(jnp.float32)
    existing = jnp.clip(total - discarded, 0.0, None)
    blocked = ((existing < 0.5) & (ranks >= height[:, None])).astype(jnp.float32)
    dead = (jnp.cumsum(blocked, axis=1) > 0.5).astype(jnp.float32)
    useless = jnp.clip((ranks < height[:, None]).astype(jnp.float32) + dead, 0.0, 1.0)
    critical = ((existing < 1.5) & (existing > 0.5)).astype(jnp.float32) * (1.0 - useless)

    # ----------------------------------------------------------- own beliefs --
    mask_self = (v0_self > 1e-9).astype(jnp.float32)
    belief = _normalize(
        mask_self * remaining_mine[None],
        (1, 2),
        _normalize(mask_self * remaining_public[None], (1, 2), v0_self),
    )
    occupied = (mask_self.sum((1, 2)) > 0.5).astype(jnp.float32)
    p_play = (belief * playable[None]).sum((1, 2)) * occupied
    p_useless = (belief * useless[None]).sum((1, 2)) * occupied
    p_critical = (belief * critical[None]).sum((1, 2)) * occupied

    # ----------------------------------------------- memory: hint bookkeeping --
    init = init_agent()
    flags = jnp.where(reset, init['flags'], memory['flags'])
    weak_flags = jnp.where(reset, init['weak_flags'], memory['weak_flags'])
    prev_touch = jnp.where(reset, init['prev_touch'], memory['prev_touch'])

    actor_self = last[0] > 0.5
    was_play = last[2] > 0.5
    was_discard = last[3] > 0.5
    was_hint = (last[4] + last[5]) > 0.5
    to_self = last[6] > 0.5
    removed_slot = jnp.argmax(last[23:28])

    removed_by_me = actor_self & (was_play | was_discard)
    flags = jnp.where(removed_by_me, _shift_left(flags, removed_slot), flags)
    weak_flags = jnp.where(removed_by_me, _shift_left(weak_flags, removed_slot), weak_flags)
    prev_touch = jnp.where(removed_by_me, _shift_left(prev_touch, removed_slot), prev_touch)

    newly_touched = jnp.clip(touch_self - prev_touch, 0.0, 1.0)
    got_hint = (~actor_self) & was_hint & to_self
    newly_touched = jnp.where(got_hint, newly_touched, 0.0)
    # A hint that freshly touches exactly one card is an unambiguous pointer;
    # a wider hint says far less about which card the partner meant.
    focused = newly_touched.sum() < 1.5
    flags = jnp.clip(flags + jnp.where(focused, newly_touched, 0.0), 0.0, 1.0) * occupied
    weak_flags = jnp.clip(weak_flags + jnp.where(focused, 0.0, newly_touched), 0.0, 1.0) * occupied
    next_memory = {'flags': flags, 'weak_flags': weak_flags, 'prev_touch': touch_self}

    # --------------------------------------------------------- play scoring ---
    threshold = jnp.where(lives > 2.5, PLAY_THRESHOLD_3,
                          jnp.where(lives > 1.5, PLAY_THRESHOLD_2, PLAY_THRESHOLD_1))
    # In the final stretch the remaining turns are few and an unplayed card
    # scores nothing, so likely cards are worth attempting.
    endgame = (deck_size < 5.5) & (lives > 1.5)
    threshold = jnp.where(endgame, threshold * ENDGAME_RELAX, threshold)
    trust = jnp.where(lives > 2.5, 1.0, jnp.where(lives > 1.5, 0.5, 0.0))
    credit = trust * (FOCUS_BONUS * flags + WEAK_BONUS * weak_flags)
    effective = p_play + credit
    floor = jnp.where(credit > 0.5 * FOCUS_BONUS, FOCUS_FLOOR, PLAY_FLOOR)
    floor = jnp.where(endgame, jnp.minimum(floor, ENDGAME_FLOOR), floor)
    play_ok = (effective >= threshold) & (p_play >= floor) & (occupied > 0.5)
    play_pref = 10.0 * effective + 0.1 * (4.0 - IDX)
    play_score = jnp.where(play_ok, 2000.0 + play_pref, 10.0 + p_play)

    # --------------------------------------------------------- hint scoring ---
    sel_color = (IDX[:, None, None] == IDX[None, :, None]) * jnp.ones((1, 1, 5), dtype=bool)
    sel_rank = (IDX[:, None, None] == IDX[None, None, :]) * jnp.ones((1, 5, 1), dtype=bool)
    sel = jnp.concatenate([sel_color, sel_rank], axis=0).astype(jnp.float32)  # [10, color, rank]

    match = ((partner_hand[None] * sel[:, None]).sum((-1, -2)) > 0.5).astype(jnp.float32)  # [10, slot]
    occupied_p = (partner_hand.sum((1, 2)) > 0.5).astype(jnp.float32)
    match = match * occupied_p[None]

    mask_partner = (v0_partner > 1e-9).astype(jnp.float32)
    keep = jnp.where(match[..., None, None] > 0.5, sel[:, None], 1.0 - sel[:, None])
    belief_after = _normalize(mask_partner[None] * keep * remaining_public[None, None],
                              (-1, -2), mask_partner[None] * keep)
    p_after = (belief_after * playable[None, None]).sum((-1, -2))          # [10, slot]
    u_after = (belief_after * useless[None, None]).sum((-1, -2))
    belief_before = _normalize(mask_partner * remaining_public[None], (1, 2), v0_partner)
    p_before = (belief_before * playable[None]).sum((1, 2))                # [slot]
    u_before = (belief_before * useless[None]).sum((1, 2))

    playable_p = (partner_hand * playable[None]).sum((1, 2))
    critical_p = (partner_hand * critical[None]).sum((1, 2))
    useless_p = (partner_hand * useless[None]).sum((1, 2))
    fresh = (1.0 - touch_partner) * occupied_p

    unlock = match * playable_p[None] * (p_after >= HINT_CONFIDENT) * (p_before < HINT_CONFIDENT)[None]
    mislead = ((1.0 - playable_p) * occupied_p)[None] * (p_after >= HINT_CONFIDENT) * (p_before < HINT_CONFIDENT)[None]
    save = match * fresh[None] * critical_p[None]
    known_trash = match * (u_after >= 0.95) * (u_before < 0.95)[None]
    touch_play = match * fresh[None] * playable_p[None]
    # A partner may read any fresh touch as "this card is playable"; touching a
    # non-playable card is therefore a bomb risk that grows with the count.
    touch_junk = (match * fresh[None] * (1.0 - playable_p)[None]).sum(-1)
    # A touched card that still looks playable invites a bomb from any partner
    # that reads touches as pointers - including this policy itself.
    bomb_invite = (match * fresh[None] * (1.0 - playable_p)[None]
                   * (p_after >= TRUST_FLOOR)).sum(-1)

    hint_value = (100.0 * unlock.sum(-1)
                  - 55.0 * mislead.sum(-1)
                  + SAVE_BONUS * save.sum(-1)
                  + 4.0 * known_trash.sum(-1)
                  + TOUCH_PLAY_BONUS * touch_play.sum(-1)
                  - TOUCH_JUNK_COST * touch_junk
                  - TOUCH_JUNK_QUADRATIC * touch_junk * touch_junk
                  - BOMB_INVITE_COST * bomb_invite
                  - 1.0 * match.sum(-1))
    gate = jnp.where(tokens > 2.5, 5.0, jnp.where(tokens > 1.5, 15.0, 45.0))
    hint_score = jnp.where(hint_value >= gate, 600.0 + hint_value, 50.0 + 0.01 * hint_value)

    # ------------------------------------------------------ discard scoring ---
    discard_score = (100.0
                     + 30.0 * p_useless
                     - 40.0 * p_critical
                     - 20.0 * p_play
                     - 12.0 * touch_self
                     - 10.0 * flags
                     + 1.0 * (4.0 - IDX))

    scores = jnp.concatenate([discard_score, play_score, hint_score,
                              jnp.asarray([-1e8], dtype=jnp.float32)])
    scores = jnp.where(legal, scores, -1e9)
    action = jnp.argmax(scores).astype(jnp.int32)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)

    trace = {
        'p_play': p_play.astype(jnp.float32),
        'p_useless': p_useless.astype(jnp.float32),
        'flags': (flags + 0.5 * weak_flags).astype(jnp.float32),
        'hint_value': hint_value.astype(jnp.float32),
        'board': jnp.asarray([height.sum(), tokens, lives, deck_size], dtype=jnp.float32),
    }
    return action, next_memory, trace
