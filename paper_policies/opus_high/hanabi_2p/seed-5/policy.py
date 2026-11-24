"""Grounded two-player Hanabi rule agent.

The policy decodes the public observation, builds a literal belief over its own
hand from the native knowledge block, and scores every action on one common
utility scale.  All meaning comes from public game facts (fireworks, discards,
literal hint content), so it does not depend on a partner sharing a private
convention.  The only partner-dependent inference is a bounded, graded boost for
cards that a partner's hint has touched.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_rule_agent', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

F32 = jnp.float32
EPS = 1e-6

TOTAL_PER_RANK = [3.0, 2.0, 2.0, 2.0, 1.0]
# Play thresholds indexed by remaining lives (0..3).
PLAY_THRESHOLD = [1.01, 1.01, 0.97, 0.93]
CUE_BOOST_FRESH = 0.75
CUE_BOOST_STALE = 0.20
SAVE_BASE = -1000.0
SAVE_URGENT = 130.0
SAVE_RANK = 15.0
ENDGAME_RELAX = 0.0
MISREAD_PENALTY = 25.0
MISREAD_LIVES = 3.0
DECK_OUT_DISCARD = 0.0


def _constants():
    total = jnp.asarray(TOTAL_PER_RANK, dtype=F32)
    rank_idx = jnp.arange(5, dtype=F32)
    slot_idx = jnp.arange(5)
    # hint index h in 0..9: 0..4 colour hints, 5..9 rank hints.
    h = jnp.arange(10)[:, None, None]
    c = jnp.arange(5)[None, :, None]
    r = jnp.arange(5)[None, None, :]
    hint_sel = jnp.where(h < 5, c == h, r == (h - 5)).astype(F32)  # [10, 5, 5]
    thresholds = jnp.asarray(PLAY_THRESHOLD, dtype=F32)
    return total, rank_idx, slot_idx, hint_sel, thresholds


def init_agent():
    return {
        'cue': jnp.zeros((5,), dtype=F32),
        'fresh': jnp.zeros((5,), dtype=F32),
    }


def _shift_out(vec, slot_onehot):
    """Remove one slot and shift later slots left, appending a zero."""
    slot_idx = jnp.arange(5)
    padded = jnp.concatenate([vec, jnp.zeros((1,), dtype=vec.dtype)])
    slot = jnp.argmax(slot_onehot)
    keep = slot_idx < slot
    return jnp.where(keep, padded[slot_idx], padded[slot_idx + 1])


def _update_memory(memory, features, reset):
    cue = jnp.where(reset, jnp.zeros((5,), F32), memory['cue'])
    fresh = jnp.where(reset, jnp.zeros((5,), F32), memory['fresh'])

    last = features[253:308]
    actor_self = last[0]
    is_play = last[2]
    is_discard = last[3]
    is_colour_hint = last[4]
    is_rank_hint = last[5]
    to_self = last[6]
    matched = last[18:23]
    slot_onehot = last[23:28]

    hint_to_me = ((is_colour_hint + is_rank_hint) > 0) & (to_self > 0)
    my_removal = (actor_self > 0) & ((is_play + is_discard) > 0)

    cue = jnp.where(hint_to_me, jnp.maximum(cue, matched), cue)
    fresh = jnp.where(hint_to_me, matched, fresh)

    cue = jnp.where(my_removal, _shift_out(cue, slot_onehot), cue)
    fresh = jnp.where(my_removal, _shift_out(fresh, slot_onehot), fresh)

    cue = jnp.where(reset, jnp.zeros((5,), F32), cue)
    fresh = jnp.where(reset, jnp.zeros((5,), F32), fresh)
    return {'cue': cue, 'fresh': fresh}


def _decode(features, rank_idx):
    partner = features[0:125].reshape(5, 5, 5)
    deck = features[127:167].sum()
    fireworks = features[167:192].reshape(5, 5)
    tokens = features[192:200].sum()
    lives = features[200:203].sum()
    discard_raw = features[203:253].reshape(5, 10)
    discards = jnp.stack(
        [
            discard_raw[:, 0:3].sum(axis=1),
            discard_raw[:, 3:5].sum(axis=1),
            discard_raw[:, 5:7].sum(axis=1),
            discard_raw[:, 7:9].sum(axis=1),
            discard_raw[:, 9:10].sum(axis=1),
        ],
        axis=1,
    )  # [colour, rank]
    knowledge = features[308:658].reshape(2, 5, 35)
    heights = (fireworks * (rank_idx + 1.0)).sum(axis=1)  # [colour]
    return partner, deck, heights, tokens, lives, discards, knowledge


def agent_act(obs, reset, memory, rng):
    features = obs.features
    legal = obs.legal_actions
    total_per_rank, RANK_IDX, SLOT_IDX, HINT_SEL, thresholds = _constants()
    memory = _update_memory(memory, features, reset)

    partner, deck, heights, tokens, lives, discards, knowledge = _decode(features, RANK_IDX)

    rank_grid = jnp.broadcast_to(RANK_IDX[None, :], (5, 5))
    height_grid = jnp.broadcast_to(heights[:, None], (5, 5))
    playable = (rank_grid == height_grid).astype(F32)
    already_played = (rank_grid < height_grid).astype(F32)

    total_grid = jnp.broadcast_to(total_per_rank[None, :], (5, 5))
    alive = ((total_grid - discards) > 0.5).astype(F32)  # copies still in play
    reachable = jnp.cumprod(alive, axis=1)  # a lower rank lost kills the rest
    dead = 1.0 - reachable
    trash = jnp.clip(already_played + dead, 0.0, 1.0)
    useful = 1.0 - trash
    critical = useful * ((total_grid - discards) < 1.5).astype(F32)

    # Public remaining counts (the partner's viewpoint, ignoring hands).
    public_count = jnp.clip(total_grid - discards - already_played, 0.0, None)
    # Own viewpoint additionally removes the partner's visible cards.
    partner_count = partner.sum(axis=0)
    own_count = jnp.clip(public_count - partner_count, 0.0, None)

    # ---- belief over own hand ----------------------------------------------
    own_mask = (knowledge[0, :, 0:25].reshape(5, 5, 5) > 0).astype(F32)
    weights = own_mask * own_count[None, :, :]
    norm = weights.sum(axis=(1, 2))
    belief = weights / jnp.maximum(norm, EPS)[:, None, None]

    occupied_own = legal[5:10].astype(F32)  # play legal exactly when occupied
    p_play = (belief * playable[None]).sum(axis=(1, 2)) * occupied_own
    p_trash = (belief * trash[None]).sum(axis=(1, 2)) * occupied_own
    p_crit = (belief * critical[None]).sum(axis=(1, 2)) * occupied_own
    exp_rank = (belief * rank_grid[None]).sum(axis=(1, 2))

    own_flags = knowledge[0, :, 25:35]
    own_hinted = (own_flags.sum(axis=1) > 0).astype(F32)

    # ---- cue-conditioned playability ---------------------------------------
    cued = jnp.clip(jnp.maximum(memory['cue'], own_hinted), 0.0, 1.0) * occupied_own
    fresh = jnp.clip(memory['fresh'], 0.0, 1.0) * occupied_own
    cue_mass = (cued * p_play).sum()
    share = (p_play * cued) / jnp.maximum(cue_mass, EPS)
    boost = jnp.where(fresh > 0, CUE_BOOST_FRESH, CUE_BOOST_STALE) * cued
    p_eff = jnp.clip(p_play + boost * share, 0.0, 1.0)

    lives_idx = jnp.clip(lives, 0, 3).astype(jnp.int32)
    threshold = thresholds[lives_idx]
    endgame = (deck < 0.5).astype(F32)
    threshold = threshold - ENDGAME_RELAX * endgame * (lives > 1.5).astype(F32)

    certain = (p_play > 0.999).astype(F32) * occupied_own
    risky = ((p_eff >= threshold) & (p_play > 0.02)).astype(F32) * occupied_own * (1.0 - certain)

    score_play = (
        certain * (1000.0 + (4.0 - exp_rank))
        + risky * (600.0 + 60.0 * p_eff)
        + (1.0 - certain) * (1.0 - risky) * (20.0 * p_eff - 40.0 * p_trash)
    )

    # ---- partner side --------------------------------------------------------
    occupied_partner = (partner.sum(axis=(1, 2)) > 0.5).astype(F32)
    partner_playable = (partner * playable[None]).sum(axis=(1, 2))
    partner_critical = (partner * critical[None]).sum(axis=(1, 2))
    partner_trash = (partner * trash[None]).sum(axis=(1, 2))

    partner_mask = (knowledge[1, :, 0:25].reshape(5, 5, 5) > 0).astype(F32)
    partner_plausible = partner_mask * (public_count > 0)[None].astype(F32)
    known_play_before = (
        ((partner_plausible * (1.0 - playable)[None]).sum(axis=(1, 2)) < 0.5)
        & ((partner_plausible * playable[None]).sum(axis=(1, 2)) > 0.5)
    ).astype(F32) * occupied_partner
    known_trash_before = (
        (partner_plausible * useful[None]).sum(axis=(1, 2)) < 0.5
    ).astype(F32) * occupied_partner

    partner_flags = knowledge[1, :, 25:35]
    partner_unhinted = (partner_flags.sum(axis=1) < 0.5).astype(F32) * occupied_partner
    # chop = oldest untouched slot
    chop_rank = jnp.cumsum(partner_unhinted) * partner_unhinted
    partner_chop = (chop_rank == 1.0).astype(F32)

    # ---- hint evaluation -----------------------------------------------------
    touch = jnp.einsum('scr,hcr->hs', partner, HINT_SEL) * occupied_partner[None, :]
    keep = jnp.where(touch[:, :, None, None] > 0.5, HINT_SEL[:, None], 1.0 - HINT_SEL[:, None])
    after = partner_mask[None] * keep * (public_count > 0)[None, None].astype(F32)
    bad_after = (after * (1.0 - playable)[None, None]).sum(axis=(2, 3))
    good_after = (after * playable[None, None]).sum(axis=(2, 3))
    known_play_after = ((bad_after < 0.5) & (good_after > 0.5)).astype(F32) * occupied_partner[None, :]
    newly_known = jnp.clip(known_play_after - known_play_before[None, :], 0.0, 1.0)
    gain_known = newly_known.sum(axis=1)

    dead_after = (after * useful[None, None]).sum(axis=(2, 3))
    known_trash_after = (dead_after < 0.5).astype(F32) * occupied_partner[None, :]
    gain_trash = jnp.clip(known_trash_after - known_trash_before[None, :], 0.0, 1.0).sum(axis=1)

    attr_flags = jnp.concatenate([partner_flags[:, 0:5], partner_flags[:, 5:10]], axis=1).T  # [10, 5]
    new_touch = touch * (attr_flags < 0.5).astype(F32)
    touched_playable = (new_touch * partner_playable[None, :]).sum(axis=1)
    touched_nonplay = (new_touch * (1.0 - partner_playable)[None, :]).sum(axis=1)
    touched_trash = (new_touch * partner_trash[None, :]).sum(axis=1)
    # Cards a partner could misread as a play cue: newly touched, not playable,
    # and still able to look playable from their own literal knowledge.
    touched_misread = (
        new_touch * (1.0 - partner_playable)[None, :] * (good_after > 0.5).astype(F32)
    ).sum(axis=1)
    partner_rank = (partner * rank_grid[None]).sum(axis=(1, 2))
    # losing a critical card of rank r costs the remaining ranks r..4 of its colour
    save_weight = (5.0 - partner_rank) * partner_critical * partner_chop * (1.0 - known_trash_before)
    save_value = (new_touch * save_weight[None, :]).sum(axis=1)
    # A save hint is only safe when no touched slot can look playable to the
    # partner afterwards, so that no reasonable partner reads it as a play cue.
    save_risk = (new_touch * good_after).sum(axis=1)
    info_value = new_touch.sum(axis=1)

    partner_has_play = (known_play_before.sum() > 0.5).astype(F32)

    tier_a = (gain_known > 0.5).astype(F32) * (
        800.0 + 25.0 * gain_known + 4.0 * touched_playable - 3.0 * touched_nonplay
    )
    tier_b = ((touched_playable > 0.5) & (touched_nonplay < 0.5)).astype(F32) * (
        500.0 + 20.0 * touched_playable
    )
    imminent = 1.0 - partner_has_play
    tier_c = ((save_value > 0.5) & (save_risk < 0.5)).astype(F32) * (
        SAVE_BASE + SAVE_URGENT * imminent + SAVE_RANK * save_value - 5.0 * touched_nonplay
    )
    tier_d = (touched_playable > 0.5).astype(F32) * (
        320.0 + 15.0 * touched_playable - 12.0 * touched_nonplay
    )
    tier_e = 60.0 + 4.0 * info_value + 10.0 * gain_trash - 2.0 * touched_trash

    score_hint = jnp.maximum(jnp.maximum(jnp.maximum(tier_a, tier_b), jnp.maximum(tier_c, tier_d)), tier_e)
    weak = (gain_known < 0.5).astype(F32)
    # With lives already lost, a partner that misreads a hint is more costly,
    # so ambiguous hints are discouraged more strongly.
    misread_scale = 1.0 + MISREAD_LIVES * (3.0 - lives)
    score_hint = score_hint - MISREAD_PENALTY * misread_scale * touched_misread
    score_hint = score_hint - 220.0 * weak * (tokens < 1.5).astype(F32)
    score_hint = score_hint - 130.0 * weak * partner_has_play * (tokens < 4.5).astype(F32)

    # ---- discard evaluation ---------------------------------------------------
    own_unhinted = (1.0 - own_hinted) * occupied_own
    own_chop_rank = jnp.cumsum(own_unhinted) * own_unhinted
    own_chop = (own_chop_rank == 1.0).astype(F32)

    score_discard = (
        150.0
        + 250.0 * (p_trash > 0.999).astype(F32)
        + 60.0 * p_trash
        - 150.0 * p_crit
        - 70.0 * p_play
        - 25.0 * own_hinted
        + 10.0 * own_chop
        + 0.5 * (4.0 - SLOT_IDX.astype(F32))
        - DECK_OUT_DISCARD * (deck < 0.5).astype(F32)
    )

    # Floors keep the desperate ordering sane: any legal hint or discard is
    # preferred over an uninformed play.
    score_hint = jnp.maximum(score_hint, 45.0)
    score_discard = jnp.maximum(score_discard, 40.0)
    score_play = jnp.minimum(score_play, jnp.where(certain + risky > 0.5, 1e6, 25.0))
    scores = jnp.concatenate([score_discard, score_play, score_hint, jnp.asarray([-1e8], dtype=F32)])
    scores = jnp.where(legal, scores, -1e9)
    action = jnp.argmax(scores).astype(jnp.int32)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)

    chosen_slot = jnp.clip(action, 0, 9) % 5
    trace = {
        'reset': jnp.asarray(reset, dtype=F32),
        'action': action,
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'fw': heights.sum(),
        'reachable': reachable.sum(),
        'certain': certain[chosen_slot],
        'risky': risky[chosen_slot],
        'p_eff_sel': p_eff[chosen_slot],
        'p_play_sel': p_play[chosen_slot],
        'p_crit_sel': p_crit[chosen_slot],
        'p_trash_sel': p_trash[chosen_slot],
        'partner_playable': partner_playable.sum(),
        'partner_known_play': known_play_before.sum(),
        'best_hint': jnp.max(score_hint),
    }
    return action, memory, trace
