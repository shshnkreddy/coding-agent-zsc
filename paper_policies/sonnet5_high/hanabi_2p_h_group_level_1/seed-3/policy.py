"""H-group Level 1 heuristic Hanabi policy.

Vectorized, fixed-shape JAX implementation. All control flow is expressed via
jnp.where / boolean masks / static (compile-time) Python loops over the fixed
5-slot hand and 21-action space; no data-dependent Python branching is used.

Design summary (see ledger A01-A03 for the full rationale and self-play
evidence behind each choice):
  - Own-hand play decisions use the native per-slot V0 possibility
    distribution (a normalized literal-possibility posterior that already
    reflects every positive/negative hint plus public discard/firework
    counts, per spec). A slot is played once the probability mass on
    currently-playable identities (rank == that color's firework height)
    clears a fixed threshold. A pure "is it merely possible" (nonzero
    support) test is unsound on its own: it fired on a freshly color-clued
    card at game start, whose true rank was still highly ambiguous, and
    caused a first-turn misplay. A fully strict "zero remaining ambiguity"
    test is overly paralytic (self-play mean return collapsed because almost
    nothing was ever recognized as safe to play, tokens ran out, and clued
    important cards ended up force-discarded). The probability threshold
    (tuned in self-play across A02-A03) is a deliberately explicit, documented
    risk-threshold choice, which the shared conventions explicitly leave to
    each implementation.
  - Clue-giving fully knows the partner's exact hand, so it directly
    simulates the H-group clue-focus algorithm (leftmost-new / chop-priority
    / leftmost-retouch, using the runtime mapping's "leftmost == highest
    native index") for all 10 possible clues and scores them: on-chop Save
    (5-Save / 2-Save / Critical-Save) highest priority, then partner Play
    Clues, then discards (known-trash > chop > other), then a minimal
    fallback clue for when discarding is illegal (8 info tokens). Among
    otherwise-equally-valid play-clue attributes, rank clues are preferred
    over color clues: a rank clue concentrates a receiver's possibility mass
    on the (often few) candidate colors already at that height, while a
    color clue splits mass across many ranks and is recognized far more
    slowly by any receiver using a similar elimination/threshold rule. This
    is a tactical, risk-threshold-adjacent choice the conventions leave open
    (docs: "don't be afraid to disregard [prefer color clues] ... if it gets
    additional cards or helps fill in").
  - No cross-turn private memory is needed for this design (every input to
    the decision is recomputable from the current public observation), so
    memory is a fixed-shape placeholder.
"""

import jax.numpy as jnp

from llm_policy.policy_api import HanabiObservation


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_l1_heuristic',
    'adapter': 'hanabi',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 5,
    'num_ranks': 5,
    'hand_size': 5,
    'observation_size': 658,
    'num_actions': 21,
    'policy_structure': 'shared_local',
}

_N = 5


def init_agent():
    return jnp.zeros((), dtype=jnp.int32)


def agent_act(obs: HanabiObservation, reset, memory, rng):
    del rng

    _ARANGE5 = jnp.arange(_N, dtype=jnp.int32)
    _MULT = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)

    f = obs.features

    # ---- Decode public blocks ----------------------------------------------
    partner_hand = f[0:125].reshape(5, 5, 5)            # [slot, color, rank]
    fireworks_onehot = f[167:192].reshape(5, 5)         # [color, rank]
    discards = f[203:253].reshape(5, 10)                # [color, subblocks]
    knowledge = f[308:658].reshape(2, 5, 35)            # [rel_player, slot, entry]

    height = jnp.sum(fireworks_onehot * (_ARANGE5.astype(jnp.float32) + 1.0), axis=-1)
    height_i = jnp.clip(jnp.round(height).astype(jnp.int32), 0, 5)

    dc0 = discards[:, 0:3].sum(-1)
    dc1 = discards[:, 3:5].sum(-1)
    dc2 = discards[:, 5:7].sum(-1)
    dc3 = discards[:, 7:9].sum(-1)
    dc4 = discards[:, 9:10].sum(-1)
    discard_count = jnp.stack([dc0, dc1, dc2, dc3, dc4], axis=-1)  # [color, rank]

    played_mask = _ARANGE5[None, :] < height_i[:, None]             # [color, rank]
    playable_mask = _ARANGE5[None, :] == height_i[:, None]          # [color, rank]
    remaining_in_play = (_MULT[None, :] - discard_count
                          - played_mask.astype(jnp.float32))        # [color, rank]

    # ---- Own hand -----------------------------------------------------------
    own_knowledge = knowledge[0]
    own_V0 = own_knowledge[:, 0:25].reshape(5, 5, 5)    # [slot, color, rank]
    own_color_hint = own_knowledge[:, 25:30]
    own_rank_hint = own_knowledge[:, 30:35]
    own_clued = (own_color_hint.sum(-1) + own_rank_hint.sum(-1)) > 0.5
    own_occupied = own_V0.sum(axis=(1, 2)) > 1e-6
    own_support = own_V0 > 1e-6

    own_chop_idx = jnp.min(jnp.where(own_occupied & (~own_clued), _ARANGE5, 5))
    own_chop_exists = own_chop_idx <= 4
    own_is_chop = (_ARANGE5 == own_chop_idx) & own_chop_exists

    own_known_dead = (own_occupied
                       & (~jnp.any(own_support & (~played_mask)[None], axis=(1, 2)))
                       & jnp.any(own_support, axis=(1, 2)))

    # Probability-mass play test: own_V0 is a normalized literal-possibility
    # distribution over this slot's identity (given all positive/negative
    # hints so far). Play once enough of that mass sits on currently-playable
    # identities. A pure "is it merely possible" test is unsound (it fired on
    # a freshly color-clued card at game start, whose true rank was almost
    # certainly still ambiguous, and caused a first-turn misplay); a fully
    # strict "zero remaining ambiguity" test is overly paralytic in
    # self-play testing. The probability threshold is a documented
    # implementation-specific risk-threshold choice.
    mass_playable = jnp.sum(own_V0 * playable_mask[None].astype(jnp.float32), axis=(1, 2))
    playable_confidence = own_occupied & (mass_playable >= 0.75)

    # ---- Partner hand (exactly known identities) -----------------------------
    partner_occupied = partner_hand.sum(axis=(1, 2)) > 0.5
    partner_color = jnp.argmax(partner_hand.sum(-1), axis=-1)
    partner_rank = jnp.argmax(partner_hand.sum(-2), axis=-1)

    partner_knowledge = knowledge[1]
    partner_color_hint = partner_knowledge[:, 25:30]
    partner_rank_hint = partner_knowledge[:, 30:35]
    partner_clued = (partner_color_hint.sum(-1) + partner_rank_hint.sum(-1)) > 0.5

    partner_chop_idx = jnp.min(jnp.where(partner_occupied & (~partner_clued), _ARANGE5, 5))
    partner_chop_exists = partner_chop_idx <= 4

    partner_height_at = height_i[partner_color]
    partner_playable = partner_occupied & (partner_rank == partner_height_at)
    partner_already_played = partner_occupied & (partner_rank < partner_height_at)
    partner_is5 = partner_occupied & (partner_rank == 4)

    same_identity = ((partner_color[:, None] == partner_color[None, :])
                      & (partner_rank[:, None] == partner_rank[None, :])
                      & partner_occupied[:, None] & partner_occupied[None, :])
    not_self = ~jnp.eye(5, dtype=jnp.bool_)
    dup_count = jnp.sum(same_identity & not_self, axis=-1)
    partner_unique2 = partner_occupied & (partner_rank == 1) & (dup_count == 0)

    rip_per_slot = remaining_in_play[partner_color, partner_rank]
    partner_critical = partner_occupied & (rip_per_slot <= 1.0) & (~partner_already_played)

    # =========================================================================
    # Score all 21 actions.
    # =========================================================================
    play_score = jnp.where(
        own_occupied,
        jnp.where(playable_confidence, 1500.0 + _ARANGE5.astype(jnp.float32) * 0.01, -1e6),
        -1e9)

    # When forced to discard a clued (non-chop, non-dead) card -- i.e. the
    # hand is fully clued -- prefer the one least likely to be near-term
    # useful (lower play-confidence mass), rather than an arbitrary one.
    other_clued_score = 50.0 * (1.0 - jnp.clip(mass_playable, 0.0, 1.0))
    discard_score = jnp.where(
        own_occupied,
        jnp.where(own_known_dead, 700.0,
                  jnp.where(own_is_chop, 200.0,
                            jnp.where(~own_clued, 150.0, other_clued_score))),
        -1e9)

    is_color_arr = jnp.concatenate([jnp.ones(5, jnp.bool_), jnp.zeros(5, jnp.bool_)])
    attr_idx_arr = jnp.concatenate([_ARANGE5, _ARANGE5])

    touched_all = jnp.where(
        is_color_arr[:, None],
        partner_color[None, :] == attr_idx_arr[:, None],
        partner_rank[None, :] == attr_idx_arr[:, None]) & partner_occupied[None, :]
    new_touched_all = touched_all & (~partner_clued)[None, :]
    has_new_all = jnp.any(new_touched_all, axis=-1)

    chop_touched_all = partner_chop_exists & jnp.take(
        touched_all, jnp.clip(partner_chop_idx, 0, 4), axis=1)
    idx_new = jnp.where(new_touched_all, _ARANGE5[None, :], -1)
    focus_via_new_all = jnp.max(idx_new, axis=-1)
    idx_re = jnp.where(touched_all, _ARANGE5[None, :], -1)
    focus_via_retouch_all = jnp.max(idx_re, axis=-1)
    focus_all = jnp.where(has_new_all,
                           jnp.where(chop_touched_all, partner_chop_idx, focus_via_new_all),
                           focus_via_retouch_all)
    focus_all_clip = jnp.clip(focus_all, 0, 4)
    focus_is_chop_all = partner_chop_exists & (focus_all == partner_chop_idx)

    focus_playable_all = jnp.take(partner_playable, focus_all_clip)
    focus_clued_all = jnp.take(partner_clued, focus_all_clip)
    focus_is5_all = jnp.take(partner_is5, focus_all_clip)
    focus_unique2_all = jnp.take(partner_unique2, focus_all_clip)
    focus_critical_all = jnp.take(partner_critical, focus_all_clip)

    target_is_play_all = (has_new_all & focus_playable_all & (~focus_clued_all)
                           & (~(focus_is_chop_all & (focus_is5_all | focus_unique2_all))))

    rank_is_2save_give = (~is_color_arr) & (attr_idx_arr == 1)
    rank_is_5save_give = (~is_color_arr) & (attr_idx_arr == 4)
    target_is_5save_all = rank_is_5save_give & focus_is_chop_all & focus_is5_all
    target_is_2save_all = rank_is_2save_give & focus_is_chop_all & focus_unique2_all
    target_is_criticalsave_all = (focus_is_chop_all & focus_critical_all
                                   & (~focus_is5_all) & (~focus_unique2_all)
                                   & (~rank_is_2save_give) & (~rank_is_5save_give))

    non_focus_touched_all = touched_all & (_ARANGE5[None, :] != focus_all[:, None])
    bad_touch_all = jnp.sum(non_focus_touched_all & partner_already_played[None, :], axis=-1)
    bad_touch_all = bad_touch_all.astype(jnp.float32)
    # Prefer rank clues for play targets: with a probability-mass receiver
    # test, a rank clue concentrates mass on colors already at that height
    # (often unambiguous very early), while a color clue splits mass across
    # many ranks and is recognized far more slowly by a matching receiver.
    color_bonus = jnp.where(is_color_arr, 0.0, 1.0)

    play_value_all = jnp.where(target_is_play_all,
                                1000.0 - bad_touch_all * 50.0 + color_bonus, -1e6)
    save_needed_all = target_is_5save_all | target_is_2save_all | target_is_criticalsave_all
    save_value_all = jnp.where(save_needed_all,
                                2000.0 - bad_touch_all * 50.0 + color_bonus, -1e6)
    fallback_value_all = jnp.where(has_new_all, -100.0 - bad_touch_all * 10.0, -300.0)

    clue_value_all = jnp.maximum(jnp.maximum(play_value_all, save_value_all), fallback_value_all)

    scores = jnp.full((21,), -jnp.inf, dtype=jnp.float32)
    scores = scores.at[0:5].set(discard_score)
    scores = scores.at[5:10].set(play_score)
    scores = scores.at[10:20].set(clue_value_all)
    scores = scores.at[20].set(-1e9)

    final_scores = jnp.where(obs.legal_actions, scores, -jnp.inf)
    action = jnp.argmax(final_scores).astype(jnp.int32)

    next_memory = memory

    lives = f[200:203]
    info_tokens = f[192:200]
    category = jnp.where(action <= 4, 0, jnp.where(action <= 9, 1,
                          jnp.where(action <= 19, 2, 3))).astype(jnp.int32)
    trace = {
        'lives': jnp.sum(lives).astype(jnp.float32),
        'tokens': jnp.sum(info_tokens).astype(jnp.float32),
        'category': category,
        'play_confident': playable_confidence.astype(jnp.float32),
        'reset_flag': jnp.asarray(reset, dtype=jnp.float32),
    }

    return action, next_memory, trace
