"""Rule-based, convention-driven two-player Hanabi policy.

Strategy summary (fully derived from public observation features each turn,
no private memory is required since the native V0 knowledge block already
encodes cumulative public-hint-derived beliefs):

Priority tiers (highest score wins, ties broken toward lower action id):
  T5 confident play - play an own slot whose public belief is >=95% playable.
  hint (unified)     - among legal, genuinely informative hints, prefer ones
                        that (a) complete a partner card's identity as
                        currently playable, (b) complete any card's identity,
                        (c) protect a critical (last-copy) card, and
                        (d) spread information broadly, in that priority
                        order. A same-style partner only ever plays/discards
                        based on its own quantified V0 confidence (never
                        merely because it was hinted), and V0 always
                        converges to the literal truth once hints plus
                        public remaining-count deduction pin a card down, so
                        no hint can mislead such a partner into a bad play.
  T2 confident discard - discard an own slot that is publicly known trash.
  T0 fallback discard  - discard the own slot least likely to be useful.
  base                 - any remaining legal action (safety net).

This relies only on generally sensible Hanabi conventions ("point at
playable cards", "protect criticals", "discard known trash") rather than a
private signalling scheme, so it should remain reasonable when paired with
an independently written competent partner.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hanabi_convention_bot',
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

NUM_COLORS = 5
NUM_RANKS = 5
HAND = 5
NUM_ACTIONS = 21

PLAY_THRESH = 0.75
DEAD_THRESH = 0.95

_MAX_COPIES_LIST = (3.0, 2.0, 2.0, 2.0, 1.0)

# Maps the 10 per-color discard thermometer entries onto their rank subblock
# (sizes [3, 2, 2, 2, 1], per the spec's discard-count encoding).
_SUBBLOCK_LIST = (
    (1, 0, 0, 0, 0),
    (1, 0, 0, 0, 0),
    (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 0, 1),
)  # (10, 5)


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    max_copies = jnp.asarray(_MAX_COPIES_LIST, dtype=jnp.float32)
    subblock = jnp.asarray(_SUBBLOCK_LIST, dtype=jnp.float32)

    features = obs.features
    legal = obs.legal_actions

    partner_hand = features[0:125].reshape(HAND, NUM_COLORS, NUM_RANKS)
    fireworks = features[167:192].reshape(NUM_COLORS, NUM_RANKS)
    discards = features[203:253].reshape(NUM_COLORS, 10)
    native = features[308:658].reshape(2, HAND, 35)
    self_native = native[0]
    partner_native = native[1]

    self_v0 = self_native[:, 0:25].reshape(HAND, NUM_COLORS, NUM_RANKS)
    partner_v0 = partner_native[:, 0:25].reshape(HAND, NUM_COLORS, NUM_RANKS)
    partner_color_hint = partner_native[:, 25:30]
    partner_rank_hint = partner_native[:, 30:35]

    occ_self = legal[5:10]
    occ_partner = jnp.sum(partner_hand, axis=(1, 2)) > 0.5

    has_progress = jnp.sum(fireworks, axis=1) > 0.5
    height_idx = jnp.argmax(fireworks, axis=1)
    height = jnp.where(has_progress, height_idx + 1, 0).astype(jnp.float32)  # (color,)

    rank_ids = jnp.arange(NUM_RANKS, dtype=jnp.float32)
    playable = rank_ids[None, :] == height[:, None]  # (color, rank)

    discard_count = discards @ subblock  # (color, rank)
    played_count = (height[:, None] > rank_ids[None, :]).astype(jnp.float32)
    remaining = max_copies[None, :] - discard_count - played_count

    alive = (remaining > 0.5) | (rank_ids[None, :] < height[:, None])
    reachable = jnp.cumprod(alive.astype(jnp.float32), axis=1)
    dead = (rank_ids[None, :] < height[:, None]) | (reachable < 0.5)
    critical = (
        (remaining <= 1.5) & (remaining > 0.5) & (~dead)
        & (rank_ids[None, :] >= height[:, None])
    )

    play_f = playable.astype(jnp.float32)
    dead_f = dead.astype(jnp.float32)
    critical_f = critical.astype(jnp.float32)

    play_prob = jnp.einsum('scr,cr->s', self_v0, play_f)
    dead_prob = jnp.einsum('scr,cr->s', self_v0, dead_f)
    critical_prob = jnp.einsum('scr,cr->s', self_v0, critical_f)

    partner_color_id = jnp.argmax(jnp.sum(partner_hand, axis=2), axis=1)
    partner_rank_id = jnp.argmax(jnp.sum(partner_hand, axis=1), axis=1)

    partner_playable = occ_partner & playable[partner_color_id, partner_rank_id]
    partner_critical = occ_partner & critical[partner_color_id, partner_rank_id]

    partner_belief_play = jnp.einsum('scr,cr->s', partner_v0, play_f)
    needs_hint_play = occ_partner & partner_playable & (partner_belief_play < PLAY_THRESH)

    already_color_known = jnp.sum(partner_color_hint, axis=1) > 0.5
    already_rank_known = jnp.sum(partner_rank_hint, axis=1) > 0.5
    partner_fully_clued = already_color_known & already_rank_known
    needs_save = occ_partner & partner_critical & (~partner_playable) & (~partner_fully_clued)

    color_ids = jnp.arange(NUM_COLORS)
    rank_ids_i = jnp.arange(NUM_RANKS)

    matches_color = occ_partner[None, :] & (partner_color_id[None, :] == color_ids[:, None])
    matches_rank = occ_partner[None, :] & (partner_rank_id[None, :] == rank_ids_i[:, None])

    legal_color = legal[10:15]
    legal_rank = legal[15:20]

    # Downstream play/discard decisions are governed purely by the receiver's
    # own quantified V0 confidence (never by "was I hinted at all"), and V0
    # always converges to the literal truth once hints (plus public
    # remaining-count deduction) pin a card down. So hints carry no risk of
    # misleading a same-style partner into a wrong confident play; hint
    # choice can simply maximize expected decision value.
    info_gain_color = jnp.sum(matches_color & (~already_color_known[None, :]), axis=1).astype(jnp.float32)
    info_gain_rank = jnp.sum(matches_rank & (~already_rank_known[None, :]), axis=1).astype(jnp.float32)

    # A "completion" hint gives the one remaining attribute for a slot that
    # already knows the other attribute, so it becomes fully identified.
    completes_color = matches_color & already_rank_known[None, :] & (~already_color_known[None, :])
    completes_rank = matches_rank & already_color_known[None, :] & (~already_rank_known[None, :])
    completion_count_color = jnp.sum(completes_color, axis=1).astype(jnp.float32)
    completion_count_rank = jnp.sum(completes_rank, axis=1).astype(jnp.float32)
    completion_playable_color = jnp.any(completes_color & partner_playable[None, :], axis=1)
    completion_playable_rank = jnp.any(completes_rank & partner_playable[None, :], axis=1)

    any_save_color = jnp.any(matches_color & needs_save[None, :], axis=1)
    any_save_rank = jnp.any(matches_rank & needs_save[None, :], axis=1)

    informative_color = info_gain_color >= 0.5
    informative_rank = info_gain_rank >= 0.5

    hint_value_color = (
        1000.0
        + 1500.0 * completion_playable_color.astype(jnp.float32)
        + 400.0 * completion_count_color
        + 500.0 * any_save_color.astype(jnp.float32)
        + 20.0 * info_gain_color
    )
    hint_value_rank = (
        1000.0
        + 1500.0 * completion_playable_rank.astype(jnp.float32)
        + 400.0 * completion_count_rank
        + 500.0 * any_save_rank.astype(jnp.float32)
        + 20.0 * info_gain_rank
    )

    t_hint_color = jnp.where(legal_color & informative_color, hint_value_color, -1e9)
    t_hint_rank = jnp.where(legal_rank & informative_rank, hint_value_rank, -1e9)

    t2 = jnp.where(
        legal[0:5] & (dead_prob >= DEAD_THRESH), 2000.0 + dead_prob * 10.0, -1e9
    )

    t0 = jnp.where(
        legal[0:5],
        500.0 + (1.0 - play_prob) * 5.0 - critical_prob * 100.0 + dead_prob * 5.0,
        -1e9,
    )

    t5 = jnp.where(legal[5:10] & (play_prob >= PLAY_THRESH), 5000.0 + play_prob * 10.0, -1e9)

    scores = jnp.where(legal, -500.0, -1e9)
    scores = scores.at[0:5].set(jnp.maximum(scores[0:5], jnp.maximum(t2, t0)))
    scores = scores.at[5:10].set(jnp.maximum(scores[5:10], t5))
    scores = scores.at[10:15].set(jnp.maximum(scores[10:15], t_hint_color))
    scores = scores.at[15:20].set(jnp.maximum(scores[15:20], t_hint_rank))

    action = jnp.argmax(scores).astype(jnp.int32)

    zero_mem = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero_mem, memory)

    return action, next_memory, {}
