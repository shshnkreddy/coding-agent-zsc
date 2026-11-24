"""Convention-light Hanabi policy.

Every decision is derived from literal public information only (the
env-provided V0 possibility block, firework heights, and discard counts), so
it does not depend on a private signalling scheme shared with a specific
partner implementation. All action utilities below are combined into one
21-length vector, masked by the legal-action mask, and argmax-selected --
there is no data-dependent Python branching.

- Play a card only when every color/rank possibility still consistent with
  public hint history and remaining-card counts is currently playable
  (i.e. provably safe). Never gamble on an uncertain play, so a play chosen
  by this policy can never fail.
- Discard a card only when every remaining possibility for it is already
  below its color's firework height, or belongs to a color that is
  permanently stuck (its next-needed rank has zero copies left anywhere,
  deck or hands, so that color can never progress further) -- both are
  provably useless already, not merely probably so.
- Otherwise, prefer giving a hint (color or rank) that either fully resolves
  a partner slot to "certainly playable"/"certainly useless", or makes
  literal progress toward a partner slot that is *actually* playable right
  now (narrows its possibility set without yet reaching full certainty).
  Both computations replay the exact literal hint-update rule from the
  environment spec on the partner's real (observed) hand, so they only rely
  on a partner tracking public information correctly, not on any bespoke
  convention.
- If no hint helps, fall back to discarding the "chop": the own slot with
  the fewest received hints and lowest estimated risk of being a
  still-needed, hard-to-replace copy.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
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

EPS = 1e-6
NUM_COLORS = 5
NUM_RANKS = 5
NUM_SLOTS = 5

_MULT_BY_RANK_LIST = [3.0, 2.0, 2.0, 2.0, 1.0]


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _slot_all_true(support, indicator):
    """support: [.., C, R] bool, indicator: [C, R] bool -> [..] bool."""
    return jnp.all(jnp.where(support, indicator, True), axis=(-2, -1))


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    _mult_by_rank = jnp.array(_MULT_BY_RANK_LIST, dtype=jnp.float32)
    _rank_ids = jnp.arange(NUM_RANKS, dtype=jnp.float32)

    partner_hand = features[0:125].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    fireworks = features[167:192].reshape(NUM_COLORS, NUM_RANKS)
    info_tokens = jnp.sum(features[192:200])
    discards = features[203:253].reshape(NUM_COLORS, 10)
    native = features[308:658].reshape(2, NUM_SLOTS, 35)

    height = jnp.sum(fireworks * (_rank_ids + 1.0)[None, :], axis=1)  # [C]

    discarded_count = jnp.stack([
        discards[:, 0:3].sum(axis=1),
        discards[:, 3:5].sum(axis=1),
        discards[:, 5:7].sum(axis=1),
        discards[:, 7:9].sum(axis=1),
        discards[:, 9:10].sum(axis=1),
    ], axis=1)  # [C, R]

    played_flag = (_rank_ids[None, :] < height[:, None]).astype(jnp.float32)
    mult = jnp.broadcast_to(_mult_by_rank[None, :], (NUM_COLORS, NUM_RANKS))
    remaining_count = mult - discarded_count - played_flag

    playable_indicator = (_rank_ids[None, :] == height[:, None]) & (height[:, None] < 4.5)
    trash_indicator = _rank_ids[None, :] < height[:, None]

    # A color is permanently stuck once its next-needed rank has no copies
    # left anywhere (deck or either hand); every remaining card of that color
    # is then provably trash, not just the already-played ranks below it.
    remaining_at_needed = jnp.sum(remaining_count * playable_indicator.astype(jnp.float32), axis=1)
    dead_color = (height < 4.5) & (remaining_at_needed <= 0.5)
    trash_indicator = trash_indicator | dead_color[:, None]

    # --- own hand (relative player 0) ---
    self_native = native[0]
    self_v0 = self_native[:, 0:25].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    self_color_flags = self_native[:, 25:30]
    self_rank_flags = self_native[:, 30:35]
    support_self = self_v0 > EPS
    any_support_self = jnp.any(support_self, axis=(1, 2))

    known_playable = _slot_all_true(support_self, playable_indicator) & any_support_self
    known_trash = _slot_all_true(support_self, trash_indicator) & any_support_self

    still_needed_unique = (
        (remaining_count <= 1.0) & (_rank_ids[None, :] >= height[:, None]) & (~dead_color[:, None]))
    risk = jnp.sum(self_v0 * still_needed_unique.astype(jnp.float32), axis=(1, 2))
    clue_count = jnp.sum(self_color_flags, axis=1) + jnp.sum(self_rank_flags, axis=1)

    slot_idx = jnp.arange(NUM_SLOTS, dtype=jnp.float32)

    play_utility = jnp.where(known_playable, 10000.0, -1e9)

    discard_trash_utility = 8000.0 - slot_idx * 0.001
    discard_fallback_utility = 1000.0 - risk * 25.0 - clue_count * 10.0 + slot_idx * 0.01
    discard_utility = jnp.where(known_trash, discard_trash_utility, discard_fallback_utility)

    # --- partner hand (relative player 1) ---
    partner_native = native[1]
    partner_v0 = partner_native[:, 0:25].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    support_partner = partner_v0 > EPS
    occupied_partner = jnp.any(partner_hand > EPS, axis=(1, 2))

    partner_known_playable = _slot_all_true(support_partner, playable_indicator) & jnp.any(
        support_partner, axis=(1, 2))
    partner_known_trash = _slot_all_true(support_partner, trash_indicator) & jnp.any(
        support_partner, axis=(1, 2))

    color_onehot = partner_hand.sum(axis=2)  # [slot, color]
    rank_onehot = partner_hand.sum(axis=1)   # [slot, rank]

    old_support_size = jnp.sum(support_partner.astype(jnp.float32), axis=(1, 2))
    real_playable_partner = jnp.sum(
        partner_hand * playable_indicator[None, :, :].astype(jnp.float32), axis=(1, 2)) > 0.5

    def hint_effect(match_mask, dim_eq_mask):
        new_support = jnp.where(
            match_mask[:, None, None],
            support_partner & dim_eq_mask[None, :, :],
            support_partner & (~dim_eq_mask)[None, :, :],
        )
        any_new = jnp.any(new_support, axis=(1, 2))
        all_playable = _slot_all_true(new_support, playable_indicator)
        all_trash = _slot_all_true(new_support, trash_indicator)
        newly_playable = all_playable & any_new & occupied_partner & (~partner_known_playable)
        newly_trash = all_trash & any_new & occupied_partner & (~partner_known_trash)
        new_size = jnp.sum(new_support.astype(jnp.float32), axis=(1, 2))
        contributed = (old_support_size - new_size) > 0.5
        partial_playable = (
            match_mask & real_playable_partner & (~partner_known_playable)
            & contributed & occupied_partner)
        gain = jnp.sum(jnp.where(occupied_partner, old_support_size - new_size, 0.0))
        return (jnp.sum(newly_playable.astype(jnp.float32)),
                jnp.sum(newly_trash.astype(jnp.float32)),
                jnp.sum(partial_playable.astype(jnp.float32)),
                gain)

    def hint_color_effect(c):
        color_eq_c = jnp.broadcast_to(
            (jnp.arange(NUM_COLORS)[:, None] == c), (NUM_COLORS, NUM_RANKS))
        matches = color_onehot[:, c] > 0.5
        return hint_effect(matches, color_eq_c)

    def hint_rank_effect(r):
        rank_eq_r = jnp.broadcast_to(
            (jnp.arange(NUM_RANKS)[None, :] == r), (NUM_COLORS, NUM_RANKS))
        matches = rank_onehot[:, r] > 0.5
        return hint_effect(matches, rank_eq_r)

    color_results = [hint_color_effect(c) for c in range(NUM_COLORS)]
    rank_results = [hint_rank_effect(r) for r in range(NUM_RANKS)]

    count_playable_color = jnp.stack([res[0] for res in color_results])
    count_trash_color = jnp.stack([res[1] for res in color_results])
    count_partial_color = jnp.stack([res[2] for res in color_results])
    gain_color = jnp.stack([res[3] for res in color_results])

    count_playable_rank = jnp.stack([res[0] for res in rank_results])
    count_trash_rank = jnp.stack([res[1] for res in rank_results])
    count_partial_rank = jnp.stack([res[2] for res in rank_results])
    gain_rank = jnp.stack([res[3] for res in rank_results])

    def hint_util(count_playable, count_trash, count_partial, gain):
        return jnp.where(
            count_playable > 0.0, 9000.0 + 10.0 * count_playable,
            jnp.where(
                count_partial > 0.0, 8900.0 + 30.0 * count_partial + gain,
                jnp.where(count_trash > 0.0, 1800.0 + 10.0 * count_trash, 200.0 + 0.1 * gain)))

    hint_color_utility = hint_util(
        count_playable_color, count_trash_color, count_partial_color, gain_color)
    hint_rank_utility = hint_util(
        count_playable_rank, count_trash_rank, count_partial_rank, gain_rank)

    utilities = jnp.concatenate([
        discard_utility,
        play_utility,
        hint_color_utility,
        hint_rank_utility,
        jnp.asarray([1e6], dtype=jnp.float32),  # no-op; only ever legal while inactive
    ])

    final = jnp.where(obs.legal_actions, utilities, -1e9)
    action = jnp.argmax(final).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    return action, next_memory, {}
