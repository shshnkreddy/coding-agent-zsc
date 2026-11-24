"""Convention-shaped Hanabi policy for the local two-player task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h_group_l1_conservative', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NUM_SLOTS = 5
NUM_CARDS = 25
CARD_MULTIPLICITIES = (3., 2., 2., 2., 1.)
CARD_COLOR_IDS = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4)
CARD_RANK_IDS = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4)


def _card_colors():
    return jnp.asarray(CARD_COLOR_IDS, dtype=jnp.int32)


def _card_ranks():
    return jnp.asarray(CARD_RANK_IDS, dtype=jnp.int32)


def init_agent():
    return {
        "play": jnp.zeros((NUM_SLOTS,), dtype=jnp.bool_),
        "saved": jnp.zeros((NUM_SLOTS,), dtype=jnp.bool_),
        "touched": jnp.zeros((NUM_SLOTS,), dtype=jnp.bool_),
        "poss": jnp.zeros((NUM_SLOTS, NUM_CARDS), dtype=jnp.bool_),
    }


def _zero_memory():
    return init_agent()


def _lowest(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _highest(mask):
    return (NUM_SLOTS - 1 - jnp.argmax(mask[::-1].astype(jnp.int32))).astype(jnp.int32)


def _decode(obs):
    f = obs.features
    partner = f[0:125].reshape((5, 5, 5))
    fireworks = f[167:192].reshape((5, 5))
    lives = jnp.sum(f[200:203])
    discards = f[203:253].reshape((5, 10))
    last = f[253:308]
    knowledge = f[308:658].reshape((2, 5, 35))
    return partner, fireworks, lives, discards, last, knowledge


def _heights(fireworks):
    rank_values = jnp.asarray([1., 2., 3., 4., 5.], dtype=jnp.float32)
    return jnp.sum(fireworks * rank_values[None, :], axis=1).astype(jnp.int32)


def _playable25(heights):
    colors = _card_colors()
    ranks = _card_ranks()
    return (ranks == heights[colors]) & (heights[colors] < 5)


def _trash25(heights):
    colors = _card_colors()
    ranks = _card_ranks()
    return ranks < heights[colors]


def _one_away25(heights):
    colors = _card_colors()
    ranks = _card_ranks()
    return (ranks == (heights[colors] + 1)) & (heights[colors] < 4)


def _discard_counts(discards):
    parts = [
        jnp.sum(discards[:, 0:3], axis=1),
        jnp.sum(discards[:, 3:5], axis=1),
        jnp.sum(discards[:, 5:7], axis=1),
        jnp.sum(discards[:, 7:9], axis=1),
        discards[:, 9],
    ]
    return jnp.stack(parts, axis=1)


def _occupied_from_v0(v0):
    return jnp.sum(v0, axis=1) > 0.01


def _literal_touched(knowledge_player):
    flags = knowledge_player[:, 25:35]
    return jnp.sum(flags, axis=1) > 0.5


def _shift_slots_array(arr, removed):
    variants = []
    for s in range(NUM_SLOTS):
        rows = []
        for i in range(NUM_SLOTS):
            source = jnp.minimum(i + (i >= s), NUM_SLOTS - 1)
            value = jnp.where(i == NUM_SLOTS - 1, jnp.zeros_like(arr[0]), arr[source])
            rows.append(value)
        variants.append(jnp.stack(rows, axis=0))
    return jnp.take(jnp.stack(variants, axis=0), removed, axis=0)


def _shift_slots_vector(vec, removed):
    variants = []
    for s in range(NUM_SLOTS):
        rows = []
        for i in range(NUM_SLOTS):
            source = jnp.minimum(i + (i >= s), NUM_SLOTS - 1)
            value = jnp.where(i == NUM_SLOTS - 1, jnp.zeros_like(vec[0]), vec[source])
            rows.append(value)
        variants.append(jnp.stack(rows, axis=0))
    return jnp.take(jnp.stack(variants, axis=0), removed, axis=0)


def _focus(matches, touched, occupied):
    new = matches & (~touched) & occupied
    any_new = jnp.any(new)
    chop_candidates = occupied & (~touched)
    any_chop = jnp.any(chop_candidates)
    chop = _lowest(chop_candidates)
    retouched = matches & touched & occupied
    focus_nonew = _highest(retouched)
    focus_new = _highest(new)
    focus = jnp.where(any_new & any_chop & matches[chop], chop, focus_new)
    return jnp.where(any_new, focus, focus_nonew)


def _critical25(heights, discards):
    discarded = _discard_counts(discards).reshape((NUM_CARDS,))
    mult = jnp.asarray(CARD_MULTIPLICITIES, dtype=jnp.float32)[_card_ranks()]
    return (discarded >= (mult - 1.0)) & (~_trash25(heights))


def _apply_memory_update(memory, reset, last, knowledge, playable25, one_away25, critical25):
    zero = _zero_memory()
    mem = jax.tree_util.tree_map(lambda z, m: jnp.where(reset, z, m), zero, memory)

    actor_self = last[0] > 0.5
    actor_partner = last[1] > 0.5
    was_play_or_discard = (last[2] > 0.5) | (last[3] > 0.5)
    removed = jnp.argmax(last[23:28]).astype(jnp.int32)
    do_shift = actor_self & was_play_or_discard & (~reset)

    shifted = {
        "play": _shift_slots_vector(mem["play"], removed),
        "saved": _shift_slots_vector(mem["saved"], removed),
        "touched": _shift_slots_vector(mem["touched"], removed),
        "poss": _shift_slots_array(mem["poss"], removed),
    }
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(do_shift, a, b), shifted, mem)

    is_color_hint = last[4] > 0.5
    is_rank_hint = last[5] > 0.5
    hint_to_self = actor_partner & (is_color_hint | is_rank_hint) & (last[6] > 0.5) & (~reset)
    matches = last[18:23] > 0.5
    own_v0 = knowledge[0, :, 0:25]
    occupied = _occupied_from_v0(own_v0)
    focus = _focus(matches, mem["touched"], occupied)
    focus_oh = jax.nn.one_hot(focus, NUM_SLOTS).astype(jnp.bool_)

    color = jnp.argmax(last[8:13]).astype(jnp.int32)
    rank = jnp.argmax(last[13:18]).astype(jnp.int32)
    clue_mask = jnp.where(is_color_hint, _card_colors() == color, _card_ranks() == rank)
    direct_poss = clue_mask & playable25
    has_direct = jnp.any(direct_poss)
    delayed_poss = clue_mask & one_away25
    has_delayed = jnp.any(delayed_poss)

    chop_candidates = occupied & (~mem["touched"])
    any_chop = jnp.any(chop_candidates)
    chop = _lowest(chop_candidates)
    focus_is_chop = any_chop & (focus == chop)
    rank_save = is_rank_hint & ((rank == 1) | (rank == 4)) & focus_is_chop
    critical_save = focus_is_chop & jnp.any(clue_mask & critical25)
    save_possible = rank_save | critical_save
    direct_play = hint_to_self & has_direct & (~save_possible)
    delayed_play = hint_to_self & (~has_direct) & has_delayed & (~save_possible) & (~focus_is_chop)
    save = hint_to_self & (save_possible | (focus_is_chop & (~has_direct)))

    mem = {
        "play": jnp.where(focus_oh & (direct_play | delayed_play), True, mem["play"]),
        "saved": jnp.where(focus_oh & save, True, mem["saved"]),
        "touched": jnp.where(hint_to_self, mem["touched"] | matches, mem["touched"]),
        "poss": jnp.where(
            (focus_oh[:, None] & (direct_play | delayed_play)),
            jnp.where(direct_play, direct_poss, delayed_poss)[None, :],
            mem["poss"],
        ),
    }
    return mem


def _partner_cards(partner):
    flat = partner.reshape((NUM_SLOTS, NUM_CARDS))
    occupied = jnp.sum(flat, axis=1) > 0.5
    idx = jnp.argmax(flat, axis=1)
    colors = (idx // 5).astype(jnp.int32)
    ranks = (idx % 5).astype(jnp.int32)
    return occupied, colors, ranks


def _clue_choice(legal, partner, partner_touched, heights, discards):
    occupied, colors, ranks = _partner_cards(partner)
    playable25 = _playable25(heights)
    trash25 = _trash25(heights)
    discard_count = _discard_counts(discards)
    rank_mult = jnp.asarray(CARD_MULTIPLICITIES, dtype=jnp.float32)[ranks]
    focus_discarded = discard_count[colors, ranks]
    critical = (focus_discarded >= (rank_mult - 1.0)) & (~trash25[colors * 5 + ranks])

    best_score = jnp.asarray(-1.0e6, dtype=jnp.float32)
    best_action = jnp.asarray(20, dtype=jnp.int32)

    for action in range(10, 20):
        is_color = action < 15
        value = action - 10 if is_color else action - 15
        matches = jnp.where(is_color, colors == value, ranks == value) & occupied
        new = matches & (~partner_touched)
        any_new = jnp.any(new)
        any_retouch = jnp.any(matches & partner_touched)
        focus = _focus(matches, partner_touched, occupied)
        focus_card = colors[focus] * 5 + ranks[focus]
        focus_playable = playable25[focus_card] & occupied[focus]
        focus_chop_candidates = occupied & (~partner_touched)
        any_chop = jnp.any(focus_chop_candidates)
        chop = _lowest(focus_chop_candidates)
        focus_is_chop = any_chop & (focus == chop)

        touched_trash = jnp.any(matches & trash25[colors * 5 + ranks])
        rank_save_action = (~is_color) & ((value == 1) | (value == 4)) & focus_is_chop
        ambiguous_save_rank = rank_save_action & focus_playable
        play_valid = focus_playable & (any_new | any_retouch) & (~touched_trash) & (~ambiguous_save_rank)

        same_identity = occupied & (colors == colors[focus]) & (ranks == ranks[focus])
        duplicate_visible = jnp.sum(same_identity.astype(jnp.int32)) > 1
        save_rank = (~is_color) & ((ranks[focus] == 4) | ((ranks[focus] == 1) & (~duplicate_visible)))
        save_valid = focus_is_chop & any_new & (~focus_playable) & (save_rank | critical[focus])
        save_valid = save_valid & (~touched_trash)

        match_count = jnp.sum(matches.astype(jnp.float32))
        new_count = jnp.sum(new.astype(jnp.float32))
        play_score = 120.0 + focus.astype(jnp.float32) * 0.2 + new_count - match_count * 0.25
        save_score = 95.0 + (ranks[focus] == 4).astype(jnp.float32) * 10.0 + critical[focus].astype(jnp.float32) * 5.0
        score = jnp.where(play_valid, play_score, save_score)
        valid = legal[action] & (play_valid | save_valid)
        better = valid & (score > best_score)
        best_score = jnp.where(better, score, best_score)
        best_action = jnp.where(better, action, best_action)

    has_choice = best_score > -1.0e5
    return best_action, has_choice


def _choose_discard(legal, occupied, touched, protected, v0, trash25):
    trash_prob = jnp.sum(v0 * trash25[None, :].astype(jnp.float32), axis=1)
    known_trash = occupied & (trash_prob > 0.999)
    legal_discards = legal[0:5]
    trash_discards = known_trash & legal_discards
    chop_candidates = occupied & (~touched) & (~protected) & legal_discards
    fallback_discards = occupied & (~protected) & legal_discards
    has_trash = jnp.any(trash_discards)
    has_chop = jnp.any(chop_candidates)
    has_fallback = jnp.any(fallback_discards)
    slot = jnp.where(has_trash, _lowest(trash_discards), jnp.where(has_chop, _lowest(chop_candidates), _lowest(fallback_discards)))
    return slot.astype(jnp.int32), (has_trash | has_chop | has_fallback)


def _first_legal(legal):
    return jnp.argmax(legal.astype(jnp.int32)).astype(jnp.int32)


def _first_legal_hint_or_action(legal):
    hints = legal[10:20]
    has_hint = jnp.any(hints)
    return jnp.where(has_hint, 10 + jnp.argmax(hints.astype(jnp.int32)), _first_legal(legal)).astype(jnp.int32)


def _safe_forced_hint(legal, partner, partner_touched, heights, discards):
    occupied, colors, ranks = _partner_cards(partner)
    playable25 = _playable25(heights)
    critical25 = _critical25(heights, discards)
    best_score = jnp.asarray(-1.0e6, dtype=jnp.float32)
    best_action = _first_legal_hint_or_action(legal)

    for action in range(10, 20):
        is_color = action < 15
        value = action - 10 if is_color else action - 15
        matches = jnp.where(is_color, colors == value, ranks == value) & occupied
        any_match = jnp.any(matches)
        focus = _focus(matches, partner_touched, occupied)
        focus_chop_candidates = occupied & (~partner_touched)
        any_chop = jnp.any(focus_chop_candidates)
        chop = _lowest(focus_chop_candidates)
        focus_is_chop = any_chop & (focus == chop)

        card_colors = _card_colors()
        card_ranks = _card_ranks()
        clue_mask = jnp.where(is_color, card_colors == value, card_ranks == value)
        has_direct = jnp.any(clue_mask & playable25)
        rank_save = (~is_color) & ((value == 1) | (value == 4)) & focus_is_chop
        critical_save = focus_is_chop & jnp.any(clue_mask & critical25)
        receiver_would_play = has_direct & (~(rank_save | critical_save))

        match_count = jnp.sum(matches.astype(jnp.float32))
        new_count = jnp.sum((matches & (~partner_touched)).astype(jnp.float32))
        score = jnp.where(receiver_would_play, 0.0, 50.0)
        score = score + (rank_save | critical_save).astype(jnp.float32) * 10.0 + new_count - match_count * 0.25
        valid = legal[action] & any_match
        better = valid & (score > best_score)
        best_score = jnp.where(better, score, best_score)
        best_action = jnp.where(better, action, best_action)

    return best_action.astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    partner, fireworks, lives, discards, last, knowledge = _decode(obs)
    heights = _heights(fireworks)
    playable25 = _playable25(heights)
    trash25 = _trash25(heights)
    one_away25 = _one_away25(heights)
    critical25 = _critical25(heights, discards)
    memory = _apply_memory_update(memory, reset, last, knowledge, playable25, one_away25, critical25)

    legal = obs.legal_actions
    inactive = legal[20]

    own_v0 = knowledge[0, :, 0:25]
    own_occ = _occupied_from_v0(own_v0)
    partner_touched = _literal_touched(knowledge[1])

    poss_count = jnp.sum(memory["poss"].astype(jnp.int32), axis=1)
    poss_playable_count = jnp.sum((memory["poss"] & playable25[None, :]).astype(jnp.int32), axis=1)
    convention_playable = memory["play"] & (poss_count > 0) & (poss_count == poss_playable_count) & own_occ
    literal_play_prob = jnp.sum(own_v0 * playable25[None, :].astype(jnp.float32), axis=1)
    literal_playable = own_occ & (literal_play_prob > 0.999)
    play_slots = (convention_playable | literal_playable) & legal[5:10]
    has_play = jnp.any(play_slots)
    play_action = 5 + _highest(play_slots)

    clue_action, has_clue = _clue_choice(legal, partner, partner_touched, heights, discards)
    forced_hint_action = _safe_forced_hint(legal, partner, partner_touched, heights, discards)

    protected = memory["play"] | memory["saved"]
    discard_slot, has_discard = _choose_discard(legal, own_occ, memory["touched"], protected, own_v0, trash25)

    active_action = jnp.where(
        has_play,
        play_action,
        jnp.where(has_clue, clue_action, jnp.where(has_discard, discard_slot, forced_hint_action)),
    ).astype(jnp.int32)
    action = jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), active_action)
    return action, memory, {}
