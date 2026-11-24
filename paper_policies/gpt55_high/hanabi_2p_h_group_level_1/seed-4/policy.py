"""H-group Level 1 oriented policy for the two-player Hanabi task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'level1_literal', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NUM_COLORS = 5
NUM_RANKS = 5
HAND_SIZE = 5
NUM_IDENTITIES = 25
NEG = -1.0e9


def _colors():
    return jnp.arange(NUM_COLORS, dtype=jnp.int32)


def _ranks():
    return jnp.arange(NUM_RANKS, dtype=jnp.int32)


def _slots():
    return jnp.arange(HAND_SIZE, dtype=jnp.int32)


def _ident_colors():
    return jnp.repeat(_colors(), NUM_RANKS)


def _ident_ranks():
    return jnp.tile(_ranks(), NUM_COLORS)


def _mults():
    return jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)


def init_agent():
    return {
        "play": jnp.zeros((HAND_SIZE, NUM_IDENTITIES), dtype=bool),
        "save": jnp.zeros((HAND_SIZE,), dtype=bool),
        "touched": jnp.zeros((HAND_SIZE,), dtype=bool),
    }


def _decode(obs):
    f = obs.features
    partner_cards = f[0:125].reshape((HAND_SIZE, NUM_COLORS, NUM_RANKS))
    partner_occ = jnp.sum(partner_cards, axis=(1, 2)) > 0.5
    partner_flat = partner_cards.reshape((HAND_SIZE, NUM_IDENTITIES))
    partner_ident = jnp.argmax(partner_flat, axis=1).astype(jnp.int32)
    partner_color = partner_ident // NUM_RANKS
    partner_rank = partner_ident % NUM_RANKS

    fw = f[167:192].reshape((NUM_COLORS, NUM_RANKS))
    heights = jnp.sum(fw * jnp.arange(1, NUM_RANKS + 1, dtype=jnp.float32), axis=1).astype(jnp.int32)
    info = jnp.sum(f[192:200] > 0.5).astype(jnp.int32)
    lives = jnp.sum(f[200:203] > 0.5).astype(jnp.int32)
    deck = jnp.sum(f[127:167] > 0.5).astype(jnp.int32)

    disc_blocks = f[203:253].reshape((NUM_COLORS, 10)) > 0.5
    disc = jnp.stack(
        [
            jnp.sum(disc_blocks[:, 0:3], axis=1),
            jnp.sum(disc_blocks[:, 3:5], axis=1),
            jnp.sum(disc_blocks[:, 5:7], axis=1),
            jnp.sum(disc_blocks[:, 7:9], axis=1),
            jnp.sum(disc_blocks[:, 9:10], axis=1),
        ],
        axis=1,
    ).astype(jnp.int32)

    knowledge = f[308:658].reshape((2, HAND_SIZE, 35))
    own_v0 = knowledge[0, :, 0:25]
    own_possible = own_v0 > 0.0
    own_occ = jnp.sum(own_v0, axis=1) > 0.0
    own_literal = jnp.sum(knowledge[0, :, 25:35], axis=1) > 0.5
    partner_literal = jnp.sum(knowledge[1, :, 25:35], axis=1) > 0.5

    la = f[253:308]
    return (
        partner_occ,
        partner_color,
        partner_rank,
        partner_literal,
        heights,
        info,
        lives,
        deck,
        disc,
        own_possible,
        own_occ,
        own_literal,
        la,
    )


def _lowest(mask):
    slots = _slots()
    return jnp.min(jnp.where(mask, slots, HAND_SIZE)).astype(jnp.int32)


def _highest(mask):
    slots = _slots()
    return jnp.max(jnp.where(mask, slots, -1)).astype(jnp.int32)


def _one_hot_slot(slot):
    return _slots() == slot


def _shift_slots(arr, removed, fill):
    slots = _slots()
    pad = jnp.expand_dims(fill, axis=0)
    padded = jnp.concatenate([arr, pad], axis=0)
    idx = jnp.where(slots < removed, slots, slots + 1)
    return padded[idx]


def _public_remaining(heights, disc):
    played = (_ranks()[None, :] < heights[:, None]).astype(jnp.int32)
    return _mults()[None, :] - disc - played


def _identity_playable(heights):
    ident_colors = _ident_colors()
    ident_ranks = _ident_ranks()
    return (ident_ranks == heights[ident_colors]) & (heights[ident_colors] < NUM_RANKS)


def _identity_useful(heights, remaining):
    ident_colors = _ident_colors()
    ident_ranks = _ident_ranks()
    rem_flat = remaining.reshape((NUM_IDENTITIES,))
    return (ident_ranks >= heights[ident_colors]) & (heights[ident_colors] < NUM_RANKS) & (rem_flat > 0)


def _clue_focus(matches, touched, occ):
    new = matches & (~touched)
    any_new = jnp.any(new)
    chop = _lowest(occ & (~touched))
    fallback_chop = _lowest(occ)
    chop = jnp.where(chop < HAND_SIZE, chop, fallback_chop)
    chop = jnp.minimum(chop, HAND_SIZE - 1)
    focus_new = _highest(new)
    focus_old = _highest(matches)
    focus = jnp.where(any_new, jnp.where(matches[chop], chop, focus_new), focus_old)
    focus = jnp.clip(focus, 0, HAND_SIZE - 1)
    return focus.astype(jnp.int32), any_new, chop.astype(jnp.int32)


def _count_same_visible(color, rank, p_color, p_rank, p_occ):
    return jnp.sum((p_color == color) & (p_rank == rank) & p_occ).astype(jnp.int32)


def _duplicate_identity_touched(matches, p_color, p_rank, p_occ):
    ids = p_color * NUM_RANKS + p_rank
    counts = jnp.sum(((ids[:, None] == jnp.arange(NUM_IDENTITIES)[None, :]) & matches[:, None] & p_occ[:, None]), axis=0)
    return jnp.any(counts > 1)


def _interpret_self_clue(mem, own_occ, own_literal, heights, disc, la):
    ident_colors = _ident_colors()
    ident_ranks = _ident_ranks()
    actor_partner = la[1] > 0.5
    is_color = la[4] > 0.5
    is_rank = la[5] > 0.5
    recipient_self = la[6] > 0.5
    clue_to_self = actor_partner & recipient_self & (is_color | is_rank)
    matches = la[18:23] > 0.5
    clue_color = jnp.argmax(la[8:13]).astype(jnp.int32)
    clue_rank = jnp.argmax(la[13:18]).astype(jnp.int32)

    focus, any_new, chop = _clue_focus(matches, mem["touched"], own_occ)
    focus_oh = _one_hot_slot(focus) & clue_to_self & (focus >= 0) & any_new

    remaining = _public_remaining(heights, disc)
    critical = remaining == 1
    color_play = (ident_colors == clue_color) & (ident_ranks == heights[clue_color])
    rank_play = (ident_ranks == clue_rank) & (heights[ident_colors] == clue_rank)
    play_candidates = jnp.where(is_color, color_play, rank_play)

    color_save = (ident_colors == clue_color) & critical.reshape((NUM_IDENTITIES,))
    rank_save = (ident_ranks == clue_rank) & critical.reshape((NUM_IDENTITIES,))
    five_save = is_rank & (clue_rank == 4) & (ident_ranks == 4)
    two_save = is_rank & (clue_rank == 1) & (ident_ranks == 1)
    chop_save = matches[chop] & (focus == chop)
    save_candidates = (jnp.where(is_color, color_save, rank_save) | five_save | two_save) & chop_save
    effective_play = play_candidates & (~jnp.any(save_candidates))

    new_play = jnp.where(focus_oh[:, None], effective_play[None, :], mem["play"])
    new_save = mem["save"] | (focus_oh & jnp.any(save_candidates))
    new_touched = mem["touched"] | own_literal | (matches & clue_to_self)
    return {"play": new_play, "save": new_save, "touched": new_touched}


def _advance_memory(mem, reset, own_occ, own_literal, heights, disc, la):
    empty_play = jnp.zeros((HAND_SIZE, NUM_IDENTITIES), dtype=bool)
    empty_bool = jnp.zeros((HAND_SIZE,), dtype=bool)
    base = {
        "play": jnp.where(reset, empty_play, mem["play"]),
        "save": jnp.where(reset, empty_bool, mem["save"]),
        "touched": jnp.where(reset, own_literal, mem["touched"]),
    }

    actor_self = la[0] > 0.5
    removed_action = (la[2] > 0.5) | (la[3] > 0.5)
    self_removed = actor_self & removed_action
    removed_slot = jnp.argmax(la[23:28]).astype(jnp.int32)
    shifted = {
        "play": jnp.where(self_removed, _shift_slots(base["play"], removed_slot, jnp.zeros((NUM_IDENTITIES,), dtype=bool)), base["play"]),
        "save": jnp.where(self_removed, _shift_slots(base["save"], removed_slot, jnp.asarray(False)), base["save"]),
        "touched": jnp.where(self_removed, _shift_slots(base["touched"], removed_slot, jnp.asarray(False)), base["touched"]),
    }
    shifted = {
        "play": shifted["play"] & own_occ[:, None],
        "save": shifted["save"] & own_occ,
        "touched": shifted["touched"] & own_occ,
    }
    return _interpret_self_clue(shifted, own_occ, own_literal, heights, disc, la)


def _own_action_scores(mem, legal, own_possible, own_occ, own_literal, heights, info, lives, deck):
    slots = _slots()
    playable_id = _identity_playable(heights)
    remaining_dummy = jnp.ones((NUM_COLORS, NUM_RANKS), dtype=jnp.int32)
    useful_id = _identity_useful(heights, remaining_dummy)

    possible_count = jnp.sum(own_possible, axis=1)
    literal_safe = own_occ & (possible_count > 0) & (~jnp.any(own_possible & (~playable_id[None, :]), axis=1))
    touched = (mem["touched"] | own_literal) & own_occ
    useful_possible = own_possible & useful_id[None, :]
    useful_count = jnp.sum(useful_possible, axis=1)
    gtp_safe = touched & (~mem["save"]) & (useful_count > 0) & (~jnp.any(useful_possible & (~playable_id[None, :]), axis=1))
    noted = mem["play"] & own_possible
    noted_count = jnp.sum(noted, axis=1)
    noted_safe = own_occ & (noted_count > 0) & (~jnp.any(noted & (~playable_id[None, :]), axis=1))
    safe_play = literal_safe | ((noted_safe | gtp_safe) & (lives > 1))

    trash = own_occ & (possible_count > 0) & (~jnp.any(own_possible & useful_id[None, :], axis=1)) & (~mem["save"])
    chop = _lowest(own_occ & (~touched) & (~mem["save"]))
    fallback = _lowest(own_occ & (~mem["save"]))
    discard_slot = jnp.where(chop < HAND_SIZE, chop, fallback)
    discard_mask = _one_hot_slot(discard_slot) & (discard_slot < HAND_SIZE) & (info < 8)

    scores = jnp.full((21,), NEG, dtype=jnp.float32)
    play_scores = jnp.where(safe_play, 900.0 + slots.astype(jnp.float32), NEG)
    play_scores = jnp.where(legal[5:10], play_scores, NEG)
    scores = scores.at[5:10].set(play_scores)

    trash_scores = jnp.where(trash & (info < 8), 820.0 - slots.astype(jnp.float32) * 0.1, NEG)
    chop_scores = jnp.where(discard_mask, 120.0 - slots.astype(jnp.float32) * 0.1, NEG)
    discard_scores = jnp.maximum(trash_scores, chop_scores)
    discard_scores = jnp.where(legal[0:5], discard_scores, NEG)
    scores = scores.at[0:5].set(discard_scores)

    playable_prob = jnp.sum(jnp.where(own_possible, playable_id[None, :], 0.0), axis=1) / jnp.maximum(possible_count.astype(jnp.float32), 1.0)
    endgame_pressure = ((deck <= 1) | (info >= 8)) & (lives > 1)
    risky = jnp.where(own_occ & endgame_pressure, 20.0 + playable_prob * 40.0 + slots.astype(jnp.float32), NEG)
    risky = jnp.where(legal[5:10], risky, NEG)
    scores = scores.at[5:10].set(jnp.maximum(scores[5:10], risky))
    return scores


def _clue_action_scores(legal, p_occ, p_color, p_rank, p_touched, heights, disc, info, lives):
    remaining = _public_remaining(heights, disc)
    p_playable = p_occ & (p_rank == heights[p_color]) & (heights[p_color] < NUM_RANKS)
    p_useful = p_occ & (p_rank >= heights[p_color]) & (heights[p_color] < NUM_RANKS)
    p_critical = p_occ & (remaining[p_color, p_rank] == 1)

    chop = _lowest(p_occ & (~p_touched))
    fallback_chop = _lowest(p_occ)
    chop = jnp.where(chop < HAND_SIZE, chop, fallback_chop)
    chop_oh = _one_hot_slot(chop) & (chop < HAND_SIZE)
    chop_playable = jnp.any(chop_oh & p_playable)
    chop_five = jnp.any(chop_oh & (p_rank == 4))
    chop_two = jnp.any(chop_oh & (p_rank == 1))
    chop_critical = jnp.any(chop_oh & p_critical)
    chop_color = p_color[chop]
    chop_rank = p_rank[chop]
    same_visible = _count_same_visible(chop_color, chop_rank, p_color, p_rank, p_occ)
    two_save_ok = chop_two & (same_visible <= 1)

    scores = jnp.full((21,), NEG, dtype=jnp.float32)

    color_scores = []
    for c in range(NUM_COLORS):
        matches = p_occ & (p_color == c)
        focus, any_new, focus_chop = _clue_focus(matches, p_touched, p_occ)
        good_touch = jnp.all(jnp.where(matches, p_useful, True)) & (~_duplicate_identity_touched(matches, p_color, p_rank, p_occ))
        focus_play = matches[focus] & p_playable[focus]
        is_chop_focus = focus == focus_chop
        immediate_chop = is_chop_focus & focus_play
        single_touch = jnp.sum(matches) == 1
        clear_color_play = single_touch | immediate_chop
        play_score = jnp.where(
            any_new & good_touch & focus_play & clear_color_play,
            jnp.where(immediate_chop, 1110.0, 700.0) + focus.astype(jnp.float32),
            NEG,
        )
        critical_save = any_new & is_chop_focus & chop_critical & (chop_rank != 4) & (chop_color == c)
        save_score = jnp.where(critical_save, 1010.0, NEG)
        useful_new = any_new & good_touch
        stall_score = jnp.where(useful_new, 70.0 + focus.astype(jnp.float32), NEG)
        forced_score = jnp.where((info >= 8) | (lives <= 1), 30.0 + focus.astype(jnp.float32), NEG)
        sc = jnp.maximum(jnp.maximum(jnp.maximum(play_score, save_score), stall_score), forced_score)
        sc = jnp.where((info > 0) & legal[10 + c], sc, NEG)
        color_scores.append(sc)
    scores = scores.at[10:15].set(jnp.stack(color_scores))

    rank_scores = []
    for r in range(NUM_RANKS):
        matches = p_occ & (p_rank == r)
        focus, any_new, focus_chop = _clue_focus(matches, p_touched, p_occ)
        good_touch = jnp.all(jnp.where(matches, p_useful, True)) & (~_duplicate_identity_touched(matches, p_color, p_rank, p_occ))
        focus_play = matches[focus] & p_playable[focus]
        is_chop_focus = focus == focus_chop
        immediate_chop = is_chop_focus & focus_play
        play_score = jnp.where(
            any_new & good_touch & focus_play,
            jnp.where(immediate_chop, 1120.0, 760.0) + focus.astype(jnp.float32),
            NEG,
        )
        five_save = any_new & is_chop_focus & chop_five & (r == 4)
        two_save = any_new & is_chop_focus & two_save_ok & (r == 1)
        critical_save = any_new & is_chop_focus & chop_critical & (chop_rank == r)
        save_score = jnp.where(five_save | two_save | critical_save, 1000.0 + float(r), NEG)
        useful_new = any_new & good_touch
        stall_score = jnp.where(useful_new, 60.0 + focus.astype(jnp.float32), NEG)
        forced_score = jnp.where((info >= 8) | (lives <= 1), 25.0 + focus.astype(jnp.float32), NEG)
        sc = jnp.maximum(jnp.maximum(jnp.maximum(play_score, save_score), stall_score), forced_score)
        sc = jnp.where((info > 0) & legal[15 + r], sc, NEG)
        rank_scores.append(sc)
    scores = scores.at[15:20].set(jnp.stack(rank_scores))
    return scores


def agent_act(obs, reset, memory, rng):
    (
        p_occ,
        p_color,
        p_rank,
        p_touched,
        heights,
        info,
        lives,
        deck,
        disc,
        own_possible,
        own_occ,
        own_literal,
        la,
    ) = _decode(obs)

    mem = _advance_memory(memory, reset, own_occ, own_literal, heights, disc, la)

    legal = obs.legal_actions
    own_scores = _own_action_scores(mem, legal, own_possible, own_occ, own_literal, heights, info, lives, deck)
    clue_scores = _clue_action_scores(legal, p_occ, p_color, p_rank, p_touched, heights, disc, info, lives)
    scores = jnp.maximum(own_scores, clue_scores)
    scores = jnp.where(legal, scores, NEG)

    fallback = jnp.argmax(legal.astype(jnp.int32)).astype(jnp.int32)
    active_action = jnp.argmax(scores).astype(jnp.int32)
    has_scored = jnp.max(scores) > (NEG / 2.0)
    active_action = jnp.where(has_scored, active_action, fallback)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), active_action)

    trace = {
        "info": info,
        "lives": lives,
        "deck": deck,
        "best_score": jnp.max(scores),
        "action": action,
    }
    return action, mem, trace
