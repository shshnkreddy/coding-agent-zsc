"""H-group Level 1 compatible Hanabi policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h_group_l1_focus_chop_policy', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

N_SLOTS = 5
N_COLORS = 5
N_RANKS = 5
N_IDS = 25
BIG_NEG = -1000000.0


def init_agent():
    return {
        "touched": jnp.zeros((N_SLOTS,), dtype=jnp.bool_),
        "play": jnp.zeros((N_SLOTS,), dtype=jnp.bool_),
        "save": jnp.zeros((N_SLOTS,), dtype=jnp.bool_),
        "note": jnp.zeros((N_SLOTS, N_IDS), dtype=jnp.bool_),
    }


def _zero_memory():
    return init_agent()


def _reset_memory(memory, reset):
    zero = _zero_memory()
    return {
        "touched": jnp.where(reset, zero["touched"], memory["touched"]),
        "play": jnp.where(reset, zero["play"], memory["play"]),
        "save": jnp.where(reset, zero["save"], memory["save"]),
        "note": jnp.where(reset, zero["note"], memory["note"]),
    }


def _shift_slots(values, removed):
    idx = jnp.arange(N_SLOTS)
    take = jnp.where(idx < removed, idx, idx + 1)
    take = jnp.minimum(take, N_SLOTS - 1)
    shifted = values[take]
    keep_last = idx < (N_SLOTS - 1)
    return jnp.where(keep_last, shifted, jnp.zeros_like(shifted))


def _shift_notes(notes, removed):
    idx = jnp.arange(N_SLOTS)
    take = jnp.where(idx < removed, idx, idx + 1)
    take = jnp.minimum(take, N_SLOTS - 1)
    shifted = notes[take, :]
    keep_last = (idx < (N_SLOTS - 1))[:, None]
    return jnp.where(keep_last, shifted, jnp.zeros_like(shifted))


def _first_low(cond):
    idx = jnp.arange(N_SLOTS)
    return jnp.argmin(jnp.where(cond, idx, N_SLOTS + 1))


def _last_high(cond):
    idx = jnp.arange(N_SLOTS)
    return jnp.argmax(jnp.where(cond, idx, -1))


def _decode(obs):
    f = obs.features
    partner = f[0:125].reshape((N_SLOTS, N_COLORS, N_RANKS))
    fw = f[167:192].reshape((N_COLORS, N_RANKS))
    has_fw = jnp.sum(fw, axis=1) > 0.5
    heights = jnp.where(has_fw, jnp.argmax(fw, axis=1) + 1, 0).astype(jnp.int32)
    info = jnp.sum(f[192:200] > 0.5).astype(jnp.int32)
    disc = f[203:253].reshape((N_COLORS, 10))
    disc_counts = jnp.stack(
        [
            jnp.sum(disc[:, 0:3], axis=1),
            jnp.sum(disc[:, 3:5], axis=1),
            jnp.sum(disc[:, 5:7], axis=1),
            jnp.sum(disc[:, 7:9], axis=1),
            jnp.sum(disc[:, 9:10], axis=1),
        ],
        axis=1,
    ).astype(jnp.int32)
    last = f[253:308]
    know = f[308:658].reshape((2, N_SLOTS, 35))
    return partner, heights, info, disc_counts, last, know


def _partner_id(partner):
    flat = partner.reshape((N_SLOTS, N_IDS))
    occ = jnp.sum(flat, axis=1) > 0.5
    ident = jnp.argmax(flat, axis=1)
    color = (ident // N_RANKS).astype(jnp.int32)
    rank = (ident % N_RANKS).astype(jnp.int32)
    return occ, color, rank


def _literal_touched(know_player):
    return jnp.any(know_player[:, 25:35] > 0.5, axis=1)


def _clue_focus(touch, pre_touched, occupied):
    new = touch & (~pre_touched)
    has_new = jnp.any(new)
    chop_cond = occupied & (~pre_touched)
    has_chop = jnp.any(chop_cond)
    chop = _first_low(chop_cond)
    retouch_focus = _last_high(touch)
    new_focus = _last_high(new)
    focus = jnp.where(has_new & has_chop & touch[chop], chop, jnp.where(has_new, new_focus, retouch_focus))
    return focus.astype(jnp.int32), jnp.any(touch)


def _id_grids():
    colors = (jnp.arange(N_IDS) // N_RANKS).astype(jnp.int32)
    ranks = (jnp.arange(N_IDS) % N_RANKS).astype(jnp.int32)
    return colors, ranks


def _playable_ids(heights):
    colors, ranks = _id_grids()
    return ranks == heights[colors]


def _nontrash_ids(heights):
    colors, ranks = _id_grids()
    return ranks >= heights[colors]


def _critical_matrix(disc_counts, heights):
    copies = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    remaining = copies[None, :] - disc_counts
    ranks = jnp.arange(N_RANKS)[None, :]
    return (remaining <= 1) & (ranks >= heights[:, None])


def _connector_ok_for_id(color, rank, heights, partner_occ, partner_color, partner_rank, partner_touched):
    need = jnp.arange(N_RANKS)
    in_gap = (need >= heights[color]) & (need < rank)
    visible_connector = jnp.zeros((N_RANKS,), dtype=jnp.bool_)
    for k in range(N_RANKS):
        visible_connector = visible_connector.at[k].set(
            jnp.any(partner_occ & partner_touched & (partner_color == color) & (partner_rank == k))
        )
    return jnp.all((~in_gap) | visible_connector)


def _convention_note_mask(is_color_hint, clue_value, heights, partner_occ, partner_color, partner_rank, partner_touched):
    colors, ranks = _id_grids()
    attr_match = jnp.where(is_color_hint, colors == clue_value, ranks == clue_value)
    ok = jnp.zeros((N_IDS,), dtype=jnp.bool_)
    for i in range(N_IDS):
        c = colors[i]
        r = ranks[i]
        can_be_play = (r >= heights[c]) & _connector_ok_for_id(c, r, heights, partner_occ, partner_color, partner_rank, partner_touched)
        ok = ok.at[i].set(can_be_play)
    return attr_match & ok


def _apply_memory_updates(memory, obs, partner, heights, disc_counts, last, know):
    actor = last[0:2]
    typ = last[2:6]
    recipient = last[6:8]
    matches = last[18:23] > 0.5
    removed = jnp.argmax(last[23:28]).astype(jnp.int32)

    own_removed = (actor[0] > 0.5) & ((typ[0] > 0.5) | (typ[1] > 0.5))
    touched = jnp.where(own_removed, _shift_slots(memory["touched"], removed), memory["touched"])
    play = jnp.where(own_removed, _shift_slots(memory["play"], removed), memory["play"])
    save = jnp.where(own_removed, _shift_slots(memory["save"], removed), memory["save"])
    note = jnp.where(own_removed, _shift_notes(memory["note"], removed), memory["note"])

    own_v0 = know[0, :, 0:25]
    own_occ = jnp.sum(own_v0, axis=1) > 0.0
    recv_hint = (actor[1] > 0.5) & ((typ[2] > 0.5) | (typ[3] > 0.5)) & (recipient[0] > 0.5)
    focus, has_focus = _clue_focus(matches, touched, own_occ)
    focus_is_chop = focus == _first_low(own_occ & (~touched))

    partner_occ, partner_color, partner_rank = _partner_id(partner)
    partner_touched = _literal_touched(know[1])
    is_color_hint = typ[2] > 0.5
    clue_color = jnp.argmax(last[8:13]).astype(jnp.int32)
    clue_rank = jnp.argmax(last[13:18]).astype(jnp.int32)
    clue_value = jnp.where(is_color_hint, clue_color, clue_rank).astype(jnp.int32)
    note_mask = _convention_note_mask(is_color_hint, clue_value, heights, partner_occ, partner_color, partner_rank, partner_touched)

    colors, ranks = _id_grids()
    crit = _critical_matrix(disc_counts, heights).reshape((N_IDS,))
    attr = jnp.where(is_color_hint, colors == clue_value, ranks == clue_value)
    save_possible = (focus_is_chop & ((typ[3] > 0.5) & ((clue_rank == 4) | (clue_rank == 1))) |
                     (focus_is_chop & jnp.any(attr & crit & (own_v0[focus] > 0.0))))
    play_possible = jnp.any(note_mask)
    set_focus = recv_hint & has_focus

    row = jnp.arange(N_SLOTS) == focus
    touched = touched | (set_focus & matches)
    play = play | (set_focus & play_possible & row)
    save = save | (set_focus & save_possible & row)
    note = jnp.where((set_focus & play_possible & row)[:, None], note_mask[None, :], note)

    return {"touched": touched, "play": play, "save": save, "note": note}


def _clue_candidate_score(action, touch, target, partner_touched, partner_occ, partner_color, partner_rank, heights, legal, prefer_color):
    focus, has_focus = _clue_focus(touch, partner_touched, partner_occ)
    new_count = jnp.sum(touch & (~partner_touched))
    touched_count = jnp.sum(touch)
    bad_touch = touch & (partner_rank < heights[partner_color])
    bad_count = jnp.sum(bad_touch)
    valid = legal[action] & has_focus & (focus == target) & (new_count > 0) & (bad_count == 0)
    color_bonus = jnp.where(prefer_color, 2.0, 0.0)
    return jnp.where(valid, 100.0 + color_bonus - touched_count.astype(jnp.float32), BIG_NEG)


def _best_target_clue(target, allow_color, allow_rank, partner_occ, partner_color, partner_rank, partner_touched, heights, legal):
    best_score = jnp.asarray(BIG_NEG)
    best_action = jnp.asarray(20, dtype=jnp.int32)
    for c in range(N_COLORS):
        touch = partner_occ & (partner_color == c)
        score = _clue_candidate_score(10 + c, touch, target, partner_touched, partner_occ, partner_color, partner_rank, heights, legal, True)
        score = jnp.where(allow_color & (partner_color[target] == c), score, BIG_NEG)
        take = score > best_score
        best_score = jnp.where(take, score, best_score)
        best_action = jnp.where(take, 10 + c, best_action)
    for r in range(N_RANKS):
        touch = partner_occ & (partner_rank == r)
        score = _clue_candidate_score(15 + r, touch, target, partner_touched, partner_occ, partner_color, partner_rank, heights, legal, False)
        score = jnp.where(allow_rank & (partner_rank[target] == r), score, BIG_NEG)
        take = score > best_score
        best_score = jnp.where(take, score, best_score)
        best_action = jnp.where(take, 15 + r, best_action)
    return best_action.astype(jnp.int32), best_score


def _best_play_clue(partner_occ, partner_color, partner_rank, partner_touched, heights, legal):
    best_score = jnp.asarray(BIG_NEG)
    best_action = jnp.asarray(20, dtype=jnp.int32)
    playable = partner_occ & (partner_rank == heights[partner_color])
    for s in range(N_SLOTS):
        action, score = _best_target_clue(
            jnp.asarray(s, dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(True),
            partner_occ,
            partner_color,
            partner_rank,
            partner_touched,
            heights,
            legal,
        )
        score = jnp.where(playable[s] & (~partner_touched[s]), score + s * 0.1, BIG_NEG)
        take = score > best_score
        best_score = jnp.where(take, score, best_score)
        best_action = jnp.where(take, action, best_action)
    return best_action, best_score


def _tempo_candidate_score(action, touch, target, partner_touched, partner_occ, partner_color, partner_rank, heights, legal):
    focus, has_focus = _clue_focus(touch, partner_touched, partner_occ)
    touched_count = jnp.sum(touch)
    bad_touch = touch & (partner_rank < heights[partner_color])
    bad_count = jnp.sum(bad_touch)
    no_new = ~jnp.any(touch & (~partner_touched))
    valid = legal[action] & has_focus & no_new & (focus == target) & (bad_count == 0)
    return jnp.where(valid, 80.0 - touched_count.astype(jnp.float32), BIG_NEG)


def _best_tempo_clue(partner_occ, partner_color, partner_rank, partner_touched, heights, legal):
    discard_legal = jnp.any(legal[0:5])
    best_score = jnp.asarray(BIG_NEG)
    best_action = jnp.asarray(20, dtype=jnp.int32)
    playable_touched = partner_occ & partner_touched & (partner_rank == heights[partner_color])
    for s in range(N_SLOTS):
        for c in range(N_COLORS):
            touch = partner_occ & (partner_color == c)
            score = _tempo_candidate_score(10 + c, touch, jnp.asarray(s, dtype=jnp.int32), partner_touched, partner_occ, partner_color, partner_rank, heights, legal)
            score = jnp.where(playable_touched[s] & (partner_color[s] == c), score + s * 0.1, BIG_NEG)
            take = score > best_score
            best_score = jnp.where(take, score, best_score)
            best_action = jnp.where(take, 10 + c, best_action)
        for r in range(N_RANKS):
            touch = partner_occ & (partner_rank == r)
            score = _tempo_candidate_score(15 + r, touch, jnp.asarray(s, dtype=jnp.int32), partner_touched, partner_occ, partner_color, partner_rank, heights, legal)
            score = jnp.where(playable_touched[s] & (partner_rank[s] == r), score + s * 0.1, BIG_NEG)
            take = score > best_score
            best_score = jnp.where(take, score, best_score)
            best_action = jnp.where(take, 15 + r, best_action)
    return best_action, jnp.where(discard_legal, BIG_NEG, best_score)


def _best_chop_save_or_play(partner_occ, partner_color, partner_rank, partner_touched, heights, disc_counts, legal):
    chop_cond = partner_occ & (~partner_touched)
    has_chop = jnp.any(chop_cond)
    chop = _first_low(chop_cond)
    c = partner_color[chop]
    r = partner_rank[chop]
    same_visible = partner_occ & (partner_color == c) & (partner_rank == r) & (jnp.arange(N_SLOTS) != chop)
    unique_visible = ~jnp.any(same_visible)
    crit = _critical_matrix(disc_counts, heights)
    save5 = r == 4
    save2 = (r == 1) & unique_visible
    critical = crit[c, r]
    playable_unique = (r == heights[c]) & unique_visible
    danger = has_chop & (save5 | save2 | critical | playable_unique)
    allow_color = danger & ((critical & (~save5)) | playable_unique)
    allow_rank = danger
    action, score = _best_target_clue(chop, allow_color, allow_rank, partner_occ, partner_color, partner_rank, partner_touched, heights, legal)
    return action, jnp.where(danger, score + 20.0, BIG_NEG)


def _own_play_action(memory, know, heights):
    own_v0 = know[0, :, 0:25]
    own_literal_touched = _literal_touched(know[0])
    own_occ = jnp.sum(own_v0, axis=1) > 0.0
    playable = _playable_ids(heights)
    nontrash = _nontrash_ids(heights)
    note_live = memory["note"] & nontrash[None, :]
    note_count = jnp.sum(note_live, axis=1)
    note_play_count = jnp.sum(note_live & playable[None, :], axis=1)
    convention_play = memory["play"] & (note_count > 0) & (note_play_count == note_count)
    literal_mass = jnp.sum(own_v0, axis=1)
    literal_play_mass = jnp.sum(jnp.where(playable[None, :], own_v0, 0.0), axis=1)
    literal_sure_play = own_literal_touched & (literal_mass > 0.0) & (literal_play_mass >= literal_mass - 1.0e-5)
    can_play = own_occ & (convention_play | literal_sure_play)
    slot = _last_high(can_play)
    score = jnp.where(jnp.any(can_play), 100.0 + slot.astype(jnp.float32), BIG_NEG)
    return (5 + slot).astype(jnp.int32), score


def _discard_action(memory, know, heights, legal):
    own_v0 = know[0, :, 0:25]
    own_occ = jnp.sum(own_v0, axis=1) > 0.0
    literal_touched = _literal_touched(know[0])
    protected = literal_touched | memory["touched"] | memory["play"] | memory["save"] | jnp.any(memory["note"], axis=1)
    nontrash = _nontrash_ids(heights)
    live_mass = jnp.sum(jnp.where(nontrash[None, :], own_v0, 0.0), axis=1)
    trash = own_occ & (live_mass <= 1.0e-6) & (~memory["save"])
    trash_slot = _first_low(trash)
    chop_cond = own_occ & (~protected)
    chop_slot = _first_low(chop_cond)
    fallback_cond = own_occ & (~memory["save"]) & (~memory["play"])
    fallback_slot = _first_low(fallback_cond)
    slot = jnp.where(jnp.any(trash), trash_slot, jnp.where(jnp.any(chop_cond), chop_slot, fallback_slot))
    valid = legal[slot] & (jnp.any(trash) | jnp.any(chop_cond) | jnp.any(fallback_cond))
    score = jnp.where(valid, 10.0, BIG_NEG)
    return slot.astype(jnp.int32), score


def _fallback_action(obs, know, heights):
    legal = obs.legal_actions
    hint_legal = legal[10:20]
    hint_action = 10 + jnp.argmax(hint_legal).astype(jnp.int32)
    play_action, _ = _own_play_action(_zero_memory(), know, heights)
    any_hint = jnp.any(hint_legal)
    any_play = jnp.any(legal[5:10])
    first_legal = jnp.argmax(legal).astype(jnp.int32)
    return jnp.where(any_hint, hint_action, jnp.where(any_play, play_action, first_legal)).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    memory = _reset_memory(memory, reset)
    partner, heights, info, disc_counts, last, know = _decode(obs)
    memory = _apply_memory_updates(memory, obs, partner, heights, disc_counts, last, know)

    legal = obs.legal_actions
    inactive = legal[20] & (~jnp.any(legal[0:20]))
    partner_occ, partner_color, partner_rank = _partner_id(partner)
    partner_touched = _literal_touched(know[1])

    save_action, save_score = _best_chop_save_or_play(partner_occ, partner_color, partner_rank, partner_touched, heights, disc_counts, legal)
    own_action, own_score = _own_play_action(memory, know, heights)
    clue_action, clue_score = _best_play_clue(partner_occ, partner_color, partner_rank, partner_touched, heights, legal)
    tempo_action, tempo_score = _best_tempo_clue(partner_occ, partner_color, partner_rank, partner_touched, heights, legal)
    discard_action, discard_score = _discard_action(memory, know, heights, legal)

    best_action = own_action
    best_score = own_score
    take = clue_score > best_score
    best_action = jnp.where(take, clue_action, best_action)
    best_score = jnp.where(take, clue_score, best_score)
    take = save_score > best_score
    best_action = jnp.where(take, save_action, best_action)
    best_score = jnp.where(take, save_score, best_score)
    take = tempo_score > best_score
    best_action = jnp.where(take, tempo_action, best_action)
    best_score = jnp.where(take, tempo_score, best_score)
    take = own_score > best_score
    best_action = jnp.where(take, own_action, best_action)
    best_score = jnp.where(take, own_score, best_score)
    take = discard_score > best_score
    best_action = jnp.where(take, discard_action, best_action)
    best_score = jnp.where(take, discard_score, best_score)

    fallback = _fallback_action(obs, know, heights)
    active_action = jnp.where(best_score > BIG_NEG / 2.0, best_action, fallback)
    action = jnp.where(inactive, 20, active_action).astype(jnp.int32)
    return action, memory, {}
