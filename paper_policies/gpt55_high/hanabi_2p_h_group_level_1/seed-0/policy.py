"""H-group Level 1 compatible Hanabi heuristic policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h_group_l1_baseline', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

N = 5
NUM_ACTIONS = 21
NEG = -1.0e9


def init_agent():
    return {
        "status": jnp.zeros((N,), dtype=jnp.int32),  # 0 unknown, 1 play, 2 save/hold
        "color": jnp.full((N,), -1, dtype=jnp.int32),
        "rank": jnp.full((N,), -1, dtype=jnp.int32),
        "clued": jnp.zeros((N,), dtype=jnp.bool_),
    }


def _empty_memory():
    return init_agent()


def _features(obs):
    return obs.features


def _fireworks(f):
    fw = f[167:192].reshape((N, N))
    return (fw * jnp.arange(1, N + 1, dtype=jnp.float32)).sum(axis=1).astype(jnp.int32)


def _tokens(f):
    return f[192:200].sum().astype(jnp.int32)


def _partner_hand(f):
    h = f[0:125].reshape((N, N, N))
    occ = h.sum(axis=(1, 2)) > 0.5
    flat = jnp.argmax(h.reshape((N, N * N)), axis=1).astype(jnp.int32)
    return occ, flat // N, flat % N


def _knowledge(f, rel_player):
    k = f[308:658].reshape((2, N, 35))[rel_player]
    v0 = k[:, 0:25].reshape((N, N, N))
    color_pos = k[:, 25:30] > 0.5
    rank_pos = k[:, 30:35] > 0.5
    occ = v0.sum(axis=(1, 2)) > 0.01
    clued = jnp.logical_or(color_pos.any(axis=1), rank_pos.any(axis=1))
    return v0, occ, clued


def _last_action(f):
    return f[253:308]


def _discard_counts(f):
    raw = f[203:253].reshape((N, 10))
    return jnp.stack(
        [
            raw[:, 0:3].sum(axis=1),
            raw[:, 3:5].sum(axis=1),
            raw[:, 5:7].sum(axis=1),
            raw[:, 7:9].sum(axis=1),
            raw[:, 9:10].sum(axis=1),
        ],
        axis=1,
    ).astype(jnp.int32)


def _multiplicities():
    return jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)


def _highest(mask):
    idx = jnp.arange(N, dtype=jnp.int32)
    return jnp.argmax(jnp.where(mask, idx, -1)).astype(jnp.int32)


def _lowest(mask):
    idx = jnp.arange(N, dtype=jnp.int32)
    return jnp.argmax(jnp.where(mask, N - 1 - idx, -1)).astype(jnp.int32)


def _choose(mask, scores, fallback):
    has = mask.any()
    action = jnp.argmax(jnp.where(mask, scores, NEG)).astype(jnp.int32)
    return jnp.where(has, action, fallback).astype(jnp.int32), has


def _shift_vec(arr, removed, fill):
    idx = jnp.arange(N, dtype=jnp.int32)
    src = jnp.where(idx < removed, idx, idx + 1)
    shifted = jnp.where(idx < N - 1, arr[jnp.minimum(src, N - 1)], fill)
    return shifted


def _reset_memory(memory, reset):
    empty = _empty_memory()
    return {
        "status": jnp.where(reset, empty["status"], memory["status"]),
        "color": jnp.where(reset, empty["color"], memory["color"]),
        "rank": jnp.where(reset, empty["rank"], memory["rank"]),
        "clued": jnp.where(reset, empty["clued"], memory["clued"]),
    }


def _update_memory(obs, reset, memory):
    f = _features(obs)
    la = _last_action(f)
    v0_self, own_occ, own_clued_now = _knowledge(f, 0)
    heights = _fireworks(f)
    discards = _discard_counts(f)
    mult = _multiplicities()

    mem = _reset_memory(memory, reset)
    actor_self = la[0] > 0.5
    did_remove = jnp.logical_and(actor_self, jnp.logical_or(la[2] > 0.5, la[3] > 0.5))
    removed = jnp.argmax(la[23:28]).astype(jnp.int32)

    shifted = {
        "status": jnp.where(did_remove, _shift_vec(mem["status"], removed, jnp.asarray(0, dtype=jnp.int32)), mem["status"]),
        "color": jnp.where(did_remove, _shift_vec(mem["color"], removed, jnp.asarray(-1, dtype=jnp.int32)), mem["color"]),
        "rank": jnp.where(did_remove, _shift_vec(mem["rank"], removed, jnp.asarray(-1, dtype=jnp.int32)), mem["rank"]),
        "clued": jnp.where(did_remove, _shift_vec(mem["clued"], removed, jnp.asarray(False)), mem["clued"]),
    }

    hint_color = la[4] > 0.5
    hint_rank = la[5] > 0.5
    hint_to_self = jnp.logical_and(jnp.logical_or(hint_color, hint_rank), la[6] > 0.5)
    match = la[18:23] > 0.5
    old_clued = shifted["clued"]
    new = jnp.logical_and(match, jnp.logical_not(old_clued))
    new_any = new.any()
    chop_mask = jnp.logical_and(own_occ, jnp.logical_not(old_clued))
    chop = _lowest(chop_mask)
    focus_retouched = _highest(match)
    focus_new = _highest(new)
    focus = jnp.where(new_any, jnp.where(match[chop], chop, focus_new), focus_retouched)
    valid_focus = match.any()

    clue_color = jnp.argmax(la[8:13]).astype(jnp.int32)
    clue_rank = jnp.argmax(la[13:18]).astype(jnp.int32)
    possible = v0_self[focus] > 0.0
    non5 = jnp.arange(N, dtype=jnp.int32) != 4
    critical = jnp.logical_and(discards >= (mult[None, :] - 1), non5[None, :])
    clue_possible = jnp.where(
        hint_color,
        jnp.logical_and(possible, jnp.arange(N, dtype=jnp.int32)[:, None] == clue_color),
        jnp.logical_and(possible, jnp.arange(N, dtype=jnp.int32)[None, :] == clue_rank),
    )
    critical_possible = jnp.logical_and(clue_possible, critical).any()
    rank_save = jnp.logical_and(hint_rank, jnp.logical_or(clue_rank == 1, clue_rank == 4))
    could_save = jnp.logical_and(match[chop], jnp.logical_or(rank_save, critical_possible))
    new_status = jnp.where(could_save, 2, 1).astype(jnp.int32)
    expected_color = jnp.where(hint_color, clue_color, jnp.asarray(-1, dtype=jnp.int32))
    expected_rank = jnp.where(hint_color, heights[clue_color], clue_rank).astype(jnp.int32)

    slot = jnp.arange(N, dtype=jnp.int32)
    is_focus = slot == focus
    apply_note = jnp.logical_and(hint_to_self, valid_focus)
    status = jnp.where(jnp.logical_and(apply_note, is_focus), new_status, shifted["status"])
    color = jnp.where(jnp.logical_and(apply_note, is_focus), expected_color, shifted["color"])
    rank = jnp.where(jnp.logical_and(apply_note, is_focus), expected_rank, shifted["rank"])
    known_color = color >= 0
    stale_play = jnp.logical_and(status == 1, jnp.logical_and(known_color, rank < heights[jnp.maximum(color, 0)]))
    status = jnp.where(stale_play, jnp.asarray(0, dtype=jnp.int32), status)

    return {"status": status, "color": color, "rank": rank, "clued": own_clued_now}


def _playable_prob(v0, heights):
    ranks = jnp.arange(N, dtype=jnp.int32)
    playable = ranks[None, :] == heights[:, None]
    mass = v0.sum(axis=(1, 2))
    p = (v0 * playable[None, :, :]).sum(axis=(1, 2))
    return jnp.where(mass > 0.01, p / mass, 0.0)


def _trash_prob(v0, heights, discards):
    ranks = jnp.arange(N, dtype=jnp.int32)
    trash = jnp.logical_or(ranks[None, :] < heights[:, None], discards >= _multiplicities()[None, :])
    mass = v0.sum(axis=(1, 2))
    p = (v0 * trash[None, :, :]).sum(axis=(1, 2))
    return jnp.where(mass > 0.01, p / mass, 0.0)


def _focus_for_match(match, clued, occupied):
    new = jnp.logical_and(match, jnp.logical_not(clued))
    new_any = new.any()
    chop = _lowest(jnp.logical_and(occupied, jnp.logical_not(clued)))
    return jnp.where(new_any, jnp.where(match[chop], chop, _highest(new)), _highest(match))


def _clue_scores(obs):
    f = _features(obs)
    legal = obs.legal_actions
    heights = _fireworks(f)
    discards = _discard_counts(f)
    mult = _multiplicities()
    occ, colors, ranks = _partner_hand(f)
    _, _, p_clued = _knowledge(f, 1)

    playable = jnp.logical_and(occ, ranks == heights[colors])
    useful = jnp.logical_and(occ, ranks >= heights[colors])
    idx = jnp.arange(N, dtype=jnp.float32)
    chop_mask = jnp.logical_and(occ, jnp.logical_not(p_clued))
    chop = _lowest(chop_mask)
    chop_occ = chop_mask.any()
    chop_color = colors[chop]
    chop_rank = ranks[chop]
    visible_same = jnp.logical_and(occ, jnp.logical_and(colors == chop_color, ranks == chop_rank)).sum()
    chop_critical = jnp.logical_and(chop_rank != 4, discards[chop_color, chop_rank] >= (mult[chop_rank] - 1))
    chop_danger = jnp.logical_or(chop_rank == 4, jnp.logical_or(jnp.logical_and(chop_rank == 1, visible_same == 1), chop_critical))

    def one_clue(action, is_color, value):
        match = jnp.logical_and(occ, jnp.where(is_color, colors == value, ranks == value))
        new = jnp.logical_and(match, jnp.logical_not(p_clued))
        focus = _focus_for_match(match, p_clued, occ)
        touches = match.sum().astype(jnp.float32)
        new_count = new.sum().astype(jnp.float32)
        focus_chop = jnp.logical_and(chop_occ, focus == chop)
        focus_playable = playable[focus]
        all_good = jnp.logical_not(jnp.logical_and(match, jnp.logical_not(useful))).all()
        focus_critical = jnp.logical_and(ranks[focus] != 4, discards[colors[focus], ranks[focus]] >= (mult[ranks[focus]] - 1))
        rank_save_shape = jnp.logical_and(jnp.logical_not(is_color), jnp.logical_or(value == 1, value == 4))
        ambiguous_save = jnp.logical_and(focus_chop, jnp.logical_or(rank_save_shape, focus_critical))
        play_valid = jnp.logical_and(
            legal[action],
            jnp.logical_and(match.any(), jnp.logical_and(new.any(), jnp.logical_and(focus_playable, jnp.logical_and(all_good, jnp.logical_not(ambiguous_save))))),
        )
        play_score = jnp.where(play_valid, 100.0 + 10.0 * new_count - touches + 0.1 * idx[focus], NEG)

        focus_is_chop = jnp.logical_and(focus_chop, match.any())
        save_rank5 = jnp.logical_and(chop_rank == 4, jnp.logical_and(jnp.logical_not(is_color), value == 4))
        save_rank2 = jnp.logical_and(chop_rank == 1, jnp.logical_and(visible_same == 1, jnp.logical_and(jnp.logical_not(is_color), value == 1)))
        save_critical = jnp.logical_and(chop_critical, jnp.logical_or(jnp.logical_and(is_color, value == chop_color), jnp.logical_and(jnp.logical_not(is_color), value == chop_rank)))
        save_valid = jnp.logical_and(legal[action], jnp.logical_and(chop_occ, jnp.logical_and(chop_danger, jnp.logical_and(focus_is_chop, jnp.logical_or(save_rank5, jnp.logical_or(save_rank2, save_critical))))))
        save_score = jnp.where(save_valid, 200.0 - touches + 0.01 * idx[chop], NEG)
        forced_valid = jnp.logical_and(legal[action], jnp.logical_and(match.any(), jnp.logical_and(focus_playable, all_good)))
        forced_score = jnp.where(forced_valid, 50.0 - touches + 0.1 * idx[focus], NEG)
        return jnp.maximum(play_score, save_score), forced_score

    good_scores = []
    forced_scores = []
    for c in range(N):
        good, forced = one_clue(jnp.asarray(10 + c, dtype=jnp.int32), True, jnp.asarray(c, dtype=jnp.int32))
        good_scores.append(good)
        forced_scores.append(forced)
    for r in range(N):
        good, forced = one_clue(jnp.asarray(15 + r, dtype=jnp.int32), False, jnp.asarray(r, dtype=jnp.int32))
        good_scores.append(good)
        forced_scores.append(forced)

    scores = jnp.full((NUM_ACTIONS,), NEG)
    forced = jnp.full((NUM_ACTIONS,), NEG)
    clue_actions = jnp.arange(10, 20, dtype=jnp.int32)
    scores = scores.at[clue_actions].set(jnp.asarray(good_scores))
    forced = forced.at[clue_actions].set(jnp.asarray(forced_scores))
    return scores, forced


def _discard_action(obs, memory, v0_self, own_occ, own_clued, heights, discards):
    legal = obs.legal_actions
    idx = jnp.arange(N, dtype=jnp.float32)
    trash_p = _trash_prob(v0_self, heights, discards)
    known_trash = jnp.logical_and(own_occ, trash_p > 0.995)
    not_saved = memory["status"] != 2
    chop_mask = jnp.logical_and(own_occ, jnp.logical_and(jnp.logical_not(own_clued), not_saved))
    discard_legal = legal[0:5]
    scores = jnp.where(jnp.logical_and(discard_legal, known_trash), 100.0 + (N - 1) - idx, NEG)
    scores = jnp.maximum(scores, jnp.where(jnp.logical_and(discard_legal, chop_mask), 10.0 + (N - 1) - idx, NEG))
    scores = jnp.maximum(scores, jnp.where(jnp.logical_and(discard_legal, own_occ), idx * 0.01, NEG))
    return _choose(scores > NEG / 2.0, scores, jnp.asarray(0, dtype=jnp.int32))


def agent_act(obs, reset, memory, rng):
    f = _features(obs)
    legal = obs.legal_actions
    memory = _reset_memory(memory, reset)
    memory = _update_memory(obs, reset, memory)

    inactive = legal[20]
    heights = _fireworks(f)
    discards = _discard_counts(f)
    v0_self, own_occ, own_clued = _knowledge(f, 0)
    idx = jnp.arange(N, dtype=jnp.float32)

    p_play = _playable_prob(v0_self, heights)
    known_color = memory["color"] >= 0
    color_note_due = memory["rank"] == heights[jnp.maximum(memory["color"], 0)]
    note_due = jnp.logical_or(jnp.logical_not(known_color), color_note_due)
    note_play = jnp.logical_and(memory["status"] == 1, jnp.logical_and(own_occ, note_due))
    certain_play = jnp.logical_and(p_play > 0.995, own_occ)
    play_mask = jnp.logical_and(legal[5:10], jnp.logical_or(note_play, certain_play))
    play_scores = jnp.where(play_mask, idx + 10.0 * note_play.astype(jnp.float32) + p_play, NEG)
    play_slot, has_play = _choose(play_mask, play_scores, jnp.asarray(0, dtype=jnp.int32))
    play_action = play_slot + 5

    clue_scores, forced_clues = _clue_scores(obs)
    clue_action, has_clue = _choose(clue_scores > 0.0, clue_scores, jnp.asarray(10, dtype=jnp.int32))
    discard_action, has_discard = _discard_action(obs, memory, v0_self, own_occ, own_clued, heights, discards)
    forced_action, has_forced = _choose(forced_clues > NEG / 2.0, forced_clues, jnp.asarray(10, dtype=jnp.int32))

    risky_scores = jnp.where(jnp.logical_and(legal[5:10], own_occ), p_play + idx * 0.001, NEG)
    risky_slot, has_risky = _choose(risky_scores > NEG / 2.0, risky_scores, jnp.asarray(0, dtype=jnp.int32))
    risky_action = risky_slot + 5
    first_legal = jnp.argmax(legal.astype(jnp.int32)).astype(jnp.int32)

    action = jnp.where(has_risky, risky_action, first_legal)
    action = jnp.where(has_forced, forced_action, action)
    action = jnp.where(has_discard, discard_action, action)
    action = jnp.where(has_clue, clue_action, action)
    action = jnp.where(has_play, play_action, action)
    action = jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), action.astype(jnp.int32))

    trace = {
        "has_play": has_play,
        "has_clue": has_clue,
        "has_discard": has_discard,
        "p_play_max": p_play.max(),
        "info_tokens": _tokens(f),
    }
    return action, memory, trace
