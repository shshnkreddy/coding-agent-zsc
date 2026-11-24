"""H-group Level 1 convention policy for two-player Hanabi (pure JAX, fixed shape).

Shared protocol (documented Level 1 meanings only):
  * chop = lowest occupied native index among unclued cards; finesse position unused (no 2p finesses)
  * clue focus: chop if newly touched, else leftmost (highest index) new card, else leftmost re-touched
  * non-chop focus = play clue (possibly delayed through cards already "gotten"); chop focus = play or save
  * saves: 5 with rank-5 clue, 2 with rank-2 clue, critical (non-5) with colour or rank
  * Good Touch: non-focus touched cards are noted as "useful, will play later"
  * self-prompt (documented rank-4 example): if a play clue cannot be direct, the leftmost compatible
    clued card is the connector
  * discard: known trash first, else chop
Private memory holds card notes (possibility masks) and play flags for both hands, derived only from
public history, so the same interpretation routine is used to read partner clues and to predict how
the partner will read mine.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_level1_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2,
                   'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21,
                   'policy_structure': 'shared_local'}

NS = 5
NI = 25


def consts():
    idx = jnp.arange(NI, dtype=jnp.int32)
    idr = idx % 5
    mult = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    return dict(IDX=idx, IDC=idx // 5, IDR=idr, MULT_ID=mult[idr], SLOT=jnp.arange(NS, dtype=jnp.int32),
                RANKS=jnp.arange(5, dtype=jnp.int32), NEG=jnp.float32(-1e9))


def leftmost(mask):
    """Highest index with True (H-group leftmost)."""
    return (NS - 1 - jnp.argmax(mask[::-1])).astype(jnp.int32)


def rightmost(mask):
    """Lowest index with True (H-group rightmost)."""
    return jnp.argmax(mask).astype(jnp.int32)


def init_agent():
    return {
        'note': jnp.ones((2, NS, NI), dtype=jnp.bool_),
        'play': jnp.zeros((2, NS), dtype=jnp.bool_),
        'clued': jnp.zeros((2, NS), dtype=jnp.bool_),
        'deck': jnp.asarray(40, dtype=jnp.int32),
        'endgame': jnp.asarray(0, dtype=jnp.int32),   # 0 none, 1 I drew the last card, 2 partner did
    }


# ----------------------------------------------------------------------------- observation decoding

def decode(f, c):
    IDX, IDC, IDR, MULT_ID, SLOT, RANKS, NEG = c['IDX'], c['IDC'], c['IDR'], c['MULT_ID'], c['SLOT'], c['RANKS'], c['NEG']
    ph = f[0:125].reshape(NS, NI) > 0.5
    p_occ = ph.any(1)
    p_id = jnp.argmax(ph, axis=1).astype(jnp.int32)
    p_count = ph.sum(0).astype(jnp.int32)
    deck = jnp.sum(f[127:167])
    fw = f[167:192].reshape(5, 5)
    height = jnp.sum(fw * (RANKS[None, :] + 1), axis=1).astype(jnp.int32)
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    d = f[203:253].reshape(5, 10)
    disc = jnp.stack([d[:, 0:3].sum(1), d[:, 3:5].sum(1), d[:, 5:7].sum(1), d[:, 7:9].sum(1), d[:, 9:10].sum(1)],
                     axis=1).astype(jnp.int32)
    remaining = MULT_ID - disc.reshape(NI)
    id_h = height[IDC]
    playable_now = IDR == id_h
    blocked = ((IDR >= id_h) & (remaining <= 0)).reshape(5, 5)
    reach = jnp.min(jnp.where(blocked, RANKS[None, :], 5), axis=1)
    useful = (IDR >= id_h) & (IDR < reach[IDC])
    critical = useful & (remaining == 1)
    know = f[308:658].reshape(2, NS, 35)
    v0 = know[:, :, :NI]
    lit = v0 > 0
    occ = v0.sum(-1) > 0
    clued = know[:, :, NI:].sum(-1) > 0
    cnt = jnp.maximum(remaining - (IDR < id_h).astype(jnp.int32) - p_count, 0)
    return dict(ph=ph, p_occ=p_occ, p_id=p_id, deck=deck, height=height, tokens=tokens, lives=lives,
                playable_now=playable_now, useful=useful, critical=critical, lit=lit, occ=occ, clued=clued,
                cnt=cnt, la=f[253:308])


# ----------------------------------------------------------------------------- convention inference

def delayed_from_gotten(gotten, height, useful, c):
    """gotten: [n,25] -> (delayed-playable [n,25], first-not-gotten rank per colour [n,5])."""
    IDX, IDC, IDR, MULT_ID, SLOT, RANKS, NEG = c['IDX'], c['IDC'], c['IDR'], c['MULT_ID'], c['SLOT'], c['RANKS'], c['NEG']
    g = gotten.reshape((-1, 5, 5))
    ok = (RANKS[None, None, :] < height[None, :, None]) | g
    fno = jnp.min(jnp.where(~ok, RANKS[None, None, :], 5), axis=-1).astype(jnp.int32)
    dp = (IDR[None, :] >= height[IDC][None, :]) & (IDR[None, :] <= fno[:, IDC]) & useful[None, :]
    return dp, fno


def dp_for_hand(h, note, play, clued, b, c):
    """Delayed-playable sets for hand h as clue recipient, excluding each slot's own contribution."""
    lit, occ = b['lit'], b['occ']
    base_h = note[h] & lit[h]
    if h == 0:
        # partner (giver) cards count as gotten only if their public note promises/identifies them and it is true
        base1 = note[1] & lit[1]
        certain1 = base1.sum(1) == 1
        giver = ((b['p_occ'] & clued[1] & (play[1] | certain1))[:, None] & base1 & b['ph']).any(0)
    else:
        # I am the giver: only cards I know for certain (strict, so the partner's true reading is at least as wide)
        base0 = note[0] & lit[0]
        certain0 = base0.sum(1) == 1
        giver = ((occ[0] & clued[0] & certain0)[:, None] & base0).any(0)
    certain = base_h.sum(1) == 1
    contrib = ((occ[h] & (play[h] | (clued[h] & certain)))[:, None] & base_h).astype(jnp.int32)
    total = contrib.sum(0)
    excl = ((total[None, :] - contrib) > 0) | giver[None, :]
    dp, fno = delayed_from_gotten(excl, b['height'], b['useful'], c)
    contrib_s = ((occ[h] & clued[h] & certain)[:, None] & base_h).astype(jnp.int32)
    total_s = contrib_s.sum(0)
    excl_s = ((total_s[None, :] - contrib_s) > 0) | giver[None, :]
    dps, _ = delayed_from_gotten(excl_s, b['height'], b['useful'], c)
    return dp, fno, dps


def giver_certain_ids(h, note, play, clued, b):
    """Identities the clue giver (the other hand) holds as certainly-known clued cards (public inference)."""
    lit, occ = b['lit'], b['occ']
    g = 1 - h
    base_g = note[g] & lit[g]
    certain_g = base_g.sum(1) == 1
    return ((occ[g] & clued[g] & certain_g)[:, None] & base_g).any(0)


def interpret(matched, is_rank, value, pre_clued, occ_h, lit_h, note_h, play_h, dp_excl, fno_excl, useful, critical, dps_excl, c,
              allow_prompt=True):
    """Level 1 reading of one clue to hand h. Returns (note, play, info)."""
    IDX, IDC, IDR, MULT_ID, SLOT, RANKS, NEG = c['IDX'], c['IDC'], c['IDR'], c['MULT_ID'], c['SLOT'], c['RANKS'], c['NEG']
    new = matched & ~pre_clued & occ_h
    any_new = new.any()
    unclued = occ_h & ~pre_clued
    has_chop = unclued.any()
    chop = rightmost(unclued)
    chop_touched = has_chop & matched[chop]
    is_chop_focus = any_new & chop_touched
    focus = jnp.where(~any_new, leftmost(matched), jnp.where(chop_touched, chop, leftmost(new)))
    base = note_h & lit_h
    base = jnp.where(base.any(1, keepdims=True), base, lit_h)
    L = base[focus]
    dp = dp_excl[focus]
    dps = dps_excl[focus]
    fno = fno_excl[focus]
    save_set = jnp.where(is_rank & (value == 4), IDR == 4,
                         jnp.where(is_rank & (value == 1), IDR == 1, critical & (IDR != 4)))
    save_set = save_set & useful & is_chop_focus
    cand = L & (dp | save_set)
    # self-prompt: exactly one missing connector held as a clued, not-yet-promised card
    conn_rank = fno[IDC]
    needs_one = (IDR == conn_rank + 1) & useful
    conn_id = IDC * 5 + jnp.maximum(IDR - 1, 0)
    elig = occ_h & pre_clued & ~play_h & (SLOT != focus)
    conn_avail = (elig[:, None] & base[:, conn_id]).any(0)
    pcand = L & needs_one & conn_avail
    use_prompt = (~cand.any()) & pcand.any() & allow_prompt
    conn_set = jnp.zeros(NI, jnp.int32).at[conn_id].max(pcand.astype(jnp.int32)) > 0
    compat = elig & (base & conn_set[None, :]).any(1)
    prompted = leftmost(compat)
    has_prompt = use_prompt & compat.any()
    # uninterpretable clue (partner error or forced clue): keep literal information only, no promise
    fnote = jnp.where(cand.any(), cand, jnp.where(use_prompt, pcand, L))
    # rank-2 / rank-5 chop clues are saves by definition; other chop clues are held while a save reading remains
    # that is not strictly (certain connectors only) delayed-playable
    is_save_rank = is_chop_focus & is_rank & ((value == 1) | (value == 4))
    is_play = cand.any() & ~is_save_rank & ~(cand & save_set & ~dps).any()
    fplay = is_play | use_prompt
    # Good Touch applies to newly touched non-focus cards at clue time only (re-touched cards keep their notes);
    # a clue that may be a save is allowed to violate Good Touch, so no narrowing is inferred from it
    others = new & (SLOT != focus)
    onote = base & useful[None, :]
    onote = jnp.where(onote.any(1, keepdims=True), onote, base)
    onote = jnp.where((cand & save_set).any(), base, onote)
    new_note = jnp.where(others[:, None], onote, note_h)
    new_note = new_note.at[focus].set(fnote)
    new_play = play_h.at[focus].set(fplay | play_h[focus])
    pnote = base[prompted] & conn_set
    new_note = jnp.where(has_prompt, new_note.at[prompted].set(pnote), new_note)
    new_play = jnp.where(has_prompt, new_play.at[prompted].set(True), new_play)
    info = dict(focus=focus, is_chop_focus=is_chop_focus, is_play=is_play, cand_any=cand.any(),
                use_prompt=use_prompt, any_new=any_new)
    return new_note, new_play, info


def refilter(note_h, play_h, lit_h, occ_h, clued_h, dp_excl, useful):
    """Literal info always applies; only promised (play-flagged) cards keep narrowing to what can still play."""
    base = note_h & lit_h
    base = jnp.where(base.any(1, keepdims=True), base, lit_h)
    pb = base & dp_excl
    play_ok = play_h & pb.any(1) & clued_h & occ_h
    nn = jnp.where(play_ok[:, None], pb, base)
    nn = jnp.where((clued_h & occ_h)[:, None], nn, jnp.ones_like(nn))
    return nn, play_ok


def shift_hand(arr, slot, fresh, c):
    SLOT = c['SLOT']
    src = SLOT + (SLOT >= slot).astype(jnp.int32)
    padded = jnp.concatenate([arr, fresh[None]], axis=0)
    return padded[src]


def update_memory(mem, b, c):
    la = b['la']
    actor_me = la[0] > 0.5
    removal = (la[2] > 0.5) | (la[3] > 0.5)
    is_hint = (la[4] > 0.5) | (la[5] > 0.5)
    is_rank = la[5] > 0.5
    rem_slot = jnp.argmax(la[23:28]).astype(jnp.int32)
    recip_me = la[6] > 0.5
    value = jnp.where(is_rank, jnp.argmax(la[13:18]), jnp.argmax(la[8:13])).astype(jnp.int32)
    matched = la[18:23] > 0.5

    note, play, clued_prev = mem['note'], mem['play'], mem['clued']
    fresh_note = jnp.ones((NI,), dtype=jnp.bool_)
    fresh_flag = jnp.asarray(False)
    for h in range(2):
        cond = removal & (actor_me == (h == 0))
        note = note.at[h].set(jnp.where(cond, shift_hand(note[h], rem_slot, fresh_note, c), note[h]))
        play = play.at[h].set(jnp.where(cond, shift_hand(play[h], rem_slot, fresh_flag, c), play[h]))
        clued_prev = clued_prev.at[h].set(jnp.where(cond, shift_hand(clued_prev[h], rem_slot, fresh_flag, c),
                                                    clued_prev[h]))
    for h in range(2):
        cond = is_hint & (recip_me == (h == 0))
        g = 1 - h
        # a prompt is only read when the giver was not plausibly forced (had < 8 tokens and a card to discard)
        giver_free = (b['tokens'] < 7) & (b['occ'][g] & ~clued_prev[g]).any()
        dp_excl, fno_excl, dps_excl = dp_for_hand(h, note, play, clued_prev, b, c)
        nn, npl, _ = interpret(matched, is_rank, value, clued_prev[h], b['occ'][h], b['lit'][h], note[h], play[h],
                               dp_excl, fno_excl, b['useful'], b['critical'], dps_excl, c, allow_prompt=giver_free)
        note = note.at[h].set(jnp.where(cond, nn, note[h]))
        play = play.at[h].set(jnp.where(cond, npl, play[h]))
    for h in range(2):
        dp_excl, _, _ = dp_for_hand(h, note, play, b['clued'], b, c)
        nn, npl = refilter(note[h], play[h], b['lit'][h], b['occ'][h], b['clued'][h], dp_excl, b['useful'])
        note = note.at[h].set(nn)
        play = play.at[h].set(npl)
    deck_now = b['deck'].astype(jnp.int32)
    last_draw = removal & (mem['deck'] > 0) & (deck_now == 0)
    endgame = jnp.where(last_draw, jnp.where(actor_me, 1, 2), mem['endgame']).astype(jnp.int32)
    return {'note': note, 'play': play, 'clued': b['clued'], 'deck': deck_now, 'endgame': endgame}


# ----------------------------------------------------------------------------- decision

def decide(mem, b, legal, c):
    IDX, IDC, IDR, MULT_ID, SLOT, RANKS, NEG = c['IDX'], c['IDC'], c['IDR'], c['MULT_ID'], c['SLOT'], c['RANKS'], c['NEG']
    note, play, clued = mem['note'], mem['play'], mem['clued']
    lit, occ = b['lit'], b['occ']
    ph, p_occ, p_id = b['ph'], b['p_occ'], b['p_id']
    useful, critical, playable_now = b['useful'], b['critical'], b['playable_now']
    height, tokens, lives = b['height'], b['tokens'], b['lives']
    occ0, clued0, occ1, clued1 = occ[0], clued[0], p_occ, clued[1]

    # ---- my hand
    base0 = note[0] & lit[0]
    base0 = jnp.where(base0.any(1, keepdims=True), base0, lit[0])
    certain0 = base0.sum(1) == 1
    w = base0.astype(jnp.float32) * b['cnt'].astype(jnp.float32)[None, :]
    wsum = w.sum(1)
    w = jnp.where((wsum > 0)[:, None], w, base0.astype(jnp.float32))
    wsum = jnp.maximum(w.sum(1), 1e-6)
    p_play = (w * playable_now[None, :]).sum(1) / wsum
    p_useful = (w * useful[None, :]).sum(1) / wsum
    certain_play = occ0 & (p_play > 0.999)
    trash0 = occ0 & (p_useful < 1e-6)
    minrank = jnp.min(jnp.where(base0, IDR[None, :], 9), axis=1)
    play_prio = jnp.where(certain_play, 100.0 - 10.0 * minrank - 0.1 * SLOT + 5.0 * play[0], -1.0)
    own_play_slot = jnp.argmax(play_prio).astype(jnp.int32)
    has_own_play = certain_play.any()

    # ---- partner hand model
    base1 = note[1] & lit[1]
    base1 = jnp.where(base1.any(1, keepdims=True), base1, lit[1])
    certain1 = base1.sum(1) == 1
    p_known_play = occ1 & base1.any(1) & ~(base1 & ~playable_now[None, :]).any(1)
    partner_will_play = p_known_play.any()
    unclued1 = occ1 & ~clued1
    has_chop1 = unclued1.any()
    chop1 = rightmost(unclued1)
    T_chop = p_id[chop1]
    my_certain_ids = ((occ0 & clued0 & (certain0 | play[0]))[:, None] & base0).any(0)
    # for each partner slot: is its identity already clued elsewhere (partner other slots, or my certain cards)
    clued_ids_partner = ((occ1 & clued1)[:, None] & ph).astype(jnp.int32).sum(0)
    dup_pre = (clued_ids_partner[p_id] - (clued1 & occ1).astype(jnp.int32) > 0) | my_certain_ids[p_id]
    copies_elsewhere = (ph[:, T_chop] & (SLOT != chop1)).any()
    r_chop = IDR[T_chop]
    needs_save = has_chop1 & useful[T_chop] & ~dup_pre[chop1] & (
        (r_chop == 4) | critical[T_chop] | ((r_chop == 1) & ~copies_elsewhere) |
        (playable_now[T_chop] & ~copies_elsewhere))

    # ---- strict "gotten" for validating my own play clues (exclude each partner slot)
    contrib_p = ((occ1 & clued1 & (play[1] | certain1))[:, None] & ph & base1).astype(jnp.int32)
    total_p = contrib_p.sum(0)
    mine = ((occ0 & (play[0] | clued0) & certain0)[:, None] & base0).any(0)
    excl_strict = ((total_p[None, :] - contrib_p) > 0) | mine[None, :]
    dp_strict, _ = delayed_from_gotten(excl_strict, height, useful, c)
    dp_excl1, fno_excl1, dps_excl1 = dp_for_hand(1, note, play, clued, b, c)

    def eval_clue(k):
        is_rank = k >= 5
        value = k % 5
        attr_id = jnp.where(is_rank, IDR, IDC)
        attr = attr_id[p_id]
        matched = occ1 & (attr == value)
        lit_after = lit[1] & jnp.where(matched[:, None], attr_id[None, :] == value, attr_id[None, :] != value)
        nn, npl, info = interpret(matched, is_rank, value, clued1, occ1, lit_after, note[1], play[1],
                                  dp_excl1, fno_excl1, useful, critical, dps_excl1, c)
        fcs = info['focus']
        T = p_id[fcs]
        new = matched & ~clued1 & occ1
        valid_play = dp_strict[fcs, T] & useful[T] & info['any_new'] & ~info['use_prompt']
        pn = nn[fcs] & lit_after[fcs]
        partner_immediate = pn.any() & ~(pn & ~playable_now[None, :]).any()
        partner_immediate = partner_immediate & (npl[fcs] | info['cand_any'])
        misread = ~pn[T]
        others_new = new & (SLOT != fcs)
        on = nn & lit_after
        other_known_play = others_new & on.any(1) & ~(on & ~playable_now[None, :]).any(1) & playable_now[p_id]
        n_bonus_play = other_known_play.sum()
        count_new = (new[:, None] & ph).astype(jnp.int32).sum(0)
        n_dup_within = jnp.maximum(count_new - 1, 0).sum()
        bad_other = others_new & (~useful[p_id] | dup_pre)
        n_bad = bad_other.sum() + n_dup_within
        n_extra = jnp.maximum((others_new & useful[p_id] & ~dup_pre).sum() - n_dup_within, 0)
        valid_play = valid_play & (n_bad == 0) & ~dup_pre[fcs]
        r_T = IDR[T]
        other_copy = (ph[:, T] & (SLOT != fcs)).any()
        save_ok = info['is_chop_focus'] & useful[T] & ~dup_pre[fcs] & (
            (is_rank & (value == 4) & (r_T == 4)) |
            (is_rank & (value == 1) & (r_T == 1) & ~other_copy) |
            (critical[T] & (r_T != 4)))
        play_score = (20.0 + 10.0 * partner_immediate + 2.0 * n_extra + 6.0 * n_bonus_play + 0.5 * (~is_rank)
                      + 1.0 * info['is_chop_focus']
                      + 0.3 * (4 - r_T) - 30.0 * (n_bad + dup_pre[fcs]) - 100.0 * misread)
        save_score = 10.0 + 2.0 * n_extra - 15.0 * n_bad - 100.0 * misread
        tempo_ok = (~info['any_new']) & partner_immediate & dp_strict[fcs, T] & ~dup_pre[fcs]
        retouched = matched & clued1 & occ1
        known_after = retouched & on.any(1) & ~(on & ~playable_now[None, :]).any(1) & playable_now[p_id]
        tempo_score = jnp.where(tempo_ok, 5.0 + 3.0 * known_after.sum(), NEG)
        clog = new.sum() * ~(valid_play | save_ok)
        forced_score = (-30.0 * (n_bad + dup_pre[fcs]) - 100.0 * misread + 5.0 * tempo_ok
                        - 40.0 * ~(valid_play | save_ok | tempo_ok) - 4.0 * clog - 100.0 * info['use_prompt'])
        return dict(valid_play=valid_play, play_score=play_score, save_ok=save_ok, save_score=save_score,
                    forced_score=forced_score, imm=partner_immediate, tempo_score=tempo_score)

    ev = jax.vmap(eval_clue)(jnp.arange(10, dtype=jnp.int32))
    clue_legal = legal[10:20]
    ps = jnp.where(clue_legal & ev['valid_play'], ev['play_score'], NEG)
    best_play_k = jnp.argmax(ps).astype(jnp.int32)
    has_play_clue = ps[best_play_k] > -1e8
    imm = ev['imm'][best_play_k]
    ss = jnp.where(clue_legal & ev['save_ok'], ev['save_score'], NEG)
    best_save_k = jnp.argmax(ss).astype(jnp.int32)
    has_save_clue = ss[best_save_k] > -1e8
    fs = jnp.where(clue_legal, ev['forced_score'], NEG)
    best_forced_k = jnp.argmax(fs).astype(jnp.int32)
    forced_ok = fs[best_forced_k] >= 0.0
    safe_forced = fs[best_forced_k] >= -65.0
    ts = jnp.where(clue_legal, ev['tempo_score'], NEG)
    best_tempo_k = jnp.argmax(ts).astype(jnp.int32)
    has_tempo = ts[best_tempo_k] > 0.0

    # ---- discard choice
    unclued0 = occ0 & ~clued0
    has_chop0 = unclued0.any()
    chop0 = rightmost(unclued0)
    trash_slot = rightmost(trash0)
    p_crit = (w * critical[None, :]).sum(1) / wsum
    low_val = jnp.where(occ0, p_useful + 5.0 * p_crit + 10.0 * play[0] + 0.01 * SLOT, 1e9)
    low_slot = jnp.argmin(low_val).astype(jnp.int32)
    all_clued_risky = ~trash0.any() & ~has_chop0 & (p_crit[low_slot] >= 0.5)
    disc_slot = jnp.where(trash0.any(), trash_slot, jnp.where(has_chop0, chop0, low_slot))
    can_discard = legal[0:5].any()

    # ---- risky play (only when forced)
    risky_slot = jnp.argmax(jnp.where(occ0, p_play, -1.0)).astype(jnp.int32)
    # forced situation only: a misleading clue is a near-certain partner bomb, so a gamble is preferred when lives allow
    risky_thr = jnp.where(lives >= 3, 0.35, jnp.where(lives >= 2, 0.5, 1.5))
    risky_ok = p_play[risky_slot] >= risky_thr

    can_clue = clue_legal.any()
    urgent = needs_save & ~partner_will_play & can_clue
    last_turn = mem['endgame'] == 1
    second_last = mem['endgame'] == 2
    gamble_slot = risky_slot
    gamble_ok = (lives >= 2) & (p_play[gamble_slot] > 0.0) & occ0[gamble_slot]

    # priority ladder (first true wins)
    conds = jnp.stack([
        urgent & has_play_clue & imm & ~last_turn,                     # 1 urgent: immediate play clue
        urgent & has_save_clue & ~last_turn & ~second_last,            # 2 urgent: save clue
        has_own_play,                                                   # 3 play own certain card
        last_turn & gamble_ok,                                          # 3b last turn: free gamble
        second_last & partner_will_play & gamble_ok,                    # 3c partner already set: gamble
        has_play_clue & ~last_turn & (~second_last | imm),             # 4 play clue (must be immediate on the second-last turn)
        second_last & has_tempo & ~partner_will_play,                   # 4a second-last: fill-in guarantees a partner play
        second_last & gamble_ok,                                        # 4b second-last turn gamble
        needs_save & has_save_clue & ~last_turn & ~second_last,        # 5 save clue
        has_tempo & ~partner_will_play & (tokens >= 2) & ~last_turn,    # 5b fill-in (tempo) clue instead of discarding
        can_discard & ~all_clued_risky,                                 # 6 discard
        forced_ok,                                                      # 6b harmless clue before risky discard
        safe_forced,                                                    # 6c safe clue before risky discard
        can_discard,                                                    # 6d discard anyway
        forced_ok,                                                      # 7 harmless forced clue
        safe_forced,                                                    # 7b non-misleading forced clue
        risky_ok,                                                       # 8 risky play
        jnp.asarray(True),                                              # 9 least-bad clue
    ])
    acts = jnp.stack([
        10 + best_play_k, 10 + best_save_k, 5 + own_play_slot, 5 + gamble_slot, 5 + gamble_slot,
        10 + best_play_k, 10 + best_tempo_k, 5 + gamble_slot, 10 + best_save_k, 10 + best_tempo_k,
        disc_slot, 10 + best_forced_k, 10 + best_forced_k, disc_slot, 10 + best_forced_k, 10 + best_forced_k,
        5 + risky_slot, 10 + best_forced_k,
    ]).astype(jnp.int32)
    idx = jnp.argmax(conds).astype(jnp.int32)
    action = acts[idx]
    reason_map = jnp.asarray([1, 2, 3, 14, 15, 4, 19, 16, 5, 18, 6, 11, 17, 6, 12, 13, 8, 9], dtype=jnp.int32)
    reason = reason_map[idx]
    reason = jnp.where(reason == 6, jnp.where(trash0.any(), 6, jnp.where(has_chop0, 7, 10)), reason)
    chosen = jnp.clip(action - 5, 0, 4)
    la = b['la']
    fail = (la[0] > 0.5) & (la[2] > 0.5) & (la[53] < 0.5)
    trace = {'reason': reason, 'p_play': p_play.astype(jnp.float32), 'needs_save': needs_save,
             'partner_will_play': partner_will_play, 'lives': lives.astype(jnp.int32),
             'score': height.sum().astype(jnp.int32), 'deck': b['deck'].astype(jnp.int32),
             'tokens': tokens.astype(jnp.int32), 'fail': fail,
             'chosen_p': p_play[chosen], 'chosen_flag': play[0][chosen],
             'chosen_n': base0[chosen].sum().astype(jnp.int32), 'chosen_clued': clued0[chosen],
             'chosen_ids': jnp.where(base0[chosen], IDX, -1).astype(jnp.int32),
             'forced_scores': fs.astype(jnp.float32), 'play_scores': ps.astype(jnp.float32),
             'height': height.astype(jnp.int32)}
    return action, trace


def agent_act(obs, reset, memory, rng):
    f = obs.features.astype(jnp.float32)
    legal = obs.legal_actions.astype(jnp.bool_)
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, m: jnp.where(reset, a, m), init, memory)
    c = consts()
    b = decode(f, c)
    mem = update_memory(memory, b, c)
    action, trace = decide(mem, b, legal, c)
    action = jnp.where(legal[20], jnp.int32(20), action).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)
    return action, mem, trace
