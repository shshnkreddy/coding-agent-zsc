"""Overcooked V2 demo_cook_simple.

Roles derived from the fixed geometry:
  * agent_0 lives in the left room. It can see the always-on recipe indicator
    and the pot, and has (far) supplies of ingredient 0 and 1. It fetches the
    displayed recipe ingredient and puts it into the pot, repeating while the
    pot still needs ingredients. Wrong-type items are parked on the row-0/row-4
    counters closest to the pot and re-used later; while the pot cooks it keeps
    one unit of the other ingredient parked there as a hedge against a recipe
    change, so a change costs a short swap instead of a far supply trip.
  * agent_1 lives in the right room. It can never see the recipe indicator, but
    it always sees the pot, both nearby supplies, the plate piles and the
    delivery station. It infers the recipe from what the partner put into the
    pot (copy-the-pot), fills the remaining slots, plates the cooked dish and
    delivers pure dishes; mixed dishes are parked on counters instead of being
    delivered (wrong deliveries cost -20).

Everything is computed from the local observation with fixed-shape jnp.where
control flow; memory holds only a few remembered scalars.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'left_cook_right_copy_server',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------------------
# Constants (only literals at module scope; arrays are built inside functions)
# ---------------------------------------------------------------------------
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3

H, W = 5, 11
LAYOUT = [
    "WWWWWR2W0WW",
    "0      W  B",
    "W     APA X",
    "1      W  B",
    "WWWWWR2W1WW",
]
# static channel order inside obs[..., 20:29]
# 0 wall, 1 goal, 2 pot, 3 recipe indicator, 4 button, 5 plate pile, 6/7/8 ing supply
STATIC_CODE = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8}

A0_CELLS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
A1_CELLS = [(1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9)]

PATIENCE_LIMIT = 25   # steps an idle incompatible pot is tolerated before forcing
EMPTY_WAIT = 4        # steps after last dish sighting before seeding an empty pot
HEDGE_TIMER = 10      # minimum pot timer for starting hedge stocking
STAGE_TIMER = 12      # minimum pot timer for staging an extra recipe unit


def _templates(cells):
    """Build [N,5,5,9] static templates for candidate self positions (zero padded)."""
    rows = []
    for r, c in cells:
        grid = []
        for i in range(5):
            row = []
            for j in range(5):
                gr, gc = r + i - 2, c + j - 2
                vec = [0] * 9
                if 0 <= gr < H and 0 <= gc < W:
                    ch = LAYOUT[gr][gc]
                    if ch in STATIC_CODE:
                        vec[STATIC_CODE[ch]] = 1
                row.append(vec)
            grid.append(row)
        rows.append(grid)
    return jnp.asarray(rows, dtype=jnp.int32)


def _consts():
    k = {}
    k['a0_tpl'] = _templates(A0_CELLS)
    k['a0_rc'] = jnp.asarray(A0_CELLS, dtype=jnp.int32)
    k['a1_tpl'] = _templates(A1_CELLS)
    k['a1_rc'] = jnp.asarray(A1_CELLS, dtype=jnp.int32)
    # agent_0 stash counters: row 0 cols 1..4 (ing 0 side), row 4 cols 1..4 (ing 1 side)
    k['stash_r'] = jnp.asarray([0, 0, 0, 0, 4, 4, 4, 4], dtype=jnp.int32)
    k['stash_c'] = jnp.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=jnp.int32)
    k['cols14'] = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    # agent_1 counters and how to access them
    k['cnt_r'] = jnp.asarray([0, 4, 1, 3], dtype=jnp.int32)
    k['cnt_c'] = jnp.asarray([9, 9, 7, 7], dtype=jnp.int32)
    k['cnt_tr'] = jnp.asarray([1, 3, 1, 3], dtype=jnp.int32)
    k['cnt_tc'] = jnp.asarray([9, 9, 8, 8], dtype=jnp.int32)
    k['cnt_td'] = jnp.asarray([D_UP, D_DOWN, D_LEFT, D_LEFT], dtype=jnp.int32)
    return k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _localize(obs, tpl, rc):
    static = obs[:, :, 20:29]
    diff = jnp.abs(tpl - static[None]).sum(axis=(1, 2, 3))
    i = jnp.argmin(diff)
    return rc[i, 0], rc[i, 1]


def _cells_view(obs, r, c, grs, gcs):
    """Return [N, 40] feature vectors for global cells (zeros if not in view)."""
    lr = grs - r + 2
    lc = gcs - c + 2
    valid = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    v = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    return jnp.where(valid[:, None], v, 0), valid


def _cell_view(obs, r, c, gr, gc):
    v, valid = _cells_view(obs, r, c, _i32(jnp.asarray([gr])), _i32(jnp.asarray([gc])))
    return v[0], valid[0]


def _nav(r, c, f, tr, tc, td, final):
    """Move vertically, then horizontally, then turn to td, then do `final`."""
    dr = tr - r
    dc = tc - c
    vert = jnp.where(dr > 0, A_DOWN, A_UP)
    horz = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    turn = jnp.asarray([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)[td]
    act = jnp.where(dr != 0, vert, jnp.where(dc != 0, horz, jnp.where(f != td, turn, final)))
    return _i32(act)


def _item_code(block):
    """Encode an item block [plate, cooked, i0, i1, i2] -> 0 empty, 1 raw ing0, 2 raw ing1, 3 other."""
    total = block[2:5].sum()
    empty = (block[0] == 0) & (block[1] == 0) & (total == 0)
    raw = (block[0] == 0) & (block[1] == 0) & (total == 1)
    raw0 = raw & (block[2] == 1)
    raw1 = raw & (block[3] == 1)
    return jnp.where(empty, 0, jnp.where(raw0, 1, jnp.where(raw1, 2, 3)))


def _raw_type(block):
    """Raw ingredient id of an item block, or -1."""
    total = block[2:5].sum()
    raw = (block[0] == 0) & (block[1] == 0) & (total == 1)
    return jnp.where(raw, jnp.argmax(block[2:5]), -1)


# ---------------------------------------------------------------------------
# agent_0: left-room cook
# memory layout (int32[24]):
#  0 recipe (-1 unknown, else ingredient id)
#  1..4 codes of row-0 counters cols 1..4;  5..8 codes of row-4 counters cols 1..4
#  9 patience counter for an idle incompatible pot
#  10..12 pot contents, 13 pot cooked flag, 14 pot timer (last seen)
#  15 steps since pot last seen non-empty or partner last seen with a dish
#  16 recipe displayed when the pot was last seen becoming empty
#  17 hedge mode (1 while parking the held recipe item to stock the hedge)
#  18 partner_fills (1 once the pot was seen full with fewer than 3 own puts)
#  19 own puts into the current pot batch
#  20 previously held raw type (-1 none)
#  21 stage mode (0 off, 1 fetch+park extra recipe unit, 2 fetch+hold)
#  22..23 unused
# ---------------------------------------------------------------------------
MEM0_SIZE = 24


def init_agent_0():
    m = [0] * MEM0_SIZE
    m[0] = -1
    m[15] = 1000   # no delivery pending at reset
    m[20] = -1
    return jnp.asarray(m, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    K = _consts()
    mem = jnp.where(reset, init_agent_0(), memory)
    r, c = _localize(obs, K['a0_tpl'], K['a0_rc'])
    me = obs[2, 2]
    f = jnp.argmax(me[1:5])
    h_plate = me[5] > 0
    h_cooked = me[6] > 0
    h_ing = me[7:10]
    h_total = h_ing.sum()
    h_type = jnp.where((h_total > 0) & ~h_plate & ~h_cooked, jnp.argmax(h_ing), -1)
    h_other = h_plate | h_cooked
    at_pot = (r == 2) & (c == 6)

    # --- recipe (always-on indicators at (0,5) and (4,5)) ---
    v05, vis05 = _cell_view(obs, r, c, 0, 5)
    v45, vis45 = _cell_view(obs, r, c, 4, 5)
    rec_block = jnp.where(vis05, v05[36:39], v45[36:39])
    rec_valid = (vis05 | vis45) & (rec_block.sum() > 0)
    recipe = _i32(jnp.where(rec_valid, jnp.argmax(rec_block), mem[0]))
    known = recipe >= 0
    rk = jnp.clip(recipe, 0, 2)
    other = _i32(jnp.where(rk == 0, 1, 0))

    # --- pot at (2,7) ---
    pv, visp = _cell_view(obs, r, c, 2, 7)
    pot_now = jnp.concatenate([pv[31:34], pv[30:31], pv[39:40]]).astype(jnp.int32)
    # out of view: keep counting the remembered timer down; once it would hit
    # zero assume the pot is cooked (busy) until it is seen again
    mem_pot = mem[10:15]
    est_timer = jnp.maximum(mem_pot[4] - 1, 0)
    est_cooked = jnp.where((mem_pot[4] > 0) & (est_timer == 0), 1, mem_pot[3])
    pot_est = jnp.concatenate([mem_pot[0:3], est_cooked[None], est_timer[None]]).astype(jnp.int32)
    pot = jnp.where(visp, pot_now, pot_est)
    pot_c = pot[0:3]
    pot_total = pot_c.sum()
    pot_timer = pot[4]
    pot_idle = (pot_timer == 0) & (pot[3] == 0)
    pot_busy = ~pot_idle
    pot_empty = pot_idle & (pot_total == 0)
    pure_r = pot_c[rk] == pot_total

    # --- own puts into the current batch / partner-fills detection ---
    prev_held = mem[20]
    just_put = at_pot & (prev_held >= 0) & (h_type < 0) & visp & (pot_total > 0)
    batch_puts = jnp.where(visp & pot_empty, 0, mem[19] + jnp.where(just_put, 1, 0))
    partner_fills = jnp.where(visp & pot_busy & (batch_puts < 3), 1, mem[18])

    # --- delivery gate: do not seed an empty pot until the partner's delivery
    # (which may resample the recipe) has plausibly happened ---
    prev_nonempty = mem[10:13].sum() > 0
    became_empty = visp & pot_empty & prev_nonempty
    partner_vis = obs[:, :, 10].sum() > 0
    partner_dish = obs[:, :, 16].sum() > 0
    since_clear = jnp.where(visp & ~pot_empty, 0,
                            jnp.where(partner_dish, 0,
                                      jnp.where(visp & pot_empty, jnp.minimum(mem[15] + 1, 1000), mem[15])))
    recipe_at_empty = jnp.where(became_empty, recipe, mem[16])
    seed_ok = ((partner_vis & ~partner_dish) | (since_clear >= EMPTY_WAIT)
               | (recipe != recipe_at_empty))

    pot_accepts = pot_idle & (pot_total < 3) & pure_r & ((pot_total > 0) | seed_ok)
    pot_incompat = pot_idle & (pot_total > 0) & (pot_total < 3) & ~pure_r
    patience = jnp.where(pot_incompat, mem[9] + 1, 0)
    force = pot_incompat & (patience > PATIENCE_LIMIT)
    pot_needs = pot_accepts | force

    # --- stash counters ---
    sv, svis = _cells_view(obs, r, c, K['stash_r'], K['stash_c'])
    s_codes_now = jax.vmap(_item_code)(sv[:, 29:34])
    s_codes = jnp.where(svis, s_codes_now, mem[1:9])

    def stash_row(k):
        return jnp.where(k == 0, s_codes[0:4], s_codes[4:8])

    def stash_count(k):
        return (stash_row(k) == (k + 1)).sum()

    def fetch_nav(k, final=A_INTERACT, supply_only=False):
        codes = stash_row(k)
        row = jnp.where(k == 0, 1, 3)
        sdir = jnp.where(k == 0, D_UP, D_DOWN)
        has = codes == (k + 1)
        best = jnp.max(jnp.where(has, K['cols14'], 0))
        anyk = (best > 0) & (not supply_only)
        tc = jnp.where(anyk, best, 1)
        td = jnp.where(anyk, sdir, D_LEFT)
        return _nav(r, c, f, row, tc, td, final)

    def stash_nav(k):
        codes = stash_row(k)
        row = jnp.where(k == 0, 1, 3)
        sdir = jnp.where(k == 0, D_UP, D_DOWN)
        empty = codes == 0
        best = jnp.max(jnp.where(empty, K['cols14'], 0))
        anyk = best > 0
        act = _nav(r, c, f, row, best, sdir, A_INTERACT)
        # nowhere to stash: hold and wait near the pot
        return jnp.where(anyk, act, _nav(r, c, f, 2, 6, D_RIGHT, A_STAY))

    act_put = _nav(r, c, f, 2, 6, D_RIGHT, A_INTERACT)
    act_wait = _nav(r, c, f, 2, 6, D_RIGHT, A_STAY)

    hedge_empty = stash_count(other) == 0
    hk = jnp.clip(h_type, 0, 2)
    k_is_recipe = h_type == recipe

    # hedge mode: park the held recipe item so the other type can be stocked
    hedge_start = (h_type >= 0) & k_is_recipe & pot_busy & (pot_timer >= HEDGE_TIMER) & hedge_empty
    hedge_mode = jnp.where(hedge_empty & pot_busy, jnp.where(hedge_start, 1, mem[17]), 0)
    hedge_active = hedge_mode == 1

    # stage mode: while the pot cooks and the hedge is stocked, park one extra
    # recipe unit on the row-r counter nearest the pot (1: fetch+park, 2: fetch+hold)
    staged = stash_count(rk)
    stage_start = ((h_type < 0) & ~h_other & pot_busy & (pot_timer >= STAGE_TIMER)
                   & ~hedge_empty & (staged == 0) & (mem[21] == 0))
    stage_prev = jnp.where(pot_busy, mem[21], 0)
    stage_mode = jnp.where(stage_start, 1,
                           jnp.where((stage_prev == 1) & (h_type < 0) & (staged > 0), 2,
                                     jnp.where((stage_prev == 2) & (h_type >= 0), 0, stage_prev)))
    stage_park = (stage_mode == 1) & (h_type >= 0) & k_is_recipe
    stage_fetch = (stage_mode >= 1) & (h_type < 0)

    act_hold = jnp.where(k_is_recipe,
                         jnp.where(pot_needs, act_put,
                                   jnp.where(hedge_active | stage_park, stash_nav(hk), act_wait)),
                         jnp.where(h_type <= 1, stash_nav(hk), stash_nav(0)))
    act_other = stash_nav(0)

    # empty hands: seed/fill when the pot needs it, otherwise stock the hedge
    # (if there is time or the partner is known to fill the pot), else hold the recipe type
    stock_hedge = hedge_empty & (
        hedge_active
        | (pot_busy & (pot_timer >= HEDGE_TIMER))
        | ((pot_total > 0) & pot_idle & (partner_fills == 1)))
    act_empty = jnp.where(pot_needs & ~(stock_hedge & (pot_total > 0)), fetch_nav(rk),
                          jnp.where(stock_hedge, fetch_nav(other),
                                    jnp.where(stage_fetch, fetch_nav(rk, supply_only=True),
                                              fetch_nav(rk))))
    # unknown recipe: walk right until the indicator is in view
    act_unknown = jnp.where(c < 3, _i32(A_RIGHT), _i32(A_STAY))

    action = jnp.where(~known, act_unknown,
                       jnp.where(h_type >= 0, act_hold,
                                 jnp.where(h_other, act_other, act_empty)))
    action = _i32(action)

    new_mem = jnp.concatenate([
        recipe[None], s_codes, _i32(patience)[None], pot, _i32(since_clear)[None],
        _i32(recipe_at_empty)[None], _i32(hedge_mode)[None], _i32(partner_fills)[None],
        _i32(batch_puts)[None], _i32(h_type)[None], _i32(stage_mode)[None],
        jnp.zeros(2, dtype=jnp.int32)
    ]).astype(jnp.int32)
    trace = {
        'row': _i32(r), 'col': _i32(c), 'facing': _i32(f), 'held': _i32(h_type),
        'recipe': recipe, 'pot_total': _i32(pot_total), 'pot_needs': pot_needs,
        'seed_ok': seed_ok, 'since_clear': _i32(since_clear), 'hedge': _i32(hedge_mode),
        'pfills': _i32(partner_fills), 'stash0': _i32(stash_count(_i32(0))),
        'stash1': _i32(stash_count(_i32(1))), 'stage': _i32(stage_mode), 'action': action,
    }
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# agent_1: right-room server that copies the pot
# memory layout (int32[8]):
#  0 steps since reset
#  1 park flag for the currently held dish (partner seen holding another type)
#  2 steps the partner has stood at the pot holding a raw type with the pot empty
#  3 steps a raw item has waited on a pass counter (1,7)/(3,7) with the pot empty
#  4 seed flag (1: allowed to seed the empty pot with type mem[5])
#  5 cued seed type
#  6..9 last seen codes of counters (0,9) (4,9) (1,7) (3,7):
#       0 empty, 1 raw ing0, 2 raw ing1, 3 empty plate, 4 other (dish/junk)
#  10..11 unused
# ---------------------------------------------------------------------------
MEM1_SIZE = 12
SHOW_STEPS = 6    # partner holding a raw type at the pot while the pot stays empty
PASS_STEPS = 20   # raw item waiting on a pass counter while the pot stays empty


def init_agent_1():
    return jnp.zeros(MEM1_SIZE, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    K = _consts()
    mem = jnp.where(reset, init_agent_1(), memory)
    r, c = _localize(obs, K['a1_tpl'], K['a1_rc'])
    me = obs[2, 2]
    f = jnp.argmax(me[1:5])
    h_plate = me[5] > 0
    h_cooked = me[6] > 0
    h_ing = me[7:10]
    h_total = h_ing.sum()
    h_type = jnp.where((h_total > 0) & ~h_plate & ~h_cooked, jnp.argmax(h_ing), -1)
    h_dish = h_cooked
    h_empty_plate = h_plate & ~h_cooked & (h_total == 0)
    dish_type = jnp.where(h_ing[0] == 3, 0, jnp.where(h_ing[1] == 3, 1, -1))
    dish_pure = h_dish & (dish_type >= 0)

    # --- pot (always visible from the right room) ---
    pv, _ = _cell_view(obs, r, c, 2, 7)
    pot_c = pv[31:34]
    pot_total = pot_c.sum()
    pot_cooked = pv[30] > 0
    pot_timer = pv[39]
    pot_idle = (pot_timer == 0) & ~pot_cooked
    pot_busy = ~pot_idle
    pure0 = (pot_total > 0) & (pot_c[0] == pot_total)
    pure1 = (pot_total > 0) & (pot_c[1] == pot_total)
    pot_type = jnp.where(pure0, 0, jnp.where(pure1, 1, -1))
    pot_junk = (pot_total > 0) & (pot_type < 0)
    pot_partial = pot_idle & (pot_total > 0) & (pot_total < 3)

    # --- partner at the pot access cell (2,6) holding a raw ingredient ---
    p26, vis26 = _cell_view(obs, r, c, 2, 6)
    partner_at_pot = vis26 & (p26[10] > 0)
    partner_raw = _raw_type(p26[15:20])
    partner_other = partner_at_pot & (partner_raw >= 0) & (partner_raw <= 1) & (partner_raw != dish_type)
    park_flag = jnp.where(h_dish, jnp.where(partner_other & dish_pure, 1, mem[1]), 0)

    # --- counters (0,9) (4,9) (1,7) (3,7); rows 0/4 are not always in view,
    # so remember the last seen contents ---
    cv, cvis = _cells_view(obs, r, c, K['cnt_r'], K['cnt_c'])
    cblk = cv[:, 29:34]
    c_total_now = cblk[:, 2:5].sum(axis=1)
    empty_now = (cblk[:, 0] == 0) & (cblk[:, 1] == 0) & (c_total_now == 0)
    plate_now = (cblk[:, 0] > 0) & (cblk[:, 1] == 0) & (c_total_now == 0)
    raw_now = (cblk[:, 0] == 0) & (cblk[:, 1] == 0) & (c_total_now == 1)
    raw0_now = raw_now & (cblk[:, 2] == 1)
    raw1_now = raw_now & (cblk[:, 3] == 1)
    code_now = jnp.where(empty_now, 0, jnp.where(raw0_now, 1, jnp.where(raw1_now, 2,
                         jnp.where(plate_now, 3, 4))))
    c_code = jnp.where(cvis, code_now, mem[6:10])
    c_empty = c_code == 0
    c_plate = c_code == 3
    c_raw_type = jnp.where(c_code == 1, 0, jnp.where(c_code == 2, 1, -1))
    n_free = c_empty.sum()

    # --- cues for seeding an otherwise idle empty pot ---
    pot_empty = pot_idle & (pot_total == 0)
    show_now = pot_empty & partner_at_pot & (partner_raw >= 0) & (partner_raw <= 1)
    show_count = jnp.where(show_now, jnp.minimum(mem[2] + 1, 1000), 0)
    pass_type = jnp.where((c_raw_type[2] >= 0) & (c_raw_type[2] <= 1), c_raw_type[2],
                          jnp.where((c_raw_type[3] >= 0) & (c_raw_type[3] <= 1), c_raw_type[3], -1))
    pass_now = pot_empty & (pass_type >= 0)
    pass_count = jnp.where(pass_now, jnp.minimum(mem[3] + 1, 1000), 0)
    cue_type = jnp.where(show_count >= SHOW_STEPS, partner_raw,
                         jnp.where(pass_count >= PASS_STEPS, pass_type, -1))
    seed_flag = jnp.where(pot_empty, jnp.where(cue_type >= 0, 1, mem[4]), 0)
    seed_k = jnp.where(cue_type >= 0, cue_type, mem[5])
    may_seed = (seed_flag == 1) & pot_empty

    def counter_nav(i, final):
        return _nav(r, c, f, K['cnt_tr'][i], K['cnt_tc'][i], K['cnt_td'][i], final)

    def first(mask):
        return jnp.argmax(mask), mask.any()

    act_pot_int = _nav(r, c, f, 2, 8, D_LEFT, A_INTERACT)
    act_pot_wait = _nav(r, c, f, 2, 8, D_LEFT, A_STAY)
    act_deliver = _nav(r, c, f, 2, 9, D_RIGHT, A_INTERACT)

    # dump held item on the first empty counter (else hold near the pot)
    di, dany = first(c_empty)
    act_dump = jnp.where(dany, counter_nav(di, A_INTERACT), act_pot_wait)

    # fetch ingredient k: cached on a counter first, else the adjacent supply
    def fetch_nav(k):
        ci, cany = first(c_raw_type == k)
        sup = jnp.where(k == 0, _nav(r, c, f, 1, 8, D_UP, A_INTERACT),
                        _nav(r, c, f, 3, 8, D_DOWN, A_INTERACT))
        return jnp.where(cany, counter_nav(ci, A_INTERACT), sup)

    # fetch plate: cached plate first, else nearest plate pile
    pi, pany = first(c_plate)
    pile = jnp.where(r <= 2, _nav(r, c, f, 1, 9, D_RIGHT, A_INTERACT),
                     _nav(r, c, f, 3, 9, D_RIGHT, A_INTERACT))
    act_plate = jnp.where(pany, counter_nav(pi, A_INTERACT), pile)

    # --- holding a cooked dish: deliver if it looks right; park a suspect or
    # mixed dish only while at least two counters stay free (never deadlock) ---
    act_dish = jnp.where(dish_pure & (park_flag == 0), act_deliver,
                         jnp.where(n_free >= 2, act_dump, act_deliver))

    # --- holding an empty plate ---
    act_hplate = jnp.where(pot_cooked, act_pot_int,
                           jnp.where(pot_timer > 0, act_pot_wait, act_dump))

    # --- holding a raw ingredient ---
    hk = jnp.clip(h_type, 0, 2)
    same = pot_partial & (pot_type == hk)
    junk_fill = pot_partial & pot_junk
    cued_seed = may_seed & (hk == seed_k)
    act_hing = jnp.where(same | junk_fill | cued_seed, act_pot_int,
                         jnp.where(pot_empty, act_pot_wait, act_dump))

    # --- empty hands ---
    tk = jnp.clip(pot_type, 0, 1)
    act_empty = jnp.where(pot_busy, act_plate,
                          jnp.where(pot_partial,
                                    jnp.where(pot_type >= 0, fetch_nav(tk), fetch_nav(_i32(0))),
                                    jnp.where(may_seed, fetch_nav(jnp.clip(seed_k, 0, 1)), act_pot_wait)))

    action = jnp.where(h_dish, act_dish,
                       jnp.where(h_empty_plate, act_hplate,
                                 jnp.where(h_type >= 0, act_hing, act_empty)))
    action = _i32(action)

    new_mem = jnp.concatenate([
        (mem[0] + 1)[None], _i32(park_flag)[None], _i32(show_count)[None], _i32(pass_count)[None],
        _i32(seed_flag)[None], _i32(seed_k)[None], _i32(c_code), mem[10:12]
    ]).astype(jnp.int32)
    trace = {
        'row': _i32(r), 'col': _i32(c), 'facing': _i32(f), 'held': _i32(h_type),
        'h_plate': h_plate, 'h_dish': h_dish, 'pot_total': _i32(pot_total),
        'pot_type': _i32(pot_type), 'pot_timer': _i32(pot_timer), 'park': _i32(park_flag),
        'seedf': _i32(seed_flag), 'nfree': _i32(n_free), 'cooked': pot_cooked, 'action': action,
    }
    return action, new_mem, trace
