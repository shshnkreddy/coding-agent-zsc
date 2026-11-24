"""Overcooked-V2 ``test_time_wide``: recipe publisher + two-pot cook.

Row 3 of the board (``WPWPWW``) is solid, so the two rooms are disconnected and
each seat is locked into the room it spawned in for the whole episode.  That
fixes the division of labour without any negotiation:

* Top room (rows 1-2) owns ingredient piles 0/1, the plate pile (0,3) and the
  delivery station (0,2), but the recipe indicator at (6,2) is always outside
  its 5x5 view, so it can never read the recipe itself.
* Bottom room (rows 4-5) always sees the recipe indicator from every cell it
  can stand on and can only reach the ingredient-3 piles, the two shared
  counters (3,2)/(3,4) and the pots.

So the bottom seat is a publisher and the top seat is a cook.  The published
code is the plainest spatial one available: the ingredient index increases left
to right.  Ingredient 0 -> a token on the *left* shared counter (3,2),
ingredient 1 -> a token on the *right* shared counter (3,4); the publisher then
stands directly underneath the counter it used and faces it, so a partner that
reads shared counters, or partner position, or the pot side, decodes the same
bit.  The cook reads counters first, then a foreign ingredient left in a pot,
then partner column, and falls back to a fresh coin flip when nothing is
published (a *fixed* fallback guess is far worse than a coin: an incorrect
delivery does not resample the recipe, so a deterministic wrong guess repeats
forever).  For the same reason, a run of informed deliveries that all leave the
published bit unchanged is evidence that the partner mirrored the convention,
because a correct delivery resamples the recipe and therefore moves the token
with probability 1/2 while an incorrect one never moves it; after five such
deliveries the cook flips its decoding.

The cook is a pipeline rather than a serial loop.  Pot (3,1) is permanently the
ingredient-0 pot and pot (3,3) the ingredient-1 pot, so *filling* never needs to
know the recipe and can run while the bit is unknown or stale; only scooping and
delivering consult the belief.  Counters (0,1)/(0,4) buffer a finished dish of
each type, which keeps a pot free to restart cooking instead of holding a dish
that is not currently wanted.

Everything is fixed-shape.  Localisation is arithmetic: the top seat always
sees the unique delivery cell (0,2) and the bottom seat always sees the unique
recipe cell (6,2), which pins the self position exactly.  Navigation is also
arithmetic because each room is an open 2x4 rectangle, so moving along one axis
and then the other is always a shortest path; the axis order is chosen so the
last move usually leaves the player already facing the object it must use.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'wide_publisher_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# Observation channels.
CH_SELF_DIR = 1          # 1..4: UP, DOWN, RIGHT, LEFT
CH_SELF_INV = 5          # 5..10: plate, cooked, ingredient counts 0..3
CH_PARTNER_POS = 11
CH_GOAL = 23
CH_RECIPE_CELL = 25
CH_ITEM = 32             # 32..37: plate, cooked, ingredient counts 0..3
CH_RECIPE = 38           # 38..43: plate, cooked, ingredient counts 0..3
CH_TIMER = 44

# Facing ids (observation order) and action ids.
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5

# Fixed board cells.
DELIVERY = (0, 2)
RECIPE_CELL = (6, 2)
POT_L = (3, 1)
POT_R = (3, 3)
SHARED_L = (3, 2)
SHARED_R = (3, 4)
SLOT_A_CELL = (0, 1)
SLOT_B_CELL = (0, 4)

# Top-room stands: (row, column, facing) triples.
G_LOOK = (2, 2, UP)              # sees both counters, both pots, both slots
G_DELIVER = (1, 2, UP)           # faces delivery (0,2)
G_PLATE = (1, 3, UP)             # faces plate pile (0,3)
G_SLOT_A = (1, 1, UP)            # faces buffer counter (0,1)
G_SLOT_B = (1, 4, UP)            # faces buffer counter (0,4)
G_POT_L = (2, 1, DOWN)
G_POT_R = (2, 3, DOWN)
G_PILE_0 = (1, 1, LEFT)          # ingredient 0 pile (1,0), next to pot (3,1)
G_PILE_1 = (2, 4, RIGHT)         # ingredient 1 pile (2,5), next to pot (3,3)

# Bottom-room stands.
G_PARK_L = (4, 2, UP)            # faces shared counter (3,2)
G_PARK_R = (4, 4, UP)            # faces shared counter (3,4)
G_WATCH_R = (4, 3, UP)           # parks one step back so both pots stay in view
G_FETCH_L = (4, 1, LEFT)         # ingredient 3 pile (4,0)
G_FETCH_R = (4, 4, RIGHT)        # ingredient 3 pile (4,5)
G_PRIV_0 = (5, 1, LEFT)          # private counter (5,0)
G_PRIV_1 = (5, 1, DOWN)          # private counter (6,1)
G_PRIV_2 = (5, 3, DOWN)          # private counter (6,3)
PRIV_0_CELL = (5, 0)
PRIV_1_CELL = (6, 1)
PRIV_2_CELL = (6, 3)

# Tuning constants.
FLIP_AFTER = 5           # informed deliveries that failed to move the token
PEND_MAX = 10            # *confident* reads allowed for the partner to republish
CHG_MIN = 2              # consecutive confident reads that confirm a move
COOL = 7                 # steps a republishing partner needs after a delivery
STALL_MAX = 30           # deliver on a stale bit rather than stall forever
PUB_FIRST = 220          # steps of a frozen recipe before the publisher inverts
PUB_NEXT = 80            # ... and before it undoes an inversion that did not help
EMPTY_MIN = 12           # clear sightings of two bare counters that mean a bit
PUB_SCOOPS = 5           # dishes the cook must have taken before blaming the code

MEM_SIZE = 26
M_BELIEF, M_HAVE, M_FLIP, M_STREAK = 0, 1, 2, 3
M_LAST_READ, M_PEND, M_PREV_INV, M_DISH = 4, 5, 6, 7
M_SLOT_A, M_SLOT_B, M_COIN, M_SEEDED = 8, 9, 10, 11
M_TARGET, M_SINCE, M_LOCK, M_CHG = 12, 13, 14, 15
M_STALL, M_PFLIP, M_PSEEN = 16, 17, 18
M_SEEN_L, M_SEEN_R, M_EMPTY = 19, 20, 21
M_PCL, M_PCR, M_SCOOPS = 22, 23, 24
MEM_INIT = (0, 0, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, 60, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def _init_memory():
    return jnp.asarray(MEM_INIT, dtype=jnp.int32)


def _pick(vec, idx):
    """vec[idx] for a traced index, without relying on gather semantics."""
    return jnp.sum(jnp.where(jnp.arange(vec.shape[0]) == idx, vec, 0))


def _sel(cond, first, second):
    """Select between two (row, column, facing) goals."""
    return (jnp.where(cond, first[0], second[0]),
            jnp.where(cond, first[1], second[1]),
            jnp.where(cond, first[2], second[2]))


def _locate(obs):
    """Self position and room, from the unique landmark visible in each room."""
    goal = obs[:, :, CH_GOAL]
    recipe = obs[:, :, CH_RECIPE_CELL]
    top = jnp.sum(goal) > 0
    gflat = jnp.argmax(goal)
    rflat = jnp.argmax(recipe)
    row = jnp.where(top, DELIVERY[0] + 2 - gflat // 5, RECIPE_CELL[0] + 2 - rflat // 5)
    col = jnp.where(top, DELIVERY[1] + 2 - gflat % 5, RECIPE_CELL[1] + 2 - rflat % 5)
    return top, row.astype(jnp.int32), col.astype(jnp.int32)


def _at(obs, row, col, grow, gcol):
    """Channel vector at a global cell, zeroed when it is outside the view."""
    li = grow - row + 2
    lj = gcol - col + 2
    ok = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    vec = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), :]
    return vec * ok.astype(vec.dtype), ok


def _item(vec, ok):
    """Decode the grid-item block at a cell."""
    plate = vec[CH_ITEM] > 0
    cooked = vec[CH_ITEM + 1] > 0
    counts = vec[CH_ITEM + 2:CH_ITEM + 6]
    total = jnp.sum(counts)
    present = ok & (plate | cooked | (total > 0))
    return present, plate, cooked, counts, total


def _occupied(obs, row, col, cell):
    vec, ok = _at(obs, row, col, cell[0], cell[1])
    present, _, _, _, _ = _item(vec, ok)
    return present


def _navigate(row, col, face, goal, use_face, interact):
    """Walk inside the open 2x4 room, then turn, then interact."""
    drow = goal[0] - row
    dcol = goal[1] - col
    gface = goal[2]
    move_h = jnp.where(dcol > 0, A_RIGHT, A_LEFT)
    move_v = jnp.where(drow > 0, A_DOWN, A_UP)
    # Finish on the axis that leaves us already facing the target object.
    horizontal_last = use_face & ((gface == LEFT) | (gface == RIGHT))
    move = jnp.where(horizontal_last,
                     jnp.where(drow != 0, move_v, move_h),
                     jnp.where(dcol != 0, move_h, move_v))
    turn = jnp.where(gface == UP, A_UP,
                     jnp.where(gface == DOWN, A_DOWN,
                               jnp.where(gface == RIGHT, A_RIGHT, A_LEFT)))
    arrived = (drow == 0) & (dcol == 0)
    aligned = (~use_face) | (face == gface)
    settled = jnp.where(interact, A_INTERACT, A_STAY)
    return jnp.where(arrived,
                     jnp.where(aligned, settled, turn),
                     move).astype(jnp.int32)


def _slot_code(obs, row, col, cell, remembered):
    """What sits on a buffer counter: -1 empty, 0/1 a pure dish, 2 anything else.

    Only the cook can reach these two cells, so a remembered value stays valid
    while the crop does not cover them.
    """
    vec, ok = _at(obs, row, col, cell[0], cell[1])
    present, plate, cooked, counts, total = _item(vec, ok)
    dish = plate & cooked & (total == 3)
    code = jnp.where(~present, -1,
                     jnp.where(dish & (counts[0] == 3), 0,
                               jnp.where(dish & (counts[1] == 3), 1, 2)))
    return jnp.where(ok, code, remembered).astype(jnp.int32)


def _pot_info(obs, row, col, cell, mine_idx):
    """Flags for one pot, given the ingredient it is permanently assigned."""
    vec, ok = _at(obs, row, col, cell[0], cell[1])
    _, _, cooked, counts, total = _item(vec, ok)
    timer = vec[CH_TIMER]
    mine = _pick(counts, mine_idx)
    other = total - mine
    pure_full = (mine == 3) & (other == 0)
    good = ok & cooked & pure_full
    bad = ok & cooked & (~pure_full)
    idle = ok & (~cooked) & (timer == 0)
    fill_clean = idle & (total < 3) & (other == 0)
    fill_dirty = idle & (total < 3) & (other > 0)
    accepts = idle & (total < 3)
    foreign = ok & (jnp.sum(counts[2:4]) > 0)
    return good, bad, fill_clean, fill_dirty, accepts, foreign


# ---------------------------------------------------------------------------
# Top room: the cook
# ---------------------------------------------------------------------------
def _cook(obs, mem, row, col, face, rng):
    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 6]
    held_plate = inv[0] > 0
    held_cooked = inv[1] > 0
    held_counts = inv[2:6]
    held_total = jnp.sum(held_counts)
    holds_dish = held_plate & held_cooked
    holds_plate = held_plate & (~held_cooked)
    holds_raw = (~held_plate) & (held_total > 0)
    hands_free = (~held_plate) & (held_total == 0)
    inv_code = jnp.where(holds_dish, 3,
                         jnp.where(holds_plate, 2,
                                   jnp.where(holds_raw, 1, 0))).astype(jnp.int32)
    dish_type = jnp.where(holds_dish & (held_counts[0] == 3) & (held_total == 3), 0,
                          jnp.where(holds_dish & (held_counts[1] == 3)
                                    & (held_total == 3), 1, 2)).astype(jnp.int32)
    raw_type = jnp.argmax(held_counts).astype(jnp.int32)

    # --- decode the published recipe bit ------------------------------------
    cl_vec, cl_ok = _at(obs, row, col, SHARED_L[0], SHARED_L[1])
    cr_vec, cr_ok = _at(obs, row, col, SHARED_R[0], SHARED_R[1])
    tok_l, _, _, _, _ = _item(cl_vec, cl_ok)
    tok_r, _, _, _, _ = _item(cr_vec, cr_ok)
    # A partner may code the bit as presence/absence on a single counter rather
    # than by side.  If only one of the two counters has ever carried anything,
    # then two counters that stay bare for a while are themselves a reading: the
    # complement of the side that side-codes.  The run length keeps our own
    # publisher's four-step token swap from being read as that state.
    seen_l = jnp.maximum(mem[M_SEEN_L], tok_l.astype(jnp.int32))
    seen_r = jnp.maximum(mem[M_SEEN_R], tok_r.astype(jnp.int32))
    both_vis = (jnp.abs(col - SHARED_L[1]) <= 2) & (jnp.abs(col - SHARED_R[1]) <= 2)
    bare = both_vis & (~tok_l) & (~tok_r)
    empty_run = jnp.where(bare, mem[M_EMPTY] + 1,
                          jnp.where(both_vis, 0, mem[M_EMPTY])).astype(jnp.int32)
    one_sided = (seen_l + seen_r) == 1
    read_bare = jnp.where(one_sided & (empty_run >= EMPTY_MIN),
                          jnp.where(seen_l > 0, 1, 0), -1)
    read_counter = jnp.where(tok_l & (~tok_r), 0,
                             jnp.where(tok_r & (~tok_l), 1, read_bare))

    good_l, bad_l, fc_l, fd_l, acc_l, foreign_l = _pot_info(obs, row, col, POT_L, 0)
    good_r, bad_r, fc_r, fd_r, acc_r, foreign_r = _pot_info(obs, row, col, POT_R, 1)
    read_pot = jnp.where(foreign_l & (~foreign_r), 0,
                         jnp.where(foreign_r & (~foreign_l), 1, -1))

    partner = obs[:, :, CH_PARTNER_POS]
    partner_seen = jnp.sum(partner) > 0
    partner_col = col + (jnp.argmax(partner).astype(jnp.int32) % 5) - 2
    read_pos = jnp.where(partner_seen, jnp.where(partner_col <= 2, 0, 1), -1)

    read = jnp.where(read_counter >= 0, read_counter,
                     jnp.where(read_pot >= 0, read_pot, read_pos)).astype(jnp.int32)
    conf = jnp.where(read_counter >= 0, 2,
                     jnp.where(read_pot >= 0, 1,
                               jnp.where(read_pos >= 0, 0, -1))).astype(jnp.int32)

    # --- mirror test: a correct delivery resamples the recipe ---------------
    # A delivery that matched moves the published token with probability 1/2; a
    # delivery that missed never moves it, because the recipe is kept.  So a run
    # of informed deliveries that never move the token is evidence of a mirrored
    # convention -- but a single observed move is *proof* that the decoding was
    # right, and that proof is permanent, so the flip is locked out afterwards.
    pending = mem[M_PEND] > 0
    differs = pending & (conf >= 2) & (read != mem[M_LAST_READ])
    chg = jnp.where(differs, mem[M_CHG] + 1, 0).astype(jnp.int32)
    changed = chg >= CHG_MIN
    # The window counts chances to see the token, not bare steps: the right-hand
    # counter is out of view whenever the cook works at column 1.
    tick = (conf >= 2) & (~changed)
    expired = pending & tick & (mem[M_PEND] == 1)
    streak = jnp.where(changed, 0,
                       jnp.where(expired, mem[M_STREAK] + 1, mem[M_STREAK]))
    pend = jnp.where(changed, 0,
                     jnp.where(tick, jnp.maximum(mem[M_PEND] - 1, 0), mem[M_PEND]))
    lock = jnp.where(changed, 1, mem[M_LOCK]).astype(jnp.int32)
    do_flip = (streak >= FLIP_AFTER) & (lock == 0)
    flip = jnp.where(do_flip, 1 - mem[M_FLIP], mem[M_FLIP]).astype(jnp.int32)
    streak = jnp.where(do_flip, 0, streak).astype(jnp.int32)

    # --- belief -------------------------------------------------------------
    decoded = jnp.bitwise_xor(read, flip)
    coin_draw = jax.random.bernoulli(rng).astype(jnp.int32)
    delivered = (mem[M_PREV_INV] == 3) & hands_free & (row == G_DELIVER[0]) & (
        col == G_DELIVER[1])
    coin = jnp.where((mem[M_SEEDED] == 0) | delivered, coin_draw, mem[M_COIN])
    have = jnp.where(conf >= 1, 1, mem[M_HAVE]).astype(jnp.int32)
    belief = jnp.where(conf >= 1, decoded,
                       jnp.where(mem[M_HAVE] > 0, mem[M_BELIEF],
                                 jnp.where(conf >= 0, decoded, coin))
                       ).astype(jnp.int32)
    other = (1 - belief).astype(jnp.int32)

    # A delivery of a dish that our decoded bit chose is an informative trial.
    informed = delivered & (conf >= 2) & (mem[M_DISH] == decoded)
    last_read = jnp.where(informed, read, mem[M_LAST_READ]).astype(jnp.int32)
    pend = jnp.where(informed, PEND_MAX, pend).astype(jnp.int32)

    # A correct delivery resamples the recipe at once, but the partner needs a
    # few steps to move its token, so the bit is stale right after a delivery.
    since = jnp.where(delivered, 0,
                      jnp.minimum(mem[M_SINCE] + 1, 60)).astype(jnp.int32)
    fresh = (since >= COOL) & ((conf >= 1) | (have == 0) | (since >= STALL_MAX))

    # --- buffer slots (only this seat can reach them) -----------------------
    slot_a = _slot_code(obs, row, col, SLOT_A_CELL, mem[M_SLOT_A])
    slot_b = _slot_code(obs, row, col, SLOT_B_CELL, mem[M_SLOT_B])
    free_a = slot_a < 0
    free_b = slot_b < 0
    any_free = free_a | free_b
    g_free = _sel(free_a, G_SLOT_A, G_SLOT_B)

    # --- what is worth doing ------------------------------------------------
    good_b = jnp.where(belief == 0, good_l, good_r)
    good_o = jnp.where(belief == 0, good_r, good_l)
    fc_b = jnp.where(belief == 0, fc_l, fc_r)
    fc_o = jnp.where(belief == 0, fc_r, fc_l)
    free_o = jnp.where(other == 0, free_a, free_b)

    bank = good_o & free_o                  # bank the unwanted flavour
    clear = (bad_l | bad_r) & any_free      # get a spoiled dish out of a pot
    clear_pot = jnp.where(bad_l, 0, 1).astype(jnp.int32)
    scoop_pot = jnp.where(good_b, belief,
                          jnp.where(bank, other,
                                    jnp.where(clear, clear_pot, -1))).astype(jnp.int32)
    fill_pot = jnp.where(fc_b, belief,
                         jnp.where(fc_o, other,
                                   jnp.where(fd_l & any_free, 0,
                                             jnp.where(fd_r & any_free, 1, -1)))
                         ).astype(jnp.int32)

    stored = jnp.where(~fresh, -1,
                       jnp.where(slot_a == belief, 0,
                                 jnp.where(slot_b == belief, 1, -1))).astype(jnp.int32)
    g_stored = _sel(stored == 0, G_SLOT_A, G_SLOT_B)
    g_scoop = _sel(scoop_pot == 0, G_POT_L, G_POT_R)
    g_fill = _sel(fill_pot == 0, G_PILE_0, G_PILE_1)
    g_pot_of_raw = _sel(raw_type == 0, G_POT_L, G_POT_R)
    g_store_dish = _sel((dish_type == 0) & free_a, G_SLOT_A,
                        _sel((dish_type == 1) & free_b, G_SLOT_B, g_free))

    raw_ok = (raw_type <= 1) & jnp.where(raw_type == 0, acc_l, acc_r)
    store_ok = any_free

    # hands free: take a stored dish > scoop a ready pot > stock a pot > watch
    g_idle = _sel(hands_free & (stored >= 0), g_stored,
                  _sel(hands_free & (scoop_pot >= 0), G_PLATE,
                       _sel(hands_free & (fill_pot >= 0), g_fill, G_LOOK)))
    i_idle = (stored >= 0) | (scoop_pot >= 0) | (fill_pot >= 0)
    u_idle = i_idle

    # plate in hand: only scoop a pot we actually want emptied
    g_plated = _sel(scoop_pot >= 0, g_scoop, G_LOOK)
    i_plated = scoop_pot >= 0

    # raw ingredient in hand: it belongs to its own pot, else park it
    g_rawh = _sel(raw_ok, g_pot_of_raw, _sel(store_ok, g_free, G_LOOK))
    i_rawh = raw_ok | store_ok

    # dish in hand: deliver it if it is wanted now, otherwise buffer it
    # A dish is carried to the hatch as soon as it matches, but handed over only
    # once the published bit has had time to catch up with the last delivery.
    deliver_now = dish_type == belief
    g_dishh = _sel(deliver_now, G_DELIVER, _sel(store_ok, g_store_dish, G_LOOK))
    i_dishh = jnp.where(deliver_now, fresh, store_ok)

    goal = _sel(holds_dish, g_dishh,
                _sel(holds_plate, g_plated,
                     _sel(holds_raw, g_rawh, g_idle)))
    interact = jnp.where(holds_dish, i_dishh,
                         jnp.where(holds_plate, i_plated,
                                   jnp.where(holds_raw, i_rawh, i_idle)))
    use_face = jnp.where(holds_dish, deliver_now | store_ok,
                         jnp.where(holds_plate, i_plated,
                                   jnp.where(holds_raw, i_rawh, u_idle)))

    action = _navigate(row, col, face, goal, use_face, interact)

    new_mem = mem
    new_mem = new_mem.at[M_BELIEF].set(belief)
    new_mem = new_mem.at[M_HAVE].set(have)
    new_mem = new_mem.at[M_FLIP].set(flip)
    new_mem = new_mem.at[M_STREAK].set(streak)
    new_mem = new_mem.at[M_LAST_READ].set(last_read)
    new_mem = new_mem.at[M_PEND].set(pend)
    new_mem = new_mem.at[M_PREV_INV].set(inv_code)
    new_mem = new_mem.at[M_DISH].set(jnp.where(holds_dish, dish_type, mem[M_DISH]))
    new_mem = new_mem.at[M_SLOT_A].set(slot_a)
    new_mem = new_mem.at[M_SLOT_B].set(slot_b)
    new_mem = new_mem.at[M_COIN].set(coin)
    new_mem = new_mem.at[M_SEEDED].set(jnp.asarray(1, dtype=jnp.int32))
    new_mem = new_mem.at[M_SINCE].set(since)
    new_mem = new_mem.at[M_LOCK].set(lock)
    new_mem = new_mem.at[M_CHG].set(chg)
    new_mem = new_mem.at[M_SEEN_L].set(seen_l.astype(jnp.int32))
    new_mem = new_mem.at[M_SEEN_R].set(seen_r.astype(jnp.int32))
    new_mem = new_mem.at[M_EMPTY].set(empty_run)
    trace = {
        'role': jnp.asarray(0, dtype=jnp.int32),
        'row': row,
        'col': col,
        'belief': belief,
        'read': read,
        'conf': conf,
        'flip': flip,
        'streak': streak,
        'scoop': scoop_pot,
        'fill': fill_pot,
        'slot_a': slot_a,
        'slot_b': slot_b,
        'lock': lock,
        'goal_row': goal[0].astype(jnp.int32),
        'goal_col': goal[1].astype(jnp.int32),
    }
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# Bottom room: the recipe publisher
# ---------------------------------------------------------------------------
def _publisher(obs, mem, row, col, face):
    rec_vec, rec_ok = _at(obs, row, col, RECIPE_CELL[0], RECIPE_CELL[1])
    rec_counts = rec_vec[CH_RECIPE + 2:CH_RECIPE + 6]
    rec_seen = rec_ok & (jnp.sum(rec_counts) > 0)
    target = jnp.where(rec_seen,
                       jnp.clip(jnp.argmax(rec_counts).astype(jnp.int32), 0, 1),
                       mem[M_TARGET]).astype(jnp.int32)

    # The recipe is resampled only by a *correct* delivery, so a target that
    # never moves means the cook across the wall is never getting one right.
    # After a long enough freeze the most likely cause is that the partner reads
    # this counter code the other way round, so invert it; if inverting does not
    # unfreeze the recipe either, undo it rather than sit on a worse guess.
    moved = (mem[M_TARGET] >= 0) & (target != mem[M_TARGET])
    stall = jnp.where(moved, 0, jnp.minimum(mem[M_STALL] + 1, 999)).astype(jnp.int32)
    # Both pots are in view from every cell of this room, so a pot that goes
    # from cooked to empty counts the dishes the cook is actually taking.  Only
    # a cook that is producing dishes and still not moving the recipe is
    # evidence of a misread code; a merely slow partner must not trigger this.
    pl_vec, pl_ok = _at(obs, row, col, POT_L[0], POT_L[1])
    pr_vec, pr_ok = _at(obs, row, col, POT_R[0], POT_R[1])
    _, _, ck_l, _, tot_l = _item(pl_vec, pl_ok)
    _, _, ck_r, _, tot_r = _item(pr_vec, pr_ok)
    took = ((pl_ok & (mem[M_PCL] > 0) & (~ck_l) & (tot_l == 0)).astype(jnp.int32)
            + (pr_ok & (mem[M_PCR] > 0) & (~ck_r) & (tot_r == 0)).astype(jnp.int32))
    scoops = jnp.where(moved, 0,
                       jnp.minimum(mem[M_SCOOPS] + took, 99)).astype(jnp.int32)
    do_pflip = (stall >= jnp.where(mem[M_PSEEN] == 0, PUB_FIRST, PUB_NEXT)) & (
        scoops >= PUB_SCOOPS)
    pflip = jnp.where(do_pflip, 1 - mem[M_PFLIP], mem[M_PFLIP]).astype(jnp.int32)
    stall = jnp.where(do_pflip, 0, stall).astype(jnp.int32)
    scoops = jnp.where(do_pflip, 0, scoops).astype(jnp.int32)
    pseen = jnp.minimum(mem[M_PSEEN] + do_pflip.astype(jnp.int32), 9).astype(jnp.int32)
    code = jnp.bitwise_xor(jnp.maximum(target, 0), pflip)

    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 6]
    held_plate = inv[0] > 0
    held_counts = inv[2:6]
    held_total = jnp.sum(held_counts)
    holds_token = (~held_plate) & (inv[1] == 0) & (held_counts[3] > 0)
    holds_any = held_plate | (held_total > 0)
    holds_other = holds_any & (~holds_token)

    left_vec, left_ok = _at(obs, row, col, SHARED_L[0], SHARED_L[1])
    right_vec, right_ok = _at(obs, row, col, SHARED_R[0], SHARED_R[1])
    present_l, plate_l, cooked_l, _, tot_l = _item(left_vec, left_ok)
    present_r, plate_r, cooked_r, _, tot_r = _item(right_vec, right_ok)
    # Only a bare ingredient counts as somebody's signal token; a plate or a
    # cooked dish a partner parked there is its property and stays put.
    raw_l = present_l & (~plate_l) & (~cooked_l) & (tot_l > 0)
    raw_r = present_r & (~plate_r) & (~cooked_r) & (tot_r > 0)
    want_left = code == 0
    sig_des = jnp.where(want_left, raw_l, raw_r)
    sig_oth = jnp.where(want_left, raw_r, raw_l)
    occ_des = jnp.where(want_left, present_l, present_r)

    g_des = _sel(want_left, G_PARK_L, G_PARK_R)
    g_oth = _sel(want_left, G_PARK_R, G_PARK_L)
    # Parking one cell back from the right-hand counter still reads as "right"
    # to anybody watching the column, and keeps both pots inside the crop.
    g_park = _sel(want_left, G_PARK_L, G_WATCH_R)
    g_fetch = _sel(want_left, G_FETCH_L, G_FETCH_R)
    priv_0_free = ~_occupied(obs, row, col, PRIV_0_CELL)
    priv_1_free = ~_occupied(obs, row, col, PRIV_1_CELL)
    g_priv = _sel(priv_0_free, G_PRIV_0, _sel(priv_1_free, G_PRIV_1, G_PRIV_2))

    # publish -> reclaim the stale token -> park -> fetch a fresh token
    goal = _sel(holds_token,
                _sel(occ_des, g_priv, g_des),
                _sel(holds_other, g_priv,
                     _sel(sig_oth, g_oth, _sel(sig_des, g_park, g_fetch))))
    # Parking must not interact: that would take our own token back off the
    # counter and erase the published bit.
    interact = jnp.where(holds_token | holds_other, True,
                         jnp.where(sig_oth, True,
                                   jnp.where(sig_des | occ_des, False, True)))

    action = _navigate(row, col, face, goal, jnp.asarray(True), interact)

    new_mem = mem.at[M_TARGET].set(target)
    new_mem = new_mem.at[M_STALL].set(stall)
    new_mem = new_mem.at[M_PFLIP].set(pflip)
    new_mem = new_mem.at[M_PSEEN].set(pseen)
    new_mem = new_mem.at[M_SCOOPS].set(scoops)
    new_mem = new_mem.at[M_PCL].set(jnp.where(pl_ok, ck_l.astype(jnp.int32),
                                              mem[M_PCL]))
    new_mem = new_mem.at[M_PCR].set(jnp.where(pr_ok, ck_r.astype(jnp.int32),
                                              mem[M_PCR]))
    zero = jnp.asarray(0, dtype=jnp.int32)
    minus = jnp.asarray(-1, dtype=jnp.int32)
    trace = {
        'role': jnp.asarray(1, dtype=jnp.int32),
        'row': row,
        'col': col,
        'belief': code,
        'read': target,
        'conf': jnp.asarray(2, dtype=jnp.int32),
        'flip': pflip,
        'streak': stall,
        'scoop': scoops,
        'fill': minus,
        'slot_a': minus,
        'slot_b': minus,
        'lock': zero,
        'goal_row': goal[0].astype(jnp.int32),
        'goal_col': goal[1].astype(jnp.int32),
    }
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# Shared entry point: the room we localise in decides the role.
# ---------------------------------------------------------------------------
def _act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    memory = jnp.where(reset, _init_memory(), memory)
    top, row, col = _locate(obs)
    face = jnp.argmax(obs[2, 2, CH_SELF_DIR:CH_SELF_DIR + 4]).astype(jnp.int32)

    cook_action, cook_mem, cook_trace = _cook(obs, memory, row, col, face, rng)
    pub_action, pub_mem, pub_trace = _publisher(obs, memory, row, col, face)

    action = jnp.where(top, cook_action, pub_action).astype(jnp.int32)
    new_mem = jnp.where(top, cook_mem, pub_mem).astype(jnp.int32)
    trace = {key: jnp.where(top, cook_trace[key], pub_trace[key]).astype(jnp.int32)
             for key in cook_trace}
    return action, new_mem, trace


def init_agent_0():
    return _init_memory()


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def init_agent_1():
    return _init_memory()


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
