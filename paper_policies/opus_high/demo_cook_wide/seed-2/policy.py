"""Cooperative policy for the demo_cook_wide Overcooked-V2 layout.

Layout facts that drive the design (all taken from the supplied specification):

    WWWWBXBWWWW
    WWW0 A 1WWW
    WWWWWPWWWWW
    W    A    W
    0  W3R3W  0
    W1WWWWWWW1W

The two floor regions are disconnected, so each seat is confined to its own
room for the whole episode.  Seat 0 lives in {(1,4),(1,5),(1,6)} and is the only
seat that can touch the plate piles (0,4)/(0,6) and the delivery station (0,5);
it can also reach ingredient-0 supply (1,3), ingredient-1 supply (1,7) and the
pot (2,5).  Seat 0 can therefore cook and deliver alone -- but it can never see
the always-on recipe indicator at (4,5), which lies outside every 5x5 crop taken
from row 1.  Seat 1 sits in the corridor, reads that indicator from
(3,3)..(3,7), and can also reach the pot from (3,5).

So the only information that has to cross the rooms is the target ingredient.
This policy transmits it with the least convention-dependent signal available:
seat 1 physically fetches the required ingredient and drops it into the shared
pot.  Seat 0 reads the target off whatever physical evidence it can see -- raw
pot contents first, then a raw ingredient carried by the partner, then a raw
ingredient resting on a counter -- and never relies on a private code, a
position code or a timing code.  Seat 0 refuses to deliver a dish it believes
to be wrong (it parks it on a counter instead), so mis-coordination costs time
rather than -20 reward.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'evidence_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# --------------------------------------------------------------------------
# Constant layout tables (literals; see gen_tables.py for their derivation).
# --------------------------------------------------------------------------

# Cell codes: 0 floor, 1 wall, 2 goal, 3 pot, 4 recipe indicator,
# 5 button, 6 plate pile, 7..10 ingredient 0..3 supplies.
_CODES = (
    (1, 1, 1, 1, 6, 2, 6, 1, 1, 1, 1),
    (1, 1, 1, 7, 0, 0, 0, 8, 1, 1, 1),
    (1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (7, 0, 0, 1, 10, 4, 10, 1, 0, 0, 7),
    (1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1),
)
_FLOOR = ((1, 4), (1, 5), (1, 6),
          (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
          (4, 1), (4, 2), (4, 8), (4, 9))

# Shortest-path distance and first move between floor cells (999 = other room).
_DIST = (
    (0, 1, 2, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999),
    (1, 0, 1, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999),
    (2, 1, 0, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999),
    (999, 999, 999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 8, 9),
    (999, 999, 999, 1, 0, 1, 2, 3, 4, 5, 6, 7, 2, 1, 7, 8),
    (999, 999, 999, 2, 1, 0, 1, 2, 3, 4, 5, 6, 3, 2, 6, 7),
    (999, 999, 999, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 5, 6),
    (999, 999, 999, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 4, 5),
    (999, 999, 999, 5, 4, 3, 2, 1, 0, 1, 2, 3, 6, 5, 3, 4),
    (999, 999, 999, 6, 5, 4, 3, 2, 1, 0, 1, 2, 7, 6, 2, 3),
    (999, 999, 999, 7, 6, 5, 4, 3, 2, 1, 0, 1, 8, 7, 1, 2),
    (999, 999, 999, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 2, 1),
    (999, 999, 999, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 9, 10),
    (999, 999, 999, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 0, 8, 9),
    (999, 999, 999, 8, 7, 6, 5, 4, 3, 2, 1, 2, 9, 8, 0, 1),
    (999, 999, 999, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 1, 0),
)
_STEP = (
    (4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (2, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (4, 4, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
    (4, 4, 4, 2, 2, 4, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 4, 0, 0, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 4, 0, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 4, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 0, 2, 2, 1, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 1, 1),
    (4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0),
    (4, 4, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 3, 3),
    (4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 4, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4),
)

# Action ids.
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5
# Channel that holds "self faces the direction of move action a" (a = 0..3).
_FACE_CH = (3, 2, 4, 1, 1)

# Floor-cell ids used by name.
_A0_L = 0    # (1,4): ingredient-0 supply, plate pile (0,4), counter (2,4)
_A0_M = 1    # (1,5): delivery station above, pot below
_A0_R = 2    # (1,6): ingredient-1 supply, plate pile (0,6), counter (2,6)
_A1_WATCH = 5    # (3,3): sees recipe indicator and pot, nearest to left supplies
_A1_POT = 7      # (3,5): faces the pot upwards
_A1_PICK_L = 12  # (4,1): ing0 to the left (4,0), ing1 below (5,1)
_A1_PICK_R = 15  # (4,9): ing0 to the right (4,10), ing1 below (5,9)
# Counters seat 0 can never see, used to discard unusable items.
_DUMP_STAND = (3, 13, 11, 14)
_DUMP_DIR = (_UP, _DOWN, _UP, _DOWN)
_DUMP_ROW = (2, 5, 2, 5)
_DUMP_COL = (1, 2, 9, 8)

# target, partner hand last step, fresh-counter bits, partner-hand freshness,
# stale-pot flag, counter-occupancy bits, own-deposit bits, steps since delivery.
_INIT_0 = (-1, -1, 3, 1, 0, 0, 0, 60)
# target, pot ingredient counts, pot cooked, stall counter, delivery hold-off,
# "partner loads the pot too" flag, "we inserted last step" flag.
_INIT_1 = (-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

def _iv(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _crops():
    """Predicted static 5x5 crop for each floor cell (exact localisation key)."""
    codes = _iv(_CODES)
    planes = jnp.stack([(codes == k).astype(jnp.int32) for k in range(1, 11)], axis=-1)
    pad = jnp.zeros((10, 15, 10), dtype=jnp.int32).at[2:8, 2:13, :].set(planes)
    return jnp.stack([pad[r:r + 5, c:c + 5, :] for (r, c) in _FLOOR])


def _locate(obs):
    diff = jnp.abs(_crops() - obs[None, :, :, 22:32]).sum(axis=(1, 2, 3))
    return jnp.argmin(diff)


def _masked(obs, mask, lo, hi):
    return (obs[:, :, lo:hi] * mask[:, :, None]).sum(axis=(0, 1))


def _cell_vec(obs, row, col, r, c):
    """Channel vector of global cell (r, c); zeros when it is out of view."""
    i = r - row + 2
    j = c - col + 2
    inside = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    vec = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
    return jnp.where(inside, vec, jnp.zeros_like(vec))


def _move(obs, here, target, direction, do_interact):
    """Walk to `target`, then face `direction` and INTERACT when requested."""
    at_target = here == target
    facing = obs[2, 2, _iv(_FACE_CH)[direction]] > 0
    turn_or_act = jnp.where(
        do_interact,
        jnp.where(facing, _INTERACT, direction),
        jnp.where(direction < 4, direction, _STAY),
    )
    return jnp.where(at_target, turn_or_act, _iv(_STEP)[here, target]).astype(jnp.int32)


def _pick(conds, targets, dirs, inters):
    """First-true selection over a fixed priority list."""
    order = jnp.argmax(jnp.stack(conds))
    return (jnp.stack(targets)[order],
            jnp.stack(dirs)[order],
            jnp.stack(inters)[order],
            order.astype(jnp.int32))


def _inventory(obs):
    ing = obs[2, 2, 7:11]
    plate = obs[2, 2, 5]
    cooked = obs[2, 2, 6]
    count = ing.sum()
    return ing, count, jnp.argmax(ing), plate, cooked


# --------------------------------------------------------------------------
# Seat 0: the kitchen seat (supplies, pot, plates, delivery station).
# --------------------------------------------------------------------------

def init_agent_0():
    return _iv(_INIT_0)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, _iv(_INIT_0), memory)
    target_prev = mem[0]
    hand_prev = mem[1]
    cnt_ok = mem[2]
    hand_fresh = mem[3]
    pot_stale = mem[4]
    prev_item = mem[5]
    self_dumped = mem[6]
    settle = mem[7]

    here = _locate(obs)
    cells = _iv(_FLOOR)
    row = cells[here, 0]
    col = cells[here, 1]

    self_ing, held_count, held_type, self_plate, self_cooked = _inventory(obs)
    held_dish = (self_plate > 0) & (self_cooked > 0)
    held_plate = (self_plate > 0) & (self_cooked == 0)
    held_raw = (held_count > 0) & (self_plate == 0) & (self_cooked == 0)

    # Pot state; the pot is inside the crop from every seat-0 cell.
    pot_mask = obs[:, :, 24]
    pot_ing = _masked(obs, pot_mask, 34, 38)
    pot_total = pot_ing.sum()
    pot_cooked = (obs[:, :, 33] * pot_mask).sum()
    pot_timer = (obs[:, :, 44] * pot_mask).sum()
    pot_open = (pot_cooked == 0) & (pot_timer == 0) & (pot_total < 3)
    pot_busy = (pot_cooked > 0) | (pot_total >= 3)

    # Partner evidence: a raw ingredient carried in view.
    par_mask = obs[:, :, 11]
    par_ing = _masked(obs, par_mask, 18, 22)
    par_plate = (obs[:, :, 16] * par_mask).sum()
    par_cooked = (obs[:, :, 17] * par_mask).sum()
    par_raw = (par_plate == 0) & (par_cooked == 0)
    hand_now = jnp.where(par_raw & (par_ing[0] > 0), 0,
                         jnp.where(par_raw & (par_ing[1] > 0), 1, -1))
    # Two consecutive sightings, so one transient frame cannot flip the belief.
    hand_seen = (hand_now >= 0) & (hand_now == hand_prev)
    par_seen = par_mask.sum() > 0
    # A dish we delivered ourselves resampled the recipe, so anything the
    # partner was already carrying at that moment says nothing about the new
    # one.  Only trust carried items once the partner could have collected
    # them afterwards -- or once waiting has clearly cost more than guessing.
    patience = settle >= 14
    hand_fresh_now = (hand_fresh > 0) | (par_seen & (hand_now < 0))
    hand_ok = hand_seen & (hand_fresh_now | patience)
    hand_soft = (hand_now >= 0) & (hand_fresh_now | patience)

    # Counter evidence, per shared counter, with the same freshness test.
    cell_l = _cell_vec(obs, row, col, 2, 4)
    cell_r = _cell_vec(obs, row, col, 2, 6)
    left_item = cell_l[32:38].sum() > 0
    right_item = cell_r[32:38].sum() > 0
    left_raw = (cell_l[32] == 0) & (cell_l[33] == 0) & left_item
    right_raw = (cell_r[32] == 0) & (cell_r[33] == 0) & right_item
    mine_l = (self_dumped % 2) > 0
    mine_r = ((self_dumped // 2) % 2) > 0
    # A counter item that was already lying there when we delivered says
    # nothing about the new recipe; it is collected rather than believed --
    # until waiting has clearly become more expensive than acting on it.
    ok_l = ((cnt_ok % 2) > 0) | patience
    ok_r = (((cnt_ok // 2) % 2) > 0) | patience
    use_l = left_raw & ok_l & (~mine_l)
    use_r = right_raw & ok_r & (~mine_r)
    cnt_ing = (jnp.where(use_l, cell_l[34:38], 0)
               + jnp.where(use_r, cell_r[34:38], 0))

    # Strict evidence priority: what is already in the pot beats what the
    # partner carries, which beats what is lying on a counter.  Ambiguous
    # evidence (none, or both ingredients at once) is simply ignored.
    pot_raw = jnp.where((pot_stale > 0) & (~patience), 0, pot_ing)
    pot_says = jnp.where((pot_raw[0] > 0) & (pot_raw[1] == 0), 0,
                         jnp.where((pot_raw[1] > 0) & (pot_raw[0] == 0), 1, -1))
    cnt_says = jnp.where((cnt_ing[0] > 0) & (cnt_ing[1] == 0), 0,
                         jnp.where((cnt_ing[1] > 0) & (cnt_ing[0] == 0), 1, -1))
    # The recipe only changes when we deliver, so a belief already established
    # in this epoch outranks a weak counter reading; only the pot or a carried
    # ingredient may revise it.
    below_hand = jnp.where(target_prev >= 0, target_prev, cnt_says)
    # `hard` only trusts a partner sighting confirmed on two frames; `soft`
    # reacts one step earlier and is used for walking, never for picking up.
    target_hard = jnp.where(pot_says >= 0, pot_says,
                            jnp.where(hand_ok, hand_now, below_hand))
    target_soft = jnp.where(pot_says >= 0, pot_says,
                            jnp.where(hand_soft, hand_now, below_hand))
    target_ing = target_hard
    known = target_ing >= 0
    known_soft = target_soft >= 0
    want = jnp.clip(target_ing, 0, 3)
    want_soft = jnp.clip(target_soft, 0, 3)

    # A partner that hands ingredients over on a shared counter should be
    # taken up on it: collecting from there costs exactly as much as a trip
    # to the supply and frees the counter again.
    occ_left = left_item
    occ_right = right_item
    left_has_want = left_raw & (cell_l[34 + want_soft] > 0)
    right_has_want = right_raw & (cell_r[34 + want_soft] > 0)
    cost_left = jnp.abs(col - 4) + jnp.where(occ_left, 100, 0)
    cost_right = jnp.abs(col - 6) + jnp.where(occ_right, 100, 0)
    dump_left = cost_left <= cost_right
    dump_target = jnp.where(dump_left, _A0_L, _A0_R)
    dump_blocked = occ_left & occ_right

    # Is the dish in hand the requested one?
    dish_pure = (held_count == 3) & (self_ing[held_type] == 3) & (held_type <= 1)
    deliver_ok = dish_pure & (~known | (held_type == target_ing))

    # Is the pot ruined for the current target?
    bad_mix = (pot_ing[2] + pot_ing[3] > 0) | ((pot_ing[0] > 0) & (pot_ing[1] > 0))
    wrong_fill = known & (pot_total > 0) & (pot_ing[want] != pot_total)
    hopeless = (pot_total > 0) & (pot_cooked == 0) & (bad_mix | wrong_fill)

    compatible = pot_ing[held_type] == pot_total
    # Never commit an unverified ingredient to the pot; the only exception is
    # a very long dry spell, where guessing beats standing still for ever.
    verified = jnp.where(known, held_type == target_ing, settle >= 30)
    insert_ok = pot_open & ((compatible & verified) | hopeless)

    # An ingredient left on a shared counter before our own delivery may be
    # obsolete, and a partner that hands over on counters cannot replace it
    # while it sits there.  Collect it (that re-opens the channel) and hold it
    # until fresh evidence says whether it is the right one.
    # Only ever pick one up while the other counter is free, so that the item
    # can always be put down again.
    untrusted_l = left_raw & (~use_l) & (~mine_l) & (~right_item)
    untrusted_r = right_raw & (~use_r) & (~mine_r) & (~left_item)
    collect = (held_count == 0) & (~known_soft) & (untrusted_l | untrusted_r)
    collect_stand = jnp.where(untrusted_l, _iv(_A0_L), _iv(_A0_R))

    near_stand = jnp.where(col <= 5, _A0_L, _A0_R)
    near_dir = jnp.where(col <= 5, _iv(_LEFT), _iv(_RIGHT))
    supply_stand = jnp.where(target_soft == 0, _A0_L, _A0_R)
    supply_dir = jnp.where(target_soft == 0, _iv(_LEFT), _iv(_RIGHT))
    fetch_stand = jnp.where(left_has_want, _iv(_A0_L),
                            jnp.where(right_has_want, _iv(_A0_R), supply_stand))
    fetch_dir = jnp.where(left_has_want | right_has_want, _iv(_DOWN), supply_dir)

    one = _iv(1)
    zero = _iv(0)
    target, direction, do_interact, choice = _pick(
        [
            held_dish & (deliver_ok | (dish_pure & dump_blocked)),  # deliver
            held_dish & dump_blocked,                  # nowhere to park it: hold on
            held_dish,                                 # park a wrong dish
            held_plate & pot_busy,                     # scoop as soon as it is ready
            held_plate & dump_blocked,                 # hold the plate
            held_plate,                                # stray plate -> counter
            held_raw & insert_ok,                      # load the pot
            held_raw & (~known) & (settle < 30),       # keep it until it is verified
            held_raw & dump_blocked,                   # hold the ingredient
            held_raw,                                  # unusable ingredient -> counter
            (held_count == 0) & pot_busy,              # fetch a plate
            (held_count == 0) & hopeless,              # force a ruined pot to cook
            collect & pot_open,                        # re-open the counter channel
            (held_count == 0) & known_soft & pot_open,  # fetch the target ingredient
            jnp.asarray(True),                         # stand by, facing the pot
        ],
        [_iv(_A0_M), _iv(_A0_M), dump_target, _iv(_A0_M), _iv(_A0_M), dump_target,
         _iv(_A0_M), _iv(_A0_M), _iv(_A0_M), dump_target,
         near_stand, near_stand, collect_stand, fetch_stand, _iv(_A0_M)],
        [_iv(_UP), _iv(_DOWN), _iv(_DOWN), _iv(_DOWN), _iv(_DOWN), _iv(_DOWN),
         _iv(_DOWN), _iv(_DOWN), _iv(_DOWN), _iv(_DOWN),
         _iv(_UP), near_dir, _iv(_DOWN), fetch_dir, _iv(_DOWN)],
        # The last ingredient pick-up waits for confirmed evidence; walking
        # towards the supply already starts on the unconfirmed sighting.
        [one, zero, one, one, zero, one, one, zero, zero, one, one, one, one,
         known.astype(jnp.int32), zero],
    )
    action = _move(obs, here, target, direction, do_interact > 0)

    delivered = (held_dish & (deliver_ok | (dish_pure & dump_blocked))
                 & (here == _A0_M) & (action == _INTERACT))
    dumping_raw = (held_raw & (~insert_ok) & (~dump_blocked)
                   & (action == _INTERACT))
    # Freshness bookkeeping for the next step.
    put_l = dumping_raw & dump_left
    put_r = dumping_raw & (~dump_left)
    mine_l_next = (mine_l & left_item) | put_l
    mine_r_next = (mine_r & right_item) | put_r
    # A newly placed counter item only carries information when the partner
    # picked it up after our delivery, i.e. when it had been seen empty-handed
    # before this step.
    could_be_new = (hand_fresh > 0) & (~delivered)
    appeared_l = left_raw & (~((prev_item % 2) > 0)) & (~put_l) & could_be_new
    appeared_r = right_raw & (~(((prev_item // 2) % 2) > 0)) & (~put_r) & could_be_new
    ok_l_next = (~delivered) & left_item & (((cnt_ok % 2) > 0) | appeared_l)
    ok_r_next = (~delivered) & right_item & ((((cnt_ok // 2) % 2) > 0) | appeared_r)
    pot_stale_next = jnp.where(delivered, pot_total > 0,
                               (pot_stale > 0) & (pot_total > 0))
    hand_fresh_next = jnp.where(delivered, par_seen & (hand_now < 0),
                                hand_fresh_now)

    mem_next = jnp.stack([
        jnp.where(delivered, _iv(-1), target_ing),
        hand_now,
        ok_l_next.astype(jnp.int32) + 2 * ok_r_next.astype(jnp.int32),
        hand_fresh_next.astype(jnp.int32),
        pot_stale_next.astype(jnp.int32),
        left_item.astype(jnp.int32) + 2 * right_item.astype(jnp.int32),
        mine_l_next.astype(jnp.int32) + 2 * mine_r_next.astype(jnp.int32),
        jnp.where(delivered, _iv(0), jnp.minimum(settle + 1, 60)),
    ]).astype(jnp.int32)
    trace = {
        'cell': here.astype(jnp.int32),
        'target': target_ing.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'pot_cooked': pot_cooked.astype(jnp.int32),
        'delivered': delivered.astype(jnp.int32),
        'choice': choice,
        'held': jnp.where(held_raw, held_type, -1).astype(jnp.int32),
    }
    return action.astype(jnp.int32), mem_next, trace


# --------------------------------------------------------------------------
# Seat 1: the corridor seat (reads the recipe, feeds and signals the pot).
# --------------------------------------------------------------------------

def init_agent_1():
    return _iv(_INIT_1)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, _iv(_INIT_1), memory)
    target_prev = mem[0]
    pot_prev = mem[1:5]
    cooked_prev = mem[5]
    stall = mem[6]
    holdoff = mem[7]
    helper = mem[8]
    inserted_last = mem[9] > 0
    spoiled = mem[10]

    here = _locate(obs)
    cells = _iv(_FLOOR)
    row = cells[here, 0]
    col = cells[here, 1]

    # Recipe indicator, visible from (3,3)..(3,7) only.
    rec_mask = obs[:, :, 25]
    rec = _masked(obs, rec_mask, 40, 44)
    rec_seen = (rec_mask.sum() > 0) & (rec[0] + rec[1] > 0)
    target_ing = jnp.where(rec_seen, jnp.where(rec[0] > 0, 0, 1), target_prev)
    known = target_ing >= 0
    want = jnp.clip(target_ing, 0, 3)

    # Pot state, remembered while it is out of view.
    pot_mask = obs[:, :, 24]
    pot_seen = pot_mask.sum() > 0
    pot_now = _masked(obs, pot_mask, 34, 38)
    cooked_now = (obs[:, :, 33] * pot_mask).sum()
    pot_ing = jnp.where(pot_seen, pot_now, pot_prev)
    pot_cooked = jnp.where(pot_seen, cooked_now, cooked_prev)
    pot_total = pot_ing.sum()
    pot_open = (pot_cooked == 0) & (pot_total < 3)

    # A plated dish on its way to the delivery station means the recipe is
    # about to be resampled: keep away from the pot until that has settled.
    par_mask = obs[:, :, 11]
    par_seen = par_mask.sum() > 0
    par_plate = (obs[:, :, 16] * par_mask).sum()
    par_cooked = (obs[:, :, 17] * par_mask).sum()
    partner_dish = (par_plate > 0) & (par_cooked > 0)
    scooped = pot_seen & (cooked_prev > 0) & (pot_now.sum() == 0)
    # The short blind window only matters while the partner cannot be seen:
    # an empty-handed partner in view proves that nothing is in flight.
    danger = partner_dish | ((holdoff > 0) & (~par_seen))

    self_ing, held_count, held_type, self_plate, self_cooked = _inventory(obs)
    held_raw = (held_count > 0) & (self_plate == 0) & (self_cooked == 0)
    held_any = (held_count > 0) | (self_plate > 0)

    compatible = pot_ing[held_type] == pot_total
    carries_target = held_raw & known & (held_type == target_ing)
    # A partner that keeps dropping the wrong ingredient into a pot we seeded
    # correctly is not reading the pot at all; seeding it again only makes its
    # wrong dishes arrive faster, so switch to showing the ingredient instead.
    trusts_pot = spoiled < 2
    insert_ok = carries_target & pot_open & compatible & (~danger) & trusts_pot
    # A pot that can no longer produce the current recipe has to be emptied by
    # the other seat first.  Until then, stand at the pot holding the right
    # ingredient: that keeps the request visible and allows an instant reload.
    pot_ruined = known & (pot_total > 0) & (pot_ing[want] != pot_total)
    display = carries_target & (pot_ruined | (~trusts_pot)) & (~insert_ok)
    spoil_event = (pot_seen & pot_ruined
                   & (pot_prev.sum() > 0) & (pot_prev[want] == pot_prev.sum()))

    # Where to discard an item we cannot use.
    dump_occ = jnp.stack([
        _cell_vec(obs, row, col, _DUMP_ROW[k], _DUMP_COL[k])[32:38].sum() > 0
        for k in range(4)
    ])
    dump_cost = _iv(_DIST)[here, _iv(_DUMP_STAND)] + jnp.where(dump_occ, 100, 0)
    dump_k = jnp.argmin(dump_cost)
    dump_full = dump_occ.all()
    # With every hiding place taken, keep the item out of the partner's sight
    # rather than waving an obsolete ingredient at it.
    dump_stand = jnp.where(dump_full, _iv(3), _iv(_DUMP_STAND)[dump_k])
    dump_dir = jnp.where(dump_full, _iv(_STAY), _iv(_DUMP_DIR)[dump_k])
    dump_act = jnp.where(dump_full, _iv(0), _iv(1))

    # Fetching: always be the first ingredient in an empty pot (that is the
    # signal), and top the pot up whenever nobody else does.
    pot_fits = known & (pot_ing[want] == pot_total)
    # Give a partner that loads the pot itself time to do so, but do not idle
    # for one that never has.
    grew = pot_seen & (pot_now.sum() > pot_prev.sum())
    helped = (helper > 0) | (grew & (~inserted_last))
    wait_for = jnp.where(helped, 6, 0)
    need_fetch = known & (~danger) & (
        (pot_open & pot_fits & ((pot_total == 0) | (stall >= wait_for))) | pot_ruined)
    left_closer = _iv(_DIST)[here, _A1_PICK_L] <= _iv(_DIST)[here, _A1_PICK_R]
    pick_stand = jnp.where(left_closer, _A1_PICK_L, _A1_PICK_R)
    pick_dir = jnp.where(target_ing == 1, _iv(_DOWN),
                         jnp.where(left_closer, _iv(_LEFT), _iv(_RIGHT)))

    one = _iv(1)
    zero = _iv(0)
    target, direction, do_interact, choice = _pick(
        [
            insert_ok,                        # carry the signal into the pot
            display,                          # show the request, wait for the pot
            held_any,                         # anything unusable -> hidden counter
            (~held_any) & need_fetch,         # collect the requested ingredient
            jnp.asarray(True),                # watch the recipe indicator
        ],
        [_iv(_A1_POT), _iv(_A1_POT), dump_stand, pick_stand, _iv(_A1_WATCH)],
        [_iv(_UP), _iv(_UP), dump_dir, pick_dir, _iv(_STAY)],
        [one, zero, dump_act, one, zero],
    )
    action = _move(obs, here, target, direction, do_interact > 0)

    changed = pot_seen & (pot_now.sum() != pot_prev.sum())
    stall_next = jnp.where(pot_seen,
                           jnp.where(changed, 0, jnp.minimum(stall + 1, 100)),
                           stall)
    holdoff_next = jnp.where(scooped, _iv(2), jnp.maximum(holdoff - 1, 0))
    inserting = insert_ok & (action == _INTERACT)

    mem_next = jnp.concatenate([
        target_ing.reshape(1),
        jnp.where(pot_seen, pot_now, pot_prev),
        pot_cooked.reshape(1),
        stall_next.reshape(1),
        holdoff_next.reshape(1),
        helped.astype(jnp.int32).reshape(1),
        inserting.astype(jnp.int32).reshape(1),
        jnp.minimum(spoiled + spoil_event.astype(jnp.int32), 5).reshape(1),
    ]).astype(jnp.int32)
    trace = {
        'cell': here.astype(jnp.int32),
        'target': target_ing.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'stall': stall_next.astype(jnp.int32),
        'choice': choice,
        'held': jnp.where(held_raw, held_type, -1).astype(jnp.int32),
    }
    return action.astype(jnp.int32), mem_next, trace
