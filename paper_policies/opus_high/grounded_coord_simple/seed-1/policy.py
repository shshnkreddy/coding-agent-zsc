"""Grounded-coordination policy for overcooked_v2 layout ``grounded_coord_simple``.

Fixed geometry (from context/environment/specification.json)::

    col:  0 1 2 3 4 5 6 7
    row0  W W 2 W W W W W
    row1  W . . W B . . 0
    row2  R . . L P . . X
    row3  W . . W B . . 1
    row4  W W 2 W W W W W

The two floor regions are disconnected: nothing a player can walk on joins
columns 1-2 to columns 5-6, and no counter touches floor on both sides.  Hence
the roles are forced by the layout rather than by any private agreement:

* seat 0 lives in the left room and can only reach the always-on recipe
  indicator ``R(2,0)``, the recipe button ``L(2,3)``, ingredient-2 piles and two
  counters.  It cannot cook or deliver anything, so its only useful act is to
  broadcast the recipe by activating the button (cost 5).
* seat 1 lives in the right room and reaches the pot ``(2,4)``, both plate
  piles, ingredient 0 ``(1,7)``, ingredient 1 ``(3,7)`` and delivery ``(2,7)``.
  It must therefore cook and deliver alone.

Seat 1 can never see ``R`` (it would need column <= 2) but always sees the
button cell ``(2,3)`` while standing in column 5, and an active button displays
the *current* recipe in observation channels 36-38.  That display is the only
convention-free evidence of the target, so seat 1 keys everything on it and
never delivers a dish that contradicts the evidence it holds: a wrong delivery
costs -20 *and* leaves the recipe unchanged, which is far worse than idling.

Both seats read their own position from a landmark that is always inside their
view (the pot for seat 1, ``R`` for seat 0), so no dead reckoning is needed.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_button_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
# Facing ids as they appear in observation channels 1..4.
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3


def _i32(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _turn_action(direction):
    """Movement action that produces ``direction`` as facing."""
    table = jnp.asarray([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)
    return jnp.take(table, jnp.clip(direction, 0, 3))


def _goto(row, col, facing, t_row, t_col, t_dir, t_int):
    """Walk to ``(t_row, t_col)``, face ``t_dir``, then INTERACT or STAY.

    Rows are matched before columns; inside either 3x2 room every intermediate
    cell of such a path is floor.  Every interaction target of this policy is an
    object, so the final turn is always a blocked move that only changes facing.
    """
    vertical = jnp.where(t_row > row, _i32(A_DOWN), _i32(A_UP))
    horizontal = jnp.where(t_col > col, _i32(A_RIGHT), _i32(A_LEFT))
    on_cell = (row == t_row) & (col == t_col)
    aligned = on_cell & (facing == t_dir)
    settled = jnp.where(t_int > 0, _i32(A_INT), _i32(A_STAY))
    final = jnp.where(facing == t_dir, settled, _turn_action(t_dir))
    action = jnp.where(on_cell, final, jnp.where(row != t_row, vertical, horizontal))
    return action.astype(jnp.int32), aligned & (t_int > 0)


def _landmark(obs, channel):
    """Local (row, col) of the single static object marked by ``channel``."""
    plane = obs[:, :, channel]
    index = jnp.argmax(plane.reshape(-1))
    seen = jnp.max(plane) > 0
    return (index // 5).astype(jnp.int32), (index % 5).astype(jnp.int32), seen


def _displayed_recipe(obs):
    """Recipe counts summed over every visible display cell (R and active L)."""
    counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    return counts, jnp.sum(counts) > 0, jnp.argmax(counts).astype(jnp.int32)


# --------------------------------------------------------------------------- #
# seat 1: the cook (right room)
# --------------------------------------------------------------------------- #

def init_agent_1():
    # belief, fresh, signal_seen_ever, wait, prev_delivery_was_blind,
    # dump_slot, belief_is_blind, consecutive_silent_cycles
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)
    belief, fresh, seen, wait, prev_blind, dump_slot, blind, silent = [mem[i] for i in range(8)]

    me = obs[2, 2]
    facing = jnp.argmax(me[1:5]).astype(jnp.int32)
    held_plate = me[5] > 0
    held_cooked = me[6] > 0
    held_ing = me[7:10]
    held_total = jnp.sum(held_ing)
    held_dish = held_plate & held_cooked
    held_empty_plate = held_plate & (~held_cooked)
    holding_ing = (held_total > 0) & (~held_plate)
    held_kind = jnp.argmax(held_ing).astype(jnp.int32)

    # The pot is always inside seat 1's view, so it doubles as the localizer.
    pot_r, pot_c, _ = _landmark(obs, 22)
    row = jnp.clip(4 - pot_r, 1, 3)
    col = jnp.clip(6 - pot_c, 5, 6)
    pot = obs[pot_r, pot_c]
    pot_ing = pot[31:34]
    pot_total = jnp.sum(pot_ing)
    pot_cooked = pot[30] > 0
    pot_timer = pot[39]
    pot_cooking = pot_timer > 0
    pot_kind = jnp.argmax(pot_ing).astype(jnp.int32)

    # --- grounded recipe evidence -----------------------------------------
    _, has_display, display_kind = _displayed_recipe(obs)
    belief = jnp.where(has_display, display_kind, belief)
    blind = jnp.where(has_display, _i32(0), blind)
    fresh = jnp.where(has_display, _i32(1), fresh)
    seen = jnp.where(has_display, _i32(1), seen)
    wait = jnp.where(has_display, _i32(0), wait)
    silent = jnp.where(has_display, _i32(0), silent)

    # No evidence: watch the button from column 5 for a while.  A partner that
    # has signalled before is given much longer than one that never has, but
    # after two whole cycles without a single display the waiting itself is the
    # main loss, so stop paying for it.
    patience = jnp.where(silent >= 2, _i32(4),
                         jnp.where(seen > 0, _i32(26), _i32(18)))
    uninformed = fresh == 0
    waited = jnp.where(uninformed, wait + 1, wait)
    # Fallback guess.  After an evidence-backed delivery the recipe is freshly
    # resampled, so repeating the last target is as good as a coin flip (and is
    # right outright if the partner only signals on change).  After a *guessed*
    # delivery, flipping is strictly better: with two possible recipes a failed
    # delivery leaves exactly the other recipe in place, so alternating can
    # never fail twice in a row.
    guess = jnp.where(belief < 0, _i32(0),
                      jnp.where(prev_blind > 0, 1 - belief, belief))
    commit = uninformed & (waited >= patience)
    belief = jnp.where(commit, guess, belief)
    fresh = jnp.where(commit, _i32(1), fresh)
    blind = jnp.where(commit, _i32(1), blind)
    wait = jnp.where(commit, _i32(0), waited)

    known = fresh > 0
    target_kind = jnp.clip(belief, 0, 2)

    # What the pot still needs: keep a partially filled pot consistent with
    # whatever is already in it (raw contents can never be removed), otherwise
    # follow the believed recipe.
    need = jnp.where(pot_total > 0, pot_kind, target_kind)
    need_ok = ((pot_total > 0) | known) & (need < 2)
    pot_open = (pot_total < 3) & (~pot_cooking) & (~pot_cooked)
    dish_ok = known & (jnp.take(held_ing, target_kind) >= 3)

    # Target table: 0 watch, 1 pot, 2 plate pile, 3 ing0, 4 ing1, 5 deliver,
    # 6 dump on a free counter.
    slot = jnp.clip(dump_slot, 0, 3)
    dump_row = jnp.take(jnp.asarray([1, 3, 1, 3], jnp.int32), slot)
    dump_col = jnp.take(jnp.asarray([6, 6, 5, 5], jnp.int32), slot)
    dump_dir = jnp.take(jnp.asarray([D_UP, D_DOWN, D_UP, D_DOWN], jnp.int32), slot)
    t_rows = jnp.stack([_i32(2), _i32(2), _i32(1), _i32(1), _i32(3), _i32(2), dump_row])
    t_cols = jnp.stack([_i32(5), _i32(5), _i32(5), _i32(6), _i32(6), _i32(6), dump_col])
    t_dirs = jnp.stack([_i32(D_LEFT), _i32(D_LEFT), _i32(D_LEFT), _i32(D_RIGHT),
                        _i32(D_RIGHT), _i32(D_RIGHT), dump_dir])
    t_ints = jnp.stack([_i32(0), _i32(1), _i32(1), _i32(1), _i32(1), _i32(1), _i32(1)])

    fetch = jnp.where(need == 0, _i32(3), _i32(4))
    empty_hands = jnp.where(pot_cooked | pot_cooking | (pot_total >= 3), _i32(2),
                            jnp.where(need_ok, fetch, _i32(0)))
    with_ing = jnp.where((held_kind == need) & need_ok & pot_open, _i32(1), _i32(6))
    with_plate = jnp.where(pot_cooked, _i32(1), jnp.where(pot_cooking, _i32(0), _i32(6)))
    with_dish = jnp.where(dish_ok, _i32(5), jnp.where(known, _i32(6), _i32(0)))
    index = jnp.where(held_dish, with_dish,
                      jnp.where(held_empty_plate, with_plate,
                                jnp.where(holding_ing, with_ing, empty_hands)))

    action, acted = _goto(row, col, facing, t_rows[index], t_cols[index],
                          t_dirs[index], t_ints[index])
    delivered = acted & (index == 5)
    dumped = acted & (index == 6)

    # A correct delivery resamples the recipe, so the belief stops being
    # evidence the moment we deliver on it.
    fresh = jnp.where(delivered, _i32(0), fresh)
    prev_blind = jnp.where(delivered, blind, prev_blind)
    wait = jnp.where(delivered, _i32(0), wait)
    silent = jnp.where(delivered,
                       jnp.where(blind > 0, jnp.minimum(silent + 1, _i32(4)), _i32(0)),
                       silent)
    dump_slot = jnp.where(dumped, (slot + 1) % 4, slot)

    next_mem = jnp.stack([belief, fresh, seen, wait, prev_blind, dump_slot, blind, silent])
    trace = {
        'belief': belief.astype(jnp.int32),
        'fresh': fresh.astype(jnp.int32),
        'blind': blind.astype(jnp.int32),
        'seen_signal': seen.astype(jnp.int32),
        'wait': wait.astype(jnp.int32),
        'mode': index.astype(jnp.int32),
        'delivered': delivered.astype(jnp.int32),
        'dumped': dumped.astype(jnp.int32),
        'silent': silent.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_mem.astype(jnp.int32), trace


# --------------------------------------------------------------------------- #
# seat 0: the broadcaster (left room)
# --------------------------------------------------------------------------- #

def init_agent_0():
    # steps_since_press, press_count, ignored_count, prev_pot_total,
    # pot_ever_filled, warnings_this_cook, step_index, prev_pot_cooked,
    # collections_seen
    return jnp.asarray([300, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_0(), memory)
    (since, presses, ignored, prev_total, ever_filled, warns, step, prev_cooked,
     collections) = [mem[i] for i in range(9)]

    me = obs[2, 2]
    facing = jnp.argmax(me[1:5]).astype(jnp.int32)

    # R(2,0) is always inside seat 0's view and localizes this seat.
    ind_r, ind_c, _ = _landmark(obs, 23)
    row = jnp.clip(4 - ind_r, 1, 3)
    col = jnp.clip(2 - ind_c, 1, 2)

    # The always-on indicator tells seat 0 the current recipe every step.
    _, recipe_known, recipe_kind = _displayed_recipe(obs)

    # Is our own broadcast still up?  Only visible from column 2.
    btn_r, btn_c, btn_seen = _landmark(obs, 24)
    btn_cell = obs[btn_r, btn_c]
    button_on = btn_seen & (jnp.sum(btn_cell[36:39]) > 0)

    # The pot is visible from column 2 only; it is the sole feedback seat 0 has
    # about what the cook is doing.
    pot_r, pot_c, pot_seen = _landmark(obs, 22)
    pot = obs[pot_r, pot_c]
    pot_ing = pot[31:34]
    pot_total = jnp.where(pot_seen, jnp.sum(pot_ing), _i32(0))
    pot_cooked = pot_seen & (pot[30] > 0)
    pot_timer = jnp.where(pot_seen, pot[39], _i32(0))
    right_ing = jnp.take(pot_ing, recipe_kind)

    # Did the cook follow a recent broadcast?  If the first ingredient of a
    # batch contradicts the recipe while our display was up, the partner is not
    # using the button and paying 5 per cycle for it is waste.
    filled_now = pot_seen & (prev_total == 0) & (pot_total > 0)
    matched = right_ing == pot_total
    ignored = jnp.where(filled_now & (since <= 14),
                        jnp.clip(ignored + jnp.where(matched, _i32(-1), _i32(1)), 0, 3),
                        ignored)
    ever_filled = jnp.where(pot_seen & (pot_total > 0), _i32(1), ever_filled)
    warns = jnp.where(pot_seen & (pot_total == 0), _i32(0), warns)
    # A cooked pot that goes empty was collected with a plate: proof that the
    # partner really is running delivery cycles and that broadcasts pay off.
    collected = pot_seen & (prev_cooked > 0) & (pot_total == 0)
    collections = jnp.minimum(collections + collected.astype(jnp.int32), _i32(9))

    # Cooldown: one broadcast per empty-pot phase normally; back off hard when
    # the partner ignores the display, and also when nothing ever happens in
    # the pot (a partner that never cooks must not be paid for).
    cooldown = jnp.where(ignored >= 2, _i32(80),
                         jnp.where((ever_filled > 0) | (presses < 3), _i32(12), _i32(90)))
    pot_empty = pot_seen & (pot_total == 0) & (~pot_cooked) & (pot_timer <= 0)
    pot_wrong = pot_seen & (pot_total > 0) & (right_ing != pot_total)
    # Warn about a batch that cannot be delivered for +20, at most twice.
    warn = pot_wrong & (pot_cooked | (pot_timer <= 4)) & (warns < 1) & (ignored < 2)
    # Emptying a pot and delivering from it takes at least ~38 of the 400 steps,
    # so a broadcast this late can no longer pay for itself.
    in_time = step <= 358
    # Until a collection has been seen, spend little: a seat that never cooks
    # (or cooks at random) would otherwise drain 5 per broadcast all episode.
    # 12 covers one broadcast per delivery cycle even for a cook faster than
    # this one, while bounding what a cook that never reads the display costs.
    budget = jnp.where(collections > 0, _i32(12), _i32(5))
    want = (recipe_known & (~button_on) & (pot_empty | warn) & in_time
            & (since >= cooldown) & (presses < budget))

    action, pressed = _goto(row, col, facing, _i32(2), _i32(2), _i32(D_RIGHT),
                            want.astype(jnp.int32))

    since = jnp.where(pressed, _i32(0), jnp.minimum(since + 1, _i32(300)))
    presses = presses + pressed.astype(jnp.int32)
    warns = warns + (pressed & warn).astype(jnp.int32)
    prev_total = jnp.where(pot_seen, pot_total, prev_total)
    prev_cooked = jnp.where(pot_seen, pot_cooked.astype(jnp.int32), prev_cooked)

    step = jnp.minimum(step + 1, _i32(400))
    next_mem = jnp.stack([since, presses, ignored, prev_total, ever_filled, warns,
                          step, prev_cooked, collections])
    trace = {
        'pressed': pressed.astype(jnp.int32),
        'presses': presses.astype(jnp.int32),
        'ignored': ignored.astype(jnp.int32),
        'collections': collections.astype(jnp.int32),
        'recipe': recipe_kind.astype(jnp.int32),
        'button_on': button_on.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'pot_wrong': pot_wrong.astype(jnp.int32),
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_mem.astype(jnp.int32), trace
