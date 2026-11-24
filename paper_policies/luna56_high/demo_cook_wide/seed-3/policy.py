"""A fixed-geometry complementary policy for demo_cook_wide."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'geometry_roles_v2', 'adapter': 'overcooked_v2',
                   'adapter_version': 1, 'max_height': 6, 'max_width': 11,
                   'num_ingredients': 4}


def init_agent_0():
    # phase, local step counter
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, jnp.zeros((2,), dtype=jnp.int32), memory)
    phase, count = mem
    facing = jnp.argmax(obs[2, 2, 1:5])
    # Top-room spawn is on row 1. Four left attempts normalize to c3, then
    # right reaches the c4 plate staging square from every allowed spawn.
    plate_here = obs[1, 2, 27] > 0
    plate_right = obs[2, 3, 29] > 0
    fixed_action0 = jnp.where(count < 4, 2, jnp.where(count == 4, 0,
                               jnp.where(count == 5, 3, 5)))
    fixed_next0 = jnp.where(count < 6,
                            jnp.asarray([0, count + 1], dtype=jnp.int32),
                            jnp.asarray([1, 0], dtype=jnp.int32))
    action0 = jnp.where(plate_here, jnp.where(facing == 0, 5, 3), fixed_action0)
    next0 = jnp.where(plate_here & (facing == 0),
                      jnp.asarray([1, jnp.where(plate_right, 2, 0)], dtype=jnp.int32),
                      fixed_next0)
    # From c4 move right, or from c6 move left, then face down toward pot.
    action1 = jnp.where(count == 0, 0, jnp.where(count == 2, 2, 1))
    next1 = jnp.where((count == 0) | (count == 2),
                      jnp.asarray([1, count + 1], dtype=jnp.int32),
                      jnp.asarray([2, 0], dtype=jnp.int32))
    pot_ready = obs[3, 2, 33] > 0
    action2 = jnp.where(pot_ready, jnp.where(facing == 1, 5, 1), 4)
    next2 = jnp.where(pot_ready & (facing == 1),
                      jnp.asarray([3, 0], dtype=jnp.int32),
                      jnp.asarray([2, 0], dtype=jnp.int32))
    # Collection leaves the player at c5 facing down; face up at the goal,
    # then deliver on the following interaction.
    action3 = jnp.where(facing == 0, 5, 3)
    next3 = jnp.where(facing == 0,
                      jnp.asarray([0, 0], dtype=jnp.int32),
                      jnp.asarray([3, 0], dtype=jnp.int32))
    action = jnp.where(phase == 0, action0,
              jnp.where(phase == 1, action1,
              jnp.where(phase == 2, action2, action3)))
    next_mem = jnp.where(phase == 0, next0,
                jnp.where(phase == 1, next1,
                jnp.where(phase == 2, next2, next3)))
    trace = {
        'phase': phase, 'count': count,
        'held_plate': obs[2, 2, 5], 'held_cooked': obs[2, 2, 6],
        'plate_above': obs[1, 2, 27],
        'pot_cooked': obs[3, 2, 33], 'pot_timer': obs[3, 2, 44],
    }
    return jnp.asarray(action, dtype=jnp.int32), next_mem, trace


def init_agent_1():
    # phase, target ingredient (0/1), completed trips, local step counter
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, jnp.zeros((4,), dtype=jnp.int32), memory)
    phase, old_target, trip, count = mem
    facing = jnp.argmax(obs[2, 2, 1:5])
    recipe0 = jnp.any(obs[:, :, 40] > 0)
    recipe1 = jnp.any(obs[:, :, 41] > 0)
    target = jnp.where(recipe1, 1, jnp.where(recipe0, 0, old_target))
    has_recipe = recipe0 | recipe1

    # Phase 0 normalizes any lower-room spawn to the recipe indicator at the
    # middle of row 3. Lower side pockets first move up; row-3 positions then
    # use the visible indicator or the enclosing wall as a compass.
    below_obstacle = (obs[3, 2, 22] > 0) | jnp.any(obs[3, 2, 28:32] > 0)
    side_pocket = below_obstacle & ~(obs[1, 2, 22] > 0)
    indicator_here = obs[3, 2, 25] > 0
    indicator_visible = jnp.any(obs[:, :, 25] > 0)
    left_boundary = (obs[2, 0, 22] > 0) | (obs[2, 1, 22] > 0)
    right_boundary = (obs[2, 3, 22] > 0) | (obs[2, 4, 22] > 0)
    seek_right = jnp.where(indicator_visible,
                           jnp.argmax(obs[:, :, 25]) % 5 > 2,
                           left_boundary | ~right_boundary)
    action0 = jnp.where(side_pocket, 3,
               jnp.where(indicator_here, 4,
               jnp.where(seek_right, 0, 2)))
    next0 = jnp.where(indicator_here & has_recipe,
                      jnp.asarray([1, target, trip, 0], dtype=jnp.int32),
                      jnp.asarray([0, target, trip, 0], dtype=jnp.int32))

    # Starting at center c5: four left moves reach row 3 c1, then down to
    # the shared staging square row 4 c1.
    action1 = jnp.where(count < 4, 2, 1)
    next1 = jnp.where(count < 4,
                      jnp.asarray([1, target, trip, count + 1], dtype=jnp.int32),
                      jnp.asarray([2, target, trip, 0], dtype=jnp.int32))

    # Type 0 is adjacent on the left. Type 1 continues down one square.
    action2 = jnp.where(target == 0,
                        jnp.where(count == 0, jnp.where(facing == 3, 5, 2), 5),
                        1)
    next2 = jnp.where(target == 0,
                      jnp.where(count == 0,
                                jnp.asarray([2, target, trip, 1], dtype=jnp.int32),
                                jnp.asarray([4, target, trip, 0], dtype=jnp.int32)),
                      jnp.asarray([3, target, trip, 0], dtype=jnp.int32))

    # Type 1 pickup at row 5 c1; type 0 immediately begins its return route.
    action3 = jnp.where(target == 1, 5, 3)
    next3 = jnp.where(target == 1,
                      jnp.asarray([4, target, trip, 0], dtype=jnp.int32),
                      jnp.asarray([4, target, trip, 1], dtype=jnp.int32))

    # Return to center: type 0 needs one up; type 1 needs two. Then four right.
    up_steps = jnp.where(target == 0, 1, 2)
    action4 = jnp.where(count < up_steps, 3, 0)
    finished_return = jnp.where(target == 0, count >= 4, count >= 5)
    next4 = jnp.where(finished_return,
                      jnp.asarray([5, target, trip, 0], dtype=jnp.int32),
                      jnp.asarray([4, target, trip, count + 1], dtype=jnp.int32))

    # Face up into the pot, then add the carried unit. Three additions start
    # cooking; wait through the cook and delivery before beginning another set.
    action5 = jnp.where(facing == 0, 5, 3)
    next5 = jnp.where(facing == 0,
                      jnp.asarray([6, target, trip, 0], dtype=jnp.int32),
                      jnp.asarray([5, target, trip, 0], dtype=jnp.int32))
    completed = trip + 1
    next6 = jnp.where(completed >= 3,
                      jnp.asarray([7, target, 0, 25], dtype=jnp.int32),
                      jnp.asarray([0, target, completed, 0], dtype=jnp.int32))
    action7 = jnp.asarray(4, dtype=jnp.int32)
    pot_empty = jnp.all(obs[1, 2, 32:38] == 0)
    next7 = jnp.where(pot_empty,
                      jnp.asarray([0, target, 0, 0], dtype=jnp.int32),
                      jnp.asarray([7, target, 0, 0], dtype=jnp.int32))

    action = jnp.where(phase == 0, action0,
              jnp.where(phase == 1, action1,
              jnp.where(phase == 2, action2,
              jnp.where(phase == 3, action3,
              jnp.where(phase == 4, action4,
              jnp.where(phase == 5, action5,
              jnp.where(phase == 6, 5, action7)))))))
    next_mem = jnp.where(phase == 0, next0,
                jnp.where(phase == 1, next1,
                jnp.where(phase == 2, next2,
                jnp.where(phase == 3, next3,
                jnp.where(phase == 4, next4,
                jnp.where(phase == 5, next5,
                jnp.where(phase == 6, next6, next7)))))))
    trace = {
        'phase': phase, 'target': target, 'trip': trip, 'count': count,
        'held_i0': obs[2, 2, 7], 'held_i1': obs[2, 2, 8],
        'pot_i0': obs[1, 2, 34], 'pot_i1': obs[1, 2, 35],
        'pot_timer': obs[1, 2, 44],
    }
    return jnp.asarray(action, dtype=jnp.int32), next_mem, trace
