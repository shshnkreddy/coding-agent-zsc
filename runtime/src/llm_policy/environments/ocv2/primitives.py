"""JAX-native mechanics available to LLM-authored OvercookedV2 policies.

The functions in this module deliberately stop below task strategy. They
decode local observations, maintain a private relative belief map, select
remembered objects, and perform navigation/facing/interaction mechanics.
They never receive the hidden environment state or complete layout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import Actions, Direction, StaticObject

from llm_policy.environments.ocv2.observation import OvercookedDefaultObsDecoder
from llm_policy.policies.types import (
    BeliefMap,
    ControllerState,
    OCV2Facts,
    PolicyTrace,
    SkillResult,
    TargetQuery,
)


# Facing directions. Movement action IDs have a different ordering in OCV2.
UP = int(Direction.UP)
DOWN = int(Direction.DOWN)
RIGHT = int(Direction.RIGHT)
LEFT = int(Direction.LEFT)
ACTION_UP = int(Actions.up)
ACTION_DOWN = int(Actions.down)
ACTION_RIGHT = int(Actions.right)
ACTION_LEFT = int(Actions.left)
STAY = int(Actions.stay)
INTERACT = int(Actions.interact)

# Belief-map object types. Zero is reserved for unknown.
UNKNOWN = 0
FLOOR = 1
WALL = 2
GOAL = 3
POT = 4
RECIPE_INDICATOR = 5
RECIPE_BUTTON = 6
PLATE_PILE = 7
INGREDIENT_PILE = 8
PARTNER = 9

# Pot states and wildcard.
ANY = -1
POT_EMPTY = 0
POT_HAS_INGREDIENTS = 1
POT_COOKING = 2
POT_COOKED = 3

# Target source modes.
VISIBLE_ONLY = 0
REMEMBERED_ONLY = 1
VISIBLE_OR_REMEMBERED = 2

# Target selection modes.
NEAREST = 0
OLDEST_OBSERVATION = 1
LOWEST_COORDINATE = 2
SEEDED_RANDOM = 3

# Frontier strategies.
NEAREST_FRONTIER = 0
OLDEST_FRONTIER = 1
SEEDED_RANDOM_FRONTIER = 2

# Skill statuses.
RUNNING = 0
SUCCEEDED = 1
FAILED = 2
TIMED_OUT = 3

# Skill identifiers.
SKILL_NONE = 0
SKILL_STAY = 1
SKILL_MOVE = 2
SKILL_INTERACT = 3
SKILL_FACE_DIRECTION = 4
SKILL_FACE_TARGET = 5
SKILL_NAVIGATE = 6
SKILL_INTERACT_WITH = 7
SKILL_EXPLORE = 8
SKILL_WAIT = 9

# Recipe source codes.
RECIPE_NOT_VISIBLE = 0
RECIPE_PRIVATE = 1
RECIPE_BUTTON_SOURCE = 2
RECIPE_MULTIPLE_SOURCES = 3

_DIRECTION_ACTIONS = jnp.asarray(
    [ACTION_UP, ACTION_DOWN, ACTION_RIGHT, ACTION_LEFT], dtype=jnp.int32
)
_DIRECTION_DELTAS = jnp.asarray([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
_INF = jnp.int32(1_000_000)


def _as_i32(value):
    """Convert a Python scalar or array to the API's standard int32 dtype."""
    return jnp.asarray(value, dtype=jnp.int32)


def _empty_trace(action=STAY, skill_id=SKILL_NONE):
    """Construct a neutral successful trace for an action with no target."""
    return PolicyTrace(
        phase_id=_as_i32(0),
        decision_id=_as_i32(0),
        skill_id=_as_i32(skill_id),
        target_type=_as_i32(UNKNOWN),
        target_detail=_as_i32(ANY),
        status=_as_i32(SUCCEEDED),
        action=_as_i32(action),
        movement_blocked=jnp.asarray(False),
    )


def init_controller(max_height: int, max_width: int, num_ingredients: int):
    """Create scalar controller memory for one role.

    The belief map is larger than the environment canvas because its origin is
    the agent's unknown reset position. A relative trajectory in an HxW world
    fits inside (2H-1)x(2W-1) without needing absolute coordinates.
    """
    map_height = 2 * int(max_height) - 1
    map_width = 2 * int(max_width) - 1
    center = jnp.asarray([max_height - 1, max_width - 1], dtype=jnp.int32)
    belief = BeliefMap(
        known=jnp.zeros((map_height, map_width), dtype=jnp.bool_),
        traversable=jnp.zeros((map_height, map_width), dtype=jnp.bool_),
        object_type=jnp.zeros((map_height, map_width), dtype=jnp.int8),
        ingredient_id=jnp.full((map_height, map_width), ANY, dtype=jnp.int8),
        pot_state=jnp.full((map_height, map_width), ANY, dtype=jnp.int8),
        pot_ingredient_counts=jnp.zeros(
            (map_height, map_width, int(num_ingredients)), dtype=jnp.int8
        ),
        seen_age=jnp.zeros((map_height, map_width), dtype=jnp.int32),
    )
    return ControllerState(
        belief=belief,
        pose=center,
        last_action=_as_i32(STAY),
        has_last_action=jnp.asarray(False),
        episode_step=_as_i32(0),
        last_inventory=jnp.zeros((2 + int(num_ingredients),), dtype=jnp.int32),
        last_recipe=jnp.zeros((int(num_ingredients),), dtype=jnp.int32),
        last_partner_facing=_as_i32(Direction.UP),
        partner_was_visible=jnp.asarray(False),
        active_skill=_as_i32(SKILL_NONE),
        skill_elapsed=_as_i32(0),
        skill_phase=_as_i32(0),
        skill_target=jnp.asarray([ANY, ANY], dtype=jnp.int32),
        skill_query_signature=jnp.full((5,), ANY, dtype=jnp.int32),
        wait_remaining=_as_i32(0),
    )


def _reset_controller_like(controller: ControllerState, num_ingredients: int):
    """Recreate a blank controller using an existing controller's map size."""
    max_height = (controller.belief.known.shape[0] + 1) // 2
    max_width = (controller.belief.known.shape[1] + 1) // 2
    return init_controller(max_height, max_width, num_ingredients)


def _tree_select(condition, true_tree, false_tree):
    """Select corresponding PyTree leaves without a Python data-dependent branch."""
    condition = jnp.asarray(condition, dtype=jnp.bool_)
    return jax.tree_util.tree_map(
        lambda true_value, false_value: jnp.where(condition, true_value, false_value),
        true_tree,
        false_tree,
    )


def select_pytree(condition, true_tree, false_tree):
    """Leaf-wise JAX selection for generated policy memories."""
    return _tree_select(condition, true_tree, false_tree)


def select_trace(condition, true_trace, false_trace):
    """Select one of two fixed-shape policy traces using a scalar JAX condition."""
    return _tree_select(condition, true_trace, false_trace)


def select_result(condition, true_result, false_result):
    """Select one complete SkillResult, including its action and controller.

    This is leaf-wise selection, not short-circuit control flow: callers have
    already constructed both candidate results.
    """
    return _tree_select(condition, true_result, false_result)


def _decode_inventory(obs, position_channel, inventory_slice):
    """Read inventory layers at an agent's marked position in a local view."""
    position_mask = obs[..., position_channel] > 0
    layers = obs[..., inventory_slice]
    return jnp.sum(layers * position_mask[..., None], axis=(0, 1)).astype(jnp.int32)


def _decode_local_grid(obs, decoder: OvercookedDefaultObsDecoder):
    """Translate raw OCV2 channels into semantic local object and recipe grids."""
    static_layers = obs[..., decoder.static_slice]
    static_present = jnp.any(static_layers > 0, axis=-1)
    static_index = jnp.argmax(static_layers, axis=-1)
    static_codes = jnp.asarray(
        [WALL, GOAL, POT, RECIPE_INDICATOR, RECIPE_BUTTON, PLATE_PILE],
        dtype=jnp.int32,
    )
    object_type = jnp.where(
        static_present,
        static_codes[static_index],
        jnp.asarray(FLOOR, dtype=jnp.int32),
    )

    pile_layers = obs[..., decoder.ingredient_pile_slice]
    pile_present = jnp.any(pile_layers > 0, axis=-1)
    pile_id = jnp.argmax(pile_layers, axis=-1).astype(jnp.int32)
    object_type = jnp.where(pile_present, INGREDIENT_PILE, object_type)
    ingredient_id = jnp.where(pile_present, pile_id, ANY)

    dynamic = obs[..., decoder.dynamic_slice]
    dynamic_ingredients = dynamic[..., 2:]
    has_ingredients = jnp.any(dynamic_ingredients > 0, axis=-1)
    cooked = dynamic[..., 1] > 0
    timer = obs[..., decoder.minimum_channels - 1]
    pot_mask = object_type == POT
    pot_state = jnp.where(
        pot_mask & cooked,
        POT_COOKED,
        jnp.where(
            pot_mask & (timer > 0),
            POT_COOKING,
            jnp.where(pot_mask & has_ingredients, POT_HAS_INGREDIENTS, POT_EMPTY),
        ),
    )
    pot_state = jnp.where(pot_mask, pot_state, ANY).astype(jnp.int32)

    recipe_layers = obs[..., decoder.recipe_slice][..., 2:]
    local_visible = jnp.ones(obs.shape[:2], dtype=jnp.bool_)
    return (
        object_type.astype(jnp.int32),
        ingredient_id.astype(jnp.int32),
        pot_state,
        dynamic_ingredients.astype(jnp.int32),
        recipe_layers.astype(jnp.int32),
        local_visible,
    )


def _gather_map_at_local(belief_values, pose, self_local_position, local_shape):
    """Project a belief-map array into the coordinate frame of the local view."""
    rows = jnp.arange(local_shape[0], dtype=jnp.int32)[:, None]
    cols = jnp.arange(local_shape[1], dtype=jnp.int32)[None, :]
    map_y = pose[0] + rows - self_local_position[0]
    map_x = pose[1] + cols - self_local_position[1]
    map_y = jnp.clip(map_y, 0, belief_values.shape[0] - 1)
    map_x = jnp.clip(map_x, 0, belief_values.shape[1] - 1)
    return belief_values[map_y, map_x], map_y, map_x


def _alignment_score(belief, pose, self_local_position, local_object_type):
    """Count remembered static cells that agree with a view at a candidate pose."""
    remembered, map_y, map_x = _gather_map_at_local(
        belief.object_type,
        pose,
        self_local_position,
        local_object_type.shape,
    )
    known = belief.known[map_y, map_x]
    comparable = known & (remembered != PARTNER)
    return jnp.sum(comparable & (remembered == local_object_type)).astype(jnp.int32)


def _infer_pose(controller, self_local_position, local_object_type):
    """Infer whether the last movement succeeded by aligning consecutive views."""
    action = controller.last_action
    direction_matches = _DIRECTION_ACTIONS == action
    is_move = jnp.any(direction_matches)
    direction = jnp.argmax(direction_matches).astype(jnp.int32)
    moved_pose = controller.pose + _DIRECTION_DELTAS[direction]
    moved_pose = jnp.stack(
        [
            jnp.clip(moved_pose[0], 0, controller.belief.known.shape[0] - 1),
            jnp.clip(moved_pose[1], 0, controller.belief.known.shape[1] - 1),
        ]
    )
    stayed_score = _alignment_score(
        controller.belief,
        controller.pose,
        self_local_position,
        local_object_type,
    )
    moved_score = _alignment_score(
        controller.belief,
        moved_pose,
        self_local_position,
        local_object_type,
    )
    choose_moved = is_move & (moved_score >= stayed_score)
    pose = jnp.where(choose_moved, moved_pose, controller.pose)
    blocked = controller.has_last_action & is_move & ~choose_moved
    return pose, blocked


def _integrate_belief(
    belief,
    pose,
    self_local_position,
    object_type,
    ingredient_id,
    pot_state,
    pot_ingredient_counts,
):
    """Scatter the currently visible semantic grid into the relative belief map."""
    _, map_y, map_x = _gather_map_at_local(
        belief.object_type,
        pose,
        self_local_position,
        object_type.shape,
    )
    aged = jnp.where(
        belief.known,
        jnp.minimum(belief.seen_age + 1, jnp.iinfo(jnp.int32).max),
        belief.seen_age,
    )
    known = belief.known.at[map_y, map_x].set(True)
    traversable = belief.traversable.at[map_y, map_x].set(object_type == FLOOR)
    remembered_type = belief.object_type.at[map_y, map_x].set(
        object_type.astype(jnp.int8)
    )
    remembered_ingredient = belief.ingredient_id.at[map_y, map_x].set(
        ingredient_id.astype(jnp.int8)
    )
    remembered_pot = belief.pot_state.at[map_y, map_x].set(pot_state.astype(jnp.int8))
    remembered_pot_ingredients = belief.pot_ingredient_counts.at[map_y, map_x].set(
        pot_ingredient_counts.astype(jnp.int8)
    )
    seen_age = aged.at[map_y, map_x].set(0)
    return BeliefMap(
        known=known,
        traversable=traversable,
        object_type=remembered_type,
        ingredient_id=remembered_ingredient,
        pot_state=remembered_pot,
        pot_ingredient_counts=remembered_pot_ingredients,
        seen_age=seen_age,
    )


def observe(obs, reset, controller: ControllerState):
    """Decode one local observation and update private controller memory."""
    num_ingredients = controller.last_recipe.shape[0]
    reset_controller = _reset_controller_like(controller, num_ingredients)
    controller = _tree_select(reset, reset_controller, controller)
    decoder = OvercookedDefaultObsDecoder(num_ingredients)

    self_mask = obs[..., decoder.self_position_channel] > 0
    self_flat = jnp.argmax(self_mask.reshape(-1))
    self_position = jnp.asarray(
        [self_flat // obs.shape[1], self_flat % obs.shape[1]], dtype=jnp.int32
    )
    self_inventory = _decode_inventory(
        obs, decoder.self_position_channel, decoder.self_inventory_slice
    )
    self_facing = decoder.direction(obs)

    agent_width = decoder.partner_position_channel
    partner_inventory_slice = slice(
        decoder.partner_direction_slice.stop,
        2 * agent_width,
    )
    partner_inventory = _decode_inventory(
        obs,
        decoder.partner_position_channel,
        partner_inventory_slice,
    )
    partner_mask = obs[..., decoder.partner_position_channel] > 0
    partner_visible = jnp.any(partner_mask)
    partner_flat = jnp.argmax(partner_mask.reshape(-1))
    partner_position = jnp.asarray(
        [partner_flat // obs.shape[1], partner_flat % obs.shape[1]],
        dtype=jnp.int32,
    )
    partner_relative = partner_position - self_position
    partner_facing = decoder.partner_direction(obs)

    (
        local_object_type,
        local_ingredient_id,
        local_pot_state,
        local_pot_ingredient_counts,
        local_recipe_counts,
        local_visible,
    ) = _decode_local_grid(obs, decoder)
    previous_local_type, _, _ = _gather_map_at_local(
        controller.belief.object_type,
        controller.pose,
        self_position,
        local_object_type.shape,
    )
    pose, blocked = _infer_pose(controller, self_position, local_object_type)
    interaction_changed = (
        controller.has_last_action
        & (controller.last_action == INTERACT)
        & jnp.any(previous_local_type != local_object_type)
    )
    belief = _integrate_belief(
        controller.belief,
        pose,
        self_position,
        local_object_type,
        local_ingredient_id,
        local_pot_state,
        local_pot_ingredient_counts,
    )

    recipe_counts = decoder.visible_recipe_counts(obs).astype(jnp.int32)
    recipe_visible = jnp.any(recipe_counts > 0)
    private_mask = local_object_type == RECIPE_INDICATOR
    button_mask = local_object_type == RECIPE_BUTTON
    private_has_recipe = jnp.any(private_mask[..., None] & (local_recipe_counts > 0))
    button_has_recipe = jnp.any(button_mask[..., None] & (local_recipe_counts > 0))
    recipe_source = jnp.where(
        private_has_recipe & button_has_recipe,
        RECIPE_MULTIPLE_SOURCES,
        jnp.where(
            private_has_recipe,
            RECIPE_PRIVATE,
            jnp.where(button_has_recipe, RECIPE_BUTTON_SOURCE, RECIPE_NOT_VISIBLE),
        ),
    ).astype(jnp.int32)

    controller_next = controller.replace(
        belief=belief,
        pose=pose,
        episode_step=controller.episode_step + 1,
        last_inventory=self_inventory,
        last_recipe=recipe_counts,
        last_partner_facing=partner_facing,
        partner_was_visible=partner_visible,
    )
    facts = OCV2Facts(
        is_initial_step=controller.episode_step == 0,
        episode_step=controller.episode_step,
        self_local_position=self_position,
        self_facing=self_facing,
        self_has_plate=self_inventory[0] > 0,
        self_has_cooked_dish=(self_inventory[0] > 0) & (self_inventory[1] > 0),
        self_ingredient_counts=self_inventory[2:],
        self_inventory_empty=jnp.all(self_inventory == 0),
        partner_visible=partner_visible,
        partner_relative_position=partner_relative,
        partner_relative_direction=_relative_direction(partner_relative),
        partner_facing=partner_facing,
        partner_has_plate=partner_inventory[0] > 0,
        partner_has_cooked_dish=(partner_inventory[0] > 0) & (partner_inventory[1] > 0),
        partner_ingredient_counts=partner_inventory[2:],
        partner_inventory_empty=jnp.all(partner_inventory == 0),
        partner_direction_changed=partner_visible
        & controller.partner_was_visible
        & (partner_facing != controller.last_partner_facing),
        recipe_visible=recipe_visible,
        recipe_counts=recipe_counts,
        recipe_source=recipe_source,
        recipe_changed=recipe_visible
        & jnp.any(recipe_counts != controller.last_recipe),
        any_indicator_visible=jnp.any(private_mask),
        any_button_visible=jnp.any(button_mask),
        any_button_active=button_has_recipe,
        any_pot_has_ingredients=jnp.any(local_pot_state == POT_HAS_INGREDIENTS),
        any_pot_cooking=jnp.any(local_pot_state == POT_COOKING),
        any_pot_cooked=jnp.any(local_pot_state == POT_COOKED),
        last_action=controller.last_action,
        last_move_blocked=blocked,
        last_inventory_changed=jnp.any(self_inventory != controller.last_inventory),
        last_interaction_changed_observation=interaction_changed,
        local_object_type=local_object_type,
        local_ingredient_id=local_ingredient_id,
        local_pot_state=local_pot_state,
        local_pot_ingredient_counts=local_pot_ingredient_counts,
        local_recipe_counts=local_recipe_counts,
        local_visible=local_visible,
    )
    return facts, controller_next


def _relative_direction(relative_position):
    """Return the cardinal direction of a displacement, preferring its wider axis."""
    dy, dx = relative_position[0], relative_position[1]
    return jnp.where(
        jnp.abs(dx) >= jnp.abs(dy),
        jnp.where(dx >= 0, Direction.RIGHT, Direction.LEFT),
        jnp.where(dy >= 0, Direction.DOWN, Direction.UP),
    ).astype(jnp.int32)


def target(
    object_type,
    *,
    ingredient_id=ANY,
    pot_state=ANY,
    source=VISIBLE_OR_REMEMBERED,
    selection=NEAREST,
):
    """Build a generic fixed-shape query for a remembered or visible object."""
    return TargetQuery(
        object_type=_as_i32(object_type),
        ingredient_id=_as_i32(ingredient_id),
        pot_state=_as_i32(pot_state),
        source=_as_i32(source),
        selection=_as_i32(selection),
    )


def ingredient_pile(
    ingredient_id,
    *,
    source=VISIBLE_OR_REMEMBERED,
    selection=NEAREST,
):
    """Build a target query for a pile containing one ingredient type."""
    return target(
        INGREDIENT_PILE,
        ingredient_id=ingredient_id,
        source=source,
        selection=selection,
    )


def pot(*, state=ANY, source=VISIBLE_OR_REMEMBERED, selection=NEAREST):
    """Build a target query for a pot, optionally restricted by cooking state."""
    return target(POT, pot_state=state, source=source, selection=selection)


def _query_signature(query):
    """Pack a TargetQuery into an array used to recognize a continuing skill."""
    return jnp.stack(
        [
            query.object_type,
            query.ingredient_id,
            query.pot_state,
            query.source,
            query.selection,
        ]
    ).astype(jnp.int32)


def _query_mask(query, controller):
    """Return belief-map cells satisfying all type, detail, and source filters."""
    belief = controller.belief
    mask = belief.known & (belief.object_type == query.object_type)
    mask &= (query.ingredient_id == ANY) | (belief.ingredient_id == query.ingredient_id)
    mask &= (query.pot_state == ANY) | (belief.pot_state == query.pot_state)
    visible = belief.seen_age == 0
    source_mask = jnp.where(
        query.source == VISIBLE_ONLY,
        visible,
        jnp.where(query.source == REMEMBERED_ONLY, ~visible, jnp.ones_like(visible)),
    )
    return mask & source_mask


def _resolve_target(query, controller, rng):
    """Resolve a query to one map coordinate using its deterministic/random rank."""
    mask = _query_mask(query, controller)
    height, width = mask.shape
    rows = jnp.arange(height, dtype=jnp.int32)[:, None]
    cols = jnp.arange(width, dtype=jnp.int32)[None, :]
    manhattan = jnp.abs(rows - controller.pose[0]) + jnp.abs(cols - controller.pose[1])
    coordinate_rank = rows * width + cols
    random_rank = jax.random.randint(
        rng, mask.shape, 0, jnp.iinfo(jnp.int32).max, dtype=jnp.int32
    )
    score = jnp.where(
        query.selection == OLDEST_OBSERVATION,
        -controller.belief.seen_age,
        jnp.where(
            query.selection == LOWEST_COORDINATE,
            coordinate_rank,
            jnp.where(query.selection == SEEDED_RANDOM, random_rank, manhattan),
        ),
    )
    score = jnp.where(mask, score, _INF)
    flat = jnp.argmin(score.reshape(-1))
    position = jnp.asarray([flat // width, flat % width], dtype=jnp.int32)
    return position, jnp.any(mask)


def target_known(query, facts, controller):
    """Report whether any belief-map cell currently satisfies a target query."""
    del facts
    return jnp.any(_query_mask(query, controller))


def target_visible(query, facts, controller):
    """Report whether a matching target occurs in the current local observation."""
    del facts
    visible_query = query.replace(source=_as_i32(VISIBLE_ONLY))
    return jnp.any(_query_mask(visible_query, controller))


def target_adjacent(query, facts, controller, rng=None):
    """Report whether the selected matching target is one cardinal step away."""
    del facts
    rng = jax.random.PRNGKey(0) if rng is None else rng
    position, found = _resolve_target(query, controller, rng)
    distance = jnp.sum(jnp.abs(position - controller.pose))
    return found & (distance == 1)


def target_relative_direction(query, facts, controller, rng=None):
    """Return the selected target's relative direction and whether it was found."""
    del facts
    rng = jax.random.PRNGKey(0) if rng is None else rng
    position, found = _resolve_target(query, controller, rng)
    return _relative_direction(position - controller.pose), found


def target_known_distance(query, facts, controller, rng=None):
    """Return Manhattan distance to the selected target, or a large sentinel."""
    del facts
    rng = jax.random.PRNGKey(0) if rng is None else rng
    position, found = _resolve_target(query, controller, rng)
    distance = jnp.sum(jnp.abs(position - controller.pose)).astype(jnp.int32)
    return jnp.where(found, distance, _INF)


def missing_ingredient_counts(recipe_counts, pot_ingredient_counts):
    """Compute non-negative ingredient deficits between a recipe and a pot."""
    return jnp.maximum(
        jnp.asarray(recipe_counts, dtype=jnp.int32)
        - jnp.asarray(pot_ingredient_counts, dtype=jnp.int32),
        0,
    )


def first_nonzero_ingredient(counts):
    """Return the lowest positive-count index; returns zero if none are positive."""
    counts = jnp.asarray(counts)
    return jnp.argmax(counts > 0).astype(jnp.int32)


def sample_nonzero_ingredient(counts, rng):
    """Sample uniformly among positive-count indices; requires at least one."""
    counts = jnp.asarray(counts)
    logits = jnp.where(counts > 0, 0.0, -jnp.inf)
    return jax.random.categorical(rng, logits).astype(jnp.int32)


def _set_action(controller, action):
    """Record the emitted action so the next observation can infer its effect."""
    return controller.replace(
        last_action=_as_i32(action),
        has_last_action=jnp.asarray(True),
    )


def _trace(skill_id, action, status, facts, query=None):
    """Create the device-side diagnostic record shared by every primitive."""
    target_type = UNKNOWN if query is None else query.object_type
    target_detail = ANY
    if query is not None:
        target_detail = jnp.where(
            query.ingredient_id != ANY, query.ingredient_id, query.pot_state
        )
    return PolicyTrace(
        phase_id=_as_i32(0),
        decision_id=_as_i32(0),
        skill_id=_as_i32(skill_id),
        target_type=_as_i32(target_type),
        target_detail=_as_i32(target_detail),
        status=_as_i32(status),
        action=_as_i32(action),
        movement_blocked=facts.last_move_blocked,
    )


def make_trace(phase_id, decision_id, result):
    """Annotate a primitive result with policy-defined phase and decision IDs."""
    return result.trace.replace(
        phase_id=_as_i32(phase_id), decision_id=_as_i32(decision_id)
    )


def _result(action, controller, status, facts, skill_id, query=None):
    """Finalize a primitive action, updated controller, status, and trace."""
    controller = _set_action(controller, action)
    trace = _trace(skill_id, action, status, facts, query)
    return SkillResult(
        action=_as_i32(action),
        controller=controller,
        status=_as_i32(status),
        trace=trace,
    )


def status_is(result, status):
    """Return a JAX boolean indicating whether a skill has the given status."""
    return result.status == _as_i32(status)


def cancel_active_skill(controller):
    """Clear all generic progress fields used by a multi-step primitive."""
    return controller.replace(
        active_skill=_as_i32(SKILL_NONE),
        skill_elapsed=_as_i32(0),
        skill_phase=_as_i32(0),
        skill_target=jnp.asarray([ANY, ANY], dtype=jnp.int32),
        skill_query_signature=jnp.full((5,), ANY, dtype=jnp.int32),
        wait_remaining=_as_i32(0),
    )


def stay(facts, controller):
    """Emit STAY for one step and cancel any active multi-step skill."""
    controller = cancel_active_skill(controller)
    return _result(STAY, controller, SUCCEEDED, facts, SKILL_STAY)


def move(direction, facts, controller):
    """Emit the OCV2 movement/turn action for a cardinal facing direction."""
    controller = cancel_active_skill(controller)
    action = _DIRECTION_ACTIONS[jnp.asarray(direction, dtype=jnp.int32)]
    return _result(action, controller, SUCCEEDED, facts, SKILL_MOVE)


def interact(facts, controller):
    """Emit one raw INTERACT action and cancel any active multi-step skill."""
    controller = cancel_active_skill(controller)
    return _result(INTERACT, controller, SUCCEEDED, facts, SKILL_INTERACT)


def _continue_skill(controller, skill_id, query_signature=None):
    """Start a skill or advance elapsed state when its identity/query matches."""
    same = controller.active_skill == skill_id
    if query_signature is not None:
        same &= jnp.all(controller.skill_query_signature == query_signature)
    elapsed = jnp.where(same, controller.skill_elapsed + 1, 0)
    phase = jnp.where(same, controller.skill_phase, 0)
    target_position = jnp.where(
        same,
        controller.skill_target,
        jnp.asarray([ANY, ANY], dtype=jnp.int32),
    )
    signature = (
        controller.skill_query_signature if query_signature is None else query_signature
    )
    return controller.replace(
        active_skill=_as_i32(skill_id),
        skill_elapsed=elapsed,
        skill_phase=phase,
        skill_target=target_position,
        skill_query_signature=signature,
    )


def face_direction(direction, facts, controller, timeout=8):
    """Issue a directional action until facing it or reaching a timeout.

    OCV2 has no turn-only action, so this can also move the agent when the cell
    in that direction is traversable.
    """
    controller = _continue_skill(controller, SKILL_FACE_DIRECTION)
    direction = jnp.asarray(direction, dtype=jnp.int32)
    done = facts.self_facing == direction
    timed_out = controller.skill_elapsed >= int(timeout)
    action = jnp.where(done | timed_out, STAY, _DIRECTION_ACTIONS[direction])
    status = jnp.where(done, SUCCEEDED, jnp.where(timed_out, TIMED_OUT, RUNNING))
    controller = _tree_select(
        done | timed_out, cancel_active_skill(controller), controller
    )
    return _result(action, controller, status, facts, SKILL_FACE_DIRECTION)


def _neighbor_mask(position, shape):
    """Return a grid mask containing the four cardinal neighbors of a cell."""
    rows = jnp.arange(shape[0], dtype=jnp.int32)[:, None]
    cols = jnp.arange(shape[1], dtype=jnp.int32)[None, :]
    return (jnp.abs(rows - position[0]) + jnp.abs(cols - position[1])) == 1


def _frontier_mask(controller):
    """Find known traversable cells bordering at least one unknown map cell."""
    known = controller.belief.known
    unknown = ~known
    adjacent_unknown = (
        jnp.roll(unknown, 1, axis=0)
        | jnp.roll(unknown, -1, axis=0)
        | jnp.roll(unknown, 1, axis=1)
        | jnp.roll(unknown, -1, axis=1)
    )
    # Prevent wrap-around produced by roll from creating false frontiers.
    adjacent_unknown = adjacent_unknown.at[0, :].set(unknown[0, :] | unknown[1, :])
    adjacent_unknown = adjacent_unknown.at[-1, :].set(unknown[-1, :] | unknown[-2, :])
    adjacent_unknown = adjacent_unknown.at[:, 0].set(unknown[:, 0] | unknown[:, 1])
    adjacent_unknown = adjacent_unknown.at[:, -1].set(unknown[:, -1] | unknown[:, -2])
    return controller.belief.traversable & adjacent_unknown


def frontier_exists(controller):
    """Report whether the current belief map contains an exploration frontier."""
    return jnp.any(_frontier_mask(controller))


def _distance_map(traversable, goal_mask):
    """Compute fixed-shape shortest-path distances backward from goal cells."""
    distance = jnp.where(goal_mask, 0, _INF).astype(jnp.int32)

    def relax(_, current):
        """Perform one parallel Bellman relaxation over cardinal neighbors."""
        padded = jnp.pad(current, ((1, 1), (1, 1)), constant_values=_INF)
        neighbor = jnp.minimum(
            jnp.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
            jnp.minimum(padded[1:-1, :-2], padded[1:-1, 2:]),
        )
        candidate = jnp.minimum(current, neighbor + 1)
        return jnp.where(traversable | goal_mask, candidate, _INF)

    steps = traversable.shape[0] * traversable.shape[1]
    return jax.lax.fori_loop(0, steps, relax, distance)


def _next_action_to_goal(controller, goal_mask):
    """Choose the first shortest-path move toward any marked goal cell."""
    distance = _distance_map(controller.belief.traversable, goal_mask)
    candidate_positions = controller.pose[None, :] + _DIRECTION_DELTAS
    y = jnp.clip(candidate_positions[:, 0], 0, distance.shape[0] - 1)
    x = jnp.clip(candidate_positions[:, 1], 0, distance.shape[1] - 1)
    values = distance[y, x]
    best_direction = jnp.argmin(values).astype(jnp.int32)
    path_found = jnp.min(values) < _INF
    return _DIRECTION_ACTIONS[best_direction], path_found


def path_known(query, facts, controller, rng=None):
    """Report whether known floor connects the agent beside the selected target."""
    del facts
    rng = jax.random.PRNGKey(0) if rng is None else rng
    position, found = _resolve_target(query, controller, rng)
    goal_mask = _neighbor_mask(position, controller.belief.known.shape)
    goal_mask &= controller.belief.traversable
    _, path_found = _next_action_to_goal(controller, goal_mask)
    adjacent = jnp.sum(jnp.abs(controller.pose - position)) == 1
    return found & (adjacent | path_found)


def _navigate_position(position, found, facts, controller, timeout, skill_id, query):
    """Advance a navigation skill toward a cell adjacent to an object position."""
    adjacent = found & (jnp.sum(jnp.abs(controller.pose - position)) == 1)
    goal_mask = _neighbor_mask(position, controller.belief.known.shape)
    goal_mask &= controller.belief.traversable
    action, path_found = _next_action_to_goal(controller, goal_mask)
    timed_out = controller.skill_elapsed >= int(timeout)
    failed = ~found | (~adjacent & ~path_found)
    terminal = adjacent | timed_out | failed
    status = jnp.where(
        adjacent,
        SUCCEEDED,
        jnp.where(timed_out, TIMED_OUT, jnp.where(failed, FAILED, RUNNING)),
    )
    action = jnp.where(terminal, STAY, action)
    controller = controller.replace(skill_target=position)
    controller = _tree_select(terminal, cancel_active_skill(controller), controller)
    return _result(action, controller, status, facts, skill_id, query)


def navigate_to(query, facts, controller, rng=None, timeout=40):
    """Move over known floor until adjacent to the object selected by a query."""
    rng = jax.random.PRNGKey(0) if rng is None else rng
    signature = _query_signature(query)
    controller = _continue_skill(controller, SKILL_NAVIGATE, signature)
    position, found = _resolve_target(query, controller, rng)
    return _navigate_position(
        position, found, facts, controller, timeout, SKILL_NAVIGATE, query
    )


def face_target(query, facts, controller, rng=None, timeout=8):
    """When already adjacent, turn to face the object selected by a query."""
    rng = jax.random.PRNGKey(0) if rng is None else rng
    signature = _query_signature(query)
    controller = _continue_skill(controller, SKILL_FACE_TARGET, signature)
    position, found = _resolve_target(query, controller, rng)
    adjacent = found & (jnp.sum(jnp.abs(position - controller.pose)) == 1)
    direction = _relative_direction(position - controller.pose)
    facing = facts.self_facing == direction
    timed_out = controller.skill_elapsed >= int(timeout)
    failed = ~adjacent
    terminal = facing | failed | timed_out
    status = jnp.where(
        facing,
        SUCCEEDED,
        jnp.where(failed, FAILED, jnp.where(timed_out, TIMED_OUT, RUNNING)),
    )
    action = jnp.where(terminal, STAY, _DIRECTION_ACTIONS[direction])
    controller = _tree_select(terminal, cancel_active_skill(controller), controller)
    return _result(action, controller, status, facts, SKILL_FACE_TARGET, query)


def interact_with(query, facts, controller, rng=None, timeout=50):
    """Navigate adjacent to a selected object, face it, and interact once.

    SUCCEEDED means the INTERACT action was emitted. Its environmental effect
    becomes observable, and can be checked, on the following policy call.
    """
    rng = jax.random.PRNGKey(0) if rng is None else rng
    signature = _query_signature(query)
    controller = _continue_skill(controller, SKILL_INTERACT_WITH, signature)
    position, found = _resolve_target(query, controller, rng)
    adjacent = found & (jnp.sum(jnp.abs(position - controller.pose)) == 1)
    direction = _relative_direction(position - controller.pose)
    facing = adjacent & (facts.self_facing == direction)
    timed_out = controller.skill_elapsed >= int(timeout)

    goal_mask = _neighbor_mask(position, controller.belief.known.shape)
    goal_mask &= controller.belief.traversable
    move_action, path_found = _next_action_to_goal(controller, goal_mask)
    failed = ~found | (~adjacent & ~path_found)
    action = jnp.where(
        facing,
        INTERACT,
        jnp.where(adjacent, _DIRECTION_ACTIONS[direction], move_action),
    )
    done = facing
    terminal = done | failed | timed_out
    status = jnp.where(
        done,
        SUCCEEDED,
        jnp.where(failed, FAILED, jnp.where(timed_out, TIMED_OUT, RUNNING)),
    )
    action = jnp.where(failed | timed_out, STAY, action)
    controller = controller.replace(skill_target=position)
    controller = _tree_select(terminal, cancel_active_skill(controller), controller)
    return _result(action, controller, status, facts, SKILL_INTERACT_WITH, query)


def _select_frontier(controller, rng, strategy):
    """Choose one exploration frontier by distance, age, or seeded random rank."""
    mask = _frontier_mask(controller)
    height, width = mask.shape
    rows = jnp.arange(height, dtype=jnp.int32)[:, None]
    cols = jnp.arange(width, dtype=jnp.int32)[None, :]
    manhattan = jnp.abs(rows - controller.pose[0]) + jnp.abs(cols - controller.pose[1])
    random_rank = jax.random.randint(
        rng, mask.shape, 0, jnp.iinfo(jnp.int32).max, dtype=jnp.int32
    )
    score = jnp.where(
        strategy == OLDEST_FRONTIER,
        -controller.belief.seen_age,
        jnp.where(strategy == SEEDED_RANDOM_FRONTIER, random_rank, manhattan),
    )
    score = jnp.where(mask, score, _INF)
    flat = jnp.argmin(score.reshape(-1))
    return jnp.asarray([flat // width, flat % width], dtype=jnp.int32), jnp.any(mask)


def explore_frontier(
    facts,
    controller,
    rng,
    *,
    strategy=NEAREST_FRONTIER,
    timeout=40,
):
    """Move to a known floor cell bordering unexplored belief-map territory."""
    controller = _continue_skill(controller, SKILL_EXPLORE)
    selected, found = _select_frontier(controller, rng, _as_i32(strategy))
    continuing = jnp.all(controller.skill_target >= 0)
    position = jnp.where(continuing, controller.skill_target, selected)
    reached = found & jnp.all(position == controller.pose)
    goal_mask = (
        jnp.zeros_like(controller.belief.known).at[position[0], position[1]].set(True)
    )
    action, path_found = _next_action_to_goal(controller, goal_mask)
    timed_out = controller.skill_elapsed >= int(timeout)
    failed = ~found | (~reached & ~path_found)
    terminal = reached | timed_out | failed
    status = jnp.where(
        reached,
        SUCCEEDED,
        jnp.where(failed, FAILED, jnp.where(timed_out, TIMED_OUT, RUNNING)),
    )
    action = jnp.where(terminal, STAY, action)
    controller = controller.replace(skill_target=position)
    controller = _tree_select(terminal, cancel_active_skill(controller), controller)
    return _result(action, controller, status, facts, SKILL_EXPLORE)


def wait(steps, facts, controller):
    """Emit STAY for a fixed number of consecutive calls using controller state."""
    steps = _as_i32(steps)
    continuing = controller.active_skill == SKILL_WAIT
    remaining = jnp.where(continuing, controller.wait_remaining - 1, steps - 1)
    done = remaining <= 0
    controller = controller.replace(
        active_skill=_as_i32(SKILL_WAIT),
        skill_elapsed=jnp.where(continuing, controller.skill_elapsed + 1, 0),
        wait_remaining=jnp.maximum(remaining, 0),
    )
    controller = _tree_select(done, cancel_active_skill(controller), controller)
    status = jnp.where(done, SUCCEEDED, RUNNING)
    return _result(STAY, controller, status, facts, SKILL_WAIT)


__all__ = [
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_UP",
    "ANY",
    "BeliefMap",
    "ControllerState",
    "DOWN",
    "FAILED",
    "FLOOR",
    "GOAL",
    "INGREDIENT_PILE",
    "INTERACT",
    "LEFT",
    "LOWEST_COORDINATE",
    "NEAREST",
    "NEAREST_FRONTIER",
    "OCV2Facts",
    "OLDEST_FRONTIER",
    "OLDEST_OBSERVATION",
    "PARTNER",
    "PLATE_PILE",
    "POT",
    "POT_COOKED",
    "POT_COOKING",
    "POT_EMPTY",
    "POT_HAS_INGREDIENTS",
    "RECIPE_BUTTON",
    "RECIPE_BUTTON_SOURCE",
    "RECIPE_INDICATOR",
    "RECIPE_MULTIPLE_SOURCES",
    "RECIPE_NOT_VISIBLE",
    "RECIPE_PRIVATE",
    "REMEMBERED_ONLY",
    "RIGHT",
    "RUNNING",
    "SEEDED_RANDOM",
    "SEEDED_RANDOM_FRONTIER",
    "SKILL_EXPLORE",
    "SKILL_FACE_DIRECTION",
    "SKILL_FACE_TARGET",
    "SKILL_INTERACT",
    "SKILL_INTERACT_WITH",
    "SKILL_MOVE",
    "SKILL_NAVIGATE",
    "SKILL_NONE",
    "SKILL_STAY",
    "SKILL_WAIT",
    "STAY",
    "SUCCEEDED",
    "SkillResult",
    "TIMED_OUT",
    "TargetQuery",
    "UNKNOWN",
    "UP",
    "VISIBLE_ONLY",
    "VISIBLE_OR_REMEMBERED",
    "WALL",
    "cancel_active_skill",
    "explore_frontier",
    "face_direction",
    "face_target",
    "first_nonzero_ingredient",
    "frontier_exists",
    "ingredient_pile",
    "init_controller",
    "interact",
    "interact_with",
    "make_trace",
    "missing_ingredient_counts",
    "move",
    "navigate_to",
    "observe",
    "path_known",
    "pot",
    "sample_nonzero_ingredient",
    "select_pytree",
    "select_result",
    "select_trace",
    "status_is",
    "stay",
    "target",
    "target_adjacent",
    "target_known",
    "target_known_distance",
    "target_relative_direction",
    "target_visible",
    "wait",
]
