"""Observation-only decoding helpers for OvercookedV2 default observations.

This module deliberately contains no task-solving policy.  It is safe to expose
inside an LLM policy workspace alongside the upstream environment source.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import Direction, StaticObject


# ``get_obs_default`` concatenates these blocks in this exact order.
_NUM_DIRECTION_LAYERS = 4
_NUM_STATIC_LAYERS = 6
_STATIC_OBJECT_ORDER = (
    StaticObject.WALL,
    StaticObject.GOAL,
    StaticObject.POT,
    StaticObject.RECIPE_INDICATOR,
    StaticObject.BUTTON_RECIPE_INDICATOR,
    StaticObject.PLATE_PILE,
)


class OvercookedDefaultObsDecoder:
    """Decode semantic facts from one agent's local default observation."""

    def __init__(self, num_ingredients):
        self.num_ingredients = int(num_ingredients)
        dynamic_width = 2 + self.num_ingredients
        agent_width = 1 + _NUM_DIRECTION_LAYERS + dynamic_width

        self.self_position_channel = 0
        self.self_direction_slice = slice(1, 1 + _NUM_DIRECTION_LAYERS)
        self.self_inventory_slice = slice(1 + _NUM_DIRECTION_LAYERS, agent_width)
        self.partner_position_channel = agent_width
        self.partner_direction_slice = slice(
            agent_width + 1,
            agent_width + 1 + _NUM_DIRECTION_LAYERS,
        )
        self.static_slice = slice(
            2 * agent_width,
            2 * agent_width + _NUM_STATIC_LAYERS,
        )
        self.ingredient_pile_slice = slice(
            self.static_slice.stop,
            self.static_slice.stop + self.num_ingredients,
        )
        self.dynamic_slice = slice(
            self.ingredient_pile_slice.stop,
            self.ingredient_pile_slice.stop + dynamic_width,
        )
        self.recipe_slice = slice(
            self.dynamic_slice.stop,
            self.dynamic_slice.stop + dynamic_width,
        )
        self.minimum_channels = self.recipe_slice.stop + 1

    def _self_mask(self, obs):
        return obs[..., self.self_position_channel] > 0

    def direction(self, obs):
        self_mask = self._self_mask(obs)
        layers = obs[..., self.self_direction_slice]
        return jnp.argmax(jnp.sum(layers * self_mask[..., None], axis=(0, 1))).astype(
            jnp.int32
        )

    def partner_visible(self, obs):
        return jnp.any(obs[..., self.partner_position_channel] > 0)

    def partner_direction(self, obs):
        partner_mask = obs[..., self.partner_position_channel] > 0
        layers = obs[..., self.partner_direction_slice]
        return jnp.argmax(
            jnp.sum(layers * partner_mask[..., None], axis=(0, 1))
        ).astype(jnp.int32)

    def inventory(self, obs):
        self_mask = self._self_mask(obs)
        layers = obs[..., self.self_inventory_slice]
        return jnp.sum(layers * self_mask[..., None], axis=(0, 1))

    def has_plate(self, obs):
        return self.inventory(obs)[0] > 0

    def has_cooked_dish(self, obs):
        inventory = self.inventory(obs)
        return (inventory[0] > 0) & (inventory[1] > 0)

    def has_raw_ingredient(self, obs):
        return jnp.any(self.inventory(obs)[2:] > 0)

    def visible_recipe_counts(self, obs):
        recipe_layers = obs[..., self.recipe_slice]
        return jnp.max(recipe_layers[..., 2:], axis=(0, 1))

    def recipe_visible(self, obs):
        return jnp.any(self.visible_recipe_counts(obs) > 0)

    def visible_recipe_index(self, obs):
        return jnp.argmax(self.visible_recipe_counts(obs)).astype(jnp.int32)

    def _static_mask(self, obs, object_type):
        object_index = _STATIC_OBJECT_ORDER.index(object_type)
        return obs[..., self.static_slice.start + object_index] > 0

    def button_active(self, obs):
        button_mask = self._static_mask(
            obs, StaticObject.BUTTON_RECIPE_INDICATOR
        )
        recipe_counts = obs[..., self.recipe_slice][..., 2:]
        return jnp.any(button_mask[..., None] & (recipe_counts > 0))

    def pot_cooked(self, obs):
        pot_mask = self._static_mask(obs, StaticObject.POT)
        cooked_layer = obs[..., self.dynamic_slice.start + 1] > 0
        return jnp.any(pot_mask & cooked_layer)

    def pot_has_ingredients(self, obs):
        pot_mask = self._static_mask(obs, StaticObject.POT)
        ingredient_layers = obs[..., self.dynamic_slice][..., 2:]
        return jnp.any(pot_mask[..., None] & (ingredient_layers > 0))

    def _adjacent_mask_direction(self, obs, target_mask):
        width = obs.shape[1]
        self_flat = jnp.argmax(self._self_mask(obs).reshape(-1))
        target_flat = jnp.argmax(target_mask.reshape(-1))
        self_row, self_col = self_flat // width, self_flat % width
        target_row, target_col = target_flat // width, target_flat % width
        delta_row = target_row - self_row
        delta_col = target_col - self_col
        direction = jnp.where(
            delta_col > 0,
            Direction.RIGHT,
            jnp.where(
                delta_col < 0,
                Direction.LEFT,
                jnp.where(delta_row > 0, Direction.DOWN, Direction.UP),
            ),
        ).astype(jnp.int32)
        adjacent = jnp.any(target_mask) & (
            (jnp.abs(delta_row) + jnp.abs(delta_col)) == 1
        )
        return direction, adjacent

    def adjacent_object_direction(self, obs, object_type):
        """Return ``(direction, visible_and_adjacent)`` for a static object."""
        return self._adjacent_mask_direction(obs, self._static_mask(obs, object_type))

    def adjacent_ingredient_pile_direction(self, obs, ingredient_index):
        """Return direction/adjacency for a visible ingredient pile."""
        ingredient_index = jnp.asarray(ingredient_index, dtype=jnp.int32)
        pile_layers = obs[..., self.ingredient_pile_slice]
        target_mask = jnp.take(pile_layers, ingredient_index, axis=-1) > 0
        return self._adjacent_mask_direction(obs, target_mask)


__all__ = [
    "OvercookedDefaultObsDecoder",
    "_NUM_DIRECTION_LAYERS",
    "_NUM_STATIC_LAYERS",
    "_STATIC_OBJECT_ORDER",
]
