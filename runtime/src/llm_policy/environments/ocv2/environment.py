"""Small, repo-local extensions to JaxMARL's OvercookedV2 environment."""

import math

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import (
    Actions,
    Direction,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, State
from jaxmarl.environments.overcooked_v2.settings import (
    INDICATOR_ACTIVATION_COST,
    INDICATOR_ACTIVATION_TIME,
    POT_COOK_TIME,
)


@chex.dataclass
class ConfigurableState(State):
    """OvercookedV2 state with hidden bookkeeping for deferred button cost."""

    indicator_was_activated: bool = False


class ConfigurableOvercookedV2(OvercookedV2):
    """OvercookedV2 with configurable indicator cost and timings.

    Indicator cost can optionally be deferred until the first correct delivery
    following one or more button activations. The default remains the upstream
    immediate per-activation charge.

    Upstream assumes every pot holds exactly three ingredients. Here the pot
    capacity follows the configured recipe length, allowing reduced recipes
    such as ``[0]`` and ``[1]`` to cook after one placement. Any full pot
    starts cooking, including a wrong recipe, so negative delivery rewards
    retain their original meaning.
    """

    def __init__(
        self,
        *args,
        indicator_activation_cost=INDICATOR_ACTIVATION_COST,
        indicator_activation_time=INDICATOR_ACTIVATION_TIME,
        indicator_cost_on_successful_delivery=False,
        pot_cook_time=POT_COOK_TIME,
        terminate_on_delivery=False,
        random_agent_orientations=False,
        **kwargs,
    ):
        indicator_activation_cost = float(indicator_activation_cost)
        if not math.isfinite(indicator_activation_cost):
            raise ValueError(
                "indicator_activation_cost must be finite, "
                f"got {indicator_activation_cost!r}"
            )
        if (
            isinstance(indicator_activation_time, bool)
            or int(indicator_activation_time) != indicator_activation_time
        ):
            raise ValueError(
                "indicator_activation_time must be a positive integer, "
                f"got {indicator_activation_time!r}"
            )
        indicator_activation_time = int(indicator_activation_time)
        if indicator_activation_time <= 0:
            raise ValueError(
                "indicator_activation_time must be a positive integer, "
                f"got {indicator_activation_time!r}"
            )
        if (
            isinstance(pot_cook_time, bool)
            or int(pot_cook_time) != pot_cook_time
        ):
            raise ValueError(
                "pot_cook_time must be a positive integer, "
                f"got {pot_cook_time!r}"
            )
        pot_cook_time = int(pot_cook_time)
        if pot_cook_time <= 0:
            raise ValueError(
                "pot_cook_time must be a positive integer, "
                f"got {pot_cook_time!r}"
            )

        self.indicator_activation_cost = indicator_activation_cost
        self.indicator_activation_time = indicator_activation_time
        self.indicator_cost_on_successful_delivery = bool(
            indicator_cost_on_successful_delivery
        )
        self.pot_cook_time = pot_cook_time
        self.terminate_on_delivery = bool(terminate_on_delivery)
        self.random_agent_orientations = bool(random_agent_orientations)
        super().__init__(*args, **kwargs)
        self.pot_capacity = int(self.possible_recipes.shape[1])
        if self.pot_capacity <= 0 or self.pot_capacity > 3:
            raise ValueError(
                "OvercookedV2 recipes must contain between 1 and 3 "
                f"ingredients, got capacity {self.pot_capacity}."
            )

    def _with_indicator_flag(self, state, value=False):
        value = jnp.asarray(value, dtype=jnp.bool_)
        if isinstance(state, ConfigurableState):
            return state.replace(indicator_was_activated=value)
        return ConfigurableState(
            agents=state.agents,
            grid=state.grid,
            time=state.time,
            terminal=state.terminal,
            recipe=state.recipe,
            new_correct_delivery=state.new_correct_delivery,
            ingredient_permutations=state.ingredient_permutations,
            indicator_was_activated=value,
        )

    def reset(self, key):
        obs, state = super().reset(key)
        if self.random_agent_orientations:
            # Match legacy JaxMARL Overcooked's fixed-start reset: positions,
            # inventories, and pots remain fixed, while both headings are
            # sampled uniformly. Reproduce its key splitting so a given reset
            # key yields the same physical headings in both implementations.
            key_after_position, _ = jax.random.split(key)
            _, direction_key = jax.random.split(key_after_position)
            directions = jax.random.choice(
                direction_key,
                jnp.arange(len(Direction), dtype=jnp.int32),
                shape=(self.num_agents,),
            )
            state = state.replace(agents=state.agents.replace(dir=directions))
            obs = self.get_obs(state)
        if not self.indicator_cost_on_successful_delivery:
            return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state)
        state = self._with_indicator_flag(state, False)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
        )

    def reset_from_state(self, state, key):
        obs, state = super().reset_from_state(state, key)
        if not self.indicator_cost_on_successful_delivery:
            return obs, state
        state = self._with_indicator_flag(state, False)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
        )

    def step_env(self, key, state, actions):
        """Optionally end an episode after the first delivered dish.

        Delivery is detected from the pre-interaction state, so both correct
        and wrong dishes terminate. This retains the upstream delivery reward
        and only changes the terminal flag.
        """
        deferred_cost = self.indicator_cost_on_successful_delivery
        if not self.terminate_on_delivery and not deferred_cost:
            return super().step_env(key, state, actions)

        if deferred_cost:
            state = self._with_indicator_flag(
                state,
                getattr(state, "indicator_was_activated", False),
            )

        ordered_actions = jnp.asarray(
            [actions[f"agent_{i}"] for i in range(self.num_agents)]
        )

        def _will_activate_indicator(agent, action):
            fwd_pos = agent.get_fwd_pos()
            cell = state.grid[fwd_pos.y, fwd_pos.x]
            return (
                (action == Actions.interact)
                & (cell[0] == StaticObject.BUTTON_RECIPE_INDICATOR)
                & (agent.inventory == 0)
                & (cell[1] == 0)
            )

        def _will_deliver(agent, action):
            fwd_pos = agent.get_fwd_pos()
            object_is_goal = (
                state.grid[fwd_pos.y, fwd_pos.x, 0] == StaticObject.GOAL
            )
            inventory_is_dish = (
                agent.inventory & int(DynamicObject.COOKED)
            ) != 0
            return (
                (action == Actions.interact)
                & object_is_goal
                & inventory_is_dish
            )

        delivered = jnp.any(
            jax.vmap(_will_deliver)(state.agents, ordered_actions)
        )
        activated_indicator = jnp.any(
            jax.vmap(_will_activate_indicator)(state.agents, ordered_actions)
        )
        obs, next_state, rewards, dones, info = super().step_env(
            key,
            state,
            actions,
        )

        if deferred_cost:
            indicator_was_activated = (
                state.indicator_was_activated | activated_indicator
            )
            apply_deferred_cost = (
                next_state.new_correct_delivery & indicator_was_activated
            )
            cost = jnp.asarray(
                self.indicator_activation_cost,
                dtype=rewards[self.agents[0]].dtype,
            )
            rewards = jax.tree_util.tree_map(
                lambda reward: reward - apply_deferred_cost * cost,
                rewards,
            )
            next_state = next_state.replace(
                indicator_was_activated=(
                    indicator_was_activated & ~apply_deferred_cost
                )
            )

        if self.terminate_on_delivery:
            done = dones["__all__"] | delivered
            next_state = next_state.replace(terminal=done)
            dones = {agent: done for agent in self.agents}
            dones["__all__"] = done
        obs = self.get_obs(next_state)
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(next_state),
            rewards,
            dones,
            info,
        )

    def process_interact(self, grid, agent, all_inventories, recipe):
        """Apply upstream dynamics, then replace configured constants."""
        fwd_pos = agent.get_fwd_pos()
        interact_cell = grid[fwd_pos.y, fwd_pos.x]
        successful_indicator_activation = (
            (interact_cell[0] == StaticObject.BUTTON_RECIPE_INDICATOR)
            & (agent.inventory == 0)
            & (interact_cell[1] == 0)
        )

        object_is_pot = interact_cell[0] == StaticObject.POT
        pot_is_cooking = interact_cell[2] > 0
        pot_is_cooked = (interact_cell[1] & int(DynamicObject.COOKED)) != 0
        pot_is_idle = object_is_pot & ~pot_is_cooking & ~pot_is_cooked
        inventory_is_ingredient = DynamicObject.is_ingredient(agent.inventory)
        merged_ingredients = interact_cell[1] + agent.inventory
        recipe_capacity = DynamicObject.ingredient_count(recipe)
        placement_fills_configured_pot = (
            pot_is_idle
            & inventory_is_ingredient
            & (
                DynamicObject.ingredient_count(merged_ingredients)
                >= recipe_capacity
            )
            & ~self.start_cooking_interaction
        )

        new_grid, new_agent, correct_delivery, reward, shaped_reward = (
            super().process_interact(grid, agent, all_inventories, recipe)
        )

        # Upstream already subtracts INDICATOR_ACTIVATION_COST. Add back the
        # difference so every other part of process_interact stays unchanged.
        immediate_indicator_cost = (
            0.0
            if self.indicator_cost_on_successful_delivery
            else self.indicator_activation_cost
        )
        cost_adjustment = INDICATOR_ACTIVATION_COST - immediate_indicator_cost
        reward = reward + successful_indicator_activation * jnp.asarray(
            cost_adjustment, dtype=reward.dtype
        )

        # Replace the upstream indicator duration and three-ingredient/20-step
        # automatic pot start with configured values. ``step_agents``
        # decrements timers after interactions, so the post-step value is one
        # less than the configured activation/cook time.
        upstream_started_cooking = (
            object_is_pot
            & (interact_cell[2] == 0)
            & (new_grid[fwd_pos.y, fwd_pos.x, 2] == POT_COOK_TIME)
        )
        start_configured_cooking = (
            placement_fills_configured_pot | upstream_started_cooking
        )
        configured_extra = jnp.asarray(
            self.pot_cook_time,
            dtype=new_grid.dtype,
        )
        new_extra = jnp.where(
            start_configured_cooking,
            configured_extra,
            new_grid[fwd_pos.y, fwd_pos.x, 2],
        )
        new_extra = jnp.where(
            successful_indicator_activation,
            jnp.asarray(self.indicator_activation_time, dtype=new_grid.dtype),
            new_extra,
        )
        new_grid = new_grid.at[fwd_pos.y, fwd_pos.x, 2].set(new_extra)
        return new_grid, new_agent, correct_delivery, reward, shaped_reward
