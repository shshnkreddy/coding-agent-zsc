"""Fixed-shape public types used by generated policies and primitives."""

from dataclasses import field
from typing import Any

import chex
import jax

from llm_policy.policies.api import PolicyTrace


Array = jax.Array


@chex.dataclass
class BeliefMap:
    """Agent-relative map reconstructed only from local observations."""

    known: Array
    traversable: Array
    object_type: Array
    ingredient_id: Array
    pot_state: Array
    pot_ingredient_counts: Array
    seen_age: Array


@chex.dataclass
class ControllerState:
    """Standard private state owned independently by one generated role."""

    belief: BeliefMap
    pose: Array
    last_action: Array
    has_last_action: Array
    episode_step: Array
    last_inventory: Array
    last_recipe: Array
    last_partner_facing: Array
    partner_was_visible: Array
    active_skill: Array
    skill_elapsed: Array
    skill_phase: Array
    skill_target: Array
    skill_query_signature: Array
    wait_remaining: Array


@chex.dataclass
class OCV2Facts:
    """Semantic facts decoded from one role's local OCV2 observation."""

    is_initial_step: Array
    episode_step: Array
    self_local_position: Array
    self_facing: Array
    self_has_plate: Array
    self_has_cooked_dish: Array
    self_ingredient_counts: Array
    self_inventory_empty: Array
    partner_visible: Array
    partner_relative_position: Array
    partner_relative_direction: Array
    partner_facing: Array
    partner_has_plate: Array
    partner_has_cooked_dish: Array
    partner_ingredient_counts: Array
    partner_inventory_empty: Array
    partner_direction_changed: Array
    recipe_visible: Array
    recipe_counts: Array
    recipe_source: Array
    recipe_changed: Array
    any_indicator_visible: Array
    any_button_visible: Array
    any_button_active: Array
    any_pot_has_ingredients: Array
    any_pot_cooking: Array
    any_pot_cooked: Array
    last_action: Array
    last_move_blocked: Array
    last_inventory_changed: Array
    last_interaction_changed_observation: Array
    local_object_type: Array
    local_ingredient_id: Array
    local_pot_state: Array
    local_pot_ingredient_counts: Array
    local_recipe_counts: Array
    local_visible: Array


@chex.dataclass
class TargetQuery:
    """Fixed-shape target selector accepted by navigation primitives."""

    object_type: Array
    ingredient_id: Array
    pot_state: Array
    source: Array
    selection: Array


@chex.dataclass
class SkillResult:
    """Uniform return type for all primitive skills."""

    action: Array
    controller: ControllerState
    status: Array
    trace: PolicyTrace


@chex.dataclass
class ProgramRolloutResult:
    """Batched self-/cross-play result for generated policy modules."""

    mean_return: Array
    first_returns: Array
    captured: Array
    trace_agent_0: Any
    trace_agent_1: Any
    trace_agent_2: Any = None
    episode_metrics: dict = field(default_factory=dict)
    actions_agent_0: Any = None
    actions_agent_1: Any = None
    actions_agent_2: Any = None
    source_exchange: dict = field(default_factory=dict)
    trace_agent_3: Any = None
    trace_agent_4: Any = None
    actions_agent_3: Any = None
    actions_agent_4: Any = None


@chex.dataclass
class XPMatrixRolloutResult:
    """Return batches for every ordered pairing in a policy population."""

    first_returns: Array
    captured: Array
    episode_metrics: dict = field(default_factory=dict)


__all__ = [
    "BeliefMap",
    "ControllerState",
    "OCV2Facts",
    "PolicyTrace",
    "ProgramRolloutResult",
    "XPMatrixRolloutResult",
    "SkillResult",
    "TargetQuery",
]
