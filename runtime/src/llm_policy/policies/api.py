"""Mandatory data-only policy interface; no environment or policy helpers."""

import chex
import jax

__all__ = ["PolicyTrace", "HanabiObservation", "YokaiObservation", "CatDogObservation"]


@chex.dataclass
class PolicyTrace:
    phase_id: jax.Array
    decision_id: jax.Array
    skill_id: jax.Array
    target_type: jax.Array
    target_detail: jax.Array
    status: jax.Array
    action: jax.Array
    movement_blocked: jax.Array


@chex.dataclass
class HanabiObservation:
    features: jax.Array
    legal_actions: jax.Array


@chex.dataclass
class CatDogObservation:
    features: jax.Array
    legal_actions: jax.Array


@chex.dataclass
class YokaiObservation:
    node_features: jax.Array
    adjacency_matrix: jax.Array
    positions: jax.Array
    legal_actions: jax.Array
