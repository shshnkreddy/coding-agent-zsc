"""Load, validate, and JIT-compile free-form generated JAX policies."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import jax
import jax.numpy as jnp

from llm_policy.policies.types import PolicyTrace
from llm_policy.policies.validation import validate_policy_bundle


POLICY_API_VERSION = 1
REQUIRED_METADATA = {
    "name",
    "adapter",
    "adapter_version",
    "max_height",
    "max_width",
    "num_ingredients",
}
FORBIDDEN_JAX_PRIMITIVE_FRAGMENTS = (
    "callback",
    "io_callback",
    "host_callback",
)


class PolicyValidationError(ValueError):
    """Raised when generated source cannot enter the compiled rollout path."""


@dataclass(frozen=True)
class CompiledRolePolicy:
    role: str
    initial_memory: Any
    scalar_act: Callable
    batched_act: Callable
    output_shape: Any

    def init(self, batch_size: int):
        """Broadcast scalar initial memory over a rollout batch."""
        return jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(
                jnp.asarray(value), (int(batch_size),) + jnp.asarray(value).shape
            ),
            self.initial_memory,
        )


@dataclass(frozen=True)
class CompiledJointPolicy:
    source_path: Path
    source_hash: str
    metadata: dict
    module: ModuleType
    agent_0: CompiledRolePolicy
    agent_1: CompiledRolePolicy
    agent_2: CompiledRolePolicy | None = None
    agent_3: CompiledRolePolicy | None = None
    agent_4: CompiledRolePolicy | None = None

    @property
    def num_agents(self):
        return int(self.metadata.get("num_agents", 2))

    @property
    def roles(self):
        return tuple(self.role(f"agent_{index}") for index in range(self.num_agents))

    def role(self, role_name: str):
        if role_name in {f"agent_{i}" for i in range(self.num_agents)}:
            role = getattr(self, role_name)
            if role is not None:
                return role
        raise KeyError(f"unknown generated-policy role {role_name!r}")


def _bundle_hash(path: Path):
    from llm_policy.policies.bundle import bundle_hash
    return bundle_hash(path)


def _load_module(path: Path):
    digest = _bundle_hash(path)
    module_name = f"_llm_policy_{digest[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    parent = str(path.parent)
    old_custom = sys.modules.pop("custom_primitives", None)
    sys.path.insert(0, parent)
    try:
        custom_path = path.with_name("custom_primitives.py")
        if custom_path.exists():
            # Execute the editable helper directly from source. Avoiding Python's
            # timestamp-based bytecode cache matters here: Codex can make several
            # same-size edits within one second during a search session.
            custom_module = ModuleType("custom_primitives")
            custom_module.__file__ = str(custom_path)
            sys.modules["custom_primitives"] = custom_module
            custom_code = compile(
                custom_path.read_text(), str(custom_path), mode="exec"
            )
            exec(custom_code, custom_module.__dict__)

        # The policy is also evaluated from its exact current source so every
        # official score corresponds to the bundle hash computed above.
        policy_code = compile(path.read_text(), str(path), mode="exec")
        exec(policy_code, module.__dict__)
    finally:
        try:
            sys.path.remove(parent)
        except ValueError:
            pass
        sys.modules.pop("custom_primitives", None)
        if old_custom is not None:
            sys.modules["custom_primitives"] = old_custom
    return module, digest


def _validate_module_contract(module, path, *, only_role=None):
    version = getattr(module, "POLICY_API_VERSION", None)
    if version != POLICY_API_VERSION:
        raise PolicyValidationError(
            f"{path}: POLICY_API_VERSION must be {POLICY_API_VERSION}, got {version!r}"
        )
    metadata = getattr(module, "POLICY_METADATA", None)
    if not isinstance(metadata, dict):
        raise PolicyValidationError(f"{path}: POLICY_METADATA must be a dict")
    is_hanabi = metadata.get("adapter") == "hanabi"
    dimensions = ("num_agents", "num_colors", "num_ranks", "hand_size", "observation_size", "num_actions") if is_hanabi else ("max_height", "max_width", "num_ingredients")
    if metadata.get("adapter") == "yokai":
        dimensions = ("num_agents", "num_colors", "deck_size", "grid_size", "num_hints", "num_nodes", "node_features", "num_actions")
    if metadata.get("adapter") == "cat_dog":
        dimensions = ("num_agents", "observation_size", "num_actions")
    required = {"name", "adapter", "adapter_version", *dimensions}
    missing = sorted(required - set(metadata))
    if missing:
        raise PolicyValidationError(
            f"{path}: POLICY_METADATA is missing {', '.join(missing)}"
        )
    if metadata["adapter"] not in ("overcooked_v2", "hanabi", "yokai", "cat_dog"):
        raise PolicyValidationError(
            f"{path}: unsupported policy adapter"
        )
    if metadata["adapter_version"] != 1:
        raise PolicyValidationError(f"{path}: adapter_version must be 1")
    for field in dimensions:
        value = metadata[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PolicyValidationError(
                f"{path}: POLICY_METADATA[{field!r}] must be a positive int"
            )
    structure = metadata.get("policy_structure", "separate_roles")
    if structure not in ("shared_local", "separate_roles"):
        raise PolicyValidationError(f"{path}: invalid policy_structure {structure!r}")
    if structure == "shared_local":
        for name in (*(f"init_agent_{i}" for i in range(5)),
                     *(f"agent_{i}_act" for i in range(5))):
            if hasattr(module, name):
                raise PolicyValidationError(f"{path}: shared_local must not define role-specific entry point {name}")
    if metadata["adapter"] == "overcooked_v2":
        num_agents = metadata.get("num_agents", 2)
        if not isinstance(num_agents, int) or isinstance(num_agents, bool) or num_agents not in (2, 3):
            raise PolicyValidationError(f"{path}: OCV2 num_agents must be 2 or 3")
        if num_agents == 3 and structure != "separate_roles":
            raise PolicyValidationError(f"{path}: three-player OCV2 requires separate_roles")
    else:
        num_agents = int(metadata["num_agents"])
    role_names = (("agent",) if structure == "shared_local" else
                  tuple(f"agent_{index}" for index in range(num_agents)))
    if only_role is not None:
        role_names = (only_role,)
    for role in role_names:
        for prefix in ("init",):
            name = f"{prefix}_{role}"
            if not callable(getattr(module, name, None)):
                raise PolicyValidationError(f"{path}: missing callable {name}()")
        action_name = f"{role}_act"
        if not callable(getattr(module, action_name, None)):
            raise PolicyValidationError(f"{path}: missing callable {action_name}()")
    return dict(metadata)


def _shape_dtype_signature(tree):
    return jax.tree_util.tree_map(
        lambda value: (tuple(value.shape), jnp.dtype(value.dtype)), tree
    )


def _jaxpr_primitive_names(closed_jaxpr):
    names = []

    def walk_jaxpr(jaxpr):
        for equation in jaxpr.eqns:
            names.append(equation.primitive.name)
            for value in equation.params.values():
                if hasattr(value, "jaxpr"):
                    walk_jaxpr(value.jaxpr)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        if hasattr(item, "jaxpr"):
                            walk_jaxpr(item.jaxpr)

    walk_jaxpr(closed_jaxpr.jaxpr)
    return names


def _compile_role(module, role, sample_obs, *, action_override=None):
    entry = "agent" if module.POLICY_METADATA.get("policy_structure") == "shared_local" else role
    init_fn = getattr(module, f"init_{entry}")
    act_fn = action_override or getattr(module, f"{entry}_act")
    try:
        initial_memory = jax.tree_util.tree_map(jnp.asarray, init_fn())
    except Exception as error:
        raise PolicyValidationError(
            f"{role}: memory initialization failed: {error}"
        ) from error
    memory_leaves = jax.tree_util.tree_leaves(initial_memory)
    if not memory_leaves:
        raise PolicyValidationError(f"{role}: initial memory must contain JAX leaves")

    initial_treedef = jax.tree_util.tree_structure(initial_memory)

    def wrapped(obs, reset, memory, rng):
        reset_memory = jax.tree_util.tree_map(
            lambda initial, current: jnp.where(reset, initial, current),
            initial_memory,
            memory,
        )
        return act_fn(obs, reset, reset_memory, rng)

    reset = jnp.asarray(False)
    rng = jax.random.PRNGKey(0)
    try:
        output_shape = jax.eval_shape(wrapped, sample_obs, reset, initial_memory, rng)
    except Exception as error:
        raise PolicyValidationError(
            f"{role}: JAX shape evaluation failed; observation-dependent Python "
            f"branches must use jnp.where/lax.cond: {error}"
        ) from error
    if not isinstance(output_shape, tuple) or len(output_shape) != 3:
        raise PolicyValidationError(
            f"{role}: act must return (action, new_memory, trace)"
        )
    action_shape, next_memory_shape, trace_shape = output_shape
    if tuple(action_shape.shape) != () or not jnp.issubdtype(
        action_shape.dtype, jnp.integer
    ):
        raise PolicyValidationError(
            f"{role}: action must be one scalar integer, got {action_shape}"
        )
    if jax.tree_util.tree_structure(next_memory_shape) != initial_treedef:
        raise PolicyValidationError(f"{role}: act changed the memory PyTree structure")
    if _shape_dtype_signature(next_memory_shape) != _shape_dtype_signature(
        initial_memory
    ):
        raise PolicyValidationError(f"{role}: act changed a memory leaf shape or dtype")
    if not isinstance(trace_shape, (dict, PolicyTrace)):
        raise PolicyValidationError(
            f"{role}: trace must be a dictionary of named JAX arrays (or legacy PolicyTrace)"
        )
    if isinstance(trace_shape, dict) and not isinstance(trace_shape, PolicyTrace):
        for name, value in trace_shape.items():
            if (not isinstance(name, str) or not hasattr(value, "shape")
                    or not hasattr(value, "dtype") or value.dtype.kind not in "biuf"):
                raise PolicyValidationError(
                    f"{role}: trace must map string keys to numeric or boolean JAX arrays"
                )

    try:
        closed_jaxpr = jax.make_jaxpr(wrapped)(sample_obs, reset, initial_memory, rng)
    except Exception as error:
        raise PolicyValidationError(
            f"{role}: JAXPR construction failed: {error}"
        ) from error
    primitive_names = _jaxpr_primitive_names(closed_jaxpr)
    forbidden = sorted(
        {
            name
            for name in primitive_names
            if any(fragment in name for fragment in FORBIDDEN_JAX_PRIMITIVE_FRAGMENTS)
        }
    )
    if forbidden:
        raise PolicyValidationError(
            f"{role}: host callback primitives are forbidden: {forbidden}"
        )

    scalar_act = jax.jit(wrapped)
    batched_act = jax.jit(jax.vmap(wrapped, in_axes=(0, 0, 0, 0)))
    try:
        scalar_output = scalar_act(sample_obs, reset, initial_memory, rng)
        jax.tree_util.tree_map(lambda value: value.block_until_ready(), scalar_output)
    except Exception as error:
        raise PolicyValidationError(
            f"{role}: scalar JIT execution failed: {error}"
        ) from error
    return CompiledRolePolicy(
        role=role,
        initial_memory=initial_memory,
        scalar_act=scalar_act,
        batched_act=batched_act,
        output_shape=output_shape,
    )


def compile_policy_module(
    path: str | Path,
    sample_obs,
    *,
    expected_metadata: dict | None = None,
    primitive_mode: str = "provided",
    restrict_introspection: bool = False,
):
    """Validate and compile one generated joint-policy Python module."""
    path = Path(path).resolve()
    issues_by_path = validate_policy_bundle(path, primitive_mode=primitive_mode,
                                            restrict_introspection=restrict_introspection)
    if issues_by_path:
        formatted = "\n".join(
            f"{source_path}: {issue}"
            for source_path, issues in issues_by_path.items()
            for issue in issues
        )
        raise PolicyValidationError(f"{path}: source validation failed:\n{formatted}")
    try:
        module, source_hash = _load_module(path)
    except PolicyValidationError:
        raise
    except Exception as error:
        raise PolicyValidationError(f"{path}: module import failed: {error}") from error
    metadata = _validate_module_contract(module, path)
    if expected_metadata is not None and metadata.get("policy_structure", "separate_roles") != expected_metadata.get("policy_structure", "separate_roles"):
        raise PolicyValidationError("policy/task policy_structure mismatch")
    if metadata["adapter"] == "cat_dog":
        from llm_policy.policies.api import CatDogObservation
        from llm_policy.environments.cat_dog.adapter import DIMENSIONS

        if primitive_mode != "none":
            raise PolicyValidationError("Cat & Dog v1 requires primitive_mode=none")
        if metadata.get("policy_structure") != "separate_roles" or any(
                metadata[k] != v for k, v in DIMENSIONS.items()):
            raise PolicyValidationError("Cat & Dog v1 requires separate_roles and exact two-player dimensions")
        if expected_metadata is not None and (expected_metadata.get("adapter") != "cat_dog" or
                any(metadata[k] != expected_metadata.get(k) for k in DIMENSIONS)):
            raise PolicyValidationError("Cat & Dog policy/task contract mismatch")
        if (not isinstance(sample_obs, CatDogObservation) or sample_obs.features.shape != (10,)
                or sample_obs.legal_actions.shape != (4,)
                or sample_obs.features.dtype != jnp.float32 or sample_obs.legal_actions.dtype != jnp.bool_):
            raise PolicyValidationError("expected CatDogObservation(features: float32[10], legal_actions: bool[4])")
        agent_0 = _compile_role(module, "agent_0", sample_obs)
        agent_1 = _compile_role(module, "agent_1", sample_obs)
        return CompiledJointPolicy(source_path=path, source_hash=source_hash, metadata=metadata,
                                   module=module, agent_0=agent_0, agent_1=agent_1)
    if metadata["adapter"] == "yokai":
        from llm_policy.policies.api import YokaiObservation
        from llm_policy.environments.yokai.adapter import DIMENSIONS_BY_DECK
        dimensions = DIMENSIONS_BY_DECK.get(metadata["deck_size"])
        if dimensions is None:
            raise PolicyValidationError("Yokai supports nine or sixteen cards")
        if metadata.get("policy_structure") != "shared_local" or any(metadata[k] != v for k, v in dimensions.items()):
            raise PolicyValidationError("Yokai v1 requires shared_local and exact supported two-player dimensions")
        if expected_metadata is not None and (expected_metadata.get("adapter") != "yokai" or
                any(metadata[k] != expected_metadata.get(k) for k in dimensions)):
            raise PolicyValidationError("Yokai policy/task contract mismatch")
        if not isinstance(sample_obs, YokaiObservation) or any(
            getattr(sample_obs, field).shape != shape for field, shape in (
                ("node_features", (dimensions["num_nodes"], dimensions["node_features"])),
                ("adjacency_matrix", (dimensions["num_nodes"], dimensions["num_nodes"])),
                ("positions", (dimensions["deck_size"], 2)), ("legal_actions", (dimensions["num_actions"],)))):
            raise PolicyValidationError("expected native YokaiObservation with matching task dimensions")
        agent_0 = _compile_role(module, "agent_0", sample_obs)
        agent_1 = _compile_role(module, "agent_1", sample_obs)
        return CompiledJointPolicy(source_path=path, source_hash=source_hash, metadata=metadata,
                                   module=module, agent_0=agent_0, agent_1=agent_1)
    if metadata["adapter"] == "hanabi":
        from llm_policy.policies.api import HanabiObservation
        from llm_policy.environments.hanabi.layout import hanabi_layout
        try:
            layout = hanabi_layout(metadata["num_agents"], metadata["hand_size"])
        except ValueError as exc:
            raise PolicyValidationError(str(exc)) from exc
        dimensions = ("num_agents", "num_colors", "num_ranks", "hand_size", "observation_size", "num_actions")
        if any(metadata[k] != layout[k] for k in dimensions):
            raise PolicyValidationError("Hanabi policy dimensions do not match the standard encoding")
        if expected_metadata is not None and (expected_metadata.get("adapter") != "hanabi" or
                any(metadata[k] != expected_metadata.get(k) for k in dimensions) or
                metadata.get("policy_structure", "separate_roles") != expected_metadata.get("policy_structure", "separate_roles")):
            raise PolicyValidationError("Hanabi policy/task contract mismatch")
        if (not isinstance(sample_obs, HanabiObservation)
                or sample_obs.features.shape != (layout["observation_size"],)
                or sample_obs.legal_actions.shape != (layout["num_actions"],)):
            raise PolicyValidationError("expected HanabiObservation with matching task dimensions")
        roles = {f"agent_{i}": _compile_role(module, f"agent_{i}", sample_obs)
                 for i in range(layout["num_agents"])}
        return CompiledJointPolicy(source_path=path, source_hash=source_hash, metadata=metadata,
                                   module=module, **roles)
    if expected_metadata is not None:
        if expected_metadata.get("adapter", "overcooked_v2") != "overcooked_v2":
            raise PolicyValidationError("policy/task adapter mismatch")
        expected_ingredients = int(expected_metadata["num_ingredients"])
        if metadata["num_ingredients"] != expected_ingredients:
            raise PolicyValidationError(
                f"{path}: POLICY_METADATA['num_ingredients'] must exactly match "
                f"the environment ({expected_ingredients}), got "
                f"{metadata['num_ingredients']}"
            )
        for field in ("max_height", "max_width"):
            minimum = int(expected_metadata[field])
            if metadata[field] < minimum:
                raise PolicyValidationError(
                    f"{path}: POLICY_METADATA[{field!r}] must be at least the "
                    f"environment value {minimum}, got {metadata[field]}"
                )
        expected_num_agents = int(expected_metadata.get("num_agents", 2))
        policy_num_agents = int(metadata.get("num_agents", 2))
        if policy_num_agents != expected_num_agents:
            raise PolicyValidationError(
                f"{path}: POLICY_METADATA['num_agents'] must exactly match "
                f"the environment ({expected_num_agents}), got {policy_num_agents}"
            )
    sample_obs = jnp.asarray(sample_obs)
    expected_channels = 25 + 5 * metadata["num_ingredients"]
    if sample_obs.ndim != 3 or sample_obs.shape[-1] < expected_channels:
        raise PolicyValidationError(
            f"{path}: expected one HxWxC OCV2 observation with at least "
            f"{expected_channels} channels for {metadata['num_ingredients']} "
            f"ingredients, got {sample_obs.shape}"
        )
    agent_0 = _compile_role(module, "agent_0", sample_obs)
    agent_1 = _compile_role(module, "agent_1", sample_obs)
    agent_2 = (_compile_role(module, "agent_2", sample_obs)
               if int(metadata.get("num_agents", 2)) == 3 else None)
    return CompiledJointPolicy(
        source_path=path,
        source_hash=source_hash,
        metadata=metadata,
        module=module,
        agent_0=agent_0,
        agent_1=agent_1,
        agent_2=agent_2,
    )


__all__ = [
    "CompiledJointPolicy",
    "CompiledRolePolicy",
    "PolicyValidationError",
    "compile_policy_module",
]
