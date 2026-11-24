"""Frozen, opposite-role source exchange for Cat & Dog.

Source strings are static compilation inputs, never executed as partner code.
"""
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

from llm_policy.policies.bundle import SOURCE_MODE, bundle_hash, bundle_paths, source_bundle
from llm_policy.policies.runtime import PolicyValidationError, _compile_role, _validate_module_contract
from llm_policy.policies.validation import PolicyAstValidator

INVALID_ACTION = "__source_exchange_invalid_action"


class SourceValidator(PolicyAstValidator):
    """Expose AST inspection without exposing the ast command-line/I/O entrypoint."""
    def __init__(self):
        import ast
        super().__init__(primitive_mode="none", restrict_introspection=True)
        self.allowed_import_modules.add("ast")
        self.ast_names = {"parse", "walk", "iter_fields", "iter_child_nodes", "dump",
                          "get_docstring", "get_source_segment", "literal_eval", "unparse"}
        self.ast_names.update(name for name, value in vars(ast).items()
                              if isinstance(value, type) and issubclass(value, ast.AST))
        self.ast_aliases = set()
        self.parents = []

    def visit(self, node):
        self.parents.append(node)
        try:
            return super().visit(node)
        finally:
            self.parents.pop()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "ast":
                self.ast_aliases.add(alias.asname or "ast")
        super().visit_Import(node)

    def visit_ImportFrom(self, node):
        if node.module == "ast":
            for alias in node.names:
                if alias.name not in self.ast_names:
                    self._issue(node, "only AST parsing and inspection imports are allowed")
        super().visit_ImportFrom(node)

    def visit_Name(self, node):
        import ast
        if node.id in self.ast_aliases:
            parent = self.parents[-2] if len(self.parents) > 1 else None
            if not isinstance(parent, ast.Attribute) or parent.value is not node:
                self._issue(node, "use the ast module directly through supported inspection attributes")

    def visit_Attribute(self, node):
        import ast
        if isinstance(node.value, ast.Name) and node.value.id in self.ast_aliases:
            if node.attr not in self.ast_names:
                self._issue(node, "only AST parsing and inspection attributes are allowed")
        super().visit_Attribute(node)


def _policy_error(error):
    current = error
    while current is not None:
        if isinstance(current, (OSError, MemoryError, TimeoutError, ImportError)):
            return False
        if any(word in str(current).lower() for word in (
            "out of memory", "resource_exhausted", "failed to allocate", "pthread_create",
            "failed to create thread", "unable to initialize", "std::bad_alloc",
        )):
            return False
        current = current.__cause__
    return isinstance(error, (PolicyValidationError, ValueError, TypeError, IndexError,
                              KeyError, ArithmeticError, AttributeError, SyntaxError))


@dataclass
class SourceAwarePolicy:
    source_path: Path
    source_hash: str
    metadata: dict
    modules: tuple
    sources: tuple[str, str]
    sample_observation: object
    evaluator_identity: str
    cache: dict = field(default_factory=dict)
    last_evaluation: dict = field(default_factory=dict)
    generation_mode: str = SOURCE_MODE

    def key(self, role, source):
        return (SOURCE_MODE, self.evaluator_identity, role,
                hashlib.sha256(self.sources[role].encode()).hexdigest(),
                None if source is None else hashlib.sha256(source.encode()).hexdigest())

    def bind(self, role, source):
        key = self.key(role, source)
        if key in self.cache:
            return self.cache[key]
        module = self.modules[role]
        action_fn = getattr(module, f"agent_{role}_act")

        def act(obs, reset, memory, rng):
            action, next_memory, trace = action_fn(obs, reset, memory, rng, source)
            if not isinstance(trace, dict) or INVALID_ACTION in trace:
                raise ValueError("source-aware trace must be a dict without reserved runner keys")
            active = obs.features[2 + role] != 0
            legal = ((action >= 0) & (action < 4)
                     & obs.legal_actions[jnp.clip(action, 0, 3)])
            return action, next_memory, {**trace, INVALID_ACTION: active & ~legal}

        compiled = _compile_role(module, f"agent_{role}", self.sample_observation,
                                 action_override=act)
        self.cache[key] = compiled
        return compiled


def compile_source_bundle(path, task, evaluator_identity):
    import ast

    path = Path(path).resolve()
    if not source_bundle(path):
        raise PolicyValidationError("open_source_zsc requires its unchanged runner locator and two role files")
    if task.environment.env_name.lower() != "cat_dog" or task.primitive_mode != "none":
        raise PolicyValidationError("open_source_zsc requires Cat & Dog with no primitives")
    bundle_paths(path)
    frozen_hash = bundle_hash(path)
    modules, sources = [], []
    for role in range(2):
        role_path = path.with_name(f"agent_{role}.py")
        # Decode bytes directly: do not normalize line endings of shared source.
        source = role_path.read_bytes().decode("utf-8")
        tree = ast.parse(source, filename=str(role_path))
        validator = SourceValidator()
        validator.visit(tree)
        if validator.issues:
            raise PolicyValidationError("\n".join(str(issue) for issue in validator.issues))
        module = ModuleType(f"_source_role_{role}")
        exec(compile(tree, str(role_path), "exec"), module.__dict__)
        metadata = _validate_module_contract(module, role_path, only_role=f"agent_{role}")
        expected = task.raw_env.runtime_contract()
        for key in ("adapter", "adapter_version", "policy_structure", "num_agents",
                    "observation_size", "num_actions"):
            if metadata.get(key) != expected[key]:
                raise PolicyValidationError(f"{role_path.name}: incorrect {key}")
        if any(hasattr(module, name) for name in (f"init_agent_{1-role}", f"agent_{1-role}_act")):
            raise PolicyValidationError("each submission file must contain only its own role")
        modules.append(module)
        sources.append(source)
    if bundle_hash(path) != frozen_hash:
        raise PolicyValidationError("candidate changed while being frozen")
    policy = SourceAwarePolicy(path, frozen_hash, dict(modules[0].POLICY_METADATA),
                               tuple(modules), tuple(sources), task.sample_observation,
                               evaluator_identity)
    for role in range(2):
        policy.bind(role, None)  # Invalid null-source submissions never reach exchange.
    run_source_pair(task, policy, policy, root_seed=0, num_episodes=32, partner_source="null")
    _validate_null_bob(task, policy)
    policy.last_evaluation = {}
    return policy


def _validate_null_bob(task, policy):
    """Exercise Bob even when the submitted Alice always ends the game early."""
    from llm_policy.evaluation.api import evaluation_env, _validate_completed_result
    from llm_policy.evaluation.rollout import rollout_roles

    for action in range(4):
        def probe(obs, reset, memory, rng):
            return jnp.asarray(action, dtype=jnp.int32), memory, {}

        alice = _compile_role(policy.modules[0], "agent_0", task.sample_observation,
                              action_override=probe)
        result = rollout_roles(jax.random.PRNGKey(0), alice, policy.bind(1, None),
                               evaluation_env(task), num_envs=32, num_steps=task.max_steps)
        jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
        if np.asarray(result.trace_agent_1[INVALID_ACTION]).any():
            raise PolicyValidationError("agent_1: illegal active action on null-source path")
        _validate_completed_result(task, result)


def run_source_pair(task, alice, bob, *, root_seed, num_episodes, num_steps=None,
                    partner_source="exchange"):
    from llm_policy.evaluation.rollout import rollout_roles
    from llm_policy.evaluation.api import evaluation_env, _validate_completed_result

    if partner_source not in ("exchange", "null"):
        raise ValueError("partner_source must be exchange or null")
    if not isinstance(alice, SourceAwarePolicy) or not isinstance(bob, SourceAwarePolicy):
        raise ValueError("source-aware pair requires two source-aware submissions")
    policies = (alice, bob)
    sources = [bob.sources[1], alice.sources[0]] if partner_source == "exchange" else [None, None]
    fallback = []
    report = {"mode": SOURCE_MODE, "partner_source": partner_source,
              "role_source_sha256s": [hashlib.sha256(p.sources[i].encode()).hexdigest()
                                     for i, p in enumerate(policies)],
              "partner_source_sha256s": [None if s is None else hashlib.sha256(s.encode()).hexdigest()
                                         for s in sources],
              "fallbacks": fallback, "root_seed": int(root_seed)}

    def fail(role, error):
        if sources[role] is None or not _policy_error(error):
            raise error
        fallback.append({"role": f"agent_{role}", "reason": f"{type(error).__name__}: {error}",
                         "replacement": "null", "whole_matchup_replayed": True})
        sources[role] = None

    for attempt in range(3):
        bound = []
        for role in range(2):
            try:
                bound.append(policies[role].bind(role, sources[role]))
            except Exception as error:
                fail(role, error)
                break
        if len(bound) != 2:
            continue
        # Python policy failures during batched tracing remain attributable to a seat.
        class RoleFailure(Exception):
            def __init__(self, role, error):
                self.role, self.error = role, error
        def attributed(role, fn):
            def call(*args):
                try:
                    return fn(*args)
                except Exception as error:
                    if _policy_error(error):
                        raise RoleFailure(role, error) from error
                    raise
            return call
        roles = [replace(b, batched_act=attributed(i, b.batched_act)) for i, b in enumerate(bound)]
        try:
            result = rollout_roles(jax.random.PRNGKey(int(root_seed)), *roles,
                                   evaluation_env(task), num_envs=int(num_episodes),
                                   num_steps=task.max_steps if num_steps is None else int(num_steps))
            jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
        except RoleFailure as error:
            fail(error.role, error.error)
            continue
        invalid = [i for i in range(2) if np.asarray(
            getattr(result, f"trace_agent_{i}")[INVALID_ACTION]).any()]
        if invalid:
            for role in invalid:
                fail(role, ValueError("illegal active action"))
            continue
        _validate_completed_result(task, result)
        for role in range(2):
            getattr(result, f"trace_agent_{role}").pop(INVALID_ACTION)
        report["effective_partner_source_sha256s"] = [
            None if s is None else hashlib.sha256(s.encode()).hexdigest() for s in sources]
        report["compilation_keys"] = [list(policies[i].key(i, sources[i])) for i in range(2)]
        report["evaluation_attempts"] = attempt + 1
        result.source_exchange = report
        alice.last_evaluation = report
        return result, report
    raise PolicyValidationError("source-aware fallback exhausted")
