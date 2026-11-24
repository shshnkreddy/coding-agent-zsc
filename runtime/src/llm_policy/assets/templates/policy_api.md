# JAX-native decentralized joint-policy API

Write one Python module implementing both decentralized roles.
The rollout remains fully JIT compiled; this is ordinary Python
using fixed-shape JAX state, not a separate policy language.

## Module contract

```python
import jax
import jax.numpy as jnp
from llm_policy import overcooked_primitives as p

POLICY_API_VERSION = 1
TASK_MAX_HEIGHT = 3       # Rendered to the selected environment.
TASK_MAX_WIDTH = 5        # Rendered to the selected environment.
TASK_NUM_INGREDIENTS = 2  # Rendered to the selected environment.
POLICY_METADATA = {
    "name": "candidate",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": TASK_MAX_HEIGHT,
    "max_width": TASK_MAX_WIDTH,
    "num_ingredients": TASK_NUM_INGREDIENTS,
}

def init_agent_0(): ...
def init_agent_1(): ...
def agent_0_act(obs, reset, memory, rng): ...
def agent_1_act(obs, reset, memory, rng): ...
```

Each act function returns `(action, next_memory, trace)`. `action` is a scalar
integer. `next_memory` must have exactly the initializer's PyTree, shapes, and
dtypes. `trace` is a `PolicyTrace`, normally produced with `p.make_trace`.
Roles are compiled and vmapped independently, so neither can inspect the
other's observation or private memory.

The task bundle renders the three `TASK_*` constants from the instantiated
environment. Keep `num_ingredients` exact. Map dimensions may be increased for
a deliberately more general policy, but never reduced below the rendered
values.

Use only fixed-shape arrays and JAX control flow. `jnp.where`, `jax.lax.cond`,
`jax.lax.switch`, `chex.dataclass`, tuples, and dictionaries of fixed-shape
leaves are supported. Array-dependent Python branching, mutable globals,
dynamic arrays, I/O, subprocesses, network access, and host callbacks are not.

## Local observation and private memory

Initialize one private controller per role:

```python
controller = p.init_controller(
    TASK_MAX_HEIGHT, TASK_MAX_WIDTH, TASK_NUM_INGREDIENTS
)
```

Begin every policy call by decoding the current local observation exactly
once:

```python
facts, controller = p.observe(obs, reset, controller)
```

`facts` contains only locally recoverable semantics. High-value fields include:

- own facing, plate/dish flags, ingredient counts, and empty-inventory flag;
- visible partner relative position, facing, inventory, and facing change;
- currently visible recipe counts and whether they came from the private or
  button indicator;
- visible button activity and pot state;
- whether the previous move was blocked or the previous interaction changed
  the observation; and
- fixed-shape local object, ingredient, recipe, and pot grids.

`controller.belief` is an agent-relative fixed-size map. It integrates local
views, remembers observed floor and objects, ages observations, and infers the
role's relative pose from successive views. It never contains absolute layout
coordinates, reward, hidden environment state, or the partner's private data.
Navigation can use only traversable cells already represented in this belief.

`facts.recipe_counts` is the recipe visible on the current step. Importantly,
`controller.last_recipe` is only the previous observation used to compute
`facts.recipe_changed`; `observe` replaces it even when the current recipe is
not visible. A policy that must remember an expiring signal should therefore
add its own fixed-shape latched recipe, for example:

```python
latched_recipe = jnp.where(
    facts.recipe_visible, facts.recipe_counts, memory.latched_recipe
)
```

Policy-specific phase, handshake, retry, or latched-information state can be
stored alongside the controller in any fixed-shape PyTree. Reset it with JAX
selection from the supplied `reset` flag.

## Target queries

Queries describe locally visible or privately remembered mechanics, not a
strategy:

```python
pile = p.ingredient_pile(ingredient_id)
empty_pot = p.pot(state=p.POT_EMPTY)
cooked_pot = p.pot(state=p.POT_COOKED)
goal = p.target(p.GOAL, source=p.VISIBLE_OR_REMEMBERED)
button = p.target(p.RECIPE_BUTTON)
```

Source modes are `VISIBLE_ONLY`, `REMEMBERED_ONLY`, and
`VISIBLE_OR_REMEMBERED`. Selection modes are `NEAREST`,
`OLDEST_OBSERVATION`, `LOWEST_COORDINATE`, and `SEEDED_RANDOM`. Reuse a stable
query while continuing a multi-step skill; changing its signature starts a new
skill.

Useful predicates are `target_known`, `target_visible`, `target_adjacent`,
`target_relative_direction`, `target_known_distance`, and `path_known`.
`missing_ingredient_counts` plus `first_nonzero_ingredient` or
`sample_nonzero_ingredient` can select a currently missing recipe component.

`target_known` does not imply that navigation is possible. `navigate_to` and
`interact_with` require a known traversable path to a known floor cell adjacent
to the selected object; they return `FAILED` when the target or path is absent.

## Skills and control flow

Available skills are `stay`, `move`, `interact`, `face_direction`,
`face_target`, `navigate_to`, `interact_with`, `explore_frontier`, and `wait`.
Each returns a `SkillResult` containing an action, updated controller, status,
and compact trace. Status values are `RUNNING`, `SUCCEEDED`, `FAILED`, and
`TIMED_OUT`.

`UP`, `DOWN`, `RIGHT`, and `LEFT` are facing-direction IDs. The corresponding
raw movement action IDs are `ACTION_UP`, `ACTION_DOWN`, `ACTION_RIGHT`, and
`ACTION_LEFT`. OCV2 has no turn-only action: a directional action changes
facing and also moves when the cell is traversable. Consequently,
`face_direction` and `face_target` can change position.

Multi-step skills store progress in the returned controller. Feed the selected
result's controller into the next policy call, and keep issuing the same skill
until it reaches a terminal status or policy logic deliberately replaces it.
Calling `stay`, `move`, or `interact` cancels active generic skill state.

`interact_with(...).status == SUCCEEDED` means the `INTERACT` action was
emitted; it does not guarantee the intended environmental effect. Confirm that
effect on the next observation using inventory, recipe, pot, or observation
change facts.

`explore_frontier` navigates to a known traversable cell bordering unknown
belief space. If the role already occupies the selected frontier, it can
succeed while emitting `STAY`; that alone does not reveal another cell.

JAX constructs both ordinary function-call branches before selecting one.
Compose already-constructed fixed-shape results with `p.select_result`; later
selections can act as higher-priority rules:

```python
result = p.select_result(know_recipe, collect_result, explore_result)
result = p.select_result(holding_ingredient, pot_result, result)
trace = p.make_trace(phase_id, decision_id, result)
return result.action, next_memory, trace
```

Use `p.select_pytree` for policy-specific memory and `p.status_is` for status
tests. These are leaf-wise selections, not short-circuit evaluation.

## Validation and rollout evidence

Validation checks the required interface, metadata, memory structure, JAXPR,
JIT compilation, and vmap behavior against a real observation from the exact
task. A valid policy may still be strategically wrong.

`policy-eval score` is the only authoritative development score. Exploratory
rollouts and `policy-eval trace` are diagnostic. Traces expose the policy's
`phase_id` and `decision_id` together with primitive skill, target, status,
action, and blocked-movement fields. `summarize_episode_trace` and
`policy-eval trace` collapse consecutive identical decisions and bound their
output while preserving both ends of a long trajectory. Field values are
reported without inferring failure or timeout semantics. `select_episode` remains available when a deliberately
selected full trace is required. Assign stable IDs that make stalls and phase
transitions interpretable.

Inspect a small number of selected episodes and summarize repeated trace runs;
avoid printing complete 400-step arrays or large episode batches into the
agent context. For continuing-delivery tasks, verify repeated cycles and
inter-delivery timing rather than treating the first successful delivery as
saturation.
