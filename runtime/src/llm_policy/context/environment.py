"""Resolve an OCV2 config and describe its complete policy-relevant behavior."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from llm_policy.resources import source_path


@dataclass(frozen=True)
class ResolvedEnvironment:
    """Canonical environment inputs shared by prompting and evaluation."""

    config_path: Path
    config: dict[str, Any]
    env_name: str
    env_kwargs: dict[str, Any]


def load_environment_config(path: str | Path) -> ResolvedEnvironment:
    """Load one Hydra/OmegaConf environment file into plain Python values."""
    from omegaconf import OmegaConf

    config_path = Path(path).resolve()
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: environment config must contain a mapping")
    env_name = data.get("env_name")
    env_kwargs = data.get("env_kwargs")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError(f"{config_path}: missing non-empty env_name")
    if not isinstance(env_kwargs, dict):
        raise ValueError(f"{config_path}: env_kwargs must be a mapping")
    if env_name.lower() not in {"overcooked_v2", "hanabi", "yokai_grid_9_game", "yokai_grid_16_game", "cat_dog"}:
        raise ValueError(f"unsupported policy-search environment: {env_name}")
    if "convention_profile" in data.get("llm_policy", {}):
        from llm_policy.environments.hanabi.context import resolve_convention_profile
        resolve_convention_profile(data)
    return ResolvedEnvironment(
        config_path=config_path,
        config=data,
        env_name=env_name,
        env_kwargs=dict(env_kwargs),
    )


def make_resolved_env(environment: ResolvedEnvironment):
    """Instantiate the exact repository environment described by a config."""
    if environment.env_name.lower() == "cat_dog":
        from llm_policy.environments.cat_dog.adapter import CatDogAdapter
        return CatDogAdapter(environment)
    if environment.env_name.lower() in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.adapter import YokaiAdapter
        return YokaiAdapter(environment)
    if environment.env_name.lower() == "hanabi":
        from llm_policy.environments.hanabi.adapter import HanabiAdapter
        return HanabiAdapter(environment)
    from llm_policy.environments.registry import make_env

    return make_env(environment.env_name, **environment.env_kwargs)


def resolved_runtime_contract(runtime_env):
    """Return exact dimensions required by a generated policy controller."""
    if getattr(runtime_env, "policy_adapter", None) in ("hanabi", "yokai", "cat_dog"):
        return runtime_env.runtime_contract()
    layout = runtime_env.layout
    return {
        "max_height": int(runtime_env.height),
        "max_width": int(runtime_env.width),
        "num_ingredients": int(layout.num_ingredients),
        "observation_shape": [int(value) for value in runtime_env.obs_shape],
        "max_steps": int(runtime_env.max_steps),
        "num_agents": int(runtime_env.num_agents),
    }


def _enum_table(value):
    """Serialize enum members or public integer constants for a prompt."""
    members = getattr(value, "__members__", None)
    if members:
        return {name: int(member) for name, member in members.items()}
    table = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        member = getattr(value, name)
        if isinstance(member, int) and not isinstance(member, bool):
            table[name] = int(member)
    return table


def _module_source(module_name: str):
    """Read installed module source, failing clearly when it is unavailable."""
    module = importlib.import_module(module_name)
    try:
        return inspect.getsource(module)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"could not inspect source for {module_name}") from error


def _class_source(module_name: str, class_name: str):
    """Read one installed class definition without unrelated policy examples."""
    module = importlib.import_module(module_name)
    value = getattr(module, class_name)
    try:
        return inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"could not inspect {module_name}.{class_name}") from error


def _class_or_function_source(module_name: str, object_name: str):
    """Read one installed class or function without unrelated module globals."""
    module = importlib.import_module(module_name)
    value = getattr(module, object_name)
    try:
        return inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"could not inspect {module_name}.{object_name}") from error


def _resolved_layout(environment: ResolvedEnvironment):
    """Return only the selected layout's geometry and recipe metadata."""
    layout_name = environment.env_kwargs.get("layout")
    if not isinstance(layout_name, str):
        return None
    from llm_policy.environments.ocv2.layouts import CUSTOM_OVERCOOKED_V2_LAYOUTS

    value = CUSTOM_OVERCOOKED_V2_LAYOUTS.get(layout_name)
    if value is not None:
        ascii_grid, default_recipes = value
        source = "repository custom layout"
    else:
        layouts = importlib.import_module(
            "jaxmarl.environments.overcooked_v2.layouts"
        )
        registry = getattr(layouts, "overcooked_v2_layouts", {})
        registered = registry.get(layout_name)
        ascii_grid = getattr(layouts, layout_name, None)
        if registered is None or not isinstance(ascii_grid, str):
            return {"name": layout_name, "source": "installed JaxMARL layout"}
        default_recipes = registered.possible_recipes
        source = "installed JaxMARL layout"
    return {
        "name": layout_name,
        "source": source,
        "ascii_grid": ascii_grid.strip("\n"),
        "default_recipes": [list(recipe) for recipe in default_recipes],
        "configured_recipes": environment.env_kwargs.get(
            "possible_recipes", default_recipes
        ),
    }


def _selected_layout_markdown(environment: ResolvedEnvironment):
    layout = _resolved_layout(environment)
    if layout is None:
        raise ValueError("fixed OCV2 requires a named layout")
    if "ascii_grid" not in layout:
        return f"""# Selected layout

Name: `{layout['name']}`  
Source: {layout['source']}

The installed environment did not expose serializable ASCII geometry for this
layout name.
"""
    recipes = json.dumps(layout["configured_recipes"], indent=2)
    return f"""# Selected layout: {layout['name']}

Source: {layout['source']}

```text
{layout['ascii_grid']}
```

Legend: `W` wall/counter, `A` agent spawn, `X` delivery goal, `P` pot,
`B` plate pile, `R` recipe indicator, `L` button recipe indicator, and each
digit is the pile for that ingredient index. Spaces are traversable floor.

Configured recipes:

```json
{recipes}
```
"""


def write_environment_source_bundle(
    environment: ResolvedEnvironment,
    *,
    repo_root: str | Path,
    environment_dir: str | Path,
):
    """Expose the complete current OCV2 implementation without other layouts."""
    if environment.env_name.lower() == "cat_dog":
        raise ValueError("Cat & Dog v1 supports specification context only; source disclosure is unavailable")
    if environment.env_name.lower() in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.context import write_sources
        return write_sources(environment, environment_dir)
    if environment.env_name.lower() == "hanabi":
        from llm_policy.environments.hanabi.context import write_sources
        return write_sources(environment, environment_dir)
    repo_root = Path(repo_root).resolve()
    environment_dir = Path(environment_dir)
    source_dir = environment_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "upstream_common.py": _module_source(
            "jaxmarl.environments.overcooked_v2.common"
        ),
        "upstream_settings.py": _module_source(
            "jaxmarl.environments.overcooked_v2.settings"
        ),
        "upstream_overcooked.py": _module_source(
            "jaxmarl.environments.overcooked_v2.overcooked"
        ),
        # Include the Layout type itself, but deliberately not the upstream
        # module-level registry containing every unrelated OCV2 layout.
        "upstream_layout_type.py": _class_source(
            "jaxmarl.environments.overcooked_v2.layouts", "Layout"
        ),
        "repository_overrides.py": source_path(
            "llm_policy.environments.ocv2.environment"
        ).read_text(encoding="utf-8"),
        "repository_observation.py": source_path(
            "llm_policy.environments.ocv2.observation"
        ).read_text(encoding="utf-8"),
        "repository_shape_fix.py": _class_or_function_source(
            "llm_policy.environments.ocv2.layouts", "fix_overcooked_v2_obs_shape"
        ),
    }
    for name, source in sources.items():
        (source_dir / name).write_text(source.rstrip() + "\n", encoding="utf-8")

    (environment_dir / "layout.md").write_text(
        _selected_layout_markdown(environment), encoding="utf-8"
    )
    (environment_dir / "README.md").write_text(
        """# Environment reference

`config.yaml` is the exact selected configuration and `layout.md` contains only
the selected layout, not the complete OCV2 layout registry. `source/` contains
the complete upstream transition, reward, object-encoding, observation, and
repository-override code used to understand this environment. These files are
design-time references only: runtime policies still receive only their local
observations and private memory.
""",
        encoding="utf-8",
    )
    return {
        "layout": _resolved_layout(environment),
        "source_files": sorted(sources),
    }


def _format_resolved_properties(environment: ResolvedEnvironment):
    """Return the small set of config values that define episode semantics."""
    property_names = (
        "max_steps",
        "agent_view_size",
        "observation_type",
        "possible_recipes",
        "pot_cook_time",
        "start_cooking_interaction",
        "negative_rewards",
        "indicator_activation_cost",
        "indicator_activation_time",
        "indicator_cost_on_successful_delivery",
        "sample_recipe_on_delivery",
        "terminate_on_delivery",
        "random_reset",
        "random_agent_positions",
        "random_agent_orientations",
        "op_ingredient_permutations",
    )
    properties = {
        name: environment.env_kwargs[name]
        for name in property_names
        if name in environment.env_kwargs
    }
    for name in ("score_min", "score_max", "weight"):
        if name in environment.config:
            properties[name] = environment.config[name]
    return json.dumps(properties, indent=2, sort_keys=True)


def build_environment_briefing(
    environment: ResolvedEnvironment,
    *,
    include_source_instructions: bool = True,
):
    """Create a concise reading guide without replacing authoritative source."""
    if environment.env_name.lower() in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.context import briefing
        return briefing(environment)
    if environment.env_name.lower() == "hanabi":
        from llm_policy.environments.hanabi.context import briefing
        return briefing(environment)
    layout = _resolved_layout(environment)
    if layout is None:
        layout_description = (
            "No fixed ASCII layout is selected. Read the resolved configuration "
            "and environment source to determine the layout distribution and its "
            "invariants. Do not assume fixed coordinates, roles, or object counts."
        )
    elif "ascii_grid" not in layout:
        layout_description = (
            f"Layout name: {layout['name']}. Its geometry comes from the installed "
            "JaxMARL layout registry; consult the authoritative source below."
        )
    else:
        layout_description = f"""Layout name: {layout['name']}
ASCII grid:
{layout['ascii_grid']}
Legend: W=wall/counter, A=agent spawn, X=delivery goal, P=pot,
B=plate pile, R=recipe indicator, L=button recipe indicator, and each
digit is the pile for that ingredient index. Spaces are traversable floor.
Configured recipes: {json.dumps(layout['configured_recipes'])}"""

    source_instructions = ""
    if include_source_instructions:
        source_instructions = """

HOW TO USE THIS BRIEFING
- Treat this briefing as a navigation aid, not a replacement specification.
- The resolved configuration and raw source appendices below are authoritative.
- Reconstruct the task's information asymmetry, reachable interactions,
  complementary roles, reward tradeoffs, and required temporal sequence before
  choosing a strategy. If a property is not stated here, derive it from source.
"""

    return f"""ENVIRONMENT IDENTITY
- Config: {environment.config_path.name}
- Task label: {environment.config.get('name', environment.config_path.stem)}
- Environment implementation: {environment.env_name}
- This is a two-player cooperative OvercookedV2 task with simultaneous actions,
  decentralized observations, private recurrent memory, and a shared return.

CORE GAME SEMANTICS
- A directional action turns the acting agent and attempts movement. A blocked
  move can still change its facing direction. INTERACT affects the adjacent
  cell in the facing direction; STAY performs no interaction.
- Recipes specify ingredient multisets. Ingredients must be collected, placed
  in a pot, cooked, plated, and delivered to a goal. Exact cooking, reward,
  reset, and termination behavior is controlled by the resolved properties and
  authoritative transition source below.
- Recipe indicators expose recipe information when locally visible. A recipe
  button can make that information visible through its indicator according to
  the configured activation duration and cost.
- Default observations are spatial tensors centered on the acting agent. They
  encode locally visible self/partner pose and inventory, static objects,
  ingredient piles, dynamic objects, recipes, and pot state. They are not the
  hidden global state.
- Information and physical affordances can differ by spawn region. Do not assume
  the two agent IDs are interchangeable or can reach the same objects. Infer
  each role from its local observation, the layout or generator, and the source.

RESOLVED EPISODE PROPERTIES
{_format_resolved_properties(environment)}

LAYOUT OR LEVEL DISTRIBUTION
{layout_description}
{source_instructions}"""


def build_environment_context(
    environment: ResolvedEnvironment,
    *,
    repo_root: str | Path,
):
    """Build a behavior-complete context while excluding scripted baselines."""
    if environment.env_name.lower() in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.adapter import SOURCE_MODULES
        return "\n\n".join([build_environment_briefing(environment),
            json.dumps(environment.config, indent=2, sort_keys=True),
            *[_module_source(name) for name in SOURCE_MODULES]])
    if environment.env_name.lower() == "hanabi":
        return "\n\n".join([build_environment_briefing(environment),
            json.dumps(environment.config, indent=2, sort_keys=True),
            _module_source("jaxmarl.environments.hanabi.hanabi"),
            _module_source("jaxmarl.environments.hanabi.hanabi_game"),
            _module_source("llm_policy.environments.hanabi.adapter"),
            _module_source("llm_policy.environments.hanabi.layout")])
    repo_root = Path(repo_root).resolve()
    common = importlib.import_module("jaxmarl.environments.overcooked_v2.common")
    decoder_module = importlib.import_module("llm_policy.environments.ocv2.observation")
    enum_tables = {
        name: _enum_table(getattr(common, name))
        for name in (
            "Actions",
            "Direction",
            "DynamicObject",
            "StaticObject",
        )
    }
    config_text = environment.config_path.read_text(encoding="utf-8")
    resolved_spec = {
        "config_name": environment.config_path.name,
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "config": environment.config,
        "env_name": environment.env_name,
        "env_kwargs": environment.env_kwargs,
        "resolved_layout": _resolved_layout(environment),
        "enum_tables": enum_tables,
        "observation_decoder_constants": {
            "num_direction_layers": int(decoder_module._NUM_DIRECTION_LAYERS),
            "num_static_layers": int(decoder_module._NUM_STATIC_LAYERS),
            "static_object_order": [
                str(value) for value in decoder_module._STATIC_OBJECT_ORDER
            ],
        },
    }

    sections = [
        (
            "RESOLVED ENVIRONMENT SPECIFICATION",
            json.dumps(resolved_spec, indent=2, sort_keys=True),
        ),
        (
            "UPSTREAM OCV2 OBJECTS AND ENCODING",
            _module_source("jaxmarl.environments.overcooked_v2.common"),
        ),
        (
            "UPSTREAM OCV2 SETTINGS AND REWARDS",
            _module_source("jaxmarl.environments.overcooked_v2.settings"),
        ),
        (
            "UPSTREAM OCV2 TRANSITIONS AND OBSERVATIONS",
            _module_source("jaxmarl.environments.overcooked_v2.overcooked"),
        ),
        (
            "REPOSITORY OCV2 DYNAMICS OVERRIDES",
            source_path("llm_policy.environments.ocv2.environment").read_text(),
        ),
        (
            "LOCAL OBSERVATION DECODER",
            _class_source(
                "llm_policy.environments.ocv2.observation", "OvercookedDefaultObsDecoder"
            ),
        ),
    ]
    return "\n\n".join(
        f"===== {title} =====\n{content.rstrip()}" for title, content in sections
    )


__all__ = [
    "ResolvedEnvironment",
    "build_environment_briefing",
    "build_environment_context",
    "load_environment_config",
    "make_resolved_env",
    "resolved_runtime_contract",
    "write_environment_source_bundle",
]
