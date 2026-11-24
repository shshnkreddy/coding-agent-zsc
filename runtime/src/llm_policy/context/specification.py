"""Factual, non-executable environment contracts for restricted contexts."""

import dataclasses
import importlib.metadata
import json
from pathlib import Path

from llm_policy.context.environment import _enum_table, _resolved_layout, _selected_layout_markdown


SPECIFICATION_VERSION = 2


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _observation_schema(observation):
    if dataclasses.is_dataclass(observation):
        return {field.name: _observation_schema(getattr(observation, field.name))
                for field in dataclasses.fields(observation)}
    return {"shape": list(observation.shape), "dtype": str(observation.dtype)}


def _ocv2(environment, runtime):
    from jaxmarl.environments.overcooked_v2 import common, settings
    from llm_policy.environments.ocv2.observation import OvercookedDefaultObsDecoder

    ingredients = int(runtime.layout.num_ingredients)
    decoder = OvercookedDefaultObsDecoder(ingredients)
    extra = int(bool(runtime.indicate_successful_delivery))
    if runtime.num_agents not in (2, 3) or len(runtime.obs_shape) != 3 or runtime.obs_shape[-1] != decoder.minimum_channels + extra:
        raise ValueError("specification requires the supported two- or three-player default OCV2 observation")
    layout = _resolved_layout(environment)
    if not layout or "ascii_grid" not in layout:
        raise ValueError("specification requires serializable fixed OCV2 geometry")
    properties = ("max_steps", "agent_view_size", "pot_cook_time", "pot_capacity",
                  "start_cooking_interaction", "negative_rewards", "indicator_activation_cost",
                  "indicator_activation_time", "indicator_cost_on_successful_delivery",
                  "sample_recipe_on_delivery", "terminate_on_delivery", "random_reset",
                  "random_agent_positions", "random_agent_orientations", "op_ingredient_permutations",
                  "indicate_successful_delivery", "possible_recipes")
    native_defaults = dict(pot_cook_time=settings.POT_COOK_TIME, pot_capacity=3,
                           indicator_activation_cost=settings.INDICATOR_ACTIVATION_COST,
                           indicator_activation_time=settings.INDICATOR_ACTIVATION_TIME,
                           indicator_cost_on_successful_delivery=False,
                           terminate_on_delivery=False,
                           random_agent_orientations=bool(runtime.random_reset or runtime.random_agent_positions))
    resolved = {name: _plain(getattr(runtime, name) if hasattr(runtime, name) else native_defaults[name])
                for name in properties}
    channels = {}
    for name in ("self_direction_slice", "self_inventory_slice", "partner_direction_slice",
                 "static_slice", "ingredient_pile_slice", "dynamic_slice", "recipe_slice"):
        span = getattr(decoder, name)
        channels[name.removesuffix("_slice")] = [span.start, span.stop]
    other = "partner" if runtime.num_agents == 2 else "other_agents"
    channels.update(self_position=decoder.self_position_channel,
                    **{f"{other}_position": decoder.partner_position_channel,
                       f"{other}_direction": channels.pop("partner_direction"),
                       f"{other}_inventory": [decoder.partner_direction_slice.stop, decoder.static_slice.start]},
                    pot_timer=decoder.recipe_slice.stop)
    if extra:
        channels["successful_delivery"] = decoder.recipe_slice.stop + 1
    static_grid = _plain(runtime.layout.static_objects)
    object_positions = {}
    for row, cells in enumerate(static_grid):
        for column, value in enumerate(cells):
            if value == int(common.StaticObject.EMPTY):
                continue
            name = (f"ingredient_{value - int(common.StaticObject.INGREDIENT_PILE_BASE)}"
                    if value >= int(common.StaticObject.INGREDIENT_PILE_BASE)
                    else common.StaticObject(value).name.lower())
            object_positions.setdefault(name, []).append([row, column])
    regions = _plain(runtime.enclosed_spaces)
    spawn_regions = []
    for x, y in runtime.layout.agent_positions:
        spawn_regions.append([[row, column] for row, cells in enumerate(static_grid)
                              for column, value in enumerate(cells)
                              if value == int(common.StaticObject.EMPTY) and regions[row][column] == regions[y][x]])
    record = dict(resolved=resolved, layout=layout, observation_channels=channels,
                  actions=_enum_table(common.Actions), directions=_enum_table(common.Direction),
                  static_channel_order=["wall", "goal", "pot", "recipe_indicator", "button_indicator", "plate_pile"],
                  delivery_reward=int(settings.DELIVERY_REWARD),
                  object_positions=object_positions, spawn_regions=spawn_regions,
                  initial_agent_positions=[[int(y), int(x)] for x, y in runtime.layout.agent_positions],
                  reset_randomizes_orientations=bool(resolved["random_agent_orientations"] or runtime.random_reset or runtime.random_agent_positions))
    text = ""  # Render after the actual observation shape and dtype are recorded.
    return record, text


def _hanabi(environment, runtime):
    contract = runtime.runtime_contract()
    from llm_policy.environments.hanabi.layout import hanabi_layout
    layout = hanabi_layout(contract["num_agents"], contract["hand_size"])
    if any(contract[key] != layout[key] for key in ("observation_size", "num_actions")):
        raise ValueError("unsupported Hanabi observation encoding")
    if contract["num_agents"] > 2:
        return {"actions": layout["actions"], "observation_blocks": layout["observation_blocks"],
                "last_action_blocks": layout["last_action_blocks"],
                "card_multiplicities_per_color": [3, 2, 2, 2, 1],
                "hint_target_relative_range": [1, contract["num_agents"]]}, ""
    return {"actions": {"discard": [0, 5], "play": [5, 10], "hint_color": [10, 15],
                        "hint_rank": [15, 20], "inactive_noop": 20},
            "card_multiplicities_per_color": [3, 2, 2, 2, 1],
            "observation_blocks": {"partner_hand": [0,125], "missing_cards": [125,127],
                "deck": [127,167], "fireworks": [167,192], "information_tokens": [192,200],
                "lives": [200,203], "discards": [203,253], "last_action": [253,308],
                "native_knowledge": [308,658]}}, ""


def write_specification(environment, runtime, directory, contract):
    import jax

    directory = Path(directory)
    adapter = contract.get("adapter", "overcooked_v2")
    if adapter == "overcooked_v2":
        facts, text = _ocv2(environment, runtime)
        (directory / "layout.md").write_text(_selected_layout_markdown(environment))
    elif adapter == "hanabi":
        facts, text = _hanabi(environment, runtime)
    elif adapter == "cat_dog":
        from llm_policy.environments.cat_dog.context import specification
        facts, text = specification(runtime), ""
    elif adapter == "yokai":
        n, g, h = (contract[key] for key in ("deck_size", "grid_size", "num_hints"))
        reveal = 1+n+n*g*g
        facts, text = {"actions": {"inactive_noop": 0, "observe": [1,1+n],
            "move": [1+n,reveal], "reveal_hint": [reveal,reveal+h],
            "place_hint": [reveal+h,reveal+h+h*g*g], "end_game": contract["num_actions"]-1},
            "initial_active_player": 0, "substeps_per_turn": 4,
            "card_multiplicity_per_color": n//contract["num_colors"]}, ""
    else:
        raise ValueError(f"unsupported specification adapter: {adapter}")
    observations, _ = runtime.reset(jax.random.PRNGKey(0))
    record = dict(schema_version=SPECIFICATION_VERSION, env_name=environment.env_name,
                  jaxmarl_version=importlib.metadata.version("jaxmarl"),
                  runtime_contract={key: value for key, value in contract.items()
                                    if key not in ("source_sha256", "belief_mode")}, observation=_observation_schema(observations[runtime.agents[0]]),
                  **facts)
    if adapter == "overcooked_v2":
        from llm_policy.context.ocv2_specification import render
        text = render(record)
    elif adapter == "cat_dog":
        from llm_policy.environments.cat_dog.context import render
        text = render(record)
    else:
        from llm_policy.context import card_specification as card_specification_markdown
        text = getattr(card_specification_markdown, adapter)(record)
    (directory / "specification.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (directory / "README.md").write_text(text + "\n\nNumeric dimensions, resolved flags and array schemas: specification.json.\n")
    return {"specification_version": SPECIFICATION_VERSION, "source_files": []}
