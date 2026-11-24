"""Construct only the environments supported by policy search."""


def make_env(env_id: str, **kwargs):
    normalized = env_id.lower()
    if normalized == "cat_dog":
        from llm_policy.environments.cat_dog.native import CatDog

        return CatDog(**kwargs)
    if normalized in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.native import YokaiGrid9, YokaiGrid16

        if normalized == "yokai_grid_9_game":
            kwargs.setdefault("past_colors_revealed", True)
            return YokaiGrid9(**kwargs)
        return YokaiGrid16(**kwargs)
    if normalized == "hanabi":
        from jaxmarl.environments.hanabi.hanabi import HanabiEnv

        return HanabiEnv(**kwargs)
    if normalized != "overcooked_v2":
        raise ValueError(f"unsupported environment: {env_id}")

    from jaxmarl.environments.overcooked_v2 import OvercookedV2
    from llm_policy.environments.ocv2.environment import ConfigurableOvercookedV2
    from llm_policy.environments.ocv2.layouts import (
        fix_overcooked_v2_obs_shape,
        resolve_custom_layout,
    )

    if "scripted_policy" in kwargs:
        raise ValueError("scripted-policy environment wrappers are not supported")
    kwargs = resolve_custom_layout(kwargs)
    fix_obs_shape = kwargs.pop("fix_obs_shape", True)
    override_names = (
        "indicator_activation_cost",
        "indicator_activation_time",
        "indicator_cost_on_successful_delivery",
        "pot_cook_time",
        "terminate_on_delivery",
        "random_agent_orientations",
    )
    overrides = {name: kwargs.pop(name, None) for name in override_names}
    overrides = {name: value for name, value in overrides.items() if value is not None}
    environment = (
        ConfigurableOvercookedV2(**kwargs, **overrides)
        if overrides
        else OvercookedV2(**kwargs)
    )
    return fix_overcooked_v2_obs_shape(environment) if fix_obs_shape else environment
