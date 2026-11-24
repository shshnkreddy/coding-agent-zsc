"""Image-only import router to the installed JaxMARL distribution."""

from importlib.metadata import distribution
from pathlib import Path


__path__ = [
    str(Path(__file__).parent),
    str(distribution("jaxmarl").locate_file("jaxmarl")),
]
registered_envs = ("overcooked_v2", "hanabi")


def make(env_id: str, **kwargs):
    if env_id.lower() == "hanabi":
        from jaxmarl.environments.hanabi.hanabi import HanabiEnv
        return HanabiEnv(**kwargs)
    if env_id.lower() != "overcooked_v2":
        raise ValueError(
            "the isolated policy image supports OvercookedV2 and Hanabi"
        )
    from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

    return OvercookedV2(**kwargs)
