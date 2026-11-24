"""Namespace shim that avoids importing unrelated JaxMARL environments."""

from importlib.metadata import distribution
from pathlib import Path


__path__ = [
    str(Path(__file__).parent),
    str(distribution("jaxmarl").locate_file("jaxmarl/environments")),
]

from jaxmarl.environments.multi_agent_env import MultiAgentEnv


__all__ = ["MultiAgentEnv"]
