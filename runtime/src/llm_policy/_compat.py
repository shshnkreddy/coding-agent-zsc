"""Narrow import compatibility for previously generated policy bundles."""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys


POLICY_MODULES = {
    "policy_api": "llm_policy.policies.api",
    "types": "llm_policy.policies.types",
    "primitives": "llm_policy.policies.primitives",
    "overcooked_primitives": "llm_policy.environments.ocv2.primitives",
    "hanabi_primitives": "llm_policy.environments.hanabi.primitives",
    "yokai_primitives": "llm_policy.environments.yokai.primitives",
    "yokai16_primitives": "llm_policy.environments.yokai.primitives16",
}


class _PolicyAliasLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        target = import_module(self.target)
        module.__dict__.update({
            name: value for name, value in vars(target).items()
            if name not in {"__name__", "__loader__", "__package__", "__spec__"}
        })


class _PolicyAliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix = "llm_policy."
        if not fullname.startswith(prefix):
            return None
        name = fullname[len(prefix):]
        if name not in POLICY_MODULES:
            return None
        return spec_from_loader(fullname, _PolicyAliasLoader(POLICY_MODULES[name]))


def install_policy_aliases():
    if not any(isinstance(finder, _PolicyAliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(_PolicyAliasFinder())

