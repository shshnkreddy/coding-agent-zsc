"""LLM-authored policy search with lightweight, lazy public entrypoints."""

from importlib import import_module

from llm_policy._compat import POLICY_MODULES, install_policy_aliases

install_policy_aliases()

_EXPORTS = {
    "test_prior_knowledge": ("llm_policy.api", "test_prior_knowledge"),
    "create_preview": ("llm_policy.api", "create_preview"),
    "launch_run": ("llm_policy.api", "launch_run"),
    "open_run": ("llm_policy.api", "open_run"),
    "RunConfig": ("llm_policy.config", "RunConfig"),
    "RunHandle": ("llm_policy.api", "RunHandle"),
    "PreviewResult": ("llm_policy.api", "PreviewResult"),
}


def __getattr__(name):
    if name in POLICY_MODULES:
        value = import_module(f"llm_policy.{name}")
    elif name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_EXPORTS, *POLICY_MODULES})


__all__ = sorted(_EXPORTS)
