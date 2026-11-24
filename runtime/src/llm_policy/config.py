"""Dependency-light, credential-reference-only public run configuration.

Schema 1 has kind single/population/meta/prior_knowledge and environment, agent, objective,
context, search, evaluation, execution, output and probe mappings. Runtime-specific
section contents are interpreted by the packaged worker. Execution uses
target=docker|ssh, image, optional gpus, and credentials={container_env: host_env}.
SSH additionally requires ssh={host, root}; that host must already have the
package, Docker image, and credentials. No credentials are uploaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any


_SECTIONS = (
    "environment", "agent", "objective", "context", "search", "evaluation",
    "execution", "output", "probe",
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SECRET_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "password", "secret",
    "authorization", "oauth_token", "auth_token", "credentials_file", "env_file",
}
CREDENTIAL_TARGETS = frozenset({"LLM_POLICY_CODEX_AUTH_JSON", "OPENROUTER_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"})


class ConfigError(ValueError):
    """A public configuration is invalid or contains inline credentials."""


def validate_host(host: str) -> str:
    if not isinstance(host, str) or not _HOST.fullmatch(host):
        raise ConfigError("execution.ssh.host must be an explicit SSH host or user@host")
    return host


def _copy_json(value: Any, path: str = "config") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise ConfigError(f"{path} keys must be strings")
            normalized = name.lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith(("_api_key", "_password", "_secret", "_access_token", "_refresh_token")):
                raise ConfigError(f"{path}.{name}: inline credentials are forbidden; use execution.credentials references")
            result[name] = _copy_json(item, f"{path}.{name}")
        return result
    if isinstance(value, (tuple, list)):
        return [_copy_json(item, path) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and ("\x00" in value or value.startswith(("sk-", "sk_", "Bearer "))):
            raise ConfigError(f"{path}: invalid string or inline credential")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ConfigError(f"{path} must contain only finite JSON values")


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = 1
    kind: str = "single"
    environment: Mapping[str, Any] = field(default_factory=dict)
    agent: Mapping[str, Any] = field(default_factory=dict)
    objective: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    search: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    execution: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    probe: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("schema_version must be 1")
        if not isinstance(self.kind, str) or self.kind not in {"single", "population", "meta", "prior_knowledge"}:
            raise ConfigError("kind must be single, population, meta, or prior_knowledge")
        for section in _SECTIONS:
            value = getattr(self, section)
            if not isinstance(value, Mapping):
                raise ConfigError(f"{section} must be a mapping")
            if section == "execution":
                value = dict(value)
                credentials = value.pop("credentials", {})
                value = _copy_json(value, section)
                if not isinstance(credentials, Mapping) or any(
                    not isinstance(name, str) or not _ENV_NAME.fullmatch(name)
                    or name not in CREDENTIAL_TARGETS
                    or not isinstance(reference, str) or not _ENV_NAME.fullmatch(reference)
                    for name, reference in credentials.items()
                ):
                    raise ConfigError("execution.credentials must map supported provider variables to environment-name references")
                value["credentials"] = dict(credentials)
            else:
                value = _copy_json(value, section)
            object.__setattr__(self, section, value)
        if self.kind == "prior_knowledge":
            from llm_policy.execution.prior_knowledge import validate_config

            validate_config(self)
        else:
            if self.probe:
                raise ConfigError("probe requires kind=prior_knowledge")
            self.context.setdefault("environment_context", "specification")
            self.context.setdefault("primitive_mode", "none")
            from llm_policy.context.paraphrasing import selectors, validate_config

            if "paraphrase" in self.context:
                self.context["paraphrase"] = selectors(self.context["paraphrase"])
            validate_config(self)
            if not self.environment:
                raise ConfigError("environment must be a nonempty mapping")
        execution = self.execution
        unknown = set(execution) - {"target", "image", "gpus", "credentials", "ssh", "startup_timeout", "poll_interval"}
        if unknown:
            raise ConfigError(f"unknown execution fields: {', '.join(sorted(unknown))}")
        execution.setdefault("target", "docker")
        if not isinstance(execution["target"], str) or execution["target"] not in {"docker", "ssh"}:
            raise ConfigError("execution.target must be docker or ssh; automatic scheduling is not supported")
        image = execution.get("image")
        if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]*", image):
            raise ConfigError("execution.image must be an explicit Docker image reference")
        gpus = execution.get("gpus", [])
        if not isinstance(gpus, list) or any(type(device) is not int or device < 0 for device in gpus) or len(gpus) != len(set(gpus)):
            raise ConfigError("execution.gpus must be a list of distinct nonnegative device indices")
        execution.setdefault("gpus", [])
        for name, default in (("startup_timeout", 60), ("poll_interval", 1)):
            value = execution.setdefault(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ConfigError(f"execution.{name} must be a positive finite number")
        if execution["target"] == "ssh":
            ssh = execution.get("ssh")
            if not isinstance(ssh, dict) or set(ssh) != {"host", "root"}:
                raise ConfigError("execution.ssh requires exactly host and root")
            validate_host(ssh["host"])
            if not isinstance(ssh["root"], str) or not ssh["root"].startswith("/") or any(character in ssh["root"] for character in "\r\n\x00"):
                raise ConfigError("execution.ssh.root must be an absolute remote path")
        elif "ssh" in execution:
            raise ConfigError("execution.ssh requires target=ssh")
        if set(self.output) - {"root", "run_id"}:
            raise ConfigError("output accepts only root and run_id")
        if "root" in self.output and (not isinstance(self.output["root"], str) or not self.output["root"]):
            raise ConfigError("output.root must be a nonempty path string")
        run_id = self.output.get("run_id")
        if run_id is not None and (not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id)):
            raise ConfigError("output.run_id must be a safe single path component")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({"schema_version": self.schema_version, "kind": self.kind, **{name: getattr(self, name) for name in _SECTIONS if name != "probe" or self.kind == "prior_knowledge"}}, allow_nan=False))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunConfig:
        unknown = set(value) - {"schema_version", "kind", *_SECTIONS}
        if unknown:
            raise ConfigError(f"unknown config fields: {', '.join(sorted(map(str, unknown)))}")
        return cls(**dict(value))


def load_config(config: RunConfig | Mapping[str, Any] | str | os.PathLike[str]) -> RunConfig:
    """Load JSON/YAML without environment interpolation or credential resolution."""
    if isinstance(config, RunConfig):
        return RunConfig.from_mapping(config.to_dict())
    if isinstance(config, Mapping):
        return RunConfig.from_mapping(config)
    if not isinstance(config, (str, os.PathLike)):
        raise ConfigError("config must be a RunConfig, mapping, or JSON/YAML path")
    path = Path(config).expanduser().resolve()
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ConfigError("configuration keys must be unique strings")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        value = yaml.load(path.read_text(), Loader=UniqueLoader)
    except (yaml.YAMLError, RecursionError) as error:
        raise ConfigError("invalid configuration YAML/JSON") from error
    if not isinstance(value, Mapping):
        raise ConfigError("configuration document must be a mapping")
    try:
        value = dict(value)
        output = value.get("output", {})
        if isinstance(output, Mapping) and isinstance(output.get("root"), str):
            root = Path(output["root"]).expanduser()
            value["output"] = {**output, "root": str((path.parent / root).resolve())}
        for section in ("environment", "objective"):
            options = value.get(section, {})
            reference = options.get("config") if isinstance(options, Mapping) else None
            if isinstance(reference, str) and not reference.startswith("config/"):
                source = Path(reference).expanduser()
                source = source if source.is_absolute() else path.parent / source
                resource = yaml.load(source.read_text(), Loader=UniqueLoader)
                if not isinstance(resource, Mapping):
                    raise ConfigError(f"{section}.config must resolve to a YAML/JSON mapping")
                value[section] = {**options, "config": resource}
        return RunConfig.from_mapping(value)
    except (yaml.YAMLError, RecursionError) as error:
        raise ConfigError("invalid or recursive configuration resource") from error
