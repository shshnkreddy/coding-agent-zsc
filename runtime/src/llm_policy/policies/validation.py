"""Static safety checks for LLM-authored policy modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "chex",
    "dataclasses",
    "functools",
    "jax",
    "llm_policy",
    "math",
    "typing",
}

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "chex",
    "dataclasses",
    "functools",
    "jax",
    "jax.lax",
    "jax.numpy",
    "llm_policy.overcooked_primitives",
    "llm_policy.hanabi_primitives",
    "llm_policy.yokai_primitives",
    "llm_policy.yokai16_primitives",
    "llm_policy.primitives",
    "llm_policy.types",
    "llm_policy.policy_api",
    "math",
    "typing",
}

PUBLIC_LLM_POLICY_IMPORTS = {
    "overcooked_primitives",
    "hanabi_primitives",
    "yokai_primitives",
    "yokai16_primitives",
    "primitives",
    "types",
    "policy_api",
}

FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}

FORBIDDEN_ATTRIBUTE_CALLS = {
    "debug.print",
    "debug.callback",
    "distributed.initialize",
    "experimental.io_callback",
    "host_callback.call",
    "pure_callback",
}

RESTRICTED_MEMBERS = {
    "os", "sys", "builtins", "inspect", "importlib", "linecache", "ctypes", "pickle",
    "f_globals", "f_locals", "f_back", "f_code", "gi_frame", "cr_frame", "tb_frame",
    "load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "open", "popen",
    "system", "subprocess", "io_callback", "pure_callback", "ffi", "export",
}


@dataclass(frozen=True)
class ValidationIssue:
    line: int
    column: int
    message: str

    def __str__(self):
        return f"line {self.line}:{self.column}: {self.message}"


class PolicyAstValidator(ast.NodeVisitor):
    """Conservative contract checker for trusted generated experiment code."""

    def __init__(self, allowed_import_roots=None, primitive_mode="provided", restrict_introspection=False):
        self.issues: list[ValidationIssue] = []
        self.restrict_introspection = restrict_introspection or primitive_mode == "none"
        self.allowed_import_modules = set(ALLOWED_IMPORT_MODULES)
        self.public_imports = set(PUBLIC_LLM_POLICY_IMPORTS)
        if primitive_mode == "none":
            self.public_imports = {"policy_api"}
            self.allowed_import_modules = {
                module for module in self.allowed_import_modules
                if not module.startswith("llm_policy.") or module == "llm_policy.policy_api"
            }
        elif primitive_mode != "provided":
            raise ValueError("invalid primitive mode")
        self.extra_import_roots = set(allowed_import_roots or ())
        self.function_depth = 0

    def _issue(self, node, message):
        self.issues.append(
            ValidationIssue(
                line=int(getattr(node, "lineno", 0)),
                column=int(getattr(node, "col_offset", 0)),
                message=message,
            )
        )

    @staticmethod
    def _attribute_name(node):
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            names.append(node.id)
        return ".".join(reversed(names))

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if (
                alias.name not in self.allowed_import_modules
                and root not in self.extra_import_roots
            ):
                self._issue(node, f"import {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        root = module.split(".", 1)[0]
        if (
            module not in self.allowed_import_modules
            and root not in self.extra_import_roots
            and module != "llm_policy"
        ):
            self._issue(node, f"import from {module!r} is not allowed")
        for alias in node.names:
            if alias.name == "*":
                self._issue(node, "wildcard imports are not allowed")
            if module == "llm_policy" and alias.name not in self.public_imports:
                self._issue(
                    node,
                    f"llm_policy import {alias.name!r} is not part of the policy API",
                )
            if self.restrict_introspection and module == "llm_policy.policy_api" and alias.name not in {
                "PolicyTrace", "HanabiObservation", "YokaiObservation", "CatDogObservation",
            }:
                self._issue(node, "only mandatory data types may be imported from policy_api")
            if self.restrict_introspection and (alias.name.startswith("_") or alias.name in RESTRICTED_MEMBERS):
                self._issue(node, "private imports are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node):
        if self.function_depth == 0:
            self._issue(node, "calls at module/class scope are not allowed")
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self._issue(node, f"call to {node.func.id!r} is not allowed")
        elif isinstance(node.func, ast.Attribute):
            name = self._attribute_name(node.func)
            if any(name.endswith(forbidden) for forbidden in FORBIDDEN_ATTRIBUTE_CALLS):
                self._issue(node, f"call to {name!r} is not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_Lambda(self, node):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_Attribute(self, node):
        if self.restrict_introspection and (node.attr.startswith("_") or node.attr in RESTRICTED_MEMBERS):
            self._issue(node, "private, interpreter and I/O attributes are not allowed in restricted policies")
        if node.attr.startswith("__"):
            self._issue(node, f"dunder attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node):
        self._issue(node, "global mutation declarations are not allowed")

    def visit_Nonlocal(self, node):
        self._issue(node, "nonlocal mutation declarations are not allowed")

    def visit_With(self, node):
        self._issue(node, "context managers are not allowed in generated policies")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._issue(node, "async functions are not allowed")
        self.generic_visit(node)

    def visit_Await(self, node):
        self._issue(node, "await is not allowed")
        self.generic_visit(node)

    def visit_Yield(self, node):
        self._issue(node, "generators are not allowed")
        self.generic_visit(node)

    def visit_YieldFrom(self, node):
        self._issue(node, "generators are not allowed")
        self.generic_visit(node)

    def visit_While(self, node):
        self._issue(node, "while loops are not allowed in generated policies")
        self.generic_visit(node)


def validate_policy_source(
    path: str | Path,
    *,
    allowed_import_roots=(),
    primitive_mode="provided",
    restrict_introspection=False,
):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [
            ValidationIssue(
                line=int(error.lineno or 0),
                column=int(error.offset or 0),
                message=f"syntax error: {error.msg}",
            )
        ]
    validator = PolicyAstValidator(allowed_import_roots=allowed_import_roots, primitive_mode=primitive_mode,
                                   restrict_introspection=restrict_introspection)
    validator.visit(tree)
    return validator.issues


def validate_policy_bundle(path: str | Path, *, primitive_mode="provided", restrict_introspection=False):
    """Validate a policy and its one permitted sibling extension module.

    ``custom_primitives.py`` is deliberately the only generated local import.
    Keeping the name fixed makes the complete candidate bundle easy to hash,
    archive, and reproduce.
    """
    path = Path(path).resolve()
    custom_path = path.with_name("custom_primitives.py")
    issues = {
        path: validate_policy_source(
            path,
            allowed_import_roots={"custom_primitives"},
            primitive_mode=primitive_mode,
            restrict_introspection=restrict_introspection,
        )
    }
    if custom_path.exists():
        issues[custom_path] = validate_policy_source(custom_path, primitive_mode=primitive_mode,
                                                     restrict_introspection=restrict_introspection)
    return {source_path: values for source_path, values in issues.items() if values}


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "PolicyAstValidator",
    "ValidationIssue",
    "validate_policy_bundle",
    "validate_policy_source",
]
