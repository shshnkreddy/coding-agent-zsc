"""Fixed, auditable prose variants; never calls a model or changes game facts."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import yaml

from llm_policy.resources import asset_path


CATALOG_VERSION = 6
VARIANTS = tuple(f"p{index}" for index in range(6))


def selectors(value):
    from llm_policy.config import ConfigError

    if not isinstance(value, Mapping) or set(value) - {"objective", "environment"}:
        raise ConfigError("context.paraphrase accepts only objective and environment selectors")
    result = {name: value.get(name, "p0") for name in ("objective", "environment")}
    if any(not isinstance(item, str) or item not in VARIANTS for item in result.values()):
        raise ConfigError("paraphrase selectors must be p0 through p5")
    return result


def validate_config(config):
    if "paraphrase" not in config.context:
        return
    selectors(config.context["paraphrase"])
    from llm_policy.config import ConfigError

    if (config.kind not in {"single", "population"}
            or config.context["environment_context"] != "specification"
            or config.context["primitive_mode"] != "none"
            or config.objective.get("name", "XP") not in {"SP", "XP"}
            or "config" in config.objective
            or config.search.get("guidance_mode", "standalone_no_guidance") != "standalone_no_guidance"
            or config.search.get("policy_guidance")):
        raise ConfigError("paraphrasing requires standalone single/population search, canonical SP/XP, specification/none, and no guidance")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot(selection):
    """Stage selected templates outside the agent-visible filesystem."""
    selection = selectors(selection)
    result = {"catalog_version": CATALOG_VERSION, "selectors": selection, "assets": {}, "objective": {}, "environment": {}}
    for category, variant in selection.items():
        if variant == "p0":
            continue
        names = ("objectives",) if category == "objective" else ("overcooked_v2", "hanabi", "yokai")
        for name in names:
            raw = asset_path(f"paraphrases/{name}.yaml").read_text()
            catalog = yaml.safe_load(raw)
            result["assets"][name] = digest(raw)
            if category == "objective":
                result[category] = catalog[variant]
            else:
                extension_raw = asset_path(f"paraphrases/{name}_extended.yaml").read_text()
                result["assets"][name + "_extended"] = digest(extension_raw)
                entries = catalog["replacements"] + yaml.safe_load(extension_raw)["replacements"]
                result[category][name] = {
                    "replacements": [{"source": entry["source"], "text": entry[variant],
                                      "when": entry.get("when", {}), "optional": entry.get("optional", False)}
                                     for entry in entries],
                }
    return result


def protected_blocks(text):
    """Fenced code and layouts are immutable, including their order."""
    return tuple(re.findall(r"^```[^\n]*\n.*?^```[^\n]*$", text, re.MULTILINE | re.DOTALL))


def literal_tokens(text):
    return Counter(re.findall(r"`[^`\n]+`|(?<![\w])[-+−]?\d+(?:\.\d+)?", text))


def technical_tokens(text):
    identifiers = re.findall(r"\b(?:[A-Za-z]\w*_\w+|[A-Za-z]\w*\.\w+|[A-Za-z]+\d+\w*|[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z]+[A-Z]\w*|True|False|None|(?:u?int|float|bool)\d*)\b", text)
    identifiers = [value for value in identifiers if value not in {"ID", "IDs"}]
    operand = r"(?:\d+(?:\.\d+)?|[A-Za-z]\w*_\w+|[A-Z]{2,}|[A-Za-z])"
    formulas = re.findall(rf"\b{operand}(?:\s*[+*/=<>−×-]\s*{operand})+(?!\w)", text)
    return literal_tokens(text), Counter(identifiers), Counter(re.sub(r"\s+", "", value) for value in formulas)


def tables(text):
    return [[tuple(cell.strip() for cell in row.strip().strip("|").split("|"))
             for row in block.splitlines()]
            for block in re.findall(r"^\|[^\n]*(?:\n\|[^\n]*)*", text, re.MULTILINE)]


def prose_paragraphs(text):
    text = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.MULTILINE | re.DOTALL)
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text)
            if block.strip() and not block.lstrip().startswith(("#", "|"))]


def validate_description(original, transformed):
    headings = re.findall(r"^#{1,6} .+$", original, re.MULTILINE)
    if re.findall(r"^#{1,6} .+$", transformed, re.MULTILINE) != headings:
        raise ValueError("paraphrasing changed environment headings or their order")
    if re.findall(r"^\d+\. ", original, re.MULTILINE) != re.findall(r"^\d+\. ", transformed, re.MULTILINE):
        raise ValueError("paraphrasing changed numbered transition steps")
    if protected_blocks(transformed) != protected_blocks(original):
        raise ValueError("paraphrasing changed code or layouts")
    original_tables, selected_tables = tables(original), tables(transformed)
    if len(original_tables) != len(selected_tables):
        raise ValueError("paraphrasing changed the table count")
    eligible, edited = 0, 0
    unchanged_cells = []
    for before, after in zip(original_tables, selected_tables):
        if len(before) != len(after) or before[:2] != after[:2]:
            raise ValueError("paraphrasing changed table headers or row counts")
        for before_row, after_row in zip(before[2:], after[2:]):
            if len(before_row) != len(after_row) or before_row[0] != after_row[0]:
                raise ValueError("paraphrasing changed table columns or row keys")
            for before_cell, after_cell in zip(before_row[1:], after_row[1:]):
                if technical_tokens(before_cell) != technical_tokens(after_cell):
                    raise ValueError(f"paraphrasing changed table facts: {before_cell}")
                if len(re.findall(r"\b[A-Za-z]{2,}\b", before_cell)) >= 6:
                    eligible += 1
                    edited += before_cell != after_cell
                    if before_cell == after_cell:
                        unchanged_cells.append(before_cell)
    before_paragraphs, after_paragraphs = prose_paragraphs(original), prose_paragraphs(transformed)
    if len(before_paragraphs) != len(after_paragraphs):
        raise ValueError("paraphrasing changed paragraph structure")
    for before, after in zip(before_paragraphs, after_paragraphs):
        if technical_tokens(before) != technical_tokens(after):
            raise ValueError(f"paraphrasing changed paragraph facts: {before[:120]}")
    if literal_tokens(original) != literal_tokens(transformed):
        raise ValueError("paraphrasing changed numeric literals or inline code")
    eligible_paragraphs = [(before, after) for before, after in zip(before_paragraphs, after_paragraphs)
                          if len(re.findall(r"\b[A-Za-z]{2,}\b", before)) >= 8]
    unchanged = [before for before, after in eligible_paragraphs if before == after]
    return {"section_order": headings, "section_order_preserved": True,
            "table_structure_preserved": True, "technical_tokens_preserved": True,
            "prose_paragraphs": len(eligible_paragraphs), "rewritten_paragraphs": len(eligible_paragraphs) - len(unchanged),
            "table_prose_cells": eligible, "rewritten_table_prose_cells": edited,
            "unchanged_paragraphs": unchanged, "unchanged_table_prose_cells": unchanged_cells}


def source_pattern(source):
    parts = re.split(r"(\{n\d+\})", source)
    pattern = []
    for part in parts:
        if re.fullmatch(r"\{n\d+\}", part):
            pattern.append(f"(?P<{part[1:-1]}>\\d+)")
        else:
            pattern.append(re.sub(r"(?:\\\s)+", lambda match: r"\s+", re.escape(part)))
    return "".join(pattern)


def render_environment(original, template, contract):
    edits = []
    changed = []
    for entry in template["replacements"]:
        if any(contract.get(key) != value for key, value in entry["when"].items()):
            continue
        matches = list(re.finditer(source_pattern(entry["source"]), original))
        if not matches and not entry.get("optional", False):
            raise ValueError(f"paraphrase source is missing: {entry['source'][:90]}")
        for match in matches:
            replacement = entry["text"].strip()
            for name, value in match.groupdict().items():
                replacement = replacement.replace("{" + name + "}", value)
            if re.search(r"\{n\d+\}", replacement) or re.search(r"\n\s*\n", match[0]):
                raise ValueError("invalid numeric slot or cross-paragraph paraphrase")
            edits.append((match.start(), match.end(), replacement))
            changed.append(digest(entry["source"]))
    edits.sort()
    if any(previous[1] > current[0] for previous, current in zip(edits, edits[1:])):
        raise ValueError("paraphrase source passages overlap")
    transformed = original
    for start, end, replacement in reversed(edits):
        transformed = transformed[:start] + replacement + transformed[end:]
    audit = validate_description(original, transformed)
    return transformed, {**audit, "rewritten_passages": len(changed), "source_passages": changed}


def apply(task_root, objective, contract, convention_profile, profile, config_path):
    """Resolve selected prose before the task manifest is constructed."""
    archive = json.loads(Path(config_path).read_text())
    selection = selectors(archive["selectors"])
    active = any(value != "p0" for value in selection.values())
    if contract.get("adapter") == "hanabi" and contract.get("num_agents", 2) > 2 and active:
        raise ValueError("multiplayer Hanabi supports only canonical p0 paraphrases")
    if archive["catalog_version"] != CATALOG_VERSION:
        raise ValueError("unsupported paraphrase catalog version")
    if active and (profile.environment_context != "specification" or profile.primitive_mode != "none"):
        raise ValueError("paraphrasing requires specification/none")
    environment_path = Path(task_root) / "context/environment/README.md"
    original = environment_path.read_text()
    baseline = {"instruction": objective.instruction, "selection_rule": objective.selection_rule,
                "name": objective.name, "metric": objective.metric, "evaluator": objective.evaluator}
    record = {"catalog_version": CATALOG_VERSION, "selectors": selection,
              "catalog_sha256": digest(json.dumps(archive, sort_keys=True)),
              "baseline_objective_sha256": digest(json.dumps(baseline, sort_keys=True)),
              "baseline_environment_sha256": digest(original), "environment_audit": {}}
    if selection["objective"] != "p0":
        template = archive["objective"]
        instruction = template[objective.name]["instruction"]
        selection_rule = template[objective.name]["selection_rule"]
        if objective.name == "XP":
            pairing = ("three_role_abb" if contract.get("adapter", "overcooked_v2") == "overcooked_v2"
                       and int(contract.get("num_agents", 2)) == 3 else
                       contract.get("policy_structure", "separate_roles"))
            instruction = instruction.format(pairing=template["pairing"][pairing])
        if convention_profile != "none":
            if objective.name == "XP":
                from llm_policy.environments.hanabi.context import convention_partner_wording

                instruction = template["h_group_XP"]
                selection_rule = convention_partner_wording(selection_rule)
            else:
                instruction += " " + template["h_group_SP"]
        objective = replace(objective, instruction=instruction, selection_rule=selection_rule)
    if selection["environment"] != "p0":
        adapter = contract.get("adapter", "overcooked_v2")
        text, record["environment_audit"] = render_environment(original, archive["environment"][adapter], contract)
        environment_path.write_text(text)
    # Record the exact task-specific objective that the worker will expose.
    # This is idempotent with the workspace's final contract rendering.
    objective = objective.for_runtime_contract(contract)
    record["selected_objective"] = {**baseline, "instruction": objective.instruction,
                                    "selection_rule": objective.selection_rule,
                                    "presentation_order": ["instruction", "selection_rule"]}
    record["selected_environment"] = environment_path.read_text()
    record["environment_sha256"] = digest(record["selected_environment"])
    destination = Path(task_root).parent / "paraphrase.json"
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text() != encoded:
        raise ValueError("archived paraphrase content changed")
    destination.write_text(encoded)
    return objective


def require_same_paraphrase(source_task, destination_task):
    records = []
    for task in (source_task, destination_task):
        path = Path(task).parent / "paraphrase.json"
        records.append(json.loads(path.read_text()) if path.is_file() else None)
    if records[0] != records[1]:
        raise ValueError("cannot resume across paraphrase texts or catalogs; start a fresh run")


def conditions():
    """Original reference plus five combined wording-only paraphrases."""
    return [{"objective": variant, "environment": variant} for variant in VARIANTS]
