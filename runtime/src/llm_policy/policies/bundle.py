"""Shared, source-only candidate identity and copying (never executes code)."""
import hashlib
from pathlib import Path
import shutil

SOURCE_MODE = "open_source_zsc"
# Retain the established evaluator locator without making it a third submission.
SOURCE_LOCATOR = '# Runner-owned locator for agent_0.py and agent_1.py. Do not edit.\n'
ALL_FILES = ("policy.py", "custom_primitives.py", "agent_0.py", "agent_1.py")


def source_bundle(path):
    path = Path(path)
    path = path / "policy.py" if path.is_dir() else path
    return path.is_file() and path.read_bytes() == SOURCE_LOCATOR.encode()


def bundle_paths(path):
    path = Path(path)
    path = path / "policy.py" if path.is_dir() else path
    names = ("policy.py", "agent_0.py", "agent_1.py") if source_bundle(path) else ("policy.py", "custom_primitives.py")
    result = []
    for name in names:
        candidate = path.with_name(name)
        if name == "custom_primitives.py" and not candidate.exists():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"candidate source must be a regular file: {candidate}")
        if candidate.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"candidate source exceeds 2 MiB: {candidate}")
        result.append(candidate)
    if source_bundle(path) and path.with_name("custom_primitives.py").exists():
        raise ValueError("open_source_zsc requires inline helpers in each role file")
    return tuple(result)


def bundle_hash(path):
    digest = hashlib.sha256()
    for source in bundle_paths(path):
        digest.update(source.name.encode() + b"\0" + source.read_bytes() + b"\0")
    return digest.hexdigest()


def copy_bundle(source, destination):
    paths = bundle_paths(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = {p.name for p in paths}
    for name in ALL_FILES:
        target = destination / name
        if target.is_symlink():
            raise ValueError(f"candidate destination must not be a symlink: {target}")
        if name not in names and target.exists():
            target.unlink()
    for path in paths:
        shutil.copy2(path, destination / path.name)
