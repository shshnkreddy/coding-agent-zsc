"""Dependency-free within-group and cross-group policy evaluation reports."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real


def validate_groups(groups, num_policies):
    """Validate disjoint, nonempty groups of zero-based policy indices.

    Groups may cover a subset of the population. Overlapping groups are rejected
    so cross-group scores cannot accidentally include self-play cells.
    """
    if isinstance(num_policies, bool) or not isinstance(num_policies, Integral) or num_policies < 1:
        raise ValueError("num_policies must be a positive integer")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must be a nonempty mapping of names to index lists")
    validated = {}
    assigned = set()
    for name, indices in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group names must be nonempty strings")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"group {name!r} must contain a nonempty list of indices")
        members = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError(f"group {name!r} indices must be integers")
            if not 0 <= index < num_policies:
                raise ValueError(f"group {name!r} index {index} is out of range")
            if index in assigned:
                raise ValueError(f"policy index {index} appears more than once in groups")
            members.append(int(index))
            assigned.add(index)
        validated[name] = members
    return validated


def _matrix_rows(matrix):
    try:
        rows = [list(row) for row in matrix]
    except TypeError as error:
        raise ValueError("matrix must be a nonempty square matrix") from error
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("matrix must be a nonempty square matrix")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("matrix entries must be finite real numbers")
    return rows


def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None


def summarize_groups(matrix, groups):
    """Summarize a square matrix indexed by seat-0 rows and seat-1 columns.

    ``within[name]`` contains ``indices``, diagonal ``mean_sp``, and directed
    off-diagonal ``mean_xp`` (None for singleton groups). ``cross_directed``
    maps row-group names to column-group names and their Cartesian-cell means.
    ``cross_combined`` contains the mean of both seat orders for each distinct
    pair, repeated symmetrically for convenient lookup.
    """
    rows = _matrix_rows(matrix)
    groups = validate_groups(groups, len(rows))
    within = {
        name: {
            "indices": indices,
            "mean_sp": _mean([rows[index][index] for index in indices]),
            "mean_xp": _mean([
                rows[row][column] for row in indices for column in indices
                if row != column
            ]),
        }
        for name, indices in groups.items()
    }
    directed = {
        source: {
            target: _mean([rows[row][column] for row in source_indices for column in target_indices])
            for target, target_indices in groups.items() if target != source
        }
        for source, source_indices in groups.items()
    }
    combined = {
        source: {
            target: _mean([forward, directed[target][source]])
            for target, forward in scores.items()
        }
        for source, scores in directed.items()
    }
    return {"within": within, "cross_directed": directed, "cross_combined": combined}
