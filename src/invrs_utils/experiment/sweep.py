"""Functions that assist in defining experiments with multiple work units.

Copyright (c) 2023 The INVRS-IO authors.
"""

import itertools
from typing import Any, Dict, Sequence

Sweep = Sequence[Dict[str, Any]]

_zip = zip


def sweep(name: str, values: Sequence[Any]) -> Sweep:
    """Generate a list of dictionaries defining a sweep."""
    return [{name: v} for v in values]


def zip(*sweeps: Sweep) -> Sweep:
    """Zip sweeps of different variables."""
    return [_merge(*kw) for kw in _zip(*sweeps, strict=True)]


def product(*sweeps: Sweep) -> Sweep:
    """Return the Cartesian product of multiple sweeps."""
    return [_merge(*kw) for kw in itertools.product(*sweeps)]


def _merge(*vars: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries defining sweeps of multiple variables."""
    d = {}
    for v in vars:
        d.update(v)
    return d
