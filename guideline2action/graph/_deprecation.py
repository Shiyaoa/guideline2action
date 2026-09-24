"""Shared helpers for backward-compatible import paths."""
from __future__ import annotations

import warnings
def warn_deprecated_graph_export(name: str, target: str) -> None:
    """Emit DeprecationWarning for symbols re-exported from ``pipeline.graph``."""
    warnings.warn(
        (
            f"Importing {name!r} from guideline2action.graph.graph is deprecated; "
            f"use {target} instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )
