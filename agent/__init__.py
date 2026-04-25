"""Agent package shim for backward-compatible imports.

This package re-exports the top-level modules so code that imports
`agent.*` keeps working when the source files live at the repository root.
"""

__all__ = [
    "graph",
    "nodes",
    "state",
    "intent_classifier",
    "rag_pipeline",
]
