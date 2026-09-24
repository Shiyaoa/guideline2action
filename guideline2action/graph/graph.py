"""
LangGraph workflow compatibility facade.

Prefer explicit imports:

- Graph assembly: ``pipeline.graph_builders``
- Graph nodes: ``pipeline.graph_nodes``
- High-level runners: ``pipeline.graph_api``
- LLM clients: ``pipeline.llm_factory``
- Structured LLM calls: ``pipeline.structured_llm``
- State reducers / models: ``pipeline.models``
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from .graph_builders import (
    ASYNC_CLUSTER_SUBGRAPH,
    CLUSTER_SUBGRAPH,
    async_build_extraction_subgraph,
    aprocess_cluster,
    build_extraction_subgraph,
    build_pipeline_graph,
    build_predicates_extraction_graph,
    build_rules_extraction_graph,
    build_terms_extraction_graph,
    distribute_clusters_for_predicates,
    distribute_clusters_for_rules,
    distribute_clusters_for_terms,
    process_cluster,
    process_cluster_node,
    process_cluster_predicates,
    process_cluster_rules,
    process_cluster_terms,
    route_to_clusters,
)
from .graph_nodes import (
    asubgraph_extract_all_terms,
    async_extract_predicates_subgraph,
    async_extract_rules_subgraph,
    distribute_texts,
    do_lsh_clustering,
    extract_recommendation,
    extract_contraindications,
    extract_safety_constraints,
    compute_missing_variables,
    compute_metadata,
    serialize_graph_output,
)
from ._deprecation import warn_deprecated_graph_export

__all__ = [
    "ASYNC_CLUSTER_SUBGRAPH",
    "CLUSTER_SUBGRAPH",
    "aprocess_cluster",
    "asubgraph_extract_all_terms",
    "async_build_extraction_subgraph",
    "async_extract_predicates_subgraph",
    "async_extract_rules_subgraph",
    "build_extraction_subgraph",
    "build_pipeline_graph",
    "build_predicates_extraction_graph",
    "build_rules_extraction_graph",
    "build_terms_extraction_graph",
    "distribute_clusters_for_predicates",
    "distribute_clusters_for_rules",
    "distribute_clusters_for_terms",
    "distribute_texts",
    "do_lsh_clustering",
    "extract_recommendation",
    "extract_contraindications",
    "extract_safety_constraints",
    "compute_missing_variables",
    "compute_metadata",
    "serialize_graph_output",
    "process_cluster",
    "process_cluster_node",
    "process_cluster_predicates",
    "process_cluster_rules",
    "process_cluster_terms",
    "route_to_clusters",
]

# Deprecated names previously re-exported from this module (PEP 562).
_DEPRECATED_EXPORTS: Dict[str, Tuple[str, str]] = {
    # llm_factory
    "_chat_openai_kwargs": (".llm_factory", "_chat_openai_kwargs"),
    "_create_llm": (".llm_factory", "_create_llm"),
    "_get_default_llm": (".llm_factory", "_get_default_llm"),
    # structured_llm
    "_ainvoke_structured_list": (".structured_llm", "_ainvoke_structured_list"),
    "_ainvoke_structured_output_direct": (
        ".structured_llm",
        "_ainvoke_structured_output_direct",
    ),
    "_aretry_term_extraction_repair": (
        ".structured_llm",
        "_aretry_term_extraction_repair",
    ),
    "_context_json": (".structured_llm", "_context_json"),
    "_extract_langchain_metadata": (".structured_llm", "_extract_langchain_metadata"),
    "_extract_raw_output": (".structured_llm", "_extract_raw_output"),
    "_invoke_structured_list": (".structured_llm", "_invoke_structured_list"),
    "_invoke_structured_output_direct": (
        ".structured_llm",
        "_invoke_structured_output_direct",
    ),
    "_pydantic_to_function_schema": (".structured_llm", "_pydantic_to_function_schema"),
    "_retry_term_extraction_repair": (".structured_llm", "_retry_term_extraction_repair"),
    "_safe_items": (".structured_llm", "_safe_items"),
    # models
    "merge_by_id": (".models", "merge_by_id"),
    "merge_cluster_cache_updates": (".models", "merge_cluster_cache_updates"),
    "_to_models": (".models", "_to_models"),
}


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr = _DEPRECATED_EXPORTS[name]
    target = f"pipeline{module_path[1:]}.{attr}"
    warn_deprecated_graph_export(name, target)
    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, attr)
