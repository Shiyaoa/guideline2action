"""Langfuse tracing helpers for LangGraph/LangChain execution.

This module follows the Langfuse LangGraph integration pattern: add a
Langfuse CallbackHandler to graph invocation config, then pass that config
through nested graphs and LangChain model calls.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_langfuse_env() -> None:
    """Load local .env before Langfuse initializes and normalize host aliases."""
    try:
        from dotenv import load_dotenv

        project_env = Path(__file__).resolve().parents[1] / ".env"
        if project_env.exists():
            load_dotenv(project_env, override=False)
    except Exception:
        pass

    base_url = os.getenv("LANGFUSE_BASE_URL")
    if base_url and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base_url


def is_langfuse_enabled() -> bool:
    """Return True when tracing is enabled and credentials are configured."""
    if os.getenv("LANGFUSE_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return False
    load_langfuse_env()
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_callback_handler(trace_context: Optional[Dict[str, Any]] = None):
    """Create a Langfuse LangChain callback handler, or return None if unavailable."""
    if not is_langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        kwargs = {}
        if trace_context:
            kwargs["trace_context"] = trace_context
        return CallbackHandler(**kwargs)
    except Exception as exc:
        print(f"[langfuse] CallbackHandler unavailable: {exc}")
        return None


def build_langgraph_config(
    *,
    max_concurrency: Optional[int] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build LangGraph config with Langfuse callback, tags, and metadata."""
    config: Dict[str, Any] = dict(base or {})
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name:
        config["run_name"] = run_name
    if tags:
        config["tags"] = sorted(set([*config.get("tags", []), *tags]))
    if metadata:
        merged_metadata = dict(config.get("metadata", {}))
        merged_metadata.update({key: _metadata_value(value) for key, value in metadata.items()})
        config["metadata"] = merged_metadata

    callbacks = list(config.get("callbacks", []))
    if not callbacks:
        handler = get_langfuse_callback_handler()
        if handler is not None:
            callbacks.append(handler)
    if callbacks:
        config["callbacks"] = callbacks
    return config


def flush_langfuse() -> None:
    """Flush Langfuse events at script/process boundaries."""
    if not is_langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        print(f"[langfuse] flush failed: {exc}")


def _metadata_value(value: Any) -> Any:
    """LangChain/Langfuse metadata attributes are safest as scalar strings."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)
