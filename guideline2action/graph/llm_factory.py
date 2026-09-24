"""OpenAI-compatible ChatOpenAI factory (singleton + per-config).

This module is the LLM client layer for LangGraph extraction. Import from here
(or ``pipeline.structured_llm`` for structured calls), not from ``pipeline.graph``.
"""
from functools import lru_cache
from typing import Any, Dict, Optional

from .config import get_config, LLMConfig
from .langfuse_tracing import load_langfuse_env


@lru_cache(maxsize=1)
def _get_default_llm() -> Any:
    """获取默认 LLM 实例（单例，缓存复用）"""
    load_langfuse_env()
    from langchain_openai import ChatOpenAI

    cfg = get_config().llm
    kwargs = _chat_openai_kwargs(cfg)
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        **kwargs,
    )


def _create_llm(config: Optional[LLMConfig] = None) -> Any:
    """创建 LLM 实例（传入 config 时创建新实例，否则返回默认单例）"""
    if config is None:
        return _get_default_llm()

    load_langfuse_env()
    from langchain_openai import ChatOpenAI

    kwargs = _chat_openai_kwargs(config)
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
        **kwargs,
    )


def _is_deepseek_endpoint(config: LLMConfig) -> bool:
    base = (config.base_url or "").lower()
    model = (config.model or "").lower()
    provider = (getattr(config, "provider", "") or "").lower()
    return provider == "deepseek" or "deepseek" in base or model.startswith("deepseek")


def _chat_openai_kwargs(config: LLMConfig) -> Dict[str, Any]:
    """Provider-specific kwargs for OpenAI-compatible endpoints."""
    if "dashscope" in (config.base_url or "").lower() or (config.model or "").lower().startswith("qwen3"):
        return {"extra_body": {"enable_thinking": False}}
    # DeepSeek thinking mode rejects forced tool_choice used by LangChain structured output.
    if _is_deepseek_endpoint(config):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}
