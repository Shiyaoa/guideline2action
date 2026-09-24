"""Structured LLM helpers: function-call schema, invoke/ainvoke, term repair.

Primary entry points used by ``graph_nodes``:

- ``_invoke_structured_output_direct`` / ``_ainvoke_structured_output_direct``
- ``_invoke_structured_list`` / ``_ainvoke_structured_list``

Do not import these from ``pipeline.graph`` (deprecated); import this module directly.
"""
from typing import Optional, Any, List, Dict
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .config import get_config, LLMConfig
from .io_utils import get_failed_task_logger
from .llm_factory import _create_llm
from .models import TermExtractionResult

# ============ 自定义结构化输出 ============

def _pydantic_to_function_schema(model_cls) -> dict:
    """
    将 Pydantic 模型转换为 OpenAI function 工具格式。
    解决 $defs 引用问题，生成内联 schema。
    """
    from pydantic import BaseModel
    if not issubclass(model_cls, BaseModel):
        raise ValueError(f"{model_cls} is not a Pydantic model")

    schema = model_cls.model_json_schema()

    # 处理 $defs：将引用内联展开
    if "$defs" in schema:
        defs = schema.pop("$defs")

        def resolve_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"].split("/")[-1]
                    return resolve_refs(defs.get(ref_path, obj))
                return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            return obj

        schema = resolve_refs(schema)

    return {
        "type": "function",
        "function": {
            "name": model_cls.__name__,
            "description": model_cls.__doc__ or f"Extract {model_cls.__name__}",
            "parameters": schema
        }
    }


def _invoke_structured_output_direct(
    system_prompt: str,
    user_content: str,
    schema_cls,
    temperature: float = 0.1,
    run_config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    使用 LangChain ChatOpenAI.with_structured_output 进行结构化输出。
    Langfuse tracing follows the official LangGraph/LangChain callback path.

    Returns:
        {
            "parsed": schema_cls 实例或 None,
            "raw": 原始 JSON 字符串,
            "parsing_error": 错误信息或 None,
            "truncated": bool,
            "metadata": {finish_reason, tokens, ...}
        }
    """
    try:
        cfg = get_config().llm
        llm = _create_llm(LLMConfig(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            provider=cfg.provider,
        ))
        structured_llm = llm.with_structured_output(
            schema_cls,
            method="function_calling",
            include_raw=True,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        response = structured_llm.invoke(messages, config=run_config)
        raw = _extract_raw_output(response)
        parsed = response.get("parsed") if isinstance(response, dict) else None
        parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
        raw_message = response.get("raw") if isinstance(response, dict) else None
        metadata = _extract_langchain_metadata(raw_message)
        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        return {
            "parsed": parsed,
            "raw": raw,
            "parsing_error": str(parsing_error) if parsing_error else None,
            "truncated": finish_reason == "length",
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "parsed": None,
            "raw": None,
            "parsing_error": f"API 调用失败: {e}",
            "truncated": False,
            "metadata": {}
        }


async def _ainvoke_structured_output_direct(
    system_prompt: str,
    user_content: str,
    schema_cls,
    temperature: float = 0.1,
    run_config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    异步版本：使用 LangChain ChatOpenAI.with_structured_output。

    Returns:
        同 _invoke_structured_output_direct
    """
    try:
        cfg = get_config().llm
        llm = _create_llm(LLMConfig(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            provider=cfg.provider,
        ))
        structured_llm = llm.with_structured_output(
            schema_cls,
            method="function_calling",
            include_raw=True,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        response = await structured_llm.ainvoke(messages, config=run_config)
        raw = _extract_raw_output(response)
        parsed = response.get("parsed") if isinstance(response, dict) else None
        parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
        raw_message = response.get("raw") if isinstance(response, dict) else None
        metadata = _extract_langchain_metadata(raw_message)
        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        return {
            "parsed": parsed,
            "raw": raw,
            "parsing_error": str(parsing_error) if parsing_error else None,
            "truncated": finish_reason == "length",
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "parsed": None,
            "raw": None,
            "parsing_error": f"API 调用失败: {e}",
            "truncated": False,
            "metadata": {}
        }


TERM_EXTRACTION_REPAIR_PROMPT = """You repair malformed OpenAI tool arguments for TermExtractionResult.

Rules:
- Preserve the original clinical content, ids, names, bindings, and evidence.
- Only repair JSON shape and field types.
- Convert stringified `terms` or `med_terms` into JSON arrays of objects.
- If a list is written in Python-ish syntax such as [({...}), ({...})], convert it to valid JSON array syntax.
- Do not add new medical concepts or infer missing facts.
- Return only a TermExtractionResult tool call.
"""


def _term_repair_content(raw: Optional[str], parsing_error: str) -> str:
    return (
        "The previous TermExtractionResult tool arguments failed validation.\n"
        f"Validation error:\n{parsing_error}\n\n"
        "Repair the arguments without changing clinical content.\n"
        f"Raw tool arguments:\n{raw or ''}"
    )


def _retry_term_extraction_repair(
    res: Dict[str, Any],
    parsing_error: str,
    run_config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """One lightweight repair retry for malformed term-stage structured output."""
    if not res.get("raw"):
        return res

    repaired = _invoke_structured_output_direct(
        system_prompt=TERM_EXTRACTION_REPAIR_PROMPT,
        user_content=_term_repair_content(res.get("raw"), parsing_error),
        schema_cls=TermExtractionResult,
        temperature=0.0,
        run_config=run_config,
    )
    if not repaired.get("parsing_error"):
        print("[extract_all_terms] repaired malformed TermExtractionResult output")
        return repaired

    repaired_error = repaired.get("parsing_error", "unknown repair error")
    res["repair_error"] = repaired_error
    res["parsing_error"] = f"{parsing_error}; repair retry failed: {repaired_error}"
    return res


async def _aretry_term_extraction_repair(
    res: Dict[str, Any],
    parsing_error: str,
    run_config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """Async one-shot repair retry for malformed term-stage structured output."""
    if not res.get("raw"):
        return res

    repaired = await _ainvoke_structured_output_direct(
        system_prompt=TERM_EXTRACTION_REPAIR_PROMPT,
        user_content=_term_repair_content(res.get("raw"), parsing_error),
        schema_cls=TermExtractionResult,
        temperature=0.0,
        run_config=run_config,
    )
    if not repaired.get("parsing_error"):
        print("[extract_all_terms] repaired malformed TermExtractionResult output")
        return repaired

    repaired_error = repaired.get("parsing_error", "unknown repair error")
    res["repair_error"] = repaired_error
    res["parsing_error"] = f"{parsing_error}; repair retry failed: {repaired_error}"
    return res


def _extract_raw_output(res: Any) -> Optional[str]:
    """从 with_structured_output 的返回结果中提取原始输出文本"""
    if not isinstance(res, dict):
        return f"Type: {type(res)}, Value: {str(res)}" if res else "Empty response"
    
    raw = res.get("raw")
    if raw is None:
        return f"Raw output is None. Available keys in response: {list(res.keys())}"
    
    try:
        # 1. 如果是 AIMessage 对象 (LangChain 标准输出)
        if hasattr(raw, "content"):
            content = raw.content
            # 处理多模态或非字符串 content
            if isinstance(content, list):
                return json.dumps(content, ensure_ascii=False)
            
            # 如果 content 为空，检查是否有 tool_calls (结构化输出通常通过工具调用实现)
            if not content and hasattr(raw, "tool_calls") and raw.tool_calls:
                return json.dumps(raw.tool_calls, ensure_ascii=False)
            
            # 如果 content 为空且无工具调用，尝试获取 additional_kwargs
            if not content and hasattr(raw, "additional_kwargs"):
                return json.dumps(raw.additional_kwargs, ensure_ascii=False)
                
            return str(content) if content else "Empty message content"
        
        # 2. 如果是字典格式 (某些 provider 或转换后的格式)
        if isinstance(raw, dict):
            if "content" in raw and raw["content"]:
                return str(raw["content"])
            return json.dumps(raw, ensure_ascii=False)
            
        # 3. 兜底处理：直接转字符串
        return str(raw)
    except Exception as e:
        return f"Extraction error: {str(e)}. Raw object type: {type(raw)}"


def _extract_langchain_metadata(raw_message: Any) -> Dict[str, Any]:
    """Extract token/model/finish metadata from LangChain AIMessage."""
    metadata: Dict[str, Any] = {}
    if raw_message is None:
        return metadata

    response_metadata = getattr(raw_message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        metadata.update(response_metadata)

    usage_metadata = getattr(raw_message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        metadata["usage_metadata"] = usage_metadata
        if "input_tokens" in usage_metadata:
            metadata["prompt_tokens"] = usage_metadata.get("input_tokens")
        if "output_tokens" in usage_metadata:
            metadata["completion_tokens"] = usage_metadata.get("output_tokens")
        if "total_tokens" in usage_metadata:
            metadata["total_tokens"] = usage_metadata.get("total_tokens")

    return metadata


def _invoke_structured_list(
    prompt: str,
    schema_cls,
    content: str,
    error_tag: str,
    task_info: Optional[Dict] = None,
    run_config: Optional[RunnableConfig] = None,
):
    """
    通用的结构化调用辅助：给定系统prompt、输出schema和文本内容，返回schema实例或None。
    支持记录失败任务及其返回信息。

    使用 LangChain structured output，以便 LangGraph config 中的
    Langfuse CallbackHandler 能追踪 node 与 LLM generation。
    """
    if not content and not prompt:
        return None

    # 使用直接 API 调用
    res = _invoke_structured_output_direct(
        system_prompt=prompt,
        user_content=content,
        schema_cls=schema_cls,
        temperature=get_config().llm.temperature,
        run_config=run_config,
    )

    # 处理截断
    if res.get("truncated"):
        error_msg = res.get("parsing_error", "输出被截断")
        print(f"[{error_tag}] WARNING: {error_msg}")
        if task_info:
            get_failed_task_logger().log_generic_failure(
                stage=error_tag,
                error=error_msg,
                task_info=task_info,
                node_output=res
            )
        return {"parsed": None, "raw": res.get("raw"), "parsing_error": error_msg, "truncated": True}

    # 处理解析错误
    parsing_error = res.get("parsing_error")
    if parsing_error:
        if schema_cls is TermExtractionResult:
            res = _retry_term_extraction_repair(res, parsing_error, run_config=run_config)
            parsing_error = res.get("parsing_error")
            if not parsing_error:
                return {"parsed": res.get("parsed"), "raw": res.get("raw"), "parsing_error": None, "error": None}

        print(f"[{error_tag}] {parsing_error}")
        if task_info:
            get_failed_task_logger().log_generic_failure(
                stage=error_tag,
                error=parsing_error,
                task_info=task_info,
                node_output=res
            )
        return {"parsed": None, "raw": res.get("raw"), "parsing_error": parsing_error}

    # 成功
    return {"parsed": res.get("parsed"), "raw": res.get("raw"), "parsing_error": None, "error": None}


async def _ainvoke_structured_list(
    prompt: str,
    schema_cls,
        content: str,
    error_tag: str,
    task_info: Optional[Dict] = None,
    run_config: Optional[RunnableConfig] = None,
):
    """
    异步版本：通用的结构化调用辅助。
    """
    if not content and not prompt:
        return None

    res = await _ainvoke_structured_output_direct(
        system_prompt=prompt,
        user_content=content,
        schema_cls=schema_cls,
        temperature=get_config().llm.temperature,
        run_config=run_config,
    )

    if res.get("truncated"):
        error_msg = res.get("parsing_error", "输出被截断")
        print(f"[{error_tag}] WARNING: {error_msg}")
        if task_info:
            get_failed_task_logger().log_generic_failure(
                stage=error_tag,
                error=error_msg,
                task_info=task_info,
                node_output=res
            )
        return {"parsed": None, "raw": res.get("raw"), "parsing_error": error_msg, "truncated": True}

    parsing_error = res.get("parsing_error")
    if parsing_error:
        if schema_cls is TermExtractionResult:
            res = await _aretry_term_extraction_repair(res, parsing_error, run_config=run_config)
            parsing_error = res.get("parsing_error")
            if not parsing_error:
                return {"parsed": res.get("parsed"), "raw": res.get("raw"), "parsing_error": None, "error": None}

        print(f"[{error_tag}] {parsing_error}")
        if task_info:
            get_failed_task_logger().log_generic_failure(
                stage=error_tag,
                error=parsing_error,
                task_info=task_info,
                node_output=res
            )
        return {"parsed": None, "raw": res.get("raw"), "parsing_error": parsing_error}

    return {"parsed": res.get("parsed"), "raw": res.get("raw"), "parsing_error": None, "error": None}

def _safe_items(obj) -> List:
    """安全获取列表或 Pydantic 模型的 items 属性"""
    if obj is None:
        return []
    # 如果是列表，直接返回
    if isinstance(obj, list):
        return obj
    # 如果是 Pydantic 模型，尝试获取 .items 属性
    items = getattr(obj, 'items', None)
    return items if items is not None else []


def _context_json(obj: Any) -> str:
    """Compact model/dict JSON for LLM context."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(exclude_none=True)
    elif isinstance(obj, dict):
        data = {k: v for k, v in obj.items() if v is not None}
    else:
        return str(obj)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


