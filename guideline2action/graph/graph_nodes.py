"""LangGraph node functions for guideline extraction (cluster subgraph)."""
from typing import Optional, Any, List, Dict, Mapping, TypedDict, Annotated
import json
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send

from .models import (
    AgentState,
    ClusterState,
    Provenance,
    ProvenanceList,
    TermList,
    MedicationTermList,
    TermExtractionResult,
    SubmitSimplifiedRules,
    Predicates,
    ClinicalRule,
    PredicateExtractionBatch,
    ProvenanceCluster,
    merge_by_id,
    merge_cluster_cache_updates,
    _to_models,
    Term,
    MedicationTerm,
)
from .config import get_config, LLMConfig
from .lsh_cluster import lsh_cluster
from .io_utils import get_failed_task_logger
from .graph_prompts import (
    RECOMMENDATION_PROMPT,
    TERMS_AND_MEDS_PROMPT,
    PREDICATES_PROMPT,
    RULES_PROMPT,
    CONTRAINDICATION_PROMPT,
    SAFETY_CONSTRAINT_PROMPT,
)
from .missing_variables import derive_missing_variables
from .structured_llm import (
    _invoke_structured_output_direct,
    _ainvoke_structured_output_direct,
    _invoke_structured_list,
    _ainvoke_structured_list,
    _safe_items,
    _context_json,
)

def extract_recommendation(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    """
    主图节点：从指南文本中抽取推荐意见
    兼容两种输入：
    - 直接传入 text/text_idx
    - 传入 messages（旧接口）
    """
    text = state.get("text")
    text_idx = state.get("text_idx", 0)
    messages = list(state.get("messages", []))

    if not messages and text:
        messages = [HumanMessage(content=text)]

    if not messages:
        print(f"[extract {text_idx}] 无可用文本")
        return {"provenance_buffer": []}

    # 提取用户内容
    user_content = ""
    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            user_content += msg.content + "\n"
        elif isinstance(msg, dict) and msg.get('role') == 'user':
            user_content += msg.get('content', '') + "\n"

    # 使用 LangChain structured output；LangGraph config 会透传 Langfuse callback。
    res = _invoke_structured_output_direct(
        system_prompt=RECOMMENDATION_PROMPT,
        user_content=user_content.strip(),
        schema_cls=ProvenanceList,
        temperature=get_config().llm.temperature,
        run_config=config,
    )

    parsing_error = res.get("parsing_error")
    if parsing_error:
        get_failed_task_logger().log_failed_extraction(
            text_idx=text_idx,
            text=text or user_content[:200],
            error=parsing_error,
            node_output=res.get("raw") or res
        )
        return {"provenance_buffer": []}

    parsed = res.get("parsed")
    provenances = parsed.items if parsed else []
    print(f"  [extract {text_idx}] 抽取到 {len(provenances)} 条推荐意见")
    return {"provenance_buffer": provenances}

def distribute_texts(state: AgentState) -> List[Send]:
    """
    第一阶段的分发节点
    将多个输入文本分发到并行的 extract_recommendation 节点
    """
    input_texts = state.get("input_texts", [])
    
    # 如果只有单条消息（兼容旧接口）
    if not input_texts:
        messages = state.get("messages", [])
        if messages:
            # 从消息中提取文本
            text = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            input_texts = [text]
    
    if not input_texts:
        print("[distribute_texts] 无输入文本")
        return []
    
    sends = []
    for idx, text in enumerate(input_texts):
        sends.append(Send("extract_recommendation", {"text": text, "text_idx": idx}))
    
    print(f"[distribute_texts] 分发 {len(sends)} 个文本到并行抽取")
    return sends


# ============ 主图节点函数，LSH聚类 ============

def do_lsh_clustering(state: AgentState) -> AgentState:
    """对推荐意见进行 LSH 聚类"""
    provenances = state.get("provenance_buffer", [])
    if not provenances:
        print("[lsh_clustering] 无推荐意见，跳过聚类")
        return {"clusters": [], "lsh_bucket_index": {}}
    
    lsh_result = lsh_cluster(provenances)
    print(f"[lsh_clustering] {len(provenances)} 条推荐 -> {len(lsh_result.clusters)} 个聚类")
    # 保存bucket索引，后续Z3验证时可用于限制验证范围
    return {
        "clusters": lsh_result.clusters,
        "lsh_bucket_index": lsh_result.bucket_index
    }



# ============ 子图节点函数 ============

def subgraph_extract_all_terms(state: ClusterState, config: Optional[RunnableConfig] = None) -> dict:
    """Extract all terms (non-medication and medication) from cluster texts using LLM."""
    cluster_id = state.get("cluster_id")
    content = "\n\n".join(state.get("texts_formatted", []))
    if not content:
        return {"terms": [], "med_terms": []}

    response = _invoke_structured_list(
        prompt=TERMS_AND_MEDS_PROMPT,
        schema_cls=TermExtractionResult,
        content=content,
        error_tag="extract_all_terms",
        task_info={"cluster_id": cluster_id},
        run_config=config,
    )
    if not response:
        return {"terms": [], "med_terms": []}

    parsed = response.get("parsed")
    if not parsed:
        return {"terms": [], "med_terms": []}

    # 抽取阶段：保持语义化 ID，不进行 OMOP 标准化
    # OMOP 标准化在后处理阶段执行
    terms = _safe_items(parsed.terms)
    med_terms = _safe_items(parsed.med_terms)
    if not terms and not med_terms:
        get_failed_task_logger().log_generic_failure(
            stage="extract_all_terms",
            error="Term extraction returned no terms or medications; downstream predicate/rule extraction will be skipped.",
            task_info={"cluster_id": cluster_id},
        )
    return {
        "terms": terms,
        "med_terms": med_terms,
    }


async def asubgraph_extract_all_terms(state: ClusterState, config: Optional[RunnableConfig] = None) -> dict:
    """异步版本：从 cluster 文本中抽取术语和药物术语。"""
    cluster_id = state.get("cluster_id")
    content = "\n\n".join(state.get("texts_formatted", []))
    if not content:
        return {"terms": [], "med_terms": []}

    response = await _ainvoke_structured_list(
        prompt=TERMS_AND_MEDS_PROMPT,
        schema_cls=TermExtractionResult,
        content=content,
        error_tag="extract_all_terms",
        task_info={"cluster_id": cluster_id},
        run_config=config,
    )
    if not response:
        return {"terms": [], "med_terms": []}

    parsed = response.get("parsed")
    if not parsed:
        return {"terms": [], "med_terms": []}

    # 抽取阶段：保持语义化 ID，不进行 OMOP 标准化
    # OMOP 标准化在后处理阶段执行
    terms = _safe_items(parsed.terms)
    med_terms = _safe_items(parsed.med_terms)
    if not terms and not med_terms:
        get_failed_task_logger().log_generic_failure(
            stage="extract_all_terms",
            error="Term extraction returned no terms or medications; downstream predicate/rule extraction will be skipped.",
            task_info={"cluster_id": cluster_id},
        )
    return {
        "terms": terms,
        "med_terms": med_terms,
    }


# ============ Predicate Agent ============

class PredicateAgentState(TypedDict):
    """
    Predicate Agent State: Implements the Symbolic Operator Layer F(E × O) → L
    
    This agent extracts atomic conditions from clinical text and formalizes them
    as logical predicates using the operator layer, which are then compiled into
    SMT constraints for Z3 verification.
    """
    cluster_id: Optional[int]  # For logging
    content: str  # Clinical text input
    terms_context: str  # Available ontological entities (conditions, measures)
    med_terms_context: str  # Available medication entities
    atoms: List[dict]  # Atomic conditions (E × O pairs before formalization)
    predicates: Annotated[List[Predicates], merge_by_id]  # Formalized predicates (L)


class RuleAgentState(TypedDict):
    """
    Rule Agent State: cluster-level rule extraction subgraph state
    """
    cluster_id: Optional[int]
    content: str
    predicates_context: str
    med_terms_context: str
    quotes_map: Dict[str, Dict]
    fragments: List[dict]
    rules: Annotated[List[ClinicalRule], merge_by_id]


def extract_predicate_atoms(state: PredicateAgentState, config: Optional[RunnableConfig] = None) -> dict:
    """
    Extract predicate atoms (cluster-level).

    Args:
        state: PredicateAgentState containing cluster_id, content, terms_context, med_terms_context.

    Returns:
        dict with key "atoms" -> list of atomic condition dicts.
    """
    content = state.get("content", "")
    terms_context = state.get("terms_context", "")
    med_terms_context = state.get("med_terms_context", "")
    cluster_id = state.get("cluster_id")
    
    if not content:
        return {"atoms": []}
    
    # 构建包含术语上下文的完整内容
    full_content = content
    if terms_context:
        full_content += f"\n\nAvailable standard terms:\n{terms_context}"
    if med_terms_context:
        full_content += f"\n\nAvailable standard medications:\n{med_terms_context}"
    
    system_msg = SystemMessage(content=PREDICATES_PROMPT)
    
    
    # Use centralized structured invocation helper
    res = _invoke_structured_list(
        prompt=PREDICATES_PROMPT,
        schema_cls=PredicateExtractionBatch,
        content=full_content,
        error_tag="predicate_agent_extract_atoms",
        task_info={"cluster_id": cluster_id},
        run_config=config,
    )

    if not res:
        return {"atoms": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"atoms": []}

    predicates = getattr(parsed, "predicates", []) or []
    out_atoms = []
    for predicate in predicates:
        out_atoms.append(predicate.model_dump() if hasattr(predicate, "model_dump") else predicate)
    return {"atoms": out_atoms}


async def async_extract_predicate_atoms(state: PredicateAgentState, config: Optional[RunnableConfig] = None) -> dict:
    """异步版本：抽取谓词原子条件。"""
    content = state.get("content", "")
    terms_context = state.get("terms_context", "")
    med_terms_context = state.get("med_terms_context", "")
    cluster_id = state.get("cluster_id")

    if not content:
        return {"atoms": []}

    full_content = content
    if terms_context:
        full_content += f"\n\nAvailable standard terms:\n{terms_context}"
    if med_terms_context:
        full_content += f"\n\nAvailable standard medications:\n{med_terms_context}"

    res = await _ainvoke_structured_list(
        prompt=PREDICATES_PROMPT,
        schema_cls=PredicateExtractionBatch,
        content=full_content,
        error_tag="predicate_agent_extract_atoms",
        task_info={"cluster_id": cluster_id},
        run_config=config,
    )

    if not res:
        return {"atoms": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"atoms": []}

    predicates = getattr(parsed, "predicates", []) or []
    out_atoms = []
    for predicate in predicates:
        out_atoms.append(predicate.model_dump() if hasattr(predicate, "model_dump") else predicate)
    return {"atoms": out_atoms}


def _normalize_term(term: str) -> str:
    """
    规范化 term ID，转换为 LLM 友好的点号分隔格式
    
    Args:
        term: 原始 term ID，如 "meas.egfr", "cond.hf__", "med.class.beta_blocker"
    
    Returns:
        规范化后的 term，如 "meas.egfr", "cond.hf", "med.class.beta_blocker"
    """
    if not term:
        return ""
    # 清理末尾的下划线（常见于 Has/On 类型的 term）
    term = term.rstrip("_")
    # term 通常已经有点号分隔（如 meas.egfr），只需确保格式一致
    # 如果 term 中有下划线但没有点号，将下划线转换为点号
    if "." not in term and "_" in term:
        term = term.replace("_", ".")
    return term


def _normalize_value(val: Any) -> str:
    """
    规范化值，转换为 LLM 友好的 ID 格式
    
    Args:
        val: 原始值，可能是数字、字符串、布尔值等
    
    Returns:
        规范化后的值字符串，如 "20", "-25%", "TargetOrMaxTolerated", "true"
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    
    val_str = str(val)
    # 处理特殊字符：空格转为下划线，其他特殊字符保持不变（如 %, -, + 等）
    val_str = val_str.replace(" ", "_")
    # 移除可能存在的引号
    val_str = val_str.strip("'\"")
    return val_str


def _normalize_comparison(cmp: str, default: str = "==") -> str:
    """
    规范化比较符，转换为 LLM 友好的缩写形式
    
    Args:
        cmp: 原始比较符，如 ">=", "<=", "==" 等
        default: 默认比较符（当 cmp 为空时）
    
    Returns:
        规范化后的比较符缩写，如 "ge", "le", "eq" 等
    """
    cmp_map = {
        ">=": "ge",  # greater or equal
        "<=": "le",  # less or equal
        ">": "gt",   # greater than
        "<": "lt",   # less than
        "==": "eq",  # equal
        "!=": "ne"   # not equal
    }
    cmp = cmp or default
    return cmp_map.get(cmp, cmp)


def assemble_predicate(atom: dict) -> dict:
    """
    Assemble a single atomic condition into a Predicates model.

    Args:
        atom: dict describing an atomic condition (text, operator, term_id, comparison, target_value, etc.)

    Returns:
        dict with key "predicates" -> list containing one Predicates instance.
    """
    # v2 path: PredicateExtractionBatch already returns complete typed schemas.
    if isinstance(atom, Predicates):
        return {"predicates": [atom]}
    if isinstance(atom, dict) and atom.get("input_shape") and atom.get("reduction") and atom.get("final_output_type"):
        try:
            return {"predicates": [Predicates.model_validate(atom)]}
        except Exception as e:
            print(f"[assemble_predicate] Invalid v2 predicate schema {atom.get('id')}: {e}")
            return {"predicates": []}

    # Legacy fallback for stale prompts/caches. It emits v2 fields and keeps
    # formal_definition only as a deprecated trace field.
    text = atom.get("text", "")
    op_type = atom.get("operator", "")
    term = atom.get("term_id", "")
    cmp = atom.get("comparison", "")
    val = atom.get("target_value", "")
    
    # 统一算子名称：History -> HistoryOf（与 models.py 保持一致）
    if op_type == "History":
        op_type = "HistoryOf"
    
    # 规范化 term
    term_normalized = _normalize_term(term)
    
    # 分发逻辑 (Dispatcher)
    if op_type in ["Has", "On", "HistoryOf"]:
        # Bool 开关类，无需比较符
        # legacy input normalized to v2 id, e.g. pred.cond.hf.exists
        formal_def = f"$ {op_type}({term}) $"
        suffix = "on" if op_type == "On" else "history" if op_type == "HistoryOf" else "exists"
        pred_id = f"pred.{term_normalized}.{suffix}"
        input_shape = "List<MedicationStatement>" if term_normalized.startswith("med.") else "List<Condition>"
        entity_type = "medication" if term_normalized.startswith("med.") else "condition"
        aspect = "existence"
        reduction = {"operator": "exists", "output_type": "Bool"}
        final_output_type = "Bool"
        return_type = "Bool"
        temporal_scope = {"mode": "currently_active" if op_type == "On" else "all_time"}
        retrieve = {"resource": "MedicationStatement" if entity_type == "medication" else "Condition", "code_binding": term_normalized}
        extract = {}
        compare = None
        unit = None
        library_function = []
        
    elif op_type in ["Value", "Duration", "Delta", "Stage", "Risk"]:
        # 需要比较符和值，缺失则容错处理
        # legacy input normalized to v2 id, e.g. pred.meas.egfr.value.ge.20
        # 如果值缺失，不应盲目默认为 True（会改变数值比较语义）。
        # 将缺省值置为 None（由后续修复 agent 补全）；在构造 ID 时使用占位符 "UNSPECIFIED"。
        if val == "":
            val = None

        cmp_normalized = _normalize_comparison(cmp, default="==")
        if val is None:
            val_normalized = "UNSPECIFIED"
            formal_def = f"$ {op_type}({term}) {cmp or '=='} UNSPECIFIED $"
        else:
            val_normalized = _normalize_value(val)
            formal_def = f"$ {op_type}({term}) {cmp or '=='} {val} $"

        input_shape = "List<Observation>" if term_normalized.startswith("meas.") else "List<Resource>"
        entity_type = "observation" if term_normalized.startswith("meas.") else "condition"
        aspect_map = {"Value": "quantity", "Delta": "delta", "Duration": "duration", "Stage": "stage", "Risk": "risk"}
        aspect = aspect_map.get(op_type, op_type.lower())
        id_aspect = "value" if aspect == "quantity" else aspect
        pred_id = f"pred.{term_normalized}.{id_aspect}.{cmp_normalized}.{val_normalized}"
        reduction = {"operator": "most_recent" if op_type in ["Value", "Delta"] else "none", "output_type": "Quantity" if op_type in ["Value", "Delta", "Duration"] else "Enum"}
        final_output_type = "Bool"
        return_type = "Bool"
        temporal_scope = {"mode": "all_time"}
        retrieve = {"resource": "Observation" if entity_type == "observation" else "Condition", "code_binding": term_normalized}
        extract = {"path": "valueQuantity", "type": "Quantity"} if op_type in ["Value", "Delta"] else {}
        compare = {"operator": cmp_normalized, "value": val}
        unit = atom.get("unit")
        library_function = ["lib.fhir.most_recent"] if op_type in ["Value", "Delta"] else []
        
    elif op_type == "Assess":
        # Assess("Intolerance", med.acei) == True
        # legacy input normalized to v2 id, e.g. pred.med.class.beta_blocker.dosagestatus.eq.TargetOrMaxTolerated
        assess_type = atom.get("assess_type", "Status")
        # Assess 可能期待字符串标签或枚举（非布尔）。若缺失则设为 None，等待后续 agent 修复。
        if val == "":
            val = None

        cmp_normalized = _normalize_comparison(cmp, default="==")
        if val is None:
            val_normalized = "UNSPECIFIED"
            formal_def = f"$ Assess('{assess_type}', {term}) {cmp or '=='} UNSPECIFIED $"
        else:
            val_normalized = _normalize_value(val)
            formal_def = f"$ Assess('{assess_type}', {term}) {cmp or '=='} {val} $"

        input_shape = "List<Resource>"
        entity_type = "medication" if term_normalized.startswith("med.") else "condition"
        aspect = str(assess_type).lower()
        pred_id = f"pred.{term_normalized}.{aspect}.{cmp_normalized}.{val_normalized}"
        reduction = {"operator": "none", "output_type": "Enum"}
        final_output_type = "Bool"
        return_type = "Bool"
        temporal_scope = {"mode": "all_time"}
        retrieve = {"resource": "AllergyIntolerance" if "allerg" in aspect.lower() else "Resource", "code_binding": term_normalized}
        extract = {"path": aspect, "type": "Enum"}
        compare = {"operator": cmp_normalized, "value": val}
        unit = None
        library_function = []
        
    else:
        # 兜底：直接输出
        if cmp and val:
            cmp_normalized = _normalize_comparison(cmp, default="==")
            val_normalized = _normalize_value(val)
            formal_def = f"$ {term} {cmp} {val} $"
            pred_id = f"pred.{term_normalized}.{str(op_type).lower()}.{cmp_normalized}.{val_normalized}"
        else:
            formal_def = f"$ {term} $"
            pred_id = f"pred.{term_normalized}.{str(op_type).lower() if op_type else 'exists'}"
        input_shape = "List<Resource>"
        entity_type = "resource"
        aspect = str(op_type).lower() if op_type else "existence"
        reduction = {"operator": "none", "output_type": "Bool"}
        final_output_type = "Bool"
        return_type = "Bool"
        temporal_scope = {"mode": "all_time"}
        retrieve = {"resource": "Resource", "code_binding": term_normalized}
        extract = {}
        compare = {"operator": _normalize_comparison(cmp), "value": val} if cmp and val else None
        unit = atom.get("unit")
        library_function = []
    
    pred = Predicates(
        id=pred_id,
        name=text,
        description=text,
        source_text=text,
        entity=term_normalized,
        entity_type=entity_type,
        aspect=aspect,
        input_shape=input_shape,
        reduction=reduction,
        return_type=return_type,
        final_output_type=final_output_type,
        temporal_scope=temporal_scope,
        data_binding={
            "FHIR": {"resource": retrieve.get("resource")},
            "OMOP": {"table": "drug_exposure" if entity_type == "medication" else "measurement" if entity_type == "observation" else "condition_occurrence"}
        },
        library_function=library_function,
        value_set_binding={"type": "Unknown", "name": term_normalized, "confidence": 0.2} if term_normalized else None,
        unit=unit,
        quantity_semantics={"unit": unit, **(compare or {})} if unit or compare else {},
        retrieve=retrieve,
        extract=extract,
        compare=compare,
        null_policy="unknown",
        evidence=[{"source_text": text}] if text else [],
        source_span={"source_text": text} if text else None,
        formal_definition=formal_def,
        dependencies=[term] if term else [],
    )
    
    return {"predicates": [pred]}


def distribute_predicates(state: PredicateAgentState) -> List[Send]:
    """Distribute predicate atoms to assemble nodes in parallel."""
    atoms = state.get("atoms", [])
    sends = []
    for atom in atoms:
        sends.append(Send("predicate_agent_assemble_predicate", atom))
    return sends


def build_predicate_subgraph():
    """
    Build the predicate extraction subgraph.
    
    Workflow:
    START → extract_atoms (E × O extraction)
         → distribute (parallelization)
         → assemble_predicate (F(E, O) → L formalization)
         → END
    
    This subgraph implements the Symbolic Operator Layer, bridging the semantic gap
    between ontological entities and logical predicates for SMT-based verification.
    """
    builder = StateGraph(PredicateAgentState)

    builder.add_node("predicate_agent_extract_atoms", extract_predicate_atoms)
    builder.add_node("predicate_agent_assemble_predicate", assemble_predicate)

    builder.add_edge(START, "predicate_agent_extract_atoms")
    builder.add_conditional_edges(
        "predicate_agent_extract_atoms",
        distribute_predicates,
        ["predicate_agent_assemble_predicate"]
    )
    builder.add_edge("predicate_agent_assemble_predicate", END)

    return builder.compile()


def async_build_predicate_subgraph():
    """异步版本：构建谓词抽取子图（使用异步节点）。"""
    builder = StateGraph(PredicateAgentState)

    builder.add_node("predicate_agent_extract_atoms", async_extract_predicate_atoms)
    builder.add_node("predicate_agent_assemble_predicate", assemble_predicate)

    builder.add_edge(START, "predicate_agent_extract_atoms")
    builder.add_conditional_edges(
        "predicate_agent_extract_atoms",
        distribute_predicates,
        ["predicate_agent_assemble_predicate"]
    )
    builder.add_edge("predicate_agent_assemble_predicate", END)

    return builder.compile()


async def async_extract_predicates_subgraph(
    state: ClusterState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """异步版本：运行谓词抽取子图。"""
    content = "\n\n".join(state.get("texts_formatted", []))
    terms = state.get("terms", []) or []
    med_terms = state.get("med_terms", []) or []
    cluster_id = state.get("cluster_id")

    if not content:
        return {"predicates": []}

    if not terms and not med_terms:
        msg = "Skipping predicate extraction because term extraction produced no terms or medications for this cluster."
        print(f"[async_extract_predicates] cluster {cluster_id}: {msg}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_predicates",
            error=msg,
            task_info={"cluster_id": cluster_id},
        )
        return {"predicates": []}

    terms_content = "\n".join([_context_json(term) for term in terms])
    med_terms_content = "\n".join([_context_json(med) for med in med_terms])

    predicate_state = {
        "cluster_id": cluster_id,
        "content": content,
        "terms_context": terms_content,
        "med_terms_context": med_terms_content,
        "atoms": [],
        "predicates": []
    }

    try:
        predicate_agent = async_build_predicate_subgraph()
        result = await predicate_agent.ainvoke(predicate_state, config=config)
        predicates = result.get("predicates", [])
        return {"predicates": predicates}
    except Exception as e:
        print(f"[async_extract_predicates] Error: {e}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_predicates",
            error=str(e),
            task_info={"cluster_id": cluster_id}
        )
        return {"predicates": []}


def extract_predicates_subgraph(
    state: ClusterState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """Run the predicate extraction subgraph for a cluster and return predicates."""
    content = "\n\n".join(state.get("texts_formatted", []))
    terms = state.get("terms", []) or []
    med_terms = state.get("med_terms", []) or []
    cluster_id = state.get("cluster_id")

    if not content:
        return {"predicates": []}

    if not terms and not med_terms:
        msg = "Skipping predicate extraction because term extraction produced no terms or medications for this cluster."
        print(f"[extract_predicates] cluster {cluster_id}: {msg}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_predicates",
            error=msg,
            task_info={"cluster_id": cluster_id},
        )
        return {"predicates": []}

    # Build terms/med terms context
    terms_content = "\n".join([_context_json(term) for term in terms])
    med_terms_content = "\n".join([_context_json(med) for med in med_terms])

    predicate_state = {
        "cluster_id": cluster_id,
        "content": content,
        "terms_context": terms_content,
        "med_terms_context": med_terms_content,
        "atoms": [],
        "predicates": []
    }

    try:
        predicate_agent = build_predicate_subgraph()
        result = predicate_agent.invoke(predicate_state, config=config)
        predicates = result.get("predicates", [])
        return {"predicates": predicates}
    except Exception as e:
        print(f"[extract_predicates] Error: {e}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_predicates",
            error=str(e),
            task_info={"cluster_id": cluster_id}
        )
        return {"predicates": []}

# ============ rules node ============

def extract_rule_fragments(state: RuleAgentState, config: Optional[RunnableConfig] = None) -> dict:
    """Extract simplified rule fragments for a cluster using LLM structured output."""

    content = state.get("content", "")
    cluster_id = state.get("cluster_id")
    quotes_map = state.get("quotes_map", {})

    # Debug: 验证 quotes_map 是否正确传递
    if not quotes_map:
        print(f"[extract_rule_fragments] WARNING: quotes_map is empty for cluster {cluster_id}")
    else:
        print(f"[extract_rule_fragments] quotes_map has {len(quotes_map)} entries for cluster {cluster_id}")

    if not content:
        return {"fragments": [], "quotes_map": quotes_map}

    # content 已经包含了完整的上下文
    full_content = content

    # Use centralized structured invocation helper
    res = _invoke_structured_list(
        prompt=RULES_PROMPT,
        schema_cls=SubmitSimplifiedRules,
        content=full_content,
        error_tag="rule_agent_extract_fragments",
        task_info={"cluster_id": cluster_id},
        run_config=config,
    )

    if not res:
        return {"fragments": [], "quotes_map": quotes_map}

    parsed = res.get("parsed")
    if not parsed:
        return {"fragments": [], "quotes_map": quotes_map}

    items = getattr(parsed, "rules", []) or []
    fragments = [item.model_dump() if hasattr(item, "model_dump") else item for item in items]
    return {"fragments": fragments, "quotes_map": quotes_map}


async def async_extract_rule_fragments(
    state: RuleAgentState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """异步版本：抽取规则片段。"""
    content = state.get("content", "")
    cluster_id = state.get("cluster_id")
    quotes_map = state.get("quotes_map", {})

    if not quotes_map:
        print(f"[async_extract_rule_fragments] WARNING: quotes_map is empty for cluster {cluster_id}")
    else:
        print(f"[async_extract_rule_fragments] quotes_map has {len(quotes_map)} entries for cluster {cluster_id}")

    if not content:
        return {"fragments": [], "quotes_map": quotes_map}

    res = await _ainvoke_structured_list(
        prompt=RULES_PROMPT,
        schema_cls=SubmitSimplifiedRules,
        content=content,
        error_tag="rule_agent_extract_fragments",
        task_info={"cluster_id": cluster_id},
        run_config=config,
    )

    if not res:
        return {"fragments": [], "quotes_map": quotes_map}

    parsed = res.get("parsed")
    if not parsed:
        return {"fragments": [], "quotes_map": quotes_map}

    items = getattr(parsed, "rules", []) or []
    fragments = [item.model_dump() if hasattr(item, "model_dump") else item for item in items]
    return {"fragments": fragments, "quotes_map": quotes_map}


def distribute_rule_fragments(state: RuleAgentState) -> List[Send]:
    """Distribute simplified fragments to assemble nodes in parallel."""
    fragments = state.get("fragments", []) or []
    quotes_map = state.get("quotes_map", {}) or {}
    cluster_id = state.get("cluster_id")
    sends = []
    for frag in fragments:
        payload = {"fragment": frag, "quotes_map": quotes_map, "cluster_id": cluster_id}
        sends.append(Send("rule_agent_assemble_rule", payload))
    return sends


def assemble_rule_fragment(payload: dict) -> dict:
    """Assemble a single simplified fragment into a ClinicalRule with provenance backfilled."""
    fragment = payload.get("fragment", {}) if isinstance(payload, dict) else payload
    quotes_map = payload.get("quotes_map", {}) if isinstance(payload, dict) else {}
    cluster_id = payload.get("cluster_id")


    try:
        rule_id = fragment.get("id") or fragment.get("rule_id")
        label = fragment.get("label", "")
        condition = fragment.get("condition")
        input_predicates = fragment.get("input_predicates", []) or []
        condition_dag = fragment.get("condition_dag")
        boolean_root = fragment.get("boolean_root") or "ROOT"
        missing_data_policy = fragment.get("missing_data_policy") or "propagate_unknown"
        scope = fragment.get("scope", {}) or {}
        action = fragment.get("action", {})
        priority = fragment.get("priority") or {}
        output_assembly = fragment.get("output_assembly")
        # 处理source_ids字段 - 必须是有效的字符串key
        source_id = fragment.get("source_ids")

        if not source_id or not isinstance(source_id, str) or not source_id.strip():
            # source_ids字段缺失、无效或为空，这是系统错误
            error_msg = f"[assemble_rule_fragment] CRITICAL ERROR: Rule {rule_id} has invalid source_ids field: {source_id}"
            print(error_msg)
            print(f"[assemble_rule_fragment] Fragment: {fragment}")
            print(f"[assemble_rule_fragment] Available quotes_map keys: {list(quotes_map.keys())}")

            # 记录失败任务
            
            get_failed_task_logger().log_generic_failure(
                stage="assemble_rule_fragment",
                error=f"Invalid source_ids '{source_id}' for rule {rule_id}",
                task_info={"cluster_id": cluster_id, "rule_id": rule_id, "source_id": source_id, "fragment": fragment}
            )
            return {"rules": []}

        # 验证source_id(s)是否存在于quotes_map中，支持单个或组合的source_ids（如'q5,q7'）
        source_ids_list = [s.strip() for s in source_id.split(',') if s.strip()]

        # 验证所有source_ids都存在
        missing_ids = [sid for sid in source_ids_list if sid not in quotes_map]
        if missing_ids:
            error_msg = f"[assemble_rule_fragment] CRITICAL ERROR: source_id(s) '{','.join(missing_ids)}' not found in quotes_map for rule {rule_id}"
            print(error_msg)
            print(f"[assemble_rule_fragment] Available keys: {list(quotes_map.keys())}")


            get_failed_task_logger().log_generic_failure(
                stage="assemble_rule_fragment",
                error=f"source_id(s) '{','.join(missing_ids)}' not in quotes_map for rule {rule_id}",
                task_info={"cluster_id": cluster_id, "rule_id": rule_id, "source_id": source_id, "missing_ids": missing_ids, "available_keys": list(quotes_map.keys())}
            )
            return {"rules": []}

        # 为每个source_id创建provenance
        provenance_list = []
        for sid in source_ids_list:
            prov_dict = quotes_map[sid]
            provenance_list.append(Provenance(**prov_dict))
        source_quotes = [prov.quote for prov in provenance_list if prov.quote]
        source_text = "\n".join(source_quotes)
        source_evidence = [
            {
                "source": prov.source,
                "quote": prov.quote,
                "source_text": prov.quote,
                "source_span": prov.source_span,
            }
            for prov in provenance_list
            if prov.quote
        ]
        source_span = {"source_text": source_text} if source_text else None

        if hasattr(priority, "model_dump"):
            priority = priority.model_dump()
        priority = dict(priority or {})
        if not priority.get("recommendation_grade"):
            priority["recommendation_grade"] = next(
                (prov.recommendation_grade for prov in provenance_list if prov.recommendation_grade),
                None,
            )
        if not priority.get("evidence_level"):
            priority["evidence_level"] = next(
                (prov.evidence_level for prov in provenance_list if prov.evidence_level),
                None,
            )

        if not condition_dag:
            error_msg = f"[assemble_rule_fragment] CRITICAL ERROR: Rule {rule_id} missing v2 condition_dag"
            print(error_msg)
            get_failed_task_logger().log_generic_failure(
                stage="assemble_rule_fragment",
                error=error_msg,
                task_info={"cluster_id": cluster_id, "rule_id": rule_id, "fragment": fragment}
            )
            return {"rules": []}

        cr = ClinicalRule(
            id=rule_id,
            label=label,
            source_text=source_text,
            source_span=source_span,
            source_evidence=source_evidence,
            input_predicates=input_predicates,
            condition_dag=condition_dag,
            boolean_root=boolean_root,
            missing_data_policy=missing_data_policy,
            action=action,
            scope=scope,
            priority=priority,
            provenance=provenance_list,
            output_assembly=output_assembly,
            condition=condition,
        )
        return {"rules": [cr]}
    except Exception as e:
        print(f"[rule_agent_assemble_rule] Failed to assemble fragment {fragment}: {e}")
        return {"rules": []}


def build_rule_subgraph():
    """Build the rule extraction subgraph (extract -> distribute -> assemble)."""
    builder = StateGraph(RuleAgentState)
    builder.add_node("rule_agent_extract_fragments", extract_rule_fragments)
    builder.add_node("rule_agent_assemble_rule", assemble_rule_fragment)

    builder.add_edge(START, "rule_agent_extract_fragments")
    builder.add_conditional_edges(
        "rule_agent_extract_fragments",
        distribute_rule_fragments,
        ["rule_agent_assemble_rule"]
    )
    builder.add_edge("rule_agent_assemble_rule", END)
    return builder.compile()


def async_build_rule_subgraph():
    """异步版本：构建规则抽取子图（使用异步节点）。"""
    builder = StateGraph(RuleAgentState)
    builder.add_node("rule_agent_extract_fragments", async_extract_rule_fragments)
    builder.add_node("rule_agent_assemble_rule", assemble_rule_fragment)

    builder.add_edge(START, "rule_agent_extract_fragments")
    builder.add_conditional_edges(
        "rule_agent_extract_fragments",
        distribute_rule_fragments,
        ["rule_agent_assemble_rule"]
    )
    builder.add_edge("rule_agent_assemble_rule", END)
    return builder.compile()



def extract_rules_subgraph(
    state: ClusterState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """Cluster-level rule extraction: build quotes map, call rule subgraph and return rules."""
    cluster_id = state.get("cluster_id")
    provenances = state.get('provenances', []) or []
    predicates = state.get("predicates", []) or []
    med_terms = state.get("med_terms", []) or []

    # Debug: 打印 provenances 数量
    print(f"[extract_rules_subgraph] cluster {cluster_id}: provenances={len(provenances)}, predicates={len(predicates)}, med_terms={len(med_terms)}")

    if not predicates:
        msg = "Skipping rule extraction because predicate extraction produced no predicates for this cluster."
        print(f"[extract_rules_subgraph] cluster {cluster_id}: {msg}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_rules",
            error=msg,
            task_info={"cluster_id": cluster_id},
        )
        return {"rules": []}

    # 动态为每个 provenance 生成 quote_id（q1, q2, ...），并构建引用映射用于回填
    quotes_map = {}
    quotes_content_lines = []
    for idx, p in enumerate(provenances):
        qid = f"q{idx+1}"
        quotes_map[qid] = {
            "source": getattr(p, "source", None) if hasattr(p, "source") else p.get("source"),
            "quote": getattr(p, "quote", None) if hasattr(p, "quote") else p.get("quote"),
            "recommendation_grade": getattr(p, "recommendation_grade", None) if hasattr(p, "recommendation_grade") else p.get("recommendation_grade"),
            "evidence_level": getattr(p, "evidence_level", None) if hasattr(p, "evidence_level") else p.get("evidence_level"),
        }

        quotes_content_lines.append(
            f"id: {qid}\nQuote: {quotes_map[qid]['quote']}\n"
        )
    # cluster-level content (原始拼接文本)

    predicates_content = "\n".join([_context_json(pred) for pred in predicates])
    med_terms_content = "\n".join([_context_json(med) for med in med_terms])

    # 在 prompt 中显式传入 quote_id 映射，指示 LLM 在输出规则时只使用 quote_id 作为 provenance 引用
    quotes_content = "\n\n".join(quotes_content_lines)
    prompt_body = (
        f"Quote IDs and texts:\n{quotes_content}\n\n"
        f"Available predicates:\n{predicates_content}\n\n"
        f"Available medications:\n{med_terms_content}\n\n"
    )

    # 使用 Rule Agent 子图（extract_fragments -> distribute -> assemble_rule）并行化处理
    rule_agent = build_rule_subgraph()
    # 把 prompt_body 放入 content 以便 LLM 看到 quote_id 映射和上下文（子节点会再次构建需要的上下文）
    rule_state = {
        "cluster_id": cluster_id,
        "content": prompt_body,
        "quotes_map": quotes_map,
        "fragments": [],
        "rules": []
    }
    print(f"[extract_rules_subgraph] Passing quotes_map with {len(quotes_map)} keys to rule_agent")
    try:
        result = rule_agent.invoke(rule_state, config=config)
        rules = result.get("rules", [])
        return {"rules": rules}
    except Exception as e:
        print(f"[extract_rules_subgraph] Error invoking rule agent subgraph: {e}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_rules",
            error=str(e),
            task_info={"cluster_id": cluster_id}
        )
        return {"rules": []}


async def async_extract_rules_subgraph(
    state: ClusterState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """异步版本：运行规则抽取子图。"""
    cluster_id = state.get("cluster_id")
    provenances = state.get('provenances', []) or []
    predicates = state.get("predicates", []) or []
    med_terms = state.get("med_terms", []) or []

    print(f"[async_extract_rules_subgraph] cluster {cluster_id}: provenances={len(provenances)}, predicates={len(predicates)}, med_terms={len(med_terms)}")

    if not predicates:
        msg = "Skipping rule extraction because predicate extraction produced no predicates for this cluster."
        print(f"[async_extract_rules_subgraph] cluster {cluster_id}: {msg}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_rules",
            error=msg,
            task_info={"cluster_id": cluster_id},
        )
        return {"rules": []}

    quotes_map = {}
    quotes_content_lines = []
    for idx, p in enumerate(provenances):
        qid = f"q{idx+1}"
        quotes_map[qid] = {
            "source": getattr(p, "source", None) if hasattr(p, "source") else p.get("source"),
            "quote": getattr(p, "quote", None) if hasattr(p, "quote") else p.get("quote"),
            "recommendation_grade": getattr(p, "recommendation_grade", None) if hasattr(p, "recommendation_grade") else p.get("recommendation_grade"),
            "evidence_level": getattr(p, "evidence_level", None) if hasattr(p, "evidence_level") else p.get("evidence_level"),
        }
        quotes_content_lines.append(f"id: {qid}\nQuote: {quotes_map[qid]['quote']}\n")

    predicates_content = "\n".join([_context_json(pred) for pred in predicates])
    med_terms_content = "\n".join([_context_json(med) for med in med_terms])

    quotes_content = "\n\n".join(quotes_content_lines)
    prompt_body = (
        f"Quote IDs and texts:\n{quotes_content}\n\n"
        f"Available predicates:\n{predicates_content}\n\n"
        f"Available medications:\n{med_terms_content}\n\n"
    )

    rule_agent = async_build_rule_subgraph()
    rule_state = {
        "cluster_id": cluster_id,
        "content": prompt_body,
        "quotes_map": quotes_map,
        "fragments": [],
        "rules": []
    }
    print(f"[async_extract_rules_subgraph] Passing quotes_map with {len(quotes_map)} keys to rule_agent")
    try:
        result = await rule_agent.ainvoke(rule_state, config=config)
        rules = result.get("rules", [])
        return {"rules": rules}
    except Exception as e:
        print(f"[async_extract_rules_subgraph] Error invoking rule agent subgraph: {e}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_rules",
            error=str(e),
            task_info={"cluster_id": cluster_id}
        )
        return {"rules": []}


# ============ Post-Cluster Extraction Nodes ============

def extract_contraindications(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Extract Contraindication objects from the full guideline text + extracted rules.

    This node runs after all cluster-level extraction stages are complete.
    It takes the aggregated rules and provenance texts as context.

    Args:
        state: A dict-like state containing at minimum 'rules', 'input_texts',
               and 'provenance_buffer'.
        config: Optional LangGraph RunnableConfig for tracing.

    Returns:
        dict with key 'contraindications' -> list of Contraindication dicts.
    """
    rules = state.get("rules", []) or []
    input_texts = state.get("input_texts", []) or []

    if not input_texts or not rules:
        print("[extract_contraindications] No text or rules available; skipping")
        return {"contraindications": []}

    # Build context: full guideline text + rules summary
    full_text = "\n\n".join(input_texts)
    rules_context = "\n".join([
        f"Rule {getattr(r, 'id', '?')}: {getattr(r, 'label', '')}"
        for r in rules
    ])

    user_content = (
        f"Guideline Text:\n{full_text[:20000]}\n\n"
        f"Extracted ClinicalRules:\n{rules_context[:10000]}\n\n"
        f"Extract all contraindications from the above guideline text."
    )

    from .models import ContraindicationList

    res = _invoke_structured_list(
        prompt=CONTRAINDICATION_PROMPT,
        schema_cls=ContraindicationList,
        content=user_content,
        error_tag="extract_contraindications",
        task_info={"num_rules": len(rules)},
        run_config=config,
    )

    if not res:
        return {"contraindications": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"contraindications": []}

    items = _safe_items(parsed.items)
    print(f"[extract_contraindications] Extracted {len(items)} contraindications")
    return {"contraindications": items}


async def async_extract_contraindications(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Async version of extract_contraindications."""
    rules = state.get("rules", []) or []
    input_texts = state.get("input_texts", []) or []

    if not input_texts or not rules:
        print("[extract_contraindications] No text or rules available; skipping")
        return {"contraindications": []}

    full_text = "\n\n".join(input_texts)
    rules_context = "\n".join([
        f"Rule {getattr(r, 'id', '?')}: {getattr(r, 'label', '')}"
        for r in rules
    ])

    user_content = (
        f"Guideline Text:\n{full_text[:20000]}\n\n"
        f"Extracted ClinicalRules:\n{rules_context[:10000]}\n\n"
        f"Extract all contraindications from the above guideline text."
    )

    from .models import ContraindicationList

    res = await _ainvoke_structured_list(
        prompt=CONTRAINDICATION_PROMPT,
        schema_cls=ContraindicationList,
        content=user_content,
        error_tag="extract_contraindications",
        task_info={"num_rules": len(rules)},
        run_config=config,
    )

    if not res:
        return {"contraindications": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"contraindications": []}

    items = _safe_items(parsed.items)
    print(f"[extract_contraindications] Extracted {len(items)} contraindications")
    return {"contraindications": items}


def extract_safety_constraints(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Extract SafetyConstraint objects from guideline text + extracted Contraindications.

    This node runs after contraindication extraction, using previously extracted
    Contraindications as context for mapping safety language to minefield labels.

    Args:
        state: A dict-like state containing 'rules', 'input_texts', and
               'contraindications'.
        config: Optional LangGraph RunnableConfig for tracing.

    Returns:
        dict with key 'safety_constraints' -> list of SafetyConstraint dicts.
    """
    input_texts = state.get("input_texts", []) or []
    contraindications = state.get("contraindications", []) or []

    if not input_texts:
        print("[extract_safety_constraints] No text available; skipping")
        return {"safety_constraints": []}

    full_text = "\n\n".join(input_texts)
    contra_context = "\n".join([
        f"Contraindication {c.get('id', '?') if isinstance(c, dict) else getattr(c, 'id', '?')}: "
        f"{c.get('description', '') if isinstance(c, dict) else getattr(c, 'description', '')}"
        for c in contraindications
    ]) if contraindications else "No contraindications extracted."

    user_content = (
        f"Guideline Text:\n{full_text[:20000]}\n\n"
        f"Extracted Contraindications:\n{contra_context[:10000]}\n\n"
        f"Extract all safety constraints from the above guideline text."
    )

    from .models import SafetyConstraintList

    res = _invoke_structured_list(
        prompt=SAFETY_CONSTRAINT_PROMPT,
        schema_cls=SafetyConstraintList,
        content=user_content,
        error_tag="extract_safety_constraints",
        task_info={"num_contraindications": len(contraindications)},
        run_config=config,
    )

    if not res:
        return {"safety_constraints": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"safety_constraints": []}

    items = _safe_items(parsed.items)
    print(f"[extract_safety_constraints] Extracted {len(items)} safety constraints")
    return {"safety_constraints": items}


async def async_extract_safety_constraints(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Async version of extract_safety_constraints."""
    input_texts = state.get("input_texts", []) or []
    contraindications = state.get("contraindications", []) or []

    if not input_texts:
        print("[extract_safety_constraints] No text available; skipping")
        return {"safety_constraints": []}

    full_text = "\n\n".join(input_texts)
    contra_context = "\n".join([
        f"Contraindication {c.get('id', '?') if isinstance(c, dict) else getattr(c, 'id', '?')}: "
        f"{c.get('description', '') if isinstance(c, dict) else getattr(c, 'description', '')}"
        for c in contraindications
    ]) if contraindications else "No contraindications extracted."

    user_content = (
        f"Guideline Text:\n{full_text[:20000]}\n\n"
        f"Extracted Contraindications:\n{contra_context[:10000]}\n\n"
        f"Extract all safety constraints from the above guideline text."
    )

    from .models import SafetyConstraintList

    res = await _ainvoke_structured_list(
        prompt=SAFETY_CONSTRAINT_PROMPT,
        schema_cls=SafetyConstraintList,
        content=user_content,
        error_tag="extract_safety_constraints",
        task_info={"num_contraindications": len(contraindications)},
        run_config=config,
    )

    if not res:
        return {"safety_constraints": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"safety_constraints": []}

    items = _safe_items(parsed.items)
    print(f"[extract_safety_constraints] Extracted {len(items)} safety constraints")
    return {"safety_constraints": items}


def compute_missing_variables(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Deterministic node: derive MissingVariableSpec lists for all ClinicalRules.

    Walks each rule's ConditionDAG, collects predicate_ref references, maps them
    to clinical variable types, and outputs MissingVariableSpec dicts per rule.

    Args:
        state: A dict-like state containing 'rules' and 'predicates'.
        config: Optional LangGraph RunnableConfig (unused, but accepted for
                LangGraph node signature compatibility).

    Returns:
        dict with key 'missing_variables' -> dict keyed by rule_id, each value
        a list of MissingVariableSpec dicts.
    """
    rules = state.get("rules", []) or []
    predicates = state.get("predicates", []) or []

    if not rules:
        print("[compute_missing_variables] No rules available; skipping")
        return {"missing_variables": {}}

    missing_vars = derive_missing_variables(rules, predicates)
    total_vars = sum(len(v) for v in missing_vars.values())
    print(f"[compute_missing_variables] Derived {total_vars} missing variable specs across {len(missing_vars)} rules")
    return {"missing_variables": missing_vars}


def compute_metadata(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Deterministic terminal node: aggregate DecisionGraphMetadata from all extraction stages.

    Collects counts, confidence statistics, and compiler version information.

    Args:
        state: A dict-like state containing all extraction outputs.
        config: Optional LangGraph RunnableConfig (unused).

    Returns:
        dict with key 'graph_metadata' -> DecisionGraphMetadata dict.
    """
    import statistics
    from datetime import datetime, timezone

    rules = state.get("rules", []) or []
    predicates = state.get("predicates", []) or []
    contraindications = state.get("contraindications", []) or []
    safety_constraints = state.get("safety_constraints", []) or []
    provenance_buffer = state.get("provenance_buffer", []) or []

    total_rules = len(rules)
    total_predicates = len(predicates)
    total_contraindications = len(contraindications)
    total_safety_constraints = len(safety_constraints)

    # Compute confidence statistics from provenance evidence or source_evidence
    confidences = []
    for comp in list(rules) + list(predicates):
        if hasattr(comp, "source_evidence"):
            for se in (comp.source_evidence or []):
                if isinstance(se, dict):
                    conf = se.get("confidence")
                else:
                    conf = getattr(se, "confidence", None)
                if conf is not None and isinstance(conf, (int, float)):
                    confidences.append(float(conf))
        if hasattr(comp, "provenance"):
            for prov in (comp.provenance or []):
                pass

    extraction_confidence_mean = None
    extraction_confidence_median = None
    if confidences:
        try:
            extraction_confidence_mean = round(float(statistics.mean(confidences)), 4)
            extraction_confidence_median = round(float(statistics.median(confidences)), 4)
        except Exception:
            pass

    # Derive guideline_id from first provenance source
    guideline_id = "unknown"
    if provenance_buffer:
        first = provenance_buffer[0]
        source = getattr(first, "source", None) if hasattr(first, "source") else first.get("source")
        if source:
            guideline_id = source.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")

    from . import COMPILER_VERSION

    metadata = {
        "guideline_id": guideline_id,
        "compiler_version": COMPILER_VERSION,
        "compilation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rules": total_rules,
        "total_predicates": total_predicates,
        "total_contraindications": total_contraindications,
        "total_safety_constraints": total_safety_constraints,
        "extraction_confidence_mean": extraction_confidence_mean,
        "extraction_confidence_median": extraction_confidence_median,
        "human_review_status": "unreviewed",
    }

    print(f"[compute_metadata] {total_rules} rules, {total_predicates} predicates, "
          f"{total_contraindications} contraindications, {total_safety_constraints} safety constraints")
    return {"graph_metadata": metadata}


# ============ Graph Output Serialization ============

def serialize_extraction(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one JSON-safe extraction payload from an ``AgentState``.

    ``ClinicalGuidelinePipeline.run`` hands back the raw agent state, whose
    components are Pydantic objects.  This is the single place that flattens
    them, and it is shared by the serializer node and by callers that want the
    payload in memory (e.g. :func:`guideline2action.compile_guideline`) without
    round-tripping through a file.
    """
    def _json_safe(value: Any) -> Any:
        """Unwrap enum members so the payload is JSON-semantics, not Python repr.

        The extraction models type several fields as ``str``-derived enums
        (``Permission`` and friends).  ``model_dump()`` keeps the members, and
        ``str(member)`` renders as ``"Permission.RECOMMEND"`` rather than
        ``"recommend"``, which downstream contracts reject.  Unwrapping here
        keeps the in-memory payload identical to the JSON written to disk.
        """
        if isinstance(value, Enum):
            return _json_safe(value.value)
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return value

    def _serialize_items(items: Any) -> List[Any]:
        return [
            _json_safe(
                item.model_dump() if hasattr(item, "model_dump")
                else item.dict() if hasattr(item, "dict")
                else item
            )
            for item in (items or [])
        ]

    graph_metadata = state.get("graph_metadata") or {}
    if hasattr(graph_metadata, "model_dump"):
        metadata: Any = graph_metadata.model_dump()
    elif isinstance(graph_metadata, Mapping):
        metadata = dict(graph_metadata)
    else:
        metadata = {}

    guideline_id = "unknown"
    for provenance in state.get("provenance_buffer") or []:
        source = (
            getattr(provenance, "source", None)
            if hasattr(provenance, "source")
            else provenance.get("source")
            if isinstance(provenance, Mapping)
            else None
        )
        if source:
            guideline_id = str(source).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
            break
    if isinstance(metadata, Mapping):
        guideline_id = str(metadata.get("guideline_id") or guideline_id)

    return {
        "guideline_id": guideline_id,
        "metadata": _json_safe(metadata),
        "terms": _serialize_items(state.get("terms")),
        "med_terms": _serialize_items(state.get("med_terms")),
        "predicates": _serialize_items(state.get("predicates")),
        "rules": _serialize_items(state.get("rules")),
        "contraindications": _serialize_items(state.get("contraindications")),
        "safety_constraints": _serialize_items(state.get("safety_constraints")),
        "missing_variables": _json_safe(state.get("missing_variables") or {}),
    }


def serialize_graph_output(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Write the serialized extraction payload into the configured ``gen_dir``.

    Terminal LangGraph node: it persists the same payload
    :func:`serialize_extraction` returns, and adds nothing to the agent state.
    """
    import json

    output = serialize_extraction(state)
    output_dir = os.path.join(get_config().paths.gen_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output['guideline_id']}_graph.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, default=str)
    logger.info("[serialize_graph_output] written to %s", output_path)
    return {}

