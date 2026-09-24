"""Compile LangGraph StateGraphs for pipeline stages."""
from typing import Optional, Any, List, Dict, TypedDict, Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send

from .models import (
    AgentState,
    ClusterState,
    ProvenanceCluster,
    Term,
    MedicationTerm,
    Predicates,
    ClinicalRule,
    merge_by_id,
    merge_cluster_cache_updates,
    _to_models,
)
from .graph_nodes import (
    subgraph_extract_all_terms,
    asubgraph_extract_all_terms,
    extract_predicates_subgraph,
    async_extract_predicates_subgraph,
    extract_rules_subgraph,
    async_extract_rules_subgraph,
    extract_recommendation,
    distribute_texts,
    do_lsh_clustering,
    extract_contraindications,
    extract_safety_constraints,
    compute_missing_variables,
    compute_metadata,
    serialize_graph_output,
)

# ============ 图构建 ============

def build_extraction_subgraph():
    builder = StateGraph(ClusterState)
    builder.add_node("extract_all_terms", subgraph_extract_all_terms)
    builder.add_node("extract_predicates", extract_predicates_subgraph)
    builder.add_node("extract_rules", extract_rules_subgraph)
    builder.add_edge(START, "extract_all_terms")
    builder.add_edge("extract_all_terms", "extract_predicates")
    builder.add_edge("extract_predicates", "extract_rules")
    builder.add_edge("extract_rules", END)
    return builder.compile()


def async_build_extraction_subgraph():
    """异步版本：构建 cluster 抽取子图（使用异步节点）。"""
    builder = StateGraph(ClusterState)
    builder.add_node("extract_all_terms", asubgraph_extract_all_terms)
    builder.add_node("extract_predicates", async_extract_predicates_subgraph)
    builder.add_node("extract_rules", async_extract_rules_subgraph)
    builder.add_edge(START, "extract_all_terms")
    builder.add_edge("extract_all_terms", "extract_predicates")
    builder.add_edge("extract_predicates", "extract_rules")
    builder.add_edge("extract_rules", END)
    return builder.compile()


CLUSTER_SUBGRAPH = build_extraction_subgraph()
ASYNC_CLUSTER_SUBGRAPH = async_build_extraction_subgraph()

# ============ 智能缓存复用的处理节点 ============
# 节点函数现在直接实现缓存复用逻辑，不再需要预定义子图

def process_cluster(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    # state 是 Send 传入的 cluster 局部 dict
    result = CLUSTER_SUBGRAPH.invoke(state, config=config)

    return {
        "terms": result.get("terms", []),
        "med_terms": result.get("med_terms", []),
        "predicates": result.get("predicates", []),
        "rules": result.get("rules", []),
    }


def process_cluster_node(state: dict) -> dict:
    """Compatibility alias for integration tests and manual callers."""
    return process_cluster(state)


async def aprocess_cluster(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """异步版本：处理单个 cluster 的完整抽取流程。"""
    result = await ASYNC_CLUSTER_SUBGRAPH.ainvoke(state, config=config)

    return {
        "terms": result.get("terms", []),
        "med_terms": result.get("med_terms", []),
        "predicates": result.get("predicates", []),
        "rules": result.get("rules", []),
    }

def process_cluster_terms(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """处理单个 cluster 的术语抽取，智能复用缓存"""
    cluster_id = state.get("cluster_id")

    # 检查缓存中是否已有数据
    terms = state.get("terms", [])
    med_terms = state.get("med_terms", [])

    # 只有在没有缓存数据时才重新抽取
    if not terms or not med_terms:
        result = subgraph_extract_all_terms(state, config=config)
        if not terms:
            terms = result.get("terms", [])
        if not med_terms:
            med_terms = result.get("med_terms", [])

    print(f"[process_cluster_terms][cluster {cluster_id}] terms={len(terms)} med_terms={len(med_terms)}")

    # 返回全局聚合结果 + cluster-specific 缓存更新
    return {
        # 全局聚合字段（供 reducer 合并）
        "terms": terms,
        "med_terms": med_terms,
        # cluster-specific 缓存更新
        "cluster_cache_updates": {
            cluster_id: {
                "terms": terms,
                "med_terms": med_terms,
            }
        }
    }

def process_cluster_predicates(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """处理单个 cluster 的谓词抽取，复用已有的术语"""
    cluster_id = state.get("cluster_id")

    # 复用已有的 terms/med_terms
    terms = state.get("terms", [])
    med_terms = state.get("med_terms", [])

    # 如果没有缓存，才重新抽取
    if not terms or not med_terms:
        result = subgraph_extract_all_terms(state, config=config)
        if not terms:
            terms = result.get("terms", [])
        if not med_terms:
            med_terms = result.get("med_terms", [])

    # 抽取谓词
    predicates_result = extract_predicates_subgraph(
        {**state, "terms": terms, "med_terms": med_terms},
        config=config,
    )
    predicates = predicates_result.get("predicates", [])

    print(f"[process_cluster_predicates][cluster {cluster_id}] preds={len(predicates)} (复用缓存: terms={bool(state.get('terms'))}, meds={bool(state.get('med_terms'))})")

    # 返回全局聚合结果 + cluster-specific 缓存更新
    return {
        # 全局聚合字段（供 reducer 合并）
        "terms": terms,
        "med_terms": med_terms,
        "predicates": predicates,
        # cluster-specific 缓存更新
        "cluster_cache_updates": {
            cluster_id: {
                "terms": terms,
                "med_terms": med_terms,
                "predicates": predicates,
            }
        }
    }

def process_cluster_rules(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """处理单个 cluster 的规则抽取，复用所有已有数据"""
    # 复用已有的所有数据
    terms = state.get("terms", [])
    med_terms = state.get("med_terms", [])
    predicates = state.get("predicates", [])
    cluster_id = state.get("cluster_id")

    # 如果没有缓存，才重新抽取
    if not terms or not med_terms:
        result = subgraph_extract_all_terms(state, config=config)
        if not terms:
            terms = result.get("terms", [])
        if not med_terms:
            med_terms = result.get("med_terms", [])

    if not predicates:
        predicates_result = extract_predicates_subgraph(
            {**state, "terms": terms, "med_terms": med_terms},
            config=config,
        )
        predicates = predicates_result.get("predicates", [])

    # 抽取规则
    rules_result = extract_rules_subgraph(
        {**state, "med_terms": med_terms, "predicates": predicates},
        config=config,
    )
    rules = rules_result.get("rules", [])

    print(f"[process_cluster_rules][cluster {state.get('cluster_id')}] rules={len(rules)} (复用缓存: terms={bool(state.get('terms'))}, meds={bool(state.get('med_terms'))}, preds={bool(state.get('predicates'))})")

    # 返回全局聚合结果 + cluster-specific 缓存更新
    return {
        # 全局聚合字段（供 reducer 合并）
        "terms": terms,
        "med_terms": med_terms,
        "predicates": predicates,
        "rules": rules,
        # cluster-specific 缓存更新
        "cluster_cache_updates": {
            cluster_id: {
                "terms": terms,
                "med_terms": med_terms,
                "predicates": predicates,
                "rules": rules,
            }
        }
    }

def route_to_clusters(state: AgentState) -> List[Send]:
    sends = []
    for cluster in state.get("clusters", []):
        sends.append(
            Send(
                "process_cluster",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,  # 修复：使用 provenances 而不是 texts
                    "texts_formatted": cluster.texts_formatted,
                    "terms": [],
                    "med_terms": [],
                    "predicates": [],
                    "rules": [],
                },
            )
        )
    return sends

def distribute_clusters_for_terms(state: dict) -> List[Send]:
    """分发 clusters 到术语抽取节点"""
    sends = []
    clusters = state.get("clusters", [])
    cluster_cache = state.get("cluster_cache", {})

    for cluster in clusters:
        # 从 cluster-specific 缓存准备状态
        cache_entry = cluster_cache.get(cluster.cluster_id, {})
        sends.append(
            Send(
                "process_cluster_terms",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                    "terms": _to_models(cache_entry.get("terms", []), Term),
                    "med_terms": _to_models(cache_entry.get("med_terms", []), MedicationTerm),
                },
            )
        )
    return sends

def distribute_clusters_for_predicates(state: dict) -> List[Send]:
    """分发 clusters 到谓词抽取节点"""
    sends = []
    clusters = state.get("clusters", [])
    cluster_cache = state.get("cluster_cache", {})

    for cluster in clusters:
        # 从 cluster-specific 缓存准备状态
        cache_entry = cluster_cache.get(cluster.cluster_id, {})
        sends.append(
            Send(
                "process_cluster_predicates",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                    "terms": _to_models(cache_entry.get("terms", []), Term),
                    "med_terms": _to_models(cache_entry.get("med_terms", []), MedicationTerm),

                },
            )
        )
    return sends

def distribute_clusters_for_rules(state: dict) -> List[Send]:
    """分发 clusters 到规则抽取节点"""
    sends = []
    clusters = state.get("clusters", [])
    cluster_cache = state.get("cluster_cache", {})

    for cluster in clusters:
        # 从 cluster-specific 缓存准备状态
        cache_entry = cluster_cache.get(cluster.cluster_id, {})
        sends.append(
            Send(
                "process_cluster_rules",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                    "terms": _to_models(cache_entry.get("terms", []), Term),
                    "med_terms": _to_models(cache_entry.get("med_terms", []), MedicationTerm),
                    "predicates": _to_models(cache_entry.get("predicates", []), Predicates)

                },
            )
        )
    return sends

# ============ 独立阶段图 ============

def build_terms_extraction_graph():
    """术语抽取专用图"""
    class TermsState(TypedDict, total=False):
        clusters: List[ProvenanceCluster]
        cluster_cache: Dict[int, Dict[str, Any]]
        terms: Annotated[List[Term], merge_by_id]
        med_terms: Annotated[List[MedicationTerm], merge_by_id]

        cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]

    builder = StateGraph(TermsState)
    builder.add_node("process_cluster_terms", process_cluster_terms)

    builder.add_conditional_edges(START, distribute_clusters_for_terms, ["process_cluster_terms"])
    builder.add_edge("process_cluster_terms", END)

    return builder.compile()

def build_predicates_extraction_graph():
    """谓词抽取专用图"""
    class PredicatesState(TypedDict, total=False):
        clusters: List[ProvenanceCluster]
        cluster_cache: Dict[int, Dict[str, Any]]
        terms: Annotated[List[Term], merge_by_id]
        med_terms: Annotated[List[MedicationTerm], merge_by_id]
        predicates: Annotated[List[Predicates], merge_by_id]

        cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]

    builder = StateGraph(PredicatesState)
    builder.add_node("process_cluster_predicates", process_cluster_predicates)

    builder.add_conditional_edges(START, distribute_clusters_for_predicates, ["process_cluster_predicates"])
    builder.add_edge("process_cluster_predicates", END)

    return builder.compile()

def build_rules_extraction_graph():
    """规则抽取专用图"""
    class RulesState(TypedDict, total=False):
        clusters: List[ProvenanceCluster]
        cluster_cache: Dict[int, Dict[str, Any]]
        terms: Annotated[List[Term], merge_by_id]
        med_terms: Annotated[List[MedicationTerm], merge_by_id]
        predicates: Annotated[List[Predicates], merge_by_id]
        rules: Annotated[List[ClinicalRule], merge_by_id]

        cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]

    builder = StateGraph(RulesState)
    builder.add_node("process_cluster_rules", process_cluster_rules)

    builder.add_conditional_edges(START, distribute_clusters_for_rules, ["process_cluster_rules"])
    builder.add_edge("process_cluster_rules", END)

    return builder.compile()

def build_pipeline_graph():
    """
    完整流水线图：两阶段 Map-Reduce 架构

    阶段1 (Map-Reduce): 并行抽取推荐意见
    START -> distribute_texts (Send) -> extract_recommendation (并行 N 个)
                                              ↓
                                     [reducer: operator.add 自动聚合到 provenance_buffer]

    阶段2 (Map-Reduce): 聚类后并行抽取知识
                                              ↓
                                       lsh_clustering
                                              ↓
                               route_to_clusters (Send) -> process_cluster (并行 M 个)
                                                                  ↓
                                                    [reducer: merge_by_id 自动聚合去重]
                                                                  ↓
    阶段3 (Sequential): Post-cluster extraction and metadata
                                                                  ↓
                                             extract_contraindications
                                                                  ↓
                                            extract_safety_constraints
                                                                  ↓
                                             compute_missing_variables
                                                                  ↓
                                                 compute_metadata
                                                                  ↓
                                             serialize_graph_output
                                                                  ↓
                                                                 END
    """
    builder = StateGraph(AgentState)
    # Stage 1-2 nodes
    builder.add_node("extract_recommendation", extract_recommendation)
    builder.add_node("lsh_clustering", do_lsh_clustering)
    builder.add_node("process_cluster", process_cluster)
    # Stage 3 nodes (post-cluster)
    builder.add_node("extract_contraindications", extract_contraindications)
    builder.add_node("extract_safety_constraints", extract_safety_constraints)
    builder.add_node("compute_missing_variables", compute_missing_variables)
    builder.add_node("compute_metadata", compute_metadata)
    builder.add_node("serialize_graph_output", serialize_graph_output)

    # Stage 1 edges
    builder.add_conditional_edges(START, distribute_texts, ["extract_recommendation"])
    builder.add_edge("extract_recommendation", "lsh_clustering")
    # Stage 2 edges
    builder.add_conditional_edges("lsh_clustering", route_to_clusters, ["process_cluster"])
    # Stage 3 edges (linear post-cluster pipeline)
    builder.add_edge("process_cluster", "extract_contraindications")
    builder.add_edge("extract_contraindications", "extract_safety_constraints")
    builder.add_edge("extract_safety_constraints", "compute_missing_variables")
    builder.add_edge("compute_missing_variables", "compute_metadata")
    builder.add_edge("compute_metadata", "serialize_graph_output")
    builder.add_edge("serialize_graph_output", END)
    return builder.compile()


# ============ Contraindication Distribution Helpers ============

def distribute_clusters_for_contraindications(state: dict) -> List[Send]:
    """Distribute clusters for per-cluster contraindication extraction (reserved for future use).

    Currently the pipeline uses a post-cluster single-pass contraindication extraction.
    This function is provided for potential future per-cluster extraction patterns.
    """
    sends = []
    clusters = state.get("clusters", [])
    for cluster in clusters:
        sends.append(
            Send(
                "extract_contraindications",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                },
            )
        )
    return sends


def process_cluster_contraindications(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Process a single cluster for contraindication extraction.

    Delegates to the shared extract_contraindications node function.
    """
    return extract_contraindications(state, config=config)

