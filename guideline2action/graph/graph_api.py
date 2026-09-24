# -*- coding: utf-8 -*-
"""High-level clinical guideline pipeline API: LangGraph wrappers and stage runners."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langgraph.graph import END, START, StateGraph

from .config import LLMConfig, PathConfig, PipelineConfig, get_config, set_config
from .graph_builders import (
    aprocess_cluster,
    build_pipeline_graph,
    build_predicates_extraction_graph,
    build_rules_extraction_graph,
    build_terms_extraction_graph,
)
from .graph_nodes import distribute_texts, extract_recommendation
from .io_utils import (
    _save_json_file,
    _to_dict_list,
    load_cluster_cache,
    load_clusters,
    load_provenances,
    save_cluster_cache,
    save_clusters,
    save_provenances,
    save_stage_results,
)
from .graph_quality import ensure_graph_references
from .langfuse_tracing import build_langgraph_config, flush_langfuse
from .lsh_cluster import lsh_cluster
from .models import (
    AgentState,
    ClinicalRule,
    MedicationTerm,
    Predicates,
    Provenance,
    ProvenanceCluster,
    Term,
    merge_by_id,
    _to_models,
)

logger = logging.getLogger(__name__)

_provenance_graph = None


def _repair_extraction_lists(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure predicate entities and rule subjects exist in term/med_term lists."""
    keys = ("terms", "med_terms", "predicates", "rules")
    if not any(payload.get(k) for k in keys):
        return payload

    repaired, report = ensure_graph_references(
        {k: list(payload.get(k) or []) for k in keys}
    )
    created = (
        report.get("auto_created_term_count", 0)
        + report.get("auto_created_med_term_count", 0)
    )
    if created:
        logger.info(
            "[graph_quality] auto-created %d term(s) and %d medication term(s): %s",
            report.get("auto_created_term_count", 0),
            report.get("auto_created_med_term_count", 0),
            {
                "terms": report.get("auto_created_terms", []),
                "med_terms": report.get("auto_created_med_terms", []),
            },
        )

    out = dict(payload)
    out.update(repaired)
    return out


def _get_provenance_graph():
    global _provenance_graph
    if _provenance_graph is None:
        builder = StateGraph(AgentState)
        builder.add_node("extract_recommendation", extract_recommendation)
        builder.add_conditional_edges(START, distribute_texts, ["extract_recommendation"])
        builder.add_edge("extract_recommendation", END)
        _provenance_graph = builder.compile()
    return _provenance_graph


def _normalize_texts(texts: Union[str, List[str]]) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _initial_agent_state(input_texts: List[str]) -> Dict[str, Any]:
    return {
        "input_texts": input_texts,
        "messages": [],
        "provenance_buffer": [],
        "clusters": [],
        "terms": [],
        "med_terms": [],
        "predicates": [],
        "rules": [],
        "cluster_cache_updates": {},
        "cluster_cache": {},
    }


def _apply_config_gen_dir(gen_dir: Optional[str]) -> None:
    if not gen_dir:
        return
    cfg = get_config()
    paths = PathConfig(
        base_dir=cfg.paths.base_dir,
        standard_dir=cfg.paths.standard_dir,
        gen_dir=gen_dir,
    )
    set_config(PipelineConfig(llm=cfg.llm, paths=paths, match=cfg.match))


def _load_clusters_from_gen(gen_dir: str) -> Tuple[List[ProvenanceCluster], Dict[int, List[int]]]:
    for name in ("clusters.json", "cluster.json"):
        path = os.path.join(gen_dir, name)
        if os.path.isfile(path):
            return load_clusters(path)
    return [], {}


def _merge_cache_updates(
    cluster_cache: Dict[int, Dict[str, Any]],
    updates: Dict[int, Dict[str, Any]],
) -> None:
    for cid, upd in (updates or {}).items():
        if cid not in cluster_cache:
            cluster_cache[cid] = {}
        cluster_cache[cid].update(upd)


def save_results_from_cache(
    cluster_cache_path: str,
    gen_dir: Optional[str] = None,
    save_provenances: bool = False,
) -> Dict[str, Any]:
    """
    Merge ``cluster_cache`` with on-disk clusters; write ``cluster_final.json`` and
    aggregated stage JSON files (append + id-merge via ``save_stage_results``).
    """
    gen_dir = gen_dir or get_config().paths.gen_dir
    Path(gen_dir).mkdir(parents=True, exist_ok=True)

    cache = load_cluster_cache(cluster_cache_path)
    clusters, bucket_index = _load_clusters_from_gen(gen_dir)

    merged_terms: List[Term] = []
    merged_med_terms: List[MedicationTerm] = []
    merged_predicates: List[Predicates] = []
    merged_rules: List[ClinicalRule] = []

    final_cluster_dicts: List[Dict[str, Any]] = []
    for cl in clusters:
        cid = cl.cluster_id
        entry = cache.get(cid, {})
        t = _to_models(entry.get("terms", []), Term)
        m = _to_models(entry.get("med_terms", []), MedicationTerm)
        p = _to_models(entry.get("predicates", []), Predicates)
        r = _to_models(entry.get("rules", []), ClinicalRule)

        merged_terms = merge_by_id(merged_terms, t)
        merged_med_terms = merge_by_id(merged_med_terms, m)
        merged_predicates = merge_by_id(merged_predicates, p)
        merged_rules = merge_by_id(merged_rules, r)

        row = cl.model_dump(mode="json", by_alias=True)
        row["terms"] = _to_dict_list(t)
        row["med_terms"] = _to_dict_list(m)
        row["predicates"] = _to_dict_list(p)
        row["rules"] = _to_dict_list(r)
        final_cluster_dicts.append(row)

    payload = {
        "clusters": final_cluster_dicts,
        "bucket_index": bucket_index or {},
    }
    _save_json_file(os.path.join(gen_dir, "cluster_final.json"), payload)
    logger.info("[save_results_from_cache] wrote cluster_final.json (%d clusters)", len(final_cluster_dicts))

    aggregated = _repair_extraction_lists(
        {
            "terms": merged_terms,
            "med_terms": merged_med_terms,
            "predicates": merged_predicates,
            "rules": merged_rules,
        }
    )
    merged_terms = aggregated["terms"]
    merged_med_terms = aggregated["med_terms"]
    merged_predicates = aggregated["predicates"]
    merged_rules = aggregated["rules"]

    save_stage_results(
        {
            "terms": merged_terms,
            "med_terms": merged_med_terms,
            "predicates": merged_predicates,
            "rules": merged_rules,
        },
        gen_dir=gen_dir,
        stage_name="aggregate_cache",
    )

    if save_provenances:
        all_prov: List[Provenance] = []
        seen = set()
        for cl in clusters:
            for pr in cl.provenances:
                q = pr.quote or ""
                if q in seen:
                    continue
                seen.add(q)
                all_prov.append(pr)
        save_provenances(all_prov, filepath=os.path.join(gen_dir, "provenances.json"))

    flush_langfuse()
    return {
        "terms": merged_terms,
        "med_terms": merged_med_terms,
        "predicates": merged_predicates,
        "rules": merged_rules,
        "clusters_final": len(final_cluster_dicts),
    }


def extract_provenances_stage(
    texts: Union[str, List[str]],
    *,
    save_to_file: bool = True,
    filepath: Optional[str] = None,
    max_concurrency: int = 5,
) -> List[Provenance]:
    input_texts = _normalize_texts(texts)
    graph = _get_provenance_graph()
    config = build_langgraph_config(
        max_concurrency=max_concurrency,
        run_name="extract_provenances",
    )
    result = graph.invoke(_initial_agent_state(input_texts), config)
    provenances: List[Provenance] = list(result.get("provenance_buffer", []))

    if save_to_file:
        out = filepath or os.path.join(get_config().paths.gen_dir, "provenances.json")
        save_provenances(provenances, filepath=out)

    flush_langfuse()
    return provenances


def cluster_provenances_stage(
    provenances: Optional[List[Provenance]] = None,
    *,
    load_from_file: bool = False,
    load_filepath: Optional[str] = None,
    save_to_file: bool = True,
    filepath: Optional[str] = None,
    **lsh_kwargs: Any,
) -> Tuple[List[ProvenanceCluster], Dict[int, List[int]]]:
    gen_dir = get_config().paths.gen_dir
    if load_from_file:
        provenances = load_provenances(load_filepath)

    if provenances is None:
        provenances = []

    lsh_result = lsh_cluster(provenances, **lsh_kwargs)
    clusters = lsh_result.clusters
    bucket_index = lsh_result.bucket_index

    if save_to_file:
        out = filepath or os.path.join(gen_dir, "clusters.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        save_clusters(clusters, bucket_index, filepath=out)
        if os.path.basename(out) != "cluster.json":
            legacy = os.path.join(os.path.dirname(out), "cluster.json")
            save_clusters(clusters, bucket_index, filepath=legacy)

    flush_langfuse()
    return clusters, bucket_index


def extract_terms_stage(
    clusters: Optional[List[ProvenanceCluster]] = None,
    *,
    max_concurrency: int = 5,
    persist_cluster_cache: bool = True,
    cluster_cache_path: Optional[str] = None,
    load_from_file: bool = False,
    save_results: bool = False,
    gen_dir: Optional[str] = None,
) -> Dict[str, Any]:
    gdir = gen_dir or get_config().paths.gen_dir
    if load_from_file:
        clusters, _ = _load_clusters_from_gen(gdir)
    if not clusters:
        return {"terms": [], "med_terms": [], "cluster_cache_updates": {}}

    cpath = cluster_cache_path or os.path.join(gdir, "cluster_cache.json")
    cluster_cache = load_cluster_cache(cpath) if os.path.isfile(cpath) else {}

    graph = build_terms_extraction_graph()
    config = build_langgraph_config(max_concurrency=max_concurrency, run_name="extract_terms")
    result = graph.invoke(
        {
            "clusters": clusters,
            "cluster_cache": cluster_cache,
            "terms": [],
            "med_terms": [],
            "cluster_cache_updates": {},
        },
        config,
    )
    _merge_cache_updates(cluster_cache, result.get("cluster_cache_updates", {}))
    if persist_cluster_cache:
        save_cluster_cache(cluster_cache, cpath)

    if save_results:
        save_stage_results(
            {"terms": result.get("terms", []), "med_terms": result.get("med_terms", [])},
            gen_dir=gdir,
            stage_name="terms",
        )

    flush_langfuse()
    return dict(result)


def extract_predicates_stage(
    clusters: Optional[List[ProvenanceCluster]] = None,
    *,
    max_concurrency: int = 5,
    persist_cluster_cache: bool = True,
    cluster_cache_path: Optional[str] = None,
    load_from_file: bool = False,
    save_results: bool = False,
    gen_dir: Optional[str] = None,
) -> Dict[str, Any]:
    gdir = gen_dir or get_config().paths.gen_dir
    if load_from_file:
        clusters, _ = _load_clusters_from_gen(gdir)
    if not clusters:
        return {"terms": [], "med_terms": [], "predicates": [], "cluster_cache_updates": {}}

    cpath = cluster_cache_path or os.path.join(gdir, "cluster_cache.json")
    cluster_cache = load_cluster_cache(cpath) if os.path.isfile(cpath) else {}

    graph = build_predicates_extraction_graph()
    config = build_langgraph_config(max_concurrency=max_concurrency, run_name="extract_predicates")
    result = graph.invoke(
        {
            "clusters": clusters,
            "cluster_cache": cluster_cache,
            "terms": [],
            "med_terms": [],
            "predicates": [],
            "cluster_cache_updates": {},
        },
        config,
    )
    _merge_cache_updates(cluster_cache, result.get("cluster_cache_updates", {}))
    if persist_cluster_cache:
        save_cluster_cache(cluster_cache, cpath)

    if save_results:
        save_stage_results(
            {
                "terms": result.get("terms", []),
                "med_terms": result.get("med_terms", []),
                "predicates": result.get("predicates", []),
            },
            gen_dir=gdir,
            stage_name="predicates",
        )

    flush_langfuse()
    return dict(result)


def extract_rules_stage(
    clusters: Optional[List[ProvenanceCluster]] = None,
    *,
    max_concurrency: int = 5,
    persist_cluster_cache: bool = True,
    cluster_cache_path: Optional[str] = None,
    load_from_file: bool = False,
    save_results: bool = False,
    gen_dir: Optional[str] = None,
) -> Dict[str, Any]:
    gdir = gen_dir or get_config().paths.gen_dir
    if load_from_file:
        clusters, _ = _load_clusters_from_gen(gdir)
    if not clusters:
        return {
            "terms": [],
            "med_terms": [],
            "predicates": [],
            "rules": [],
            "cluster_cache_updates": {},
        }

    cpath = cluster_cache_path or os.path.join(gdir, "cluster_cache.json")
    cluster_cache = load_cluster_cache(cpath) if os.path.isfile(cpath) else {}

    graph = build_rules_extraction_graph()
    config = build_langgraph_config(max_concurrency=max_concurrency, run_name="extract_rules")
    result = graph.invoke(
        {
            "clusters": clusters,
            "cluster_cache": cluster_cache,
            "terms": [],
            "med_terms": [],
            "predicates": [],
            "rules": [],
            "cluster_cache_updates": {},
        },
        config,
    )
    _merge_cache_updates(cluster_cache, result.get("cluster_cache_updates", {}))
    if persist_cluster_cache:
        save_cluster_cache(cluster_cache, cpath)

    if save_results:
        save_stage_results(
            {
                "terms": result.get("terms", []),
                "med_terms": result.get("med_terms", []),
                "predicates": result.get("predicates", []),
                "rules": result.get("rules", []),
            },
            gen_dir=gdir,
            stage_name="rules",
        )

    flush_langfuse()
    return dict(result)


def run_recommendation_clustering_pipeline(
    texts: Union[str, List[str]],
    gen_dir: str = "gen",
    max_concurrency: int = 5,
) -> Dict[str, Any]:
    _apply_config_gen_dir(gen_dir)
    Path(gen_dir).mkdir(parents=True, exist_ok=True)

    provenances = extract_provenances_stage(
        texts,
        save_to_file=True,
        max_concurrency=max_concurrency,
    )
    clusters, bucket_index = cluster_provenances_stage(
        provenances,
        save_to_file=True,
    )
    cache_path = os.path.join(gen_dir, "cluster_cache.json")
    save_cluster_cache({}, cache_path)

    flush_langfuse()
    return {
        "provenances": provenances,
        "clusters": clusters,
        "bucket_index": bucket_index,
        "cluster_cache_path": cache_path,
    }


def run_graph_extraction_pipeline(
    gen_dir: str = "gen",
    max_concurrency: int = 5,
) -> Dict[str, Any]:
    _apply_config_gen_dir(gen_dir)
    cpath = os.path.join(gen_dir, "cluster_cache.json")

    extract_terms_stage(
        load_from_file=True,
        max_concurrency=max_concurrency,
        persist_cluster_cache=True,
        cluster_cache_path=cpath,
        save_results=True,
        gen_dir=gen_dir,
    )
    extract_predicates_stage(
        load_from_file=True,
        max_concurrency=max_concurrency,
        persist_cluster_cache=True,
        cluster_cache_path=cpath,
        save_results=True,
        gen_dir=gen_dir,
    )
    extract_rules_stage(
        load_from_file=True,
        max_concurrency=max_concurrency,
        persist_cluster_cache=True,
        cluster_cache_path=cpath,
        save_results=True,
        gen_dir=gen_dir,
    )
    save_results_from_cache(cluster_cache_path=cpath, gen_dir=gen_dir, save_provenances=False)
    flush_langfuse()
    return {"gen_dir": gen_dir, "cluster_cache_path": cpath}


async def process_clusters_async(
    *,
    clusters: List[ProvenanceCluster],
    cluster_cache_path: str,
    persist_cluster_cache: bool = True,
    max_concurrency: int = 8,
    verbose: bool = False,
    save_results: bool = False,
    gen_dir: Optional[str] = None,
) -> Dict[str, Any]:
    gdir = gen_dir or get_config().paths.gen_dir
    cache = load_cluster_cache(cluster_cache_path) if os.path.isfile(cluster_cache_path) else {}
    sem = asyncio.Semaphore(max(1, max_concurrency))
    cfg = build_langgraph_config(
        max_concurrency=1,
        run_name="process_clusters_async",
    )

    async def _one(cl: ProvenanceCluster) -> Dict[str, Any]:
        async with sem:
            cid = cl.cluster_id
            prev = cache.get(cid, {})
            state: Dict[str, Any] = {
                "cluster_id": cid,
                "provenances": cl.provenances,
                "texts_formatted": cl.texts_formatted,
                "terms": _to_models(prev.get("terms", []), Term),
                "med_terms": _to_models(prev.get("med_terms", []), MedicationTerm),
                "predicates": _to_models(prev.get("predicates", []), Predicates),
                "rules": _to_models(prev.get("rules", []), ClinicalRule),
            }
            if verbose:
                logger.info("[process_clusters_async] cluster_id=%s", cid)
            out = await aprocess_cluster(state, config=cfg)
            cache[cid] = {
                "terms": out.get("terms", []),
                "med_terms": out.get("med_terms", []),
                "predicates": out.get("predicates", []),
                "rules": out.get("rules", []),
            }
            return out

    parts = await asyncio.gather(*[_one(c) for c in clusters])

    merged_terms: List[Term] = []
    merged_med_terms: List[MedicationTerm] = []
    merged_predicates: List[Predicates] = []
    merged_rules: List[ClinicalRule] = []
    for out in parts:
        merged_terms = merge_by_id(merged_terms, out.get("terms", []))
        merged_med_terms = merge_by_id(merged_med_terms, out.get("med_terms", []))
        merged_predicates = merge_by_id(merged_predicates, out.get("predicates", []))
        merged_rules = merge_by_id(merged_rules, out.get("rules", []))

    if persist_cluster_cache:
        Path(cluster_cache_path).parent.mkdir(parents=True, exist_ok=True)
        save_cluster_cache(cache, cluster_cache_path)

    repaired = _repair_extraction_lists(
        {
            "terms": merged_terms,
            "med_terms": merged_med_terms,
            "predicates": merged_predicates,
            "rules": merged_rules,
        }
    )
    merged_terms = repaired["terms"]
    merged_med_terms = repaired["med_terms"]
    merged_predicates = repaired["predicates"]
    merged_rules = repaired["rules"]

    if save_results:
        save_results_from_cache(cluster_cache_path=cluster_cache_path, gen_dir=gdir, save_provenances=False)

    flush_langfuse()
    return {
        "terms": merged_terms,
        "med_terms": merged_med_terms,
        "predicates": merged_predicates,
        "rules": merged_rules,
        "cluster_cache": cache,
    }


class ClinicalGuidelinePipeline:
    """End-to-end pipeline: provenance → LSH → per-cluster sync extraction (``CLUSTER_SUBGRAPH``)."""

    def run(
        self,
        texts: Union[str, List[str]],
        max_concurrency: int = 5,
    ) -> Dict[str, Any]:
        input_texts = _normalize_texts(texts)
        graph = build_pipeline_graph()
        config = build_langgraph_config(
            max_concurrency=max_concurrency,
            run_name="clinical_guideline_pipeline",
        )
        result = graph.invoke(_initial_agent_state(input_texts), config)
        flush_langfuse()
        return _repair_extraction_lists(dict(result))

    async def process_clusters_async(self, **kwargs: Any) -> Dict[str, Any]:
        return await process_clusters_async(**kwargs)


def create_pipeline(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    *,
    gen_dir: Optional[str] = None,
    **kwargs: Any,
) -> ClinicalGuidelinePipeline:
    llm = LLMConfig.from_env(temperature=temperature)
    if api_key is not None:
        llm.api_key = api_key
    if base_url is not None:
        llm.base_url = base_url
    if model is not None:
        llm.model = model

    cfg_existing = get_config()
    paths = PathConfig(
        base_dir=cfg_existing.paths.base_dir,
        standard_dir=cfg_existing.paths.standard_dir,
        gen_dir=gen_dir if gen_dir is not None else cfg_existing.paths.gen_dir,
    )
    set_config(
        PipelineConfig(
            llm=llm,
            paths=paths,
            match=cfg_existing.match,
        )
    )

    from .io_utils import get_failed_task_logger

    get_failed_task_logger().configure(log_dir=paths.gen_dir)

    return ClinicalGuidelinePipeline()
