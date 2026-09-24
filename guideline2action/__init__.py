"""Guideline2Action — compile guideline text into an executable decision graph.

Two calls make up the functional interface::

    from guideline2action import compile_guideline, run_graph

    graph  = compile_guideline("对于射血分数降低的心衰患者，建议使用 SGLT2i 治疗……")
    result = run_graph(graph, observations=[...])          # deterministic, no LLM

``compile_guideline`` is an LLM call (it reads the guideline text).  ``run_graph``
is a pure deterministic computation: the same graph plus the same patient facts
always yields the same assessments, actions and trace.  Findings that cannot be
settled by computation are not executed — they become ``ClinicalAssessmentSpec``
questions the caller answers with ``true``/``false``/``unknown``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .projection import (
    JUDGMENT_TOKENS,
    NON_COMPUTABLE_ASPECTS,
    RUNTIME_CONTRACT_VERSION,
    SCHEMA_VERSION,
    project_graph,
    runtime_program,
)

__version__ = "0.1.0"

TextInput = Union[str, Sequence[str]]


class Guideline2ActionError(RuntimeError):
    """Raised when a guideline cannot be compiled or a graph cannot be executed."""


class ExecutableGraph(Mapping[str, Any]):
    """Mapping view over a projected Schema-v2 graph plus its compile provenance.

    Behave like the graph dict (so it can be handed straight to :func:`run_graph`
    or serialised with :func:`json.dump`), while also exposing the extraction
    payload and a flat list of human-review items under ``.review``.
    """

    def __init__(self, graph: Mapping[str, Any], extraction: Mapping[str, Any]) -> None:
        self._graph: Dict[str, Any] = dict(graph)
        self.extraction: Dict[str, Any] = dict(extraction)

    # -- Mapping protocol --------------------------------------------------- #
    def __getitem__(self, key: str) -> Any:
        return self._graph[key]

    def __iter__(self):
        return iter(self._graph)

    def __len__(self) -> int:
        return len(self._graph)

    def __repr__(self) -> str:
        counts = self._graph.get("projection", {}).get("counts", {})
        return (
            f"ExecutableGraph(graph_id={self._graph.get('graph_id')!r}, "
            f"predicates={counts.get('predicates', 0)}, rules={counts.get('rules', 0)}, "
            f"actions={counts.get('actions', 0)}, "
            f"assessments={counts.get('clinical_assessment_specs', 0)})"
        )

    # -- convenience -------------------------------------------------------- #
    @property
    def graph(self) -> Dict[str, Any]:
        return self._graph

    @property
    def review(self) -> List[Dict[str, Any]]:
        """Human-review items produced by the projection (may be empty)."""
        return list(self._graph.get("projection", {}).get("review") or [])

    @property
    def programs(self) -> Dict[str, Any]:
        """The interpreter's own program envelope for this graph."""
        return runtime_program(self._graph)

    # -- construction helpers ---------------------------------------------- #
    @classmethod
    def from_extraction(
        cls,
        extraction: Mapping[str, Any],
        *,
        graph_id: Optional[str] = None,
        qid: Optional[str] = None,
    ) -> "ExecutableGraph":
        """Build a graph from a stored extraction payload (offline, no LLM)."""
        return cls(project_graph(extraction, graph_id=graph_id, qid=qid), extraction)

    def to_json(self, path: Optional[Union[str, Path]] = None, *, indent: int = 2) -> str:
        """Serialise the graph; optionally also write it to ``path``."""
        import json

        payload = json.dumps(self._graph, ensure_ascii=False, indent=indent, sort_keys=True)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload + "\n", encoding="utf-8")
        return payload


def compile_guideline(
    text: TextInput,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    graph_id: Optional[str] = None,
    qid: Optional[str] = None,
    gen_dir: Optional[Union[str, Path]] = None,
    max_concurrency: int = 5,
) -> ExecutableGraph:
    """Compile guideline text into an executable decision graph.

    Parameters
    ----------
    text:
        One guideline passage, or a list of passages from which recommendations
        are extracted.  Each passage should carry its source, either as
        ``"<source>\\n<recommendation>"`` or as a mapping ``{"source": ...,
        "text": ...}`` — the extraction layer records provenance from it.
    api_key / base_url / model / temperature:
        LLM overrides.  When omitted the extraction layer falls back to
        ``LLMConfig.from_env()`` (``LLM_PROVIDER``/``LLM_API_KEY``/``...``).
    graph_id / qid:
        Stable identifiers.  ``qid`` defaults to a slug of ``graph_id``; the
        interpreter requires one but text compilation has no benchmark question.
    gen_dir:
        Scratch directory for stage caches.  Defaults to a temporary directory
        so a library install never writes inside the package.
    max_concurrency:
        Parallel cluster extraction.

    Returns
    -------
    ExecutableGraph
        Mapping-compatible graph with ``terms``, ``predicates``, ``rules``,
        ``actions``, ``clinical_assessment_specs``, ``nodes``, ``edges`` and a
        ``projection`` review report.

    Raises
    ------
    Guideline2ActionError
        If no text is supplied, or if extraction yields no rule at all.
    """
    from .graph.graph_api import create_pipeline, get_config, set_config
    from .graph.graph_nodes import serialize_extraction

    passages = _normalise_input(text)
    if not passages:
        raise Guideline2ActionError("compile_guideline requires at least one non-empty passage")

    import tempfile

    previous = get_config()
    scratch = Path(gen_dir) if gen_dir is not None else Path(tempfile.mkdtemp(prefix="g2a-gen-"))
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = create_pipeline(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            gen_dir=str(scratch),
        )
        state = pipeline.run(passages, max_concurrency=max_concurrency)
        # run() returns the raw agent state (Pydantic objects); the projection
        # layer works on the JSON-safe payload, which is the same thing the
        # serializer node persists.
        extraction = serialize_extraction(state) if isinstance(state, Mapping) else state
    finally:
        # The extraction layer keeps LLM/Path configuration in a module-level
        # singleton; restore the caller's environment so repeated compilations
        # never leak credentials or scratch paths into each other.
        try:
            set_config(previous)
        except Exception:  # pragma: no cover - best-effort restoration
            pass

    if not isinstance(extraction, Mapping):
        raise Guideline2ActionError(
            f"extraction returned {type(extraction).__name__}, expected a mapping"
        )

    graph = ExecutableGraph.from_extraction(extraction, graph_id=graph_id, qid=qid)
    if not graph.graph.get("rules"):
        raise Guideline2ActionError(
            "extraction produced no decision rule; the passage carries no executable "
            "recommendation (check that each passage states a condition or an action)"
        )
    return graph


def _normalise_input(text: TextInput) -> List[str]:
    """Coerce the accepted text forms into the extraction layer's passage list."""
    if isinstance(text, str):
        return [line.strip() for line in text.splitlines() if line.strip()] or (
            [text.strip()] if text.strip() else []
        )
    if isinstance(text, Mapping):
        source = str(text.get("source") or "").strip()
        body = str(text.get("text") or text.get("quote") or "").strip()
        return [f"{source}\n{body}".strip()] if body else []
    passages: List[str] = []
    if isinstance(text, Iterable):
        for item in text:
            if isinstance(item, Mapping):
                passages.extend(_normalise_input(item))  # type: ignore[arg-type]
            elif str(item).strip():
                passages.append(str(item).strip())
    return passages


def run_graph(
    graph: Union[ExecutableGraph, Mapping[str, Any]],
    observations: Optional[Sequence[Mapping[str, Any]]] = None,
    clinical_assessments: Optional[Sequence[Mapping[str, Any]]] = None,
    clinical_assertions: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    include_clinical_trace: bool = False,
) -> Dict[str, Any]:
    """Execute a compiled graph deterministically over patient facts.

    No LLM participates in this call.  Missing or insufficient information
    propagates as ``unknown`` (strong Kleene) rather than ``false``.

    Parameters
    ----------
    graph:
        A graph from :func:`compile_guideline` / :meth:`ExecutableGraph.from_extraction`.
    observations:
        Patient facts extracted from the case, each ``{"fact_id", "concept_id",
        "value", "unit"?, "polarity"?, "temporal"?, "source_quote"?}``.  A fact's
        ``concept_id`` (or its predicate's ``retrieve.concepts`` alias) is what
        binds it to a predicate.
    clinical_assessments:
        Answers to the graph's ``clinical_assessment_specs``, each
        ``{"assessment_id", "assessment_spec_id", "value": "true|false|unknown",
        "assessment_authority": "llm|clinical_expert", "supporting_observation_ids":
        [...], "source_spans": [...], "requires_human_review": bool}``.
    clinical_assertions:
        Findings the case states outright (diagnosis, stage, grade), read
        deterministically by ``evaluation_mode=assertion_read`` predicates.
    include_clinical_trace:
        Also attach the clinical handover rendering (what an LLM would be shown)
        under ``trace["clinical_trace_for_llm"]``.

    Returns
    -------
    dict
        Execution trace with ``predicate_results``, ``condition_node_results``,
        ``rule_results``, ``actions`` (status ``required`` /
        ``pending_human_review`` / ``not_activated`` / ``unresolved`` plus
        ``execution_authorized``) and, when requested, the clinical trace.
    """
    from .runtime.guideline2graph_runtime_v1 import Guideline2GraphProgramError
    from .runtime.unified_schema_runtime_v1 import (
        UnifiedSchemaV2Runtime,
        UnifiedSchemaV2RuntimeError,
    )

    payload = dict(graph.graph if isinstance(graph, ExecutableGraph) else graph)
    payload.setdefault("conversion_status", "converted")
    if str(payload.get("schema_version") or "") != SCHEMA_VERSION:
        raise Guideline2ActionError(
            f"graph schema_version must be {SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    try:
        runtime = UnifiedSchemaV2Runtime(payload)
        return runtime.execute(
            list(observations or []),
            list(clinical_assessments or []) or None,
            list(clinical_assertions or []) or None,
            include_clinical_trace=include_clinical_trace,
        )
    except (UnifiedSchemaV2RuntimeError, Guideline2GraphProgramError, ValueError, KeyError) as exc:
        # The interpreter reports contract violations as its own error types;
        # callers of this package see one exception type.
        raise Guideline2ActionError(str(exc)) from exc


__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "RUNTIME_CONTRACT_VERSION",
    "NON_COMPUTABLE_ASPECTS",
    "JUDGMENT_TOKENS",
    "ExecutableGraph",
    "Guideline2ActionError",
    "compile_guideline",
    "project_graph",
    "run_graph",
    "runtime_program",
]
