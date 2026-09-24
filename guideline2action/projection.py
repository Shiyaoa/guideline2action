"""Compile raw guideline-text extraction output into an executable Schema-v2 graph.

The extraction layer (:mod:`guideline2action.graph`) turns guideline text into
terms, computable predicates, decision rules and actions.  That payload is
close to — but not the same as — the graph the deterministic interpreter
(:mod:`guideline2action.runtime`) consumes.  This module closes the gap with
four deterministic, non-destructive transformations:

1. **Action materialisation** — the extraction nests one ``Action`` inside each
   ``ClinicalRule``; the interpreter contract needs a top-level ``actions[]``
   plus ``rules[].action_refs``, action identity, permission provenance and an
   execution-authorisation flag.
2. **DAG vocabulary normalisation** — the extraction DAG node vocabulary (16
   types, lowercase combiners, bare predicate-id operands) is mapped onto the
   interpreter vocabulary (6 types, ``AND``/``OR``/``NOT``/``AT_LEAST``
   combiners, explicit ``predicate_ref`` nodes).
3. **ClinicalAssessment routing** — findings that cannot be settled by
   deterministic computation are not predicates.  A finding whose ``aspect`` is
   ``treatment_eligibility`` (or whose identifier/label carries a judgment token
   such as ``fitness``/``tolerab*``/``suitab*``/``feasib*``) is moved out of the
   executable predicate set and becomes a ``ClinicalAssessmentSpec`` the LLM must
   answer with ``true``/``false``/``unknown``; the DAG references it through a
   ``clinical_assessment_ref`` node instead.
4. **``schema_v2`` semantic classification** — terms gain a
   ``schema_v2.semantic_domain`` block and predicates a
   ``schema_v2.compute_family``/``compute_facets`` block, derived mechanically
   from their ids and existing facets.

Nothing here reads patient data or evaluation labels: the projection is a pure
function of the extraction payload.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "decisionkg.schema_v2.unified.20260804"
RUNTIME_CONTRACT_VERSION = "decisionkg.runtime_contract.v1"

# --------------------------------------------------------------------------- #
# ClinicalAssessment boundary
# --------------------------------------------------------------------------- #

#: ``aspect`` values that never qualify as deterministically computable.
NON_COMPUTABLE_ASPECTS = frozenset({"treatment_eligibility"})

#: Identifier/label tokens that mark a composite clinical judgement.
JUDGMENT_TOKENS = (
    "fitness",
    "tolerab",
    "suitab",
    "feasib",
    "readiness",
    "patient_specific",
    "overall_severity",
    "severity_grade",
    "stable_control",
)

# --------------------------------------------------------------------------- #
# DAG vocabulary
# --------------------------------------------------------------------------- #

COMBINE_OPERATORS = frozenset({"AND", "OR", "NOT", "AT_LEAST"})
COMPARISON_OPERATORS = frozenset({"gt", "ge", "lt", "le", "eq", "ne"})

#: non-standard comparators seen in real extraction payloads and their nearest
#: executable equivalent.  ``eq`` here means "the fact's value equals this
#: enumerated band", which is how a normal-range flag is actually carried.
COMPARATOR_ALIASES = {
    "within_normal_range": "eq",
    "within_normal": "eq",
    "normal": "eq",
    "near_normal": "eq",
    "in_range": "eq",
    "between": "eq",
    "not_within_normal_range": "ne",
    "abnormal": "ne",
    "greater_or_equal": "ge",
    "less_or_equal": "le",
    "greater_than": "gt",
    "less_than": "lt",
    "equal": "eq",
    "not_equal": "ne",
}
_RUNTIME_COMBINERS = frozenset({"AND", "ALL", "OR", "ANY", "AT_LEAST", "NOT"})

_PREDICATE_LEAF_TYPES = frozenset({"predicate_ref"})
_ASSESSMENT_LEAF_TYPE = "clinical_assessment_ref"
_PASSTHROUGH_TYPES = frozenset({"combine", "compare", "literal", "exists"})

#: DAG node types the extraction vocabulary offers that the interpreter cannot
#: execute.  They are folded into their owning predicate (handled in
#: :func:`_normalise_predicate`) and reported as review items.
_NON_EXECUTABLE_DAG_TYPES = frozenset(
    {
        "aggregate",
        "coalesce",
        "temporal_relation",
        "range_membership",
        "library_function",
        "output_assembly",
        "filter",
        "sort",
        "interval",
        "unit_convert",
        "extract",
    }
)

_ASPECT_DOMAIN = {
    "existence": "Condition",
    "status": "Condition",
    "diagnostic_criterion": "Condition",
    "molecular_marker": "Condition",
    "complication_grade": "Condition",
    "risk_group": "Condition",
    "infection_marker": "Condition",
    "contraindication": "Condition",
    "drug_interaction": "Condition",
    "organ_function": "Measurement",
    "quantity": "Measurement",
    "quantity_range": "Measurement",
    "dose_constraint": "Drug",
    "treatment_eligibility": "Assessment",
}

_REDUCTION_ALIASES = {
    "none": "none",
    "exists": "exists",
    "most_recent": "most_recent",
    "latest": "most_recent",
    "max": "most_recent",
    "min": "most_recent",
    "extremum": "most_recent",
}
_RUNTIME_REDUCTIONS = frozenset({"none", "most_recent", "exists"})

_ID_NAMESPACE_DOMAIN = {
    "cond": "Condition",
    "meas": "Measurement",
    "obs": "Observation",
    "proc": "Procedure",
    "med": "Drug",
    "assessment": "Assessment",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _records(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _slug(value: Any) -> str:
    text = _text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unnamed"


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, source: str, *, used: Optional[MutableMapping[str, int]] = None) -> str:
    """Deterministic identifier; disambiguated only on genuine collision."""
    base = f"{prefix}.{_slug(source)}"
    if used is None:
        return base
    seen = used.get(base)
    if seen is None:
        used[base] = 1
        return base
    used[base] = seen + 1
    return f"{base}.{_sha256(source)[:8]}.{seen + 1}"


def _rule_namespace(rule_id: str) -> str:
    """Readable per-rule namespace for generated DAG node ids."""
    text = _text(rule_id)
    text = re.sub(r"^rule[._:-]*", "", text, flags=re.IGNORECASE)
    return _slug(text) or "rule"


def _is_judgment(predicate: Mapping[str, Any]) -> bool:
    """True when a finding cannot be settled by deterministic computation."""
    if _text(predicate.get("aspect")) in NON_COMPUTABLE_ASPECTS:
        return True
    haystack = " ".join(
        [
            _text(predicate.get("id")),
            _text(predicate.get("name")),
            _text(predicate.get("description")),
        ]
    ).lower()
    return any(token in haystack for token in JUDGMENT_TOKENS)


def _domain_for(identifier: str, aspect: str) -> str:
    if aspect in _ASPECT_DOMAIN:
        return _ASPECT_DOMAIN[aspect]
    namespace = _text(identifier).split(".")[0]
    return _ID_NAMESPACE_DOMAIN.get(namespace, "Other")


# --------------------------------------------------------------------------- #
# (1) predicates and terms — semantic classification + executability
# --------------------------------------------------------------------------- #


#: keys a real extraction payload may use to point at the fact concept(s).
_CONCEPT_KEYS = ("concepts", "code_binding", "code_bindings", "concept", "concept_id")


def _fact_concepts(predicate: Mapping[str, Any]) -> List[str]:
    """Collect every concept identifier a predicate can bind a fact through.

    Extraction payloads express this inconsistently: some carry
    ``retrieve.concepts``, some only a ``retrieve.code_binding``, some only
    ``dependencies``, and some only a bare ``entity``.  The interpreter rejects a
    predicate with no binding at all, so all of these are folded into one list.
    """
    concepts: List[str] = []
    retrieve = predicate.get("retrieve")
    if isinstance(retrieve, Mapping):
        for key in _CONCEPT_KEYS:
            value = retrieve.get(key)
            if isinstance(value, str) and value.strip():
                concepts.append(value.strip())
            elif isinstance(value, list):
                concepts.extend(str(item).strip() for item in value if str(item).strip())
    for key in _CONCEPT_KEYS:
        value = predicate.get(key)
        if isinstance(value, str) and value.strip():
            concepts.append(value.strip())
    for key in ("dependencies", "entity", "input_id", "left_input_id"):
        value = predicate.get(key)
        if isinstance(value, str) and value.strip():
            concepts.append(value.strip())
        elif isinstance(value, list):
            concepts.extend(str(item).strip() for item in value if str(item).strip())
    entity = _text(predicate.get("entity"))
    if not entity and not concepts:
        # last resort: the predicate namespace itself (``pred.cond.apl.exists``
        # -> ``cond.apl``), so the graph still declares where its facts come from
        parts = [part for part in _text(predicate.get("id")).split(".") if part]
        if len(parts) >= 3 and parts[0] == "pred":
            concepts.append(".".join(parts[1:3]))
    seen: List[str] = []
    for item in concepts:
        if item not in seen:
            seen.append(item)
    return seen


def _normalise_retrieve(predicate: Mapping[str, Any]) -> Dict[str, Any]:
    entity_type = _text(predicate.get("entity_type")).lower()
    resource = "QuestionObservation"
    if entity_type in {"medication", "drug", "medication_statement"}:
        resource = "MedicationStatement"
    elif entity_type in {"clinical_assertion", "assertion"}:
        resource = "ClinicalAssertion"
    elif entity_type == "clinical_assessment":
        resource = "ClinicalAssessment"

    retrieve = dict(predicate.get("retrieve")) if isinstance(predicate.get("retrieve"), Mapping) else {}
    retrieve.pop("code_binding", None)
    retrieve.pop("code_bindings", None)
    if not _text(retrieve.get("resource")):
        retrieve["resource"] = resource
    concepts = _fact_concepts(predicate)
    if concepts:
        retrieve["concepts"] = concepts
    return retrieve


def _normalise_reduction(predicate: Mapping[str, Any]) -> Dict[str, Any]:
    reduction = predicate.get("reduction")
    if isinstance(reduction, Mapping) and reduction.get("operator"):
        operator = _text(reduction.get("operator")).lower()
        output_type = _text(reduction.get("output_type")) or _text(
            predicate.get("final_output_type")
        ) or "TruthValue"
    else:
        operator, output_type = "most_recent", _text(
            predicate.get("final_output_type")
        ) or "TruthValue"
    operator = _REDUCTION_ALIASES.get(operator, "most_recent")
    if operator not in _RUNTIME_REDUCTIONS:
        operator = "most_recent"
    return {"operator": operator, "output_type": output_type}


def _normalise_extract(predicate: Mapping[str, Any]) -> Dict[str, Any]:
    extract = predicate.get("extract")
    if isinstance(extract, Mapping) and extract.get("path"):
        return dict(extract)
    return {
        "path": _text(predicate.get("extract_path")) or "value",
        "type": _text(predicate.get("final_output_type")) or "TruthValue",
    }


def _compute_family(predicate: Mapping[str, Any]) -> str:
    aspect = _text(predicate.get("aspect"))
    compare = predicate.get("compare")
    if aspect == "existence":
        return "existence"
    if aspect in {"quantity", "organ_function", "dose_constraint"}:
        return "threshold"
    if aspect == "quantity_range":
        return "range"
    if aspect in {"temporal", "time_since_event"}:
        return "temporal"
    if isinstance(compare, Mapping) and compare.get("operator") in {"in", "contains_any"}:
        return "category"
    return "state"


def _compute_facets(predicate: Mapping[str, Any], domain: str) -> Dict[str, Any]:
    compare = predicate.get("compare") if isinstance(predicate.get("compare"), Mapping) else {}
    return {
        "input_entity_type": _text(predicate.get("entity_type")) or "observation",
        "input_domain": domain,
        "reduction": _normalise_reduction(predicate).get("operator"),
        "comparison": _text(compare.get("operator")) or None,
        "temporal_scope": _text(predicate.get("temporal_scope")) or "all_time",
        "output_type": _text(predicate.get("final_output_type")) or "TruthValue",
    }


def _normalise_predicate(
    source: Mapping[str, Any],
    *,
    review: List[Dict[str, Any]],
) -> Dict[str, Any]:
    predicate = copy.deepcopy(dict(source))
    predicate_id = _text(predicate.get("id"))
    if not predicate_id:
        predicate_id = _stable_id("pred", _text(predicate.get("name")) or _sha256(predicate)[:8])
        predicate["id"] = predicate_id

    predicate["entity_type"] = _text(predicate.get("entity_type")) or "observation"
    predicate["aspect"] = _text(predicate.get("aspect")) or "existence"
    predicate["input_shape"] = _text(predicate.get("input_shape")) or "single"
    predicate["return_type"] = _text(predicate.get("return_type")) or "TruthValue"
    predicate["final_output_type"] = (
        _text(predicate.get("final_output_type")) or predicate["return_type"]
    )
    predicate["null_policy"] = "unknown"
    predicate["temporal_scope"] = (
        predicate["temporal_scope"]
        if isinstance(predicate.get("temporal_scope"), Mapping)
        else {"mode": _text(predicate.get("temporal_scope")) or "all_time"}
    )
    if not _text(predicate.get("entity")):
        concepts = _fact_concepts(predicate)
        if concepts:
            predicate["entity"] = concepts[0]
    predicate["retrieve"] = _normalise_retrieve(predicate)
    predicate["extract"] = _normalise_extract(predicate)
    predicate["reduction"] = _normalise_reduction(predicate)

    compare = predicate.get("compare")
    if isinstance(compare, Mapping) and compare:
        operator = _text(compare.get("operator")).lower()
        if operator in {"equals", "==", "="}:
            operator = "eq"
        elif operator in {"!=", "<>"}:
            operator = "ne"
        elif operator in COMPARATOR_ALIASES:
            mapped = COMPARATOR_ALIASES[operator]
            review.append(
                {
                    "kind": "mapped_custom_comparator",
                    "subject": predicate_id,
                    "detail": (
                        f"comparison operator {operator!r} is not a deterministic comparator; "
                        f"mapped to {mapped!r} — verify that the fact carries the band/flag value"
                    ),
                }
            )
            operator = mapped
        if operator not in COMPARISON_OPERATORS | {"in", "contains_any"}:
            review.append(
                {
                    "kind": "unsupported_comparison",
                    "subject": predicate_id,
                    "detail": (
                        f"comparison operator {operator!r} has no executable equivalent; the "
                        "comparison was dropped and the predicate reduced to its reduction "
                        "semantics — manual review required"
                    ),
                }
            )
            predicate["compare"] = None
        else:
            predicate["compare"] = {**dict(compare), "operator": operator}
    else:
        predicate["compare"] = None

    domain = _domain_for(predicate_id, predicate["aspect"])
    predicate["schema_v2"] = {
        "semantic_domain": domain,
        "compute_family": _compute_family(predicate),
        "compute_facets": _compute_facets(predicate, domain),
    }
    predicate["source_kind"] = (
        "question_observation" if predicate["entity_type"] != "clinical_assertion"
        else "explicit_clinical_assertion"
    )
    return predicate


def _normalise_term(source: Mapping[str, Any], *, review: List[Dict[str, Any]]) -> Dict[str, Any]:
    term = copy.deepcopy(dict(source))
    term_id = _text(term.get("id"))
    if not term_id:
        term["id"] = _stable_id("term", _text(term.get("name")))
        term_id = term["id"]
    aspect_hint = _text(term.get("type")) or _text(term.get("label"))
    term["schema_v2"] = {
        "semantic_domain": _domain_for(term_id, aspect_hint),
        "node_role": "clinical_entity",
    }
    term.setdefault("binding_status", "candidate")
    return term


# --------------------------------------------------------------------------- #
# (3) ClinicalAssessment routing
# --------------------------------------------------------------------------- #


def _build_assessment_spec(
    predicate: Mapping[str, Any],
    *,
    evidence_concept_ids: Sequence[str],
    graph_id: str,
) -> Dict[str, Any]:
    """Declare a bounded clinical question the interpreter must not compute."""
    predicate_id = _text(predicate.get("id"))
    spec_id = _stable_id("assessment_spec", predicate_id)
    question = _text(predicate.get("description")) or _text(predicate.get("name"))
    if not question:
        question = f"临床条件“{predicate_id}”在当前资料下是否成立？"
    return {
        "id": f"clinical_assessment_spec::{spec_id}",
        "node_type": "clinical_assessment_spec",
        "assessment_spec_id": spec_id,
        "condition_concept_id": _stable_id("assessment", predicate_id),
        "question": (
            f"{question} 仅依据已提取且本评估允许的原始观察回答；"
            "证据不足以肯定或否定时返回 unknown。"
        ),
        "allowed_values": ["true", "false", "unknown"],
        "evidence_concept_ids": list(evidence_concept_ids),
        "source_evidence_ids": [],
        "unknown_policy": "propagate_unknown",
        "human_review_policy": "required_before_action",
        "graph_ref": graph_id,
        "derived_from_aspect": _text(predicate.get("aspect")),
    }


# --------------------------------------------------------------------------- #
# (2) DAG normalisation
# --------------------------------------------------------------------------- #



def _resolve_predicate_target(
    node: Mapping[str, Any],
    source_key: str,
    *,
    predicate_ids: Mapping[str, str],
    assessment_for: Mapping[str, str],
    source_nodes: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Resolve a predicate-leaf node to the predicate (or DAG node) it names.

    Real extraction payloads vary: the reference lives in ``predicate_ref``, in
    ``input``, in an alias field, or the leaf only carries a container id while
    another DAG node holds the actual predicate.  Returns ``None`` when nothing
    resolves, so the caller can report it instead of inventing a dangling id.
    """
    candidates = [
        _text(node.get("predicate_ref")),
        _text(node.get("input")),
        _text(node.get("predicate_id")),
        _text(node.get("input_id")),
        _text(node.get("reference_id")),
        source_key,
        _text(node.get("id")),
    ]
    for candidate in candidates:
        if candidate and (candidate in predicate_ids or candidate in assessment_for):
            return candidate
    for candidate in candidates:
        if candidate and candidate in source_nodes:
            nested = source_nodes[candidate]
            for alias in ("predicate_ref", "input", "predicate_id", "input_id", "reference_id", "id"):
                inner = _text(nested.get(alias))
                if inner and (inner in predicate_ids or inner in assessment_for):
                    return inner
    return None


class _DagCompiler:
    """Rewrite one extraction DAG into the interpreter contract."""

    def __init__(
        self,
        *,
        rule_id: str,
        predicate_ids: Mapping[str, str],
        assessment_for: Mapping[str, str],
        review: List[Dict[str, Any]],
        source_nodes: Optional[Mapping[str, Mapping[str, Any]]] = None,
        output_types: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.rule_id = rule_id
        self.predicate_ids = predicate_ids
        self.output_types = dict(output_types or {})
        self.assessment_for = assessment_for
        self.review = review
        self.source_nodes = dict(source_nodes or {})
        self.nodes: List[Dict[str, Any]] = []
        self.node_id_map: Dict[str, Optional[str]] = {}
        self._by_source: Dict[str, str] = {}
        self.predicate_refs: List[str] = []
        self.assessment_refs: List[str] = []

    # -- node emission ----------------------------------------------------- #

    def _emit(self, node: Dict[str, Any]) -> str:
        node_id = str(node["id"])
        while any(existing["id"] == node_id for existing in self.nodes):
            node_id = f"{node_id}.{_sha256(node_id)[:8]}"
            node["id"] = node_id
        self.nodes.append(node)
        return node_id

    def emit(self, node: Dict[str, Any]) -> str:
        """Emit a node verbatim (public wrapper around the de-duplicating emitter)."""
        return self._emit(node)

    def resolved(self, source_key: str) -> Optional[str]:
        """Return the emitted node id already bound to a source DAG node id."""
        return self._by_source.get(_text(source_key))

    def literal(self, value: Any) -> str:
        """Emit (or reuse) a literal node."""
        return self._literal(value)

    def operand(self, raw: Any, source_id: str) -> str:
        """Resolve one DAG operand (bare predicate id, node id, or literal)."""
        if isinstance(raw, bool) or raw is None:
            return self._literal(raw)
        if isinstance(raw, (int, float)):
            return self._literal(raw)
        if isinstance(raw, Mapping):
            # nested inline node
            return self.compile_node(dict(raw), source_id=source_id)
        text = _text(raw)
        if not text:
            return self._literal(None)
        if text in self._by_source:
            return self._by_source[text]
        if text in self.assessment_for:
            return self._assessment_node(text)
        if text in self.predicate_ids:
            return self._predicate_node(text)
        self.review.append(
            {
                "kind": "unresolved_dag_operand",
                "subject": self.rule_id,
                "detail": f"DAG operand {text!r} is neither a predicate nor a node id",
            }
        )
        return self._literal(None)

    def _literal(self, value: Any) -> str:
        node_id = self._emit(
            {
                "id": _stable_id("dag.literal", repr(value)),
                "type": "literal",
                "node_type": "literal",
                "value": value,
                "return_type": "TruthValue",
                "output_type": "TruthValue",
            }
        )
        return node_id

    def _predicate_node(self, predicate_id: str) -> str:
        self.predicate_refs.append(predicate_id)
        node_id = self._emit(
            {
                "id": _stable_id("dag.pred", predicate_id),
                "type": "predicate_ref",
                "node_type": "predicate_ref",
                "predicate_ref": predicate_id,
                "return_type": self.output_types.get(predicate_id, "TruthValue"),
                "output_type": self.output_types.get(predicate_id, "TruthValue"),
            }
        )
        self._by_source[predicate_id] = node_id
        return node_id

    def _assessment_node(self, predicate_id: str) -> str:
        spec_id = self.assessment_for[predicate_id]
        self.assessment_refs.append(spec_id)
        node_id = self._emit(
            {
                "id": _stable_id("dag.assessment", spec_id),
                "type": _ASSESSMENT_LEAF_TYPE,
                "node_type": _ASSESSMENT_LEAF_TYPE,
                "assessment_spec_id": spec_id,
                "return_type": "TruthValue",
                "output_type": "TruthValue",
            }
        )
        self._by_source[predicate_id] = node_id
        return node_id

    # -- node compilation -------------------------------------------------- #

    def compile_node(self, node: Mapping[str, Any], *, source_id: str) -> str:
        node = dict(node)
        source_key = _text(node.get("id")) or source_id
        self.node_id_map[source_key] = None  # filled on emission
        if source_key in self._by_source:
            return self._by_source[source_key]

        node_type = _text(node.get("type")) or "combine"
        if node_type in _NON_EXECUTABLE_DAG_TYPES:
            self.review.append(
                {
                    "kind": "non_executable_dag_node",
                    "subject": self.rule_id,
                    "detail": (
                        f"DAG node {source_key!r} of type {node_type!r} has no interpreter "
                        "equivalent and was folded into its predicate facet"
                    ),
                }
            )
            node_type = "combine" if node.get("inputs") or node.get("input") else "literal"

        emitted_id = _stable_id(f"dag.{node_type}", f"{_rule_namespace(self.rule_id)}.{source_key}")

        if node_type in _PREDICATE_LEAF_TYPES:
            target = _resolve_predicate_target(
                node, source_key, predicate_ids=self.predicate_ids,
                assessment_for=self.assessment_for, source_nodes=self.source_nodes,
            )
            if target is None:
                # Operand names another DAG node: inline that node in place.
                if source_key != _text(node.get("id") or source_key):
                    pass
                self.review.append(
                    {
                        "kind": "unresolved_dag_operand",
                        "subject": self.rule_id,
                        "detail": (
                            f"predicate leaf {source_key!r} references "
                            f"{_text(node.get('predicate_ref')) or _text(node.get('input'))!r}, "
                            "which is neither a declared predicate nor a DAG node"
                        ),
                    }
                )
                node_id = self._literal(None)
                self._by_source[source_key] = node_id
                return node_id
            if target in self.assessment_for:
                node_id = self._assessment_node(target)
            else:
                node_id = self._predicate_node(target)
            self._by_source[source_key] = node_id
            return node_id

        if node_type == _ASSESSMENT_LEAF_TYPE:
            spec_id = _text(node.get("assessment_spec_id"))
            if not spec_id:
                self.review.append(
                    {
                        "kind": "assessment_ref_without_spec",
                        "subject": self.rule_id,
                        "detail": f"DAG node {source_key!r} references no assessment spec",
                    }
                )
                return self._literal(None)
            if not any(existing["id"] == _stable_id("dag.assessment", spec_id) for existing in self.nodes):
                self.assessment_refs.append(spec_id)
            node_id = self._emit(
                {
                    "id": _stable_id("dag.assessment", spec_id),
                    "type": _ASSESSMENT_LEAF_TYPE,
                    "node_type": _ASSESSMENT_LEAF_TYPE,
                    "assessment_spec_id": spec_id,
                    "return_type": "TruthValue",
                    "output_type": "TruthValue",
                }
            )
            self._by_source[source_key] = node_id
            return node_id

        if node_type == "combine":
            raw_operator = _text(node.get("operator")) or "AND"
            operator = raw_operator.upper()
            if operator in {"ALL", "ANY"}:
                operator = {"ALL": "AND", "ANY": "OR"}[operator]
            operands: List[Any] = list(node.get("inputs") or [])
            if not operands and node.get("input") is not None:
                single = node.get("input")
                operands = list(single) if isinstance(single, list) else [single]
            self._by_source[source_key] = emitted_id  # guard against cycles
            children = [self.operand(item, source_id=f"{source_key}.{index}") for index, item in enumerate(operands)]
            if operator not in _RUNTIME_COMBINERS:
                self.review.append(
                    {
                        "kind": "unknown_combiner",
                        "subject": self.rule_id,
                        "detail": f"combiner {raw_operator!r} mapped to AND",
                    }
                )
                operator = "AND"
            if operator == "NOT" and len(children) != 1:
                self.review.append(
                    {
                        "kind": "invalid_not_arity",
                        "subject": self.rule_id,
                        "detail": f"NOT has {len(children)} operands; invalid arity forced to AND",
                    }
                )
                operator = "AND"
            if not children:
                self.nodes = [item for item in self.nodes if item["id"] != emitted_id]
                self._by_source.pop(source_key, None)
                return self._literal(None)
            emitted = {
                "id": emitted_id,
                "type": "combine",
                "node_type": "combine",
                "operator": operator,
                "inputs": children,
                "return_type": "TruthValue",
                "output_type": "TruthValue",
            }
            if operator == "AT_LEAST":
                parameters = node.get("parameters") if isinstance(node.get("parameters"), Mapping) else {}
                minimum = parameters.get("minimum", node.get("minimum"))
                if not isinstance(minimum, int) or isinstance(minimum, bool) or not 1 <= minimum <= len(children):
                    minimum = 1
                    self.review.append(
                        {
                            "kind": "at_least_missing_minimum",
                            "subject": self.rule_id,
                            "detail": f"AT_LEAST on {source_key!r} lacks a valid minimum; defaulted to 1",
                        }
                    )
                emitted["parameters"] = {"minimum": minimum}
            self._emit(emitted)
            self._by_source[source_key] = emitted_id
            return emitted_id

        if node_type == "compare":
            raw_operator = _text(node.get("operator")).lower().replace(">=", "ge").replace(
                "<=", "le"
            ).replace(">", "gt").replace("<", "lt").replace("==", "eq")
            operator = raw_operator if raw_operator in COMPARISON_OPERATORS else "eq"
            left = node.get("left", node.get("input"))
            right = node.get("right", node.get("value"))
            self._by_source[source_key] = emitted_id
            left_id = self.operand(left, source_id=f"{source_key}.left")
            right_id = self.operand(right, source_id=f"{source_key}.right")
            self._emit(
                {
                    "id": emitted_id,
                    "type": "compare",
                    "node_type": "compare",
                    "operator": operator,
                    "left": left_id,
                    "right": right_id,
                    "return_type": "TruthValue",
                    "output_type": "TruthValue",
                }
            )
            self._by_source[source_key] = emitted_id
            return emitted_id

        if node_type == "exists":
            self._by_source[source_key] = emitted_id
            operand = self.operand(
                node.get("input", node.get("predicate_ref")), source_id=f"{source_key}.input"
            )
            self._emit(
                {
                    "id": emitted_id,
                    "type": "exists",
                    "node_type": "exists",
                    "input": operand,
                    "return_type": "TruthValue",
                    "output_type": "TruthValue",
                }
            )
            self._by_source[source_key] = emitted_id
            return emitted_id

        # literal / anything else
        self._by_source[source_key] = emitted_id
        self._emit(
            {
                "id": emitted_id,
                "type": "literal",
                "node_type": "literal",
                "value": node.get("value"),
                "return_type": _text(node.get("return_type")) or "TruthValue",
                "output_type": _text(node.get("return_type")) or "TruthValue",
            }
        )
        return emitted_id


def _compile_dag(
    rule: Mapping[str, Any],
    *,
    rule_id: str,
    predicate_ids: Mapping[str, str],
    assessment_for: Mapping[str, str],
    review: List[Dict[str, Any]],
    output_types: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    dag = rule.get("condition_dag") if isinstance(rule.get("condition_dag"), Mapping) else {}
    compiler = _DagCompiler(
        rule_id=rule_id,
        predicate_ids=predicate_ids,
        assessment_for=assessment_for,
        review=review,
        source_nodes={
            _text(item.get("id")): item
            for item in _records(dag.get("nodes"))
            if _text(item.get("id"))
        },
        output_types=output_types,
    )
    root = _text(dag.get("root")) or "ROOT"
    nodes = _records(dag.get("nodes"))
    root_id: Optional[str] = None
    for node in nodes:
        source_key = _text(node.get("id"))
        compiled = compiler.compile_node(node, source_id=source_key)
        if source_key == root:
            root_id = compiled
    if root_id is None:
        root_id = compiler.resolved(root)
    if root_id is None and nodes:
        root_id = compiler.resolved(_text(nodes[-1].get("id")))
    if root_id is None and rule.get("condition"):
        root_id = compiler.operand(rule.get("condition"), source_id=f"{rule_id}.condition")
    if root_id is None:
        review.append(
            {
                "kind": "dag_root_unresolved",
                "subject": rule_id,
                "detail": (
                    "condition DAG declares no resolvable root; the rule is pinned to a "
                    "constant-true literal and cannot constrain any action"
                ),
            }
        )
        root_id = compiler.literal(True)
    # A rule root must be Boolean.  Real extraction payloads sometimes root a
    # rule directly on a value-typed predicate (Enum/Quantity/Code).  The
    # interpreter supports no operator that turns such a value into a truth
    # value, and inventing one (an existence test on a scalar) would change the
    # rule's clinical meaning.  The rule is therefore pinned to an unresolved
    # literal: it never activates, and the review item says why.
    root_node = next((item for item in compiler.nodes if item["id"] == root_id), None)
    if root_node is not None and _text(root_node.get("return_type")) not in {"Bool", "TruthValue"}:
        original_type = _text(root_node.get("return_type"))
        review.append(
            {
                "kind": "non_boolean_rule_root",
                "subject": rule_id,
                "detail": (
                    f"rule root returns {original_type!r}, which no interpreter operator can "
                    "reduce to a Boolean condition; the rule was pinned to 'unknown' and "
                    "requires manual modelling review"
                ),
            }
        )
        root_id = compiler.literal(None)

    return (
        {"root": root_id, "nodes": compiler.nodes},
        sorted(set(compiler.predicate_refs)),
        sorted(set(compiler.assessment_refs)),
    )


# --------------------------------------------------------------------------- #
# (1) actions
# --------------------------------------------------------------------------- #


def _action_type(permission: str, intent: str) -> str:
    if permission in {"avoid", "contraindicate", "stop"}:
        return "discontinuation_or_avoidance"
    if permission in {"reduce_dose", "increase_dose", "titrate", "max_dose_limit", "start_low_dose"}:
        return "dose_adjustment"
    if permission in {"caution"}:
        return "monitoring_or_procedure"
    if "diagnos" in intent or "derive" in intent:
        return "diagnosis_or_state_derivation"
    if intent in {"monitor", "prophylaxis"} or "monitor" in intent:
        return "monitoring_or_procedure"
    if permission in {"recommend", "require", "allow", "consider", "continue", "maintain_dose"}:
        return "therapy_initiation_or_continuation"
    return "other_clinical_guidance"


def _subject_refs(action: Mapping[str, Any], known_node_ids: Mapping[str, str]) -> List[str]:
    refs: List[str] = []
    for subject in action.get("subjects") or []:
        text = _text(subject)
        if not text:
            continue
        refs.append(known_node_ids.get(text, text))
    return refs


def _materialise_action(
    action: Mapping[str, Any],
    *,
    rule_id: str,
    graph_id: str,
    known_node_ids: Mapping[str, str],
    used_action_ids: MutableMapping[str, int],
) -> Dict[str, Any]:
    permission = _text(action.get("permission")) or "recommend"
    intent = _text(action.get("intent")) or ""
    label = _text((action.get("output") or {}).get("text")) if isinstance(
        action.get("output"), Mapping
    ) else ""
    action_id = _stable_id("action", f"{rule_id}.{permission}.{label or intent or 'do'}", used=used_action_ids)
    return {
        "id": action_id,
        "node_type": "action",
        "subject_refs": _subject_refs(action, known_node_ids),
        "permission": permission,
        "permission_source": "extraction",
        "strength": action.get("strength"),
        "intent": intent,
        "action_type": _action_type(permission, intent),
        "dose": copy.deepcopy(action.get("dose")),
        "duration": copy.deepcopy(action.get("duration")),
        "timing": copy.deepcopy(action.get("timing")),
        "monitoring": list(action.get("monitoring") or []),
        "requirements": list(action.get("requirements") or []),
        "conflict_profile": copy.deepcopy(action.get("conflict_profile") or {}),
        "output": copy.deepcopy(action.get("output")),
        # Derived at execution time from rule truth x review policy (see the
        # interpreter's status/execution_authorized contract); a static graph
        # cannot pre-authorise anything.
        "execution_authorized": False,
        "clinical_review_required": bool(action.get("clinical_review_required", False)),
        "human_review_policy": _text(action.get("human_review_policy")) or "none",
        "source_rule_ids": [rule_id],
    }


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #


def project_graph(
    extraction: Mapping[str, Any],
    *,
    graph_id: Optional[str] = None,
    qid: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile one extraction payload into an executable Schema-v2 graph.

    Parameters
    ----------
    extraction:
        Output of :meth:`guideline2action.graph.graph_api.ClinicalGuidelinePipeline.run`
        (or an equivalent mapping with ``terms``/``med_terms``/``predicates``/
        ``rules`` keys).
    graph_id:
        Stable identifier for the produced graph.  Defaults to the extraction
        payload's ``guideline_id`` or a content hash.
    qid:
        Identifier the interpreter requires on its program envelope.  For
        text-driven compilation there is no benchmark question, so a stable
        placeholder is derived from ``graph_id`` unless one is supplied.

    Returns
    -------
    dict
        Graph carrying ``schema_version``, ``graph_id``, ``predicates``,
        ``actions``, ``rules``, ``clinical_assessment_specs``, ``nodes``,
        ``edges``, ``runtime_contract`` and a ``projection`` report.  The
        ``runtime_program()`` view of this mapping is what the interpreter
        consumes.
    """
    if not isinstance(extraction, Mapping):
        raise TypeError("extraction payload must be a mapping")

    review: List[Dict[str, Any]] = []
    resolved_graph_id = _text(graph_id) or _text(extraction.get("guideline_id")) or (
        f"guideline.{_sha256(extraction)[:12]}"
    )
    resolved_qid = _text(qid) or f"guideline::{_slug(resolved_graph_id)}"

    terms = [_normalise_term(item, review=review) for item in _records(extraction.get("terms"))]
    med_terms = [
        _normalise_term(item, review=review) for item in _records(extraction.get("med_terms"))
    ]

    known_node_ids: Dict[str, str] = {}
    for term in terms:
        known_node_ids[_text(term.get("id"))] = _text(term.get("id"))
    for term in med_terms:
        known_node_ids[_text(term.get("id"))] = _text(term.get("id"))

    raw_predicates = [_normalise_predicate(item, review=review) for item in _records(extraction.get("predicates"))]

    # --- ClinicalAssessment routing: non-computable findings leave the executable set
    executable: List[Dict[str, Any]] = []
    assessment_specs: List[Dict[str, Any]] = []
    assessment_for: Dict[str, str] = {}
    for predicate in raw_predicates:
        predicate_id = _text(predicate.get("id"))
        if _is_judgment(predicate):
            spec = _build_assessment_spec(
                predicate,
                evidence_concept_ids=[_text(predicate.get("entity"))] if predicate.get("entity") else [],
                graph_id=resolved_graph_id,
            )
            assessment_specs.append(spec)
            assessment_for[predicate_id] = _text(spec["assessment_spec_id"])
            review.append(
                {
                    "kind": "routed_to_clinical_assessment",
                    "subject": predicate_id,
                    "detail": (
                        "finding is not deterministically computable "
                        f"(aspect={_text(predicate.get('aspect'))!r}); routed to "
                        f"{spec['assessment_spec_id']}"
                    ),
                }
            )
            continue
        if predicate.get("entity_type") == "clinical_assessment":
            review.append(
                {
                    "kind": "clinical_assessment_entity_without_judgment_token",
                    "subject": predicate_id,
                    "detail": (
                        "predicate declares entity_type=clinical_assessment but its aspect/id "
                        "carries no judgment token; kept executable — verify manually"
                    ),
                }
            )
        executable.append(predicate)

    predicate_ids = {_text(item.get("id")): _text(item.get("id")) for item in executable}
    predicate_output_types = {
        _text(item.get("id")): _text(item.get("final_output_type")) or "TruthValue"
        for item in executable
    }

    # --- rules + action materialisation
    used_action_ids: Dict[str, int] = {}
    actions: List[Dict[str, Any]] = []
    rules: List[Dict[str, Any]] = []
    all_predicate_refs: List[str] = []
    all_assessment_refs: List[str] = []

    for source_rule in _records(extraction.get("rules")):
        rule_id = _text(source_rule.get("id")) or _stable_id(
            "rule", _text(source_rule.get("label")) or _sha256(source_rule)[:8]
        )
        dag, predicate_refs, assessment_refs = _compile_dag(
            source_rule,
            rule_id=rule_id,
            predicate_ids=predicate_ids,
            assessment_for=assessment_for,
            review=review,
            output_types=predicate_output_types,
        )
        source_action = source_rule.get("action")
        action_ids: List[str] = []
        if isinstance(source_action, Mapping):
            action = _materialise_action(
                source_action,
                rule_id=rule_id,
                graph_id=resolved_graph_id,
                known_node_ids=known_node_ids,
                used_action_ids=used_action_ids,
            )
            actions.append(action)
            if assessment_refs:
                review.append(
                    {
                        "kind": "assessment_gated_action",
                        "subject": _text(action["id"]),
                        "detail": (
                            "action is gated by clinical assessment(s) "
                            + ", ".join(sorted(assessment_refs))
                            + "; execution depends on the runtime review policy"
                        ),
                    }
                )
            action_ids.append(_text(action["id"]))
        else:
            review.append(
                {
                    "kind": "rule_without_action",
                    "subject": rule_id,
                    "detail": "rule declares no action payload; nothing can be authorised",
                }
            )

        rules.append(
            {
                "id": rule_id,
                "node_type": "rule",
                "label": _text(source_rule.get("label")),
                "condition_dag": dag,
                "boolean_root": _text(source_rule.get("boolean_root")) or "ROOT",
                "input_predicates": predicate_refs,
                "predicate_refs": predicate_refs,
                "assessment_spec_refs": assessment_refs,
                "action_refs": action_ids,
                "missing_data_policy": _text(source_rule.get("missing_data_policy")) or "propagate_unknown",
                "scope": copy.deepcopy(source_rule.get("scope") or {}),
                "priority": copy.deepcopy(source_rule.get("priority") or {}),
                "source_text": _text(source_rule.get("source_text")),
            }
        )
        all_predicate_refs.extend(predicate_refs)
        all_assessment_refs.extend(assessment_refs)

    # --- graph projection (nodes / edges)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, label: str, node_type: str) -> None:
        if any(item["id"] == node_id for item in nodes):
            return
        nodes.append({"id": node_id, "label": label, "node_type": node_type})

    def add_edge(subject: str, predicate: str, obj: str, role: str) -> None:
        edges.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "reference_role": role,
            }
        )

    for term in [*terms, *med_terms]:
        add_node(_text(term.get("id")), _text(term.get("name")), "clinical_entity")
    for predicate in executable:
        predicate_id = _text(predicate.get("id"))
        add_node(predicate_id, _text(predicate.get("name")), "predicate")
        entity = _text(predicate.get("entity"))
        if entity and entity in known_node_ids:
            add_edge(predicate_id, "reads_entity", entity, "predicate.entity")
    for spec in assessment_specs:
        spec_id = _text(spec.get("assessment_spec_id"))
        add_node(f"clinical_assessment_spec::{spec_id}", _text(spec.get("question")), "clinical_assessment_spec")
    for action in actions:
        add_node(_text(action.get("id")), _text((action.get("output") or {}).get("text") or action.get("intent")), "action")
        for rule_id in action.get("source_rule_ids") or []:
            add_edge(_text(rule_id), "triggers_action", _text(action.get("id")), "rule.action_ref")
        for subject in action.get("subject_refs") or []:
            if subject in known_node_ids:
                add_edge(_text(action.get("id")), "acts_on", subject, "action.subject_ref")
    for rule in rules:
        add_node(_text(rule.get("id")), _text(rule.get("label")), "rule")
        for predicate_id in rule.get("predicate_refs") or []:
            add_edge(_text(rule.get("id")), "requires_predicate", predicate_id, "rule.predicate_ref")
        for spec_id in rule.get("assessment_spec_refs") or []:
            add_edge(
                _text(rule.get("id")),
                "requires_assessment",
                f"clinical_assessment_spec::{spec_id}",
                "rule.assessment_spec_ref",
            )

    orphan_predicates = sorted(set(predicate_ids) - set(all_predicate_refs))
    if orphan_predicates:
        review.append(
            {
                "kind": "orphan_predicates",
                "subject": resolved_graph_id,
                "detail": (
                    "predicates not referenced by any rule DAG remain declared but unexecuted: "
                    + ", ".join(orphan_predicates)
                ),
            }
        )

    unconsumed_assessments = sorted(set(assessment_for.values()) - set(all_assessment_refs))
    graph: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph_id": resolved_graph_id,
        "graph_instance_id": resolved_graph_id,
        "source_question": {"qid": resolved_qid},
        "metadata": {
            "entrypoint": "guideline_text",
            "projection_schema": SCHEMA_VERSION,
        },
        "terms": terms,
        "med_terms": med_terms,
        "predicates": executable,
        "actions": actions,
        "rules": rules,
        "clinical_assessment_specs": assessment_specs,
        "nodes": nodes,
        "edges": edges,
        "conversion_status": "converted",
        "runtime_contract": {
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "condition_dag_node_types": ["predicate_ref", _ASSESSMENT_LEAF_TYPE, "literal", "compare", "combine", "exists"],
            "condition_dag_combine_operators": sorted(COMBINE_OPERATORS),
            "unknown_policy": "strong_kleene_propagation",
            "qid_allowed_only_in": ["graph_instance_id", "source_question"],
        },
        "projection": {
            "review": review,
            "counts": {
                "terms": len(terms),
                "med_terms": len(med_terms),
                "predicates": len(executable),
                "rules": len(rules),
                "actions": len(actions),
                "clinical_assessment_specs": len(assessment_specs),
            },
            "routed_to_clinical_assessment": sorted(assessment_for),
            "unconsumed_assessment_specs": unconsumed_assessments,
            "orphan_predicates": orphan_predicates,
        },
    }
    return graph


def runtime_program(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the interpreter's own program envelope for a projected graph.

    The runtime validates ``schema_version``/``qid`` on its program envelope,
    which the graph carries under different keys; this view bridges the two
    without touching either contract.
    """
    question = graph.get("source_question") if isinstance(graph.get("source_question"), Mapping) else {}
    qid = _text(question.get("qid")) or f"guideline::{_slug(graph.get('graph_id'))}"
    return {
        "schema_version": "guideline2graph.fact_first_program.v1",
        "qid": qid,
        "graph_id": _text(graph.get("graph_id")) or "guideline2action.graph",
        "predicates": copy.deepcopy(list(graph.get("predicates") or [])),
        "clinical_assessment_specs": copy.deepcopy(
            list(graph.get("clinical_assessment_specs") or [])
        ),
        "rules": copy.deepcopy(list(graph.get("rules") or [])),
        "knowledge_relations": [],
    }


__all__ = [
    "SCHEMA_VERSION",
    "RUNTIME_CONTRACT_VERSION",
    "NON_COMPUTABLE_ASPECTS",
    "JUDGMENT_TOKENS",
    "project_graph",
    "runtime_program",
]
