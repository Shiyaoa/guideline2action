"""Runtime adapter for the unified DecisionKG Schema-v2 graph contract.

The adapter contains no question-specific clinical logic.  It translates the
static Schema-v2 payload into the generic fact-first runtime contract and
passes ClinicalAssessment values as an explicit, validated runtime input.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .guideline2graph_runtime_v1 import (
    Guideline2GraphProgramError,
    Guideline2GraphRuntimeV1,
    validate_clinical_assertions,
    validate_clinical_assessments,
)
from .handover import attach_clinical_trace


SCHEMA_VERSION = "decisionkg.schema_v2.unified.20260804"
TRACE_SCHEMA_VERSION = "decisionkg.schema_v2.unified_trace.v1"


class UnifiedSchemaV2RuntimeError(ValueError):
    """Raised when a Schema-v2 graph cannot be adapted for execution."""


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        return []
    return list(value)


def _observation_facts(observations: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    facts: list[Dict[str, Any]] = []
    for index, observation in enumerate(observations):
        item = dict(observation)
        observation_id = str(item.get("observation_id") or f"observation.{index + 1}")
        source_span = item.get("source_span") if isinstance(item.get("source_span"), Mapping) else {}
        fact = {
            "fact_id": observation_id,
            "concept_id": str(item.get("concept_id") or ""),
            "bound_predicate_id": str(item.get("bound_predicate_id") or ""),
            "value": copy.deepcopy(item.get("value")),
            "unit": item.get("unit"),
            "polarity": item.get("polarity", "present"),
            "temporal": copy.deepcopy(
                item.get("temporal")
                or {"relation": item.get("timepoint") or "current", "offset": None}
            ),
            "source_quote": str(source_span.get("text") or item.get("source_quote") or ""),
            "operator": item.get("operator"),
            "input_id": item.get("input_id"),
            "source_input_id": item.get("source_input_id"),
            "source_fact_id": item.get("source_fact_id"),
            "timepoint": item.get("timepoint"),
            "observation_kind": item.get("observation_kind"),
            "fact_resource": "QuestionObservation",
            "source_kind": (
                "explicit_clinical_assertion"
                if str(item.get("observation_kind") or "").strip().casefold()
                == "clinical_assertion"
                else "question_observation"
            ),
        }
        facts.append(fact)
    return facts


def _clinical_assertion_facts(
    assertions: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    facts: list[Dict[str, Any]] = []
    for assertion in assertions:
        item = dict(assertion)
        source_span = item.get("source_span") if isinstance(item.get("source_span"), Mapping) else {}
        facts.append(
            {
                "fact_id": str(item.get("assertion_id") or ""),
                "concept_id": str(
                    item.get("concept_id")
                    or item.get("condition_concept_id")
                    or ""
                ),
                "value": copy.deepcopy(item.get("value")),
                "unit": item.get("unit"),
                "polarity": item.get("polarity", "present"),
                "temporal": copy.deepcopy(
                    item.get("temporal")
                    or {"relation": item.get("timepoint") or "current", "offset": None}
                ),
                "source_quote": str(source_span.get("text") or ""),
                "source_fact_id": item.get("source_fact_id"),
                "timepoint": item.get("timepoint"),
                "asserted_by": item.get("asserted_by"),
                "fact_resource": "ClinicalAssertion",
                "source_kind": "explicit_clinical_assertion",
            }
        )
    return facts


def _program_from_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    question = graph.get("source_question") if isinstance(graph.get("source_question"), Mapping) else {}
    qid = str(question.get("qid") or "")
    actions = {
        str(item.get("id")): dict(item)
        for item in _records(graph.get("actions"))
        if str(item.get("id") or "")
    }
    predicates = []
    for item in _records(graph.get("predicates")):
        predicate = copy.deepcopy(dict(item))
        predicate.pop("node_type", None)
        source_payload = predicate.get("source_payload") or {}
        comparison = predicate.get("compare")
        if (
            isinstance(source_payload, Mapping)
            and str(source_payload.get("operator") or "") == "enum_in"
            and isinstance(comparison, Mapping)
            and comparison.get("value") is None
            and isinstance(source_payload.get("values"), list)
        ):
            predicate["compare"] = {
                "operator": "in",
                "value": list(source_payload["values"]),
            }
        comparison = predicate.get("compare")
        if isinstance(comparison, Mapping) and comparison.get("unit") is None:
            expected_unit = predicate.get("expected_unit")
            if expected_unit is None and isinstance(source_payload, Mapping):
                expected_unit = source_payload.get("expected_unit")
            if expected_unit is not None:
                predicate["compare"] = {
                    **dict(comparison),
                    "unit": expected_unit,
                }
        facets = ((predicate.get("schema_v2") or {}).get("compute_facets") or {})
        if not isinstance(facets, Mapping):
            facets = {}
        predicate.setdefault("entity_type", facets.get("input_entity_type") or "observation")
        predicate.setdefault("aspect", facets.get("aspect") or "observation")
        predicate.setdefault("input_shape", "single")
        predicate.setdefault("return_type", predicate.get("final_output_type") or "TruthValue")
        predicate.setdefault("final_output_type", "TruthValue")
        predicate.setdefault("null_policy", "unknown")
        predicate.setdefault("extract", {"path": "value", "type": predicate.get("final_output_type") or "TruthValue"})
        predicate.setdefault("retrieve", {"resource": "QuestionObservation", "concepts": [predicate.get("entity")]})
        predicate.setdefault("temporal_scope", {"mode": "all_time"})
        predicate.setdefault("reduction", {"operator": "most_recent", "output_type": predicate.get("final_output_type") or "TruthValue"})
        predicate.setdefault("compare", None)
        resource = str((predicate.get("retrieve") or {}).get("resource") or "")
        if resource.casefold() == "clinicalassertion":
            predicate["source_kind"] = "explicit_clinical_assertion"
        elif resource.casefold() == "clinicalassessment":
            predicate["source_kind"] = "clinical_assessment"
        else:
            predicate["source_kind"] = "question_observation"
        predicates.append(predicate)

    source_review = graph.get("source_review")
    graph_source_review_required = bool(
        isinstance(source_review, Mapping) and source_review.get("required")
    )
    rules = []
    for source_rule in _records(graph.get("rules")):
        dag = source_rule.get("condition_dag") if isinstance(source_rule.get("condition_dag"), Mapping) else {}
        nodes = []
        for source_node in _records(dag.get("nodes")):
            node = copy.deepcopy(dict(source_node))
            node_type = str(node.pop("node_type", node.get("type") or ""))
            node["type"] = node_type
            node["return_type"] = node.pop("output_type", node.get("return_type") or "TruthValue")
            nodes.append(node)
        action_ids = [str(item) for item in source_rule.get("action_refs") or []]
        labels = {
            action_id: str(
                ((actions.get(action_id) or {}).get("output") or {}).get("text")
                if isinstance((actions.get(action_id) or {}).get("output"), Mapping)
                else None
            )
            or action_id
            for action_id in action_ids
        }
        action_metadata = {
            action_id: {
                "permission": str((actions.get(action_id) or {}).get("permission") or "recommend"),
                "intent": str((actions.get(action_id) or {}).get("intent") or ""),
            }
            for action_id in action_ids
        }
        permission = "recommend"
        intent = ""
        human_review_reasons = set()
        if graph_source_review_required:
            human_review_reasons.add("source_review_pending")
        for action_id in action_ids:
            action = actions.get(action_id) or {}
            action_permission = str(action.get("permission") or "recommend")
            if permission == "recommend" and action_permission != "recommend":
                permission = action_permission
            if not intent:
                intent = str(action.get("intent") or "")
            if action.get("source_review_required"):
                human_review_reasons.add("source_review_pending")
            if action.get("clinical_review_required"):
                human_review_reasons.add("action_review_required")
            if str(action.get("human_review_policy") or "") == "required_before_action":
                human_review_reasons.add("action_review_required")
        rules.append(
            {
                "id": str(source_rule.get("id") or ""),
                "label": str(source_rule.get("label") or ""),
                "input_predicates": [str(item) for item in source_rule.get("predicate_refs") or []],
                "input_assessments": [str(item) for item in source_rule.get("assessment_spec_refs") or []],
                "condition_dag": {"root": str(dag.get("root") or ""), "nodes": nodes},
                "missing_data_policy": str(source_rule.get("missing_data_policy") or "propagate_unknown"),
                "action": {
                    "permission": permission,
                    "intent": intent,
                    "human_review_required": bool(human_review_reasons),
                    "human_review_reasons": sorted(human_review_reasons),
                    "output": {
                        "action_ids": action_ids,
                        "labels": labels,
                        "action_metadata": action_metadata,
                    },
                },
            }
        )
    return {
        "schema_version": "guideline2graph.fact_first_program.v1",
        "qid": qid,
        "graph_id": str(graph.get("graph_id") or "decisionkg.guideline2graph.unified"),
        "predicates": predicates,
        "clinical_assessment_specs": [
            {
                **dict(spec),
                "evidence_concept_ids": list(spec.get("evidence_concept_ids") or []),
                "source_evidence_ids": list(spec.get("source_evidence_ids") or []),
            }
            for spec in _records(graph.get("clinical_assessment_specs"))
        ],
        "rules": rules,
        "knowledge_relations": [],
    }


def _compile_graph(
    graph: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Guideline2GraphRuntimeV1]:
    program = _program_from_graph(graph)
    try:
        return program, Guideline2GraphRuntimeV1(program)
    except Guideline2GraphProgramError as exc:
        raise UnifiedSchemaV2RuntimeError(str(exc)) from exc


def dry_run_unified_graph(
    graph: Mapping[str, Any],
    patient_bundle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile and execute a graph without relying on its conversion status."""

    errors = []
    if patient_bundle is not None:
        stored_assessments = patient_bundle.get("clinical_assessments") or []
        if stored_assessments:
            errors.append(
                "patient_observations.clinical_assessments must be empty; Pass B supplies them at runtime"
            )
        observations = patient_bundle.get("observations")
        if not isinstance(observations, list) or not all(
            isinstance(item, Mapping) for item in observations
        ):
            errors.append("patient_observations.observations must be a list of objects")
            observations = []
        for observation in observations:
            if (
                observation.get("value") is None
                and str(observation.get("source_role") or "")
                == "not_asserted_by_question"
            ):
                errors.append(
                    "patient_observations.observations must not store an unreported null fact; "
                    "record it only in unknown_inputs"
                )
        clinical_assertions = patient_bundle.get("clinical_assertions") or []
        try:
            validate_clinical_assertions(clinical_assertions)
        except Guideline2GraphProgramError as exc:
            errors.append(str(exc))
    else:
        observations = []
        clinical_assertions = []

    trace = None
    try:
        _, runtime = _compile_graph(graph)
        if not errors:
            trace = runtime.execute(
                [
                    *_observation_facts(observations),
                    *_clinical_assertion_facts(clinical_assertions),
                ],
                [],
            )
    except (Guideline2GraphProgramError, UnifiedSchemaV2RuntimeError) as exc:
        errors.append(str(exc))
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "rule_count": len(graph.get("rules") or []),
        "action_count": len(graph.get("actions") or []),
        "trace": trace,
    }


class UnifiedSchemaV2Runtime:
    """Execute one converted Schema-v2 graph over observations and assessments."""

    def __init__(self, graph: Mapping[str, Any]):
        if graph.get("schema_version") != SCHEMA_VERSION:
            raise UnifiedSchemaV2RuntimeError(
                f"schema_version must equal {SCHEMA_VERSION}"
            )
        if str(graph.get("conversion_status") or "") != "converted":
            raise UnifiedSchemaV2RuntimeError("only converted graphs may enter runtime")
        self.graph = copy.deepcopy(dict(graph))
        self.program, self.runtime = _compile_graph(self.graph)

    def execute(
        self,
        observations: Sequence[Mapping[str, Any]],
        clinical_assessments: Optional[Sequence[Mapping[str, Any]]] = None,
        clinical_assertions: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        include_clinical_trace: bool = False,
    ) -> Dict[str, Any]:
        validate_clinical_assessments(clinical_assessments)
        validate_clinical_assertions(clinical_assertions)
        facts = [
            *_observation_facts(observations),
            *_clinical_assertion_facts(clinical_assertions or []),
        ]
        runtime_trace = self.runtime.execute(facts, clinical_assessments)
        assessment_authorities = sorted(
            {
                str(item.get("assessment_authority") or "")
                for item in clinical_assessments or []
                if str(item.get("assessment_authority") or "")
            }
        )
        if not assessment_authorities:
            assessment_authority = "none"
        elif len(assessment_authorities) == 1:
            assessment_authority = assessment_authorities[0]
        else:
            assessment_authority = "mixed"
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "target_schema_version": SCHEMA_VERSION,
            "qid": str(self.graph.get("source_question", {}).get("qid") or ""),
            "graph_instance_id": str(self.graph.get("graph_instance_id") or ""),
            "observations": copy.deepcopy(list(observations)),
            "clinical_assertions": copy.deepcopy(list(clinical_assertions or [])),
            "clinical_assessments": copy.deepcopy(list(clinical_assessments or [])),
            "predicate_results": runtime_trace["predicate_results"],
            "clinical_assessment_results": runtime_trace["clinical_assessment_results"],
            "condition_node_results": runtime_trace["condition_node_results"],
            "rule_results": runtime_trace["rule_results"],
            "actions": runtime_trace["actions"],
            "input_contract": {
                "question_observations_only": True,
                "clinical_assessment_authority": assessment_authority,
                "baseline_inputs_used": False,
                "oracle_or_answer_data_used": False,
            },
        }
        if include_clinical_trace:
            return attach_clinical_trace(trace, self.graph)
        return trace


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "UnifiedSchemaV2Runtime",
    "UnifiedSchemaV2RuntimeError",
    "dry_run_unified_graph",
]
