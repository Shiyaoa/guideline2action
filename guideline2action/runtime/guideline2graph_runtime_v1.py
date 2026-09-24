"""Deterministic fact-first runtime for Guideline2Graph-compatible programs.

The benchmark NS runtime consumes question-local input variables.  This module
instead executes the original Guideline2Graph contract:

question facts -> typed observation predicates -> condition DAG -> actions.

It intentionally has no patient baseline input and accepts no predicate/root
truth labels.  Missing observations propagate with Strong-Kleene semantics.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


PROGRAM_SCHEMA_VERSION = "guideline2graph.fact_first_program.v1"
OBSERVATION_DOCUMENT_SCHEMA_VERSION = "guideline2graph.standard_patient_observations.v1"
TRACE_SCHEMA_VERSION = "guideline2graph.fact_first_trace.v1"

FORBIDDEN_ONLINE_FIELDS = frozenset(
    {
        "baseline_inputs",
        "baseline_patient_state",
        "patient_baseline",
        "patient_state",
        "patient_truth",
        "predicate_truth",
        "root_truth",
        "oracle",
        "oracle_truth",
        "oracle_projection",
    }
)
SUPPORTED_DAG_NODE_TYPES = frozenset(
    {
        "predicate_ref",
        "clinical_assessment_ref",
        "literal",
        "compare",
        "combine",
        "exists",
    }
)
SUPPORTED_COMBINE_OPERATORS = frozenset({"AND", "ALL", "OR", "ANY", "AT_LEAST", "NOT"})
SUPPORTED_COMPARE_OPERATORS = frozenset(
    {"gt", ">", "ge", ">=", "lt", "<", "le", "<=", "eq", "==", "ne", "!="}
)
SUPPORTED_PREDICATE_COMPARE_OPERATORS = frozenset(
    {"contains_any", "eq", "equals", "==", "in", "gt", ">", "ge", ">=", "lt", "<", "le", "<=", "ne", "!="}
)
SUPPORTED_REDUCTIONS = frozenset({"none", "most_recent", "exists"})
SUPPORTED_CALCULATION_OPERATORS = frozenset(
    {
        "cockcroft_gault_normalized_creatinine_clearance",
        "quantity_difference_at_least",
        "quantity_multiply_by_factor",
        "relative_percentage_change",
    }
)
SUPPORTED_EXTRACT_PARSERS = frozenset(
    {"", "blood_pressure_systolic", "blood_pressure_diastolic"}
)
SUPPORTED_ACTION_PERMISSIONS = frozenset(
    {
        "allow",
        "recommend",
        "require",
        "caution",
        "avoid",
        "contraindicate",
        "continue",
        "stop",
        "consider",
        "reduce_dose",
        "increase_dose",
        "start_low_dose",
        "max_dose_limit",
        "titrate",
        "maintain_dose",
    }
)
SUPPORTED_PREDICATE_SOURCE_KINDS = frozenset(
    {
        "question_observation",
        "explicit_clinical_assertion",
        "clinical_assessment",
    }
)


class Guideline2GraphProgramError(ValueError):
    """Raised when a migrated graph is structurally invalid."""


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


def kleene_all(values: Iterable[Truth]) -> Truth:
    values = list(values)
    if any(value is Truth.FALSE for value in values):
        return Truth.FALSE
    if any(value is Truth.UNKNOWN for value in values):
        return Truth.UNKNOWN
    return Truth.TRUE


def kleene_any(values: Iterable[Truth]) -> Truth:
    values = list(values)
    if any(value is Truth.TRUE for value in values):
        return Truth.TRUE
    if any(value is Truth.UNKNOWN for value in values):
        return Truth.UNKNOWN
    return Truth.FALSE


def kleene_at_least(values: Iterable[Truth], minimum: int) -> Truth:
    values = list(values)
    true_count = sum(value is Truth.TRUE for value in values)
    unknown_count = sum(value is Truth.UNKNOWN for value in values)
    if true_count >= minimum:
        return Truth.TRUE
    if true_count + unknown_count < minimum:
        return Truth.FALSE
    return Truth.UNKNOWN


def kleene_not(value: Truth) -> Truth:
    if value is Truth.TRUE:
        return Truth.FALSE
    if value is Truth.FALSE:
        return Truth.TRUE
    return Truth.UNKNOWN


def _records(value: Any, field: str) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise Guideline2GraphProgramError(f"{field} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise Guideline2GraphProgramError(f"{field} must contain objects")
    return list(value)


def _is_forbidden_online_field(value: Any) -> bool:
    key = str(value or "").strip().casefold()
    return key in FORBIDDEN_ONLINE_FIELDS or key.startswith("oracle_")


def _forbidden_online_paths(value: Any, path: str = "$") -> List[str]:
    matches: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_forbidden_online_field(key):
                matches.append(child_path)
            matches.extend(_forbidden_online_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            matches.extend(_forbidden_online_paths(child, f"{path}[{index}]"))
    return matches


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text).split())


_UNIT_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "day": "day",
    "days": "day",
    "d": "day",
    "天": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "岁": "year",
    "umol/l": "umol/L",
    "μmol/l": "umol/L",
    "µmol/l": "umol/L",
    "10^9/l": "10^9/L",
    "ml/min": "ml/min",
    "mmhg": "mmHg",
    "ms": "ms",
    "cm": "cm",
    "fraction": "fraction",
    "c": "C",
    "°c": "C",
    "℃": "C",
}


def canonical_unit(value: Any) -> Optional[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return None
    return _UNIT_ALIASES.get(text.casefold(), text)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _quote_comparator(fact: Mapping[str, Any]) -> Optional[str]:
    supplied = str(fact.get("operator") or "").strip()
    if supplied in {">", ">=", "<", "<=", "=", "=="}:
        return supplied
    quote = unicodedata.normalize(
        "NFKC", str(fact.get("source_quote") or fact.get("quote") or "")
    )
    match = re.search(r"(?:^|\s)(>=|<=|>|<)\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)", quote)
    return match.group(1) if match else None


def _assertion_truth(fact: Mapping[str, Any]) -> bool:
    polarity = _normalize_text(fact.get("polarity"))
    value = _normalize_text(fact.get("value"))
    if polarity in {"negated", "absent", "negative"}:
        return False
    negative_values = {
        "absent",
        "excluded",
        "negative",
        "none",
        "no",
        "false",
        "无",
        "阴性",
        "未见",
        "排除",
    }
    return value not in negative_values


def _blood_pressure_component(value: Any, index: int) -> Optional[float]:
    if isinstance(value, Mapping):
        keys = ("systolic", "diastolic")
        candidate = value.get(keys[index])
        return float(candidate) if _is_number(candidate) else None
    numbers = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", str(value or ""))
    if len(numbers) <= index:
        return None
    return float(numbers[index])


def _binding_hints(predicate: Mapping[str, Any]) -> Mapping[str, Any]:
    data_binding = predicate.get("data_binding") or {}
    if not isinstance(data_binding, Mapping):
        return {}
    fhir = data_binding.get("FHIR") or data_binding.get("fhir") or {}
    return fhir if isinstance(fhir, Mapping) else {}


def _binding_section(predicate: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    data_binding = predicate.get("data_binding") or {}
    if not isinstance(data_binding, Mapping):
        return {}
    value = data_binding.get(name) or {}
    return value if isinstance(value, Mapping) else {}


def _predicate_source_kind(predicate: Mapping[str, Any]) -> str:
    configured = str(predicate.get("source_kind") or "").strip()
    if configured:
        return configured
    resource = _retrieve_resource(predicate)
    if resource == "clinicalassertion":
        return "explicit_clinical_assertion"
    if resource == "clinicalassessment":
        return "clinical_assessment"
    return "question_observation"


def _fact_source_kind(fact: Mapping[str, Any]) -> str:
    configured = str(fact.get("source_kind") or "").strip()
    if configured:
        return configured
    resource = str(fact.get("fact_resource") or "").strip().casefold()
    if resource == "clinicalassertion":
        return "explicit_clinical_assertion"
    if str(fact.get("observation_kind") or "").strip().casefold() == "clinical_assertion":
        return "explicit_clinical_assertion"
    return "question_observation"


def _retrieve_resource(predicate: Mapping[str, Any]) -> str:
    retrieve = predicate.get("retrieve") or {}
    if not isinstance(retrieve, Mapping):
        return ""
    return str(retrieve.get("resource") or "").strip().casefold()


def _uses_predicate_outputs(predicate: Mapping[str, Any]) -> bool:
    return _retrieve_resource(predicate) in {
        "predicateoutput",
        "dependencies",
        "predicateoutput/dependencies",
    } or bool(_binding_section(predicate, "PredicateOutput"))


def _predicate_dependency_ids(predicate: Mapping[str, Any]) -> List[str]:
    retrieve = predicate.get("retrieve") or {}
    binding = _binding_section(predicate, "PredicateOutput")
    candidates: List[Any] = []
    if isinstance(retrieve, Mapping):
        if isinstance(retrieve.get("predicates"), list):
            candidates.extend(retrieve["predicates"])
        if retrieve.get("predicate_id") is not None:
            candidates.append(retrieve["predicate_id"])
    if isinstance(binding.get("predicate_ids"), list):
        candidates.extend(binding["predicate_ids"])
    for field in ("predicate_id", "left_predicate_id", "right_predicate_id"):
        if binding.get(field) is not None:
            candidates.append(binding[field])
    dependencies = predicate.get("dependencies")
    if isinstance(dependencies, list):
        candidates.extend(dependencies)
    return list(dict.fromkeys(str(item) for item in candidates if str(item).strip()))


def _fact_aliases(predicate: Mapping[str, Any]) -> List[str]:
    hints = _binding_hints(predicate)
    binding_name = (
        "ClinicalAssertion"
        if _predicate_source_kind(predicate) == "explicit_clinical_assertion"
        else "QuestionObservation"
    )
    binding = _binding_section(predicate, binding_name)
    retrieve = predicate.get("retrieve") or {}
    candidates: List[Any] = []
    if isinstance(hints.get("concept_aliases"), list):
        candidates.extend(hints["concept_aliases"])
    if isinstance(retrieve, Mapping) and isinstance(retrieve.get("concepts"), list):
        candidates.extend(retrieve["concepts"])
    if isinstance(binding.get("concept_ids"), list):
        candidates.extend(binding["concept_ids"])
    for field in ("concept_id", "source_input_id", "input_id", "left_input_id"):
        if binding.get(field) is not None:
            candidates.append(binding[field])
    source_payload = predicate.get("source_payload") or {}
    if isinstance(source_payload, Mapping):
        for field in ("input_id", "left_input_id"):
            if source_payload.get(field) is not None:
                candidates.append(source_payload[field])
    return list(dict.fromkeys(_normalize_text(item) for item in candidates if str(item).strip()))


def _fact_identifiers(fact: Mapping[str, Any]) -> set[str]:
    return {
        _normalize_text(fact.get(field))
        for field in (
            "concept_id",
            "clinical_concept",
            "input_id",
            "source_input_id",
            "source_fact_id",
            "fact_id",
        )
        if str(fact.get(field) or "").strip()
    }


def _comparison_spec(predicate: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    comparison = predicate.get("compare")
    if not isinstance(comparison, Mapping):
        return None
    result = copy.deepcopy(dict(comparison))
    if result.get("value") is None and result.get("threshold") is not None:
        result["value"] = result.pop("threshold")
    if result.get("unit") is None:
        source_payload = predicate.get("source_payload") or {}
        expected_unit = predicate.get("expected_unit")
        if expected_unit is None and isinstance(source_payload, Mapping):
            expected_unit = source_payload.get("expected_unit")
        if expected_unit is not None:
            result["unit"] = expected_unit
    return result


def _calculation_spec(predicate: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    calculation = predicate.get("calculation")
    if not isinstance(calculation, Mapping):
        return None
    return calculation if str(calculation.get("operator") or "").strip() else None


def _allowed_time_scopes(predicate: Mapping[str, Any]) -> set[str]:
    hints = _binding_hints(predicate)
    retrieve = predicate.get("retrieve") or {}
    configured = hints.get("time_scopes")
    if not isinstance(configured, list) and isinstance(retrieve, Mapping):
        configured = retrieve.get("time_scopes")
    if isinstance(configured, list):
        return {_normalize_text(item) for item in configured if str(item).strip()}
    temporal = predicate.get("temporal_scope") or {}
    mode = str(temporal.get("mode") or "all_time") if isinstance(temporal, Mapping) else "all_time"
    if mode == "currently_active":
        return {"current"}
    return set()


def _fact_time_scope(fact: Mapping[str, Any]) -> str:
    temporal = fact.get("temporal") or {}
    relation = temporal.get("relation") if isinstance(temporal, Mapping) else None
    return _normalize_text(relation or fact.get("timepoint") or fact.get("time_scope"))


def _fact_time_scopes(fact: Mapping[str, Any]) -> set[str]:
    temporal = fact.get("temporal") or {}
    relation = temporal.get("relation") if isinstance(temporal, Mapping) else None
    return {
        _normalize_text(value)
        for value in (relation, fact.get("timepoint"), fact.get("time_scope"))
        if str(value or "").strip()
    }


def _dag_operand_references(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping) and value.get("id") is not None:
        return [str(value["id"])]
    return []


def _dag_cycles(adjacency: Mapping[str, Sequence[str]]) -> List[str]:
    state: Dict[str, int] = {}
    stack: List[str] = []
    cycles: set[str] = set()

    def visit(node_id: str) -> None:
        status = state.get(node_id, 0)
        if status == 2:
            return
        if status == 1:
            start = stack.index(node_id) if node_id in stack else 0
            cycles.add(" -> ".join([*stack[start:], node_id]))
            return
        state[node_id] = 1
        stack.append(node_id)
        for dependency in adjacency.get(node_id, ()):
            if dependency in adjacency:
                visit(dependency)
        stack.pop()
        state[node_id] = 2

    for node_id in adjacency:
        visit(node_id)
    return sorted(cycles)


def load_program(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise Guideline2GraphProgramError(f"{path}: program must be an object")
    program = dict(value)
    validate_program(program, source=str(path))
    return program


def load_observation_document(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise Guideline2GraphProgramError(
            f"{path}: observation document must be an object"
        )
    document = dict(value)
    if document.get("schema_version") != OBSERVATION_DOCUMENT_SCHEMA_VERSION:
        raise Guideline2GraphProgramError(
            f"{path}: schema_version must equal {OBSERVATION_DOCUMENT_SCHEMA_VERSION}"
        )
    forbidden_paths = _forbidden_online_paths(document)
    if forbidden_paths:
        raise Guideline2GraphProgramError(
            f"{path}: patient/oracle evaluation fields are forbidden: "
            + ", ".join(sorted(forbidden_paths))
        )
    if not str(document.get("qid") or "").strip():
        raise Guideline2GraphProgramError(f"{path}: qid is required")
    observations = document.get("observations")
    validate_facts(observations, source=f"{path}:observations")
    assertions = document.get("clinical_assertions") or []
    if not isinstance(assertions, list):
        raise Guideline2GraphProgramError(
            f"{path}:clinical_assertions must be a list"
        )
    validate_clinical_assertions(assertions, source=f"{path}:clinical_assertions")
    return document


def validate_facts(facts: Any, *, source: str = "facts") -> None:
    errors: List[str] = []
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)):
        raise Guideline2GraphProgramError(
            f"{source}: facts must be a sequence of structured observation records"
        )

    forbidden_paths = _forbidden_online_paths(facts)
    if forbidden_paths:
        errors.append(
            "patient/oracle evaluation fields are forbidden: "
            + ", ".join(sorted(forbidden_paths))
        )

    fact_ids: List[str] = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, Mapping):
            errors.append(f"facts[{index}] must be an object")
            continue
        fact_id = str(fact.get("fact_id") or "").strip()
        if not fact_id:
            errors.append(f"facts[{index}].fact_id is required")
        fact_ids.append(fact_id)
        concept_id = str(fact.get("concept_id") or "").strip()
        clinical_concept = str(fact.get("clinical_concept") or "").strip()
        if not concept_id and not clinical_concept:
            errors.append(
                f"facts[{index}] requires concept_id or clinical_concept"
            )
        temporal = fact.get("temporal")
        if temporal is not None and not isinstance(temporal, Mapping):
            errors.append(f"facts[{index}].temporal must be an object")

    nonempty_ids = [fact_id for fact_id in fact_ids if fact_id]
    if len(nonempty_ids) != len(set(nonempty_ids)):
        errors.append("fact ids must be unique")
    if errors:
        raise Guideline2GraphProgramError(f"{source}: " + "; ".join(errors))


def _validate_source_span(value: Any, *, source: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise Guideline2GraphProgramError(f"{source}: source_span must be an object")
    if not str(value.get("text") or value.get("source_text") or "").strip():
        raise Guideline2GraphProgramError(f"{source}: source_span.text is required")
    for field in ("start", "end"):
        if value.get(field) is not None and (
            not isinstance(value.get(field), int) or isinstance(value.get(field), bool)
        ):
            raise Guideline2GraphProgramError(f"{source}: source_span.{field} must be an integer")


def validate_clinical_assertions(assertions: Any, *, source: str = "clinical_assertions") -> None:
    if assertions is None:
        return
    records = _records(assertions, source)
    errors: List[str] = []
    ids: List[str] = []
    concepts: List[str] = []
    for index, assertion in enumerate(records):
        assertion_id = str(assertion.get("assertion_id") or "").strip()
        concept_id = str(
            assertion.get("concept_id")
            or assertion.get("condition_concept_id")
            or ""
        ).strip()
        if not assertion_id:
            errors.append(f"{source}[{index}].assertion_id is required")
        if not concept_id:
            errors.append(f"{source}[{index}].concept_id is required")
        if "value" not in assertion:
            errors.append(f"{source}[{index}].value is required")
        if str(assertion.get("asserted_by") or "") not in {
            "question_text",
            "clinician",
            "source_record",
        }:
            errors.append(
                f"{source}[{index}].asserted_by must be question_text, clinician, or source_record"
            )
        try:
            _validate_source_span(assertion.get("source_span"), source=f"{source}[{index}]")
        except Guideline2GraphProgramError as exc:
            errors.append(str(exc))
        ids.append(assertion_id)
        concepts.append(concept_id)
    if len([item for item in ids if item]) != len(set(item for item in ids if item)):
        errors.append(f"{source}: assertion_id values must be unique")
    if len([item for item in concepts if item]) != len(set(item for item in concepts if item)):
        errors.append(f"{source}: condition_concept_id values must be unique")
    forbidden_paths = _forbidden_online_paths(assertions)
    if forbidden_paths:
        errors.append(
            "patient/oracle evaluation fields are forbidden: "
            + ", ".join(sorted(forbidden_paths))
        )
    if errors:
        raise Guideline2GraphProgramError("; ".join(errors))


def validate_clinical_assessments(
    assessments: Any, *, source: str = "clinical_assessments"
) -> None:
    if assessments is None:
        return
    records = _records(assessments, source)
    errors: List[str] = []
    ids: List[str] = []
    specs: List[str] = []
    allowed_values = {"true", "false", "unknown"}
    for index, assessment in enumerate(records):
        assessment_id = str(assessment.get("assessment_id") or "").strip()
        spec_id = str(assessment.get("assessment_spec_id") or "").strip()
        value = str(assessment.get("value") or "").strip().lower()
        if not assessment_id:
            errors.append(f"{source}[{index}].assessment_id is required")
        if not spec_id:
            errors.append(f"{source}[{index}].assessment_spec_id is required")
        if value not in allowed_values:
            errors.append(f"{source}[{index}].value must be true, false, or unknown")
        authority = str(assessment.get("assessment_authority") or "").strip()
        if authority not in {"llm", "clinical_expert"}:
            errors.append(
                f"{source}[{index}].assessment_authority must be llm or clinical_expert"
            )
        for field in ("supporting_observation_ids", "source_spans"):
            if not isinstance(assessment.get(field), list):
                errors.append(f"{source}[{index}].{field} must be a list")
        if not isinstance(assessment.get("requires_human_review"), bool):
            errors.append(f"{source}[{index}].requires_human_review must be a boolean")
        for span_index, span in enumerate(assessment.get("source_spans") or []):
            try:
                _validate_source_span(
                    span,
                    source=f"{source}[{index}].source_spans[{span_index}]",
                )
            except Guideline2GraphProgramError as exc:
                errors.append(str(exc))
        if authority == "llm":
            if not str(assessment.get("model_id") or "").strip():
                errors.append(f"{source}[{index}].model_id is required for llm assessment")
            if not str(assessment.get("prompt_version") or "").strip():
                errors.append(
                    f"{source}[{index}].prompt_version is required for llm assessment"
                )
        elif authority == "clinical_expert":
            if not str(assessment.get("reviewer_id") or "").strip():
                errors.append(
                    f"{source}[{index}].reviewer_id is required for clinical_expert assessment"
                )
            if not str(assessment.get("reviewed_at") or "").strip():
                errors.append(
                    f"{source}[{index}].reviewed_at is required for clinical_expert assessment"
                )
        ids.append(assessment_id)
        specs.append(spec_id)
    if len([item for item in ids if item]) != len(set(item for item in ids if item)):
        errors.append(f"{source}: assessment_id values must be unique")
    if len([item for item in specs if item]) != len(set(item for item in specs if item)):
        errors.append(f"{source}: assessment_spec_id values must be unique")
    forbidden_paths = _forbidden_online_paths(assessments)
    if forbidden_paths:
        errors.append(
            "patient/oracle evaluation fields are forbidden: "
            + ", ".join(sorted(forbidden_paths))
        )
    if errors:
        raise Guideline2GraphProgramError("; ".join(errors))


def validate_program(program: Mapping[str, Any], *, source: str = "program") -> None:
    errors: List[str] = []
    if program.get("schema_version") != PROGRAM_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {PROGRAM_SCHEMA_VERSION}")
    qid = str(program.get("qid") or "")
    if not qid:
        errors.append("qid is required")
    forbidden_paths = _forbidden_online_paths(program)
    if forbidden_paths:
        errors.append(
            "patient/oracle evaluation fields are forbidden: "
            + ", ".join(sorted(forbidden_paths))
        )

    predicates = _records(program.get("predicates"), "predicates")
    predicate_ids = [str(item.get("id") or "") for item in predicates]
    if any(not item for item in predicate_ids):
        errors.append("every predicate requires id")
    if len(predicate_ids) != len(set(predicate_ids)):
        errors.append("predicate ids must be unique")
    required_predicate_fields = {
        "entity",
        "entity_type",
        "aspect",
        "input_shape",
        "reduction",
        "return_type",
        "final_output_type",
        "temporal_scope",
        "retrieve",
        "extract",
        "null_policy",
    }
    for predicate in predicates:
        predicate_id = str(predicate.get("id") or "<missing>")
        missing = sorted(required_predicate_fields - set(predicate))
        if missing:
            errors.append(f"{predicate_id}: missing typed fields {missing}")
        if not _uses_predicate_outputs(predicate) and not _fact_aliases(predicate):
            errors.append(f"{predicate_id}: no executable fact concept binding")
        if predicate.get("null_policy") not in {"unknown", "propagate_unknown"}:
            errors.append(f"{predicate_id}: fact-first prototype requires unknown null policy")
        source_kind = str(predicate.get("source_kind") or "question_observation")
        if source_kind not in SUPPORTED_PREDICATE_SOURCE_KINDS:
            errors.append(
                f"{predicate_id}: unsupported source_kind {source_kind!r}"
            )
        if str(predicate.get("aspect") or "") == "clinical_assertion":
            if not str(predicate.get("assertion_concept_id") or "").strip():
                errors.append(
                    f"{predicate_id}: clinical_assertion predicates require assertion_concept_id"
                )
            if not str(predicate.get("asserted_by") or "").strip():
                errors.append(
                    f"{predicate_id}: clinical_assertion predicates require asserted_by"
                )
        reduction = predicate.get("reduction") or {}
        if not isinstance(reduction, Mapping):
            errors.append(f"{predicate_id}: reduction must be an object")
        else:
            reduction_operator = str(reduction.get("operator") or "none")
            if reduction_operator not in SUPPORTED_REDUCTIONS:
                errors.append(
                    f"{predicate_id}: unsupported reduction {reduction_operator!r}"
                )
        extract = predicate.get("extract") or {}
        if not isinstance(extract, Mapping):
            errors.append(f"{predicate_id}: extract must be an object")
        else:
            parser = str(extract.get("parser") or "")
            if parser not in SUPPORTED_EXTRACT_PARSERS:
                errors.append(f"{predicate_id}: unsupported extract parser {parser!r}")
        predicate_compare = _comparison_spec(predicate)
        if predicate_compare is not None:
            if not isinstance(predicate_compare, Mapping):
                errors.append(f"{predicate_id}: compare must be an object or null")
            else:
                compare_operator = str(
                    predicate_compare.get("operator") or ""
                ).strip().lower()
                if compare_operator not in SUPPORTED_PREDICATE_COMPARE_OPERATORS:
                    errors.append(
                        f"{predicate_id}: unsupported predicate comparison "
                        f"{compare_operator!r}"
                    )
        calculation = _calculation_spec(predicate)
        if calculation is not None:
            calculation_operator = str(calculation.get("operator") or "")
            if calculation_operator not in SUPPORTED_CALCULATION_OPERATORS:
                errors.append(
                    f"{predicate_id}: unsupported calculation operator "
                    f"{calculation_operator!r}"
                )
            elif calculation_operator == "quantity_difference_at_least":
                for field in ("minuend_input_ref", "subtrahend_input_ref"):
                    if not str(calculation.get(field) or "").strip():
                        errors.append(
                            f"{predicate_id}: {calculation_operator} requires {field}"
                        )
                comparison = calculation.get("comparison") or predicate_compare
                if not isinstance(comparison, Mapping):
                    errors.append(
                        f"{predicate_id}: {calculation_operator} requires comparison"
                    )
            elif calculation_operator == "quantity_multiply_by_factor":
                input_ref = str(calculation.get("input_ref") or "").strip()
                if not input_ref:
                    errors.append(
                        f"{predicate_id}: {calculation_operator} requires input_ref"
                    )
                elif input_ref not in {
                    str(item) for item in predicate.get("dependencies") or []
                }:
                    errors.append(
                        f"{predicate_id}: {calculation_operator} input_ref must be a dependency"
                    )
                if not _is_number(calculation.get("factor")):
                    errors.append(
                        f"{predicate_id}: {calculation_operator} requires numeric factor"
                    )
                if not canonical_unit(calculation.get("result_unit")):
                    errors.append(
                        f"{predicate_id}: {calculation_operator} requires result_unit"
                    )
            elif calculation_operator == "relative_percentage_change":
                for field in ("baseline_input_ref", "current_input_ref"):
                    if not str(calculation.get(field) or "").strip():
                        errors.append(
                            f"{predicate_id}: {calculation_operator} requires {field}"
                        )
                comparison = calculation.get("comparison") or predicate_compare
                if not isinstance(comparison, Mapping):
                    errors.append(
                        f"{predicate_id}: {calculation_operator} requires comparison"
                    )
                if canonical_unit(calculation.get("result_unit")) != "percent":
                    errors.append(
                        f"{predicate_id}: {calculation_operator} result_unit must be percent"
                    )
            elif (
                calculation_operator
                == "cockcroft_gault_normalized_creatinine_clearance"
            ):
                for field in (
                    "age_input_ref",
                    "sex_input_ref",
                    "body_weight_input_ref",
                    "serum_creatinine_input_ref",
                ):
                    if not str(calculation.get(field) or "").strip():
                        errors.append(
                            f"{predicate_id}: {calculation_operator} requires {field}"
                        )
                for field in ("female_factor", "constant"):
                    if not _is_number(calculation.get(field)):
                        errors.append(
                            f"{predicate_id}: {calculation_operator} requires numeric {field}"
                        )
                if not canonical_unit(calculation.get("result_unit")):
                    errors.append(
                        f"{predicate_id}: {calculation_operator} requires result_unit"
                    )

    knowledge_relations = _records(
        program.get("knowledge_relations") or [], "knowledge_relations"
    )
    for index, relation in enumerate(knowledge_relations):
        missing_relation_fields = [
            field
            for field in ("subject", "relation", "object")
            if not str(relation.get(field) or "").strip()
        ]
        if missing_relation_fields:
            errors.append(
                f"knowledge_relations[{index}]: missing {missing_relation_fields}"
            )

    rules = _records(program.get("rules"), "rules")
    rule_ids = [str(item.get("id") or "") for item in rules]
    if any(not item for item in rule_ids) or len(rule_ids) != len(set(rule_ids)):
        errors.append("rule ids must be present and unique")
    for rule in rules:
        rule_id = str(rule.get("id") or "<missing>")
        dag = rule.get("condition_dag") or {}
        if not isinstance(dag, Mapping):
            errors.append(f"{rule_id}: condition_dag must be an object")
            continue
        nodes = dag.get("nodes") or []
        if not isinstance(nodes, list) or not nodes:
            errors.append(f"{rule_id}: condition_dag.nodes must be non-empty")
            continue
        if not all(isinstance(item, Mapping) for item in nodes):
            errors.append(f"{rule_id}: condition_dag.nodes must contain objects")
            continue
        node_records = [dict(item) for item in nodes]
        node_ids = [str(item.get("id") or "") for item in node_records]
        if len(node_ids) != len(nodes) or any(not item for item in node_ids):
            errors.append(f"{rule_id}: every DAG node requires id")
        if len(node_ids) != len(set(node_ids)):
            errors.append(f"{rule_id}: DAG node ids must be unique")
        node_by_id = {
            str(item.get("id") or ""): item
            for item in node_records
            if str(item.get("id") or "")
        }
        root = str(dag.get("root") or "")
        if root not in node_ids:
            errors.append(f"{rule_id}: DAG root {root!r} is not a node")
        raw_input_predicates = rule.get("input_predicates") or []
        if not isinstance(raw_input_predicates, list):
            errors.append(f"{rule_id}: input_predicates must be a list")
            raw_input_predicates = []
        input_predicates = {str(item) for item in raw_input_predicates}
        unknown_predicates = sorted(input_predicates - set(predicate_ids))
        if unknown_predicates:
            errors.append(f"{rule_id}: unknown input predicates {unknown_predicates}")

        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
        for node in node_records:
            node_id = str(node.get("id") or "<missing>")
            node_type = str(node.get("type") or "")
            return_type = str(node.get("return_type") or "")
            if node_type not in SUPPORTED_DAG_NODE_TYPES:
                errors.append(f"{rule_id}/{node_id}: unsupported DAG node type {node_type!r}")
                continue
            if not return_type:
                errors.append(f"{rule_id}/{node_id}: return_type is required")

            references: List[str] = []
            if node_type == "predicate_ref":
                predicate_ref = str(node.get("predicate_ref") or "")
                if predicate_ref not in predicate_ids:
                    errors.append(
                        f"{rule_id}/{node_id}: unknown predicate_ref {predicate_ref!r}"
                    )
                elif predicate_ref not in input_predicates:
                    errors.append(
                        f"{rule_id}/{node_id}: predicate_ref {predicate_ref!r} is not "
                        "declared in input_predicates"
                    )
                else:
                    predicate = predicates[predicate_ids.index(predicate_ref)]
                    expected_type = str(predicate.get("final_output_type") or "")
                    if expected_type and return_type != expected_type:
                        errors.append(
                            f"{rule_id}/{node_id}: return_type {return_type!r} differs "
                            f"from predicate {predicate_ref!r} type {expected_type!r}"
                        )
            elif node_type == "clinical_assessment_ref":
                assessment_spec_id = str(node.get("assessment_spec_id") or "")
                if not assessment_spec_id:
                    errors.append(
                        f"{rule_id}/{node_id}: clinical_assessment_ref requires assessment_spec_id"
                    )
                if return_type not in {"Bool", "TruthValue"}:
                    errors.append(
                        f"{rule_id}/{node_id}: clinical_assessment_ref must return Bool or TruthValue"
                    )
            elif node_type == "literal":
                pass
            elif node_type == "compare":
                operator = str(node.get("operator") or "").strip().lower()
                if operator not in SUPPORTED_COMPARE_OPERATORS:
                    errors.append(
                        f"{rule_id}/{node_id}: unsupported compare operator {operator!r}"
                    )
                if node.get("left") is None or node.get("right") is None:
                    errors.append(
                        f"{rule_id}/{node_id}: compare requires left and right operands"
                    )
                references = [
                    *_dag_operand_references(node.get("left")),
                    *_dag_operand_references(node.get("right")),
                ]
                if return_type not in {"Bool", "TruthValue"}:
                    errors.append(f"{rule_id}/{node_id}: compare must return Bool or TruthValue")
            elif node_type == "combine":
                operator = str(node.get("operator") or "").strip().upper()
                raw_inputs = node.get("inputs") or []
                if not isinstance(raw_inputs, list) or not raw_inputs:
                    errors.append(
                        f"{rule_id}/{node_id}: combine requires non-empty inputs"
                    )
                    raw_inputs = []
                if not all(isinstance(item, str) and item for item in raw_inputs):
                    errors.append(
                        f"{rule_id}/{node_id}: combine inputs must be node ids"
                    )
                references = [str(item) for item in raw_inputs]
                if operator not in SUPPORTED_COMBINE_OPERATORS:
                    errors.append(
                        f"{rule_id}/{node_id}: unsupported combine operator {operator!r}"
                    )
                if operator == "AT_LEAST":
                    parameters = node.get("parameters") or {}
                    minimum = (
                        parameters.get("minimum")
                        if isinstance(parameters, Mapping)
                        else None
                    )
                    if (
                        not isinstance(minimum, int)
                        or isinstance(minimum, bool)
                        or minimum < 1
                        or minimum > len(raw_inputs)
                    ):
                        errors.append(
                            f"{rule_id}/{node_id}: AT_LEAST minimum must be an "
                            "integer from 1 through the input count"
                        )
                if operator == "NOT" and len(raw_inputs) != 1:
                    errors.append(f"{rule_id}/{node_id}: NOT requires exactly one input")
                if return_type not in {"Bool", "TruthValue"}:
                    errors.append(f"{rule_id}/{node_id}: combine must return Bool or TruthValue")
            elif node_type == "exists":
                specification = node.get("input") or {}
                relation = (
                    specification.get("knowledge_relation")
                    if isinstance(specification, Mapping)
                    else None
                )
                if not isinstance(relation, Mapping) or any(
                    not str(relation.get(field) or "").strip()
                    for field in ("subject", "relation", "object")
                ):
                    errors.append(
                        f"{rule_id}/{node_id}: exists currently requires a complete "
                        "knowledge_relation input"
                    )
                if return_type not in {"Bool", "TruthValue"}:
                    errors.append(f"{rule_id}/{node_id}: exists must return Bool or TruthValue")

            adjacency[node_id] = references

        for node_id, references in adjacency.items():
            missing_references = sorted(set(references) - set(node_ids))
            if missing_references:
                errors.append(
                    f"{rule_id}/{node_id}: unresolved DAG node references "
                    f"{missing_references}"
                )
        for cycle in _dag_cycles(adjacency):
            errors.append(f"{rule_id}: condition DAG cycle: {cycle}")

        if root in node_by_id and str(node_by_id[root].get("return_type") or "") not in {"Bool", "TruthValue"}:
            errors.append(f"{rule_id}: DAG root {root!r} must return Bool or TruthValue")
        boolean_root = str(rule.get("boolean_root") or root)
        if boolean_root == "ROOT" and boolean_root not in node_by_id:
            boolean_root = root
        if boolean_root not in node_by_id:
            errors.append(
                f"{rule_id}: boolean_root {boolean_root!r} is not a DAG node"
            )
        elif str(node_by_id[boolean_root].get("return_type") or "") not in {"Bool", "TruthValue"}:
            errors.append(f"{rule_id}: boolean_root {boolean_root!r} must return Bool or TruthValue")
        if boolean_root != root:
            errors.append(
                f"{rule_id}: boolean_root must equal condition_dag.root for execution"
            )

        typed_intermediates = dag.get("typed_intermediates") or {}
        if not isinstance(typed_intermediates, Mapping):
            errors.append(f"{rule_id}: typed_intermediates must be an object")
        else:
            for node_id, declared_type in typed_intermediates.items():
                node = node_by_id.get(str(node_id))
                if node is None:
                    errors.append(
                        f"{rule_id}: typed_intermediate {node_id!r} is not a DAG node"
                    )
                elif str(node.get("return_type") or "") != str(declared_type):
                    errors.append(
                        f"{rule_id}: typed_intermediate {node_id!r} type differs "
                        "from its DAG node"
                    )

        missing_policy = str(rule.get("missing_data_policy") or "propagate_unknown")
        if missing_policy != "propagate_unknown":
            errors.append(
                f"{rule_id}: runtime supports only propagate_unknown missing data policy"
            )
        action = rule.get("action") or {}
        if not isinstance(action, Mapping):
            errors.append(f"{rule_id}: action must be an object")
            action = {}
        permission = str(action.get("permission") or "")
        if permission not in SUPPORTED_ACTION_PERMISSIONS:
            errors.append(f"{rule_id}: unsupported action permission {permission!r}")
        output = action.get("output") or {}
        emitted = output.get("action_ids") if isinstance(output, Mapping) else None
        if not isinstance(emitted, list) or not all(
            isinstance(item, str) and item for item in emitted
        ):
            errors.append(
                f"{rule_id}: action.output.action_ids must be a list of non-empty strings"
            )
        elif len(emitted) != len(set(emitted)):
            errors.append(f"{rule_id}: action.output.action_ids must be unique")
        metadata = output.get("action_metadata") if isinstance(output, Mapping) else None
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                errors.append(f"{rule_id}: action.output.action_metadata must be an object")
            else:
                for action_id, action_metadata in metadata.items():
                    if action_id not in (emitted or []):
                        errors.append(
                            f"{rule_id}: action metadata references unlisted action {action_id!r}"
                        )
                    if not isinstance(action_metadata, Mapping):
                        errors.append(
                            f"{rule_id}: action metadata for {action_id!r} must be an object"
                        )
                        continue
                    action_permission = str(action_metadata.get("permission") or permission)
                    if action_permission not in SUPPORTED_ACTION_PERMISSIONS:
                        errors.append(
                            f"{rule_id}: unsupported permission {action_permission!r} for {action_id!r}"
                        )

    assessment_specs = _records(
        program.get("clinical_assessment_specs") or [], "clinical_assessment_specs"
    )
    assessment_spec_ids: List[str] = []
    for index, spec in enumerate(assessment_specs):
        spec_id = str(spec.get("assessment_spec_id") or "").strip()
        if not spec_id:
            errors.append(f"clinical_assessment_specs[{index}].assessment_spec_id is required")
        if not str(spec.get("condition_concept_id") or "").strip():
            errors.append(
                f"clinical_assessment_specs[{index}].condition_concept_id is required"
            )
        if not str(spec.get("question") or "").strip():
            errors.append(f"clinical_assessment_specs[{index}].question is required")
        allowed = spec.get("allowed_values")
        if not isinstance(allowed, list) or set(str(item).lower() for item in allowed) != {
            "true",
            "false",
            "unknown",
        }:
            errors.append(
                f"clinical_assessment_specs[{index}].allowed_values must contain true, false, unknown"
            )
        if not isinstance(spec.get("evidence_concept_ids"), list):
            errors.append(
                f"clinical_assessment_specs[{index}].evidence_concept_ids must be a list"
            )
        if not isinstance(spec.get("source_evidence_ids"), list):
            errors.append(
                f"clinical_assessment_specs[{index}].source_evidence_ids must be a list"
            )
        assessment_spec_ids.append(spec_id)
    if len([item for item in assessment_spec_ids if item]) != len(
        set(item for item in assessment_spec_ids if item)
    ):
        errors.append("clinical_assessment_specs: assessment_spec_id values must be unique")
    for rule in rules:
        rule_id = str(rule.get("id") or "<missing>")
        for node in (rule.get("condition_dag") or {}).get("nodes") or []:
            if str(node.get("type") or "") == "clinical_assessment_ref":
                spec_id = str(node.get("assessment_spec_id") or "")
                if spec_id not in set(assessment_spec_ids):
                    errors.append(
                        f"{rule_id}/{node.get('id')}: unknown assessment_spec_id {spec_id!r}"
                    )
        input_assessments = rule.get("input_assessments")
        if input_assessments is not None and not isinstance(input_assessments, list):
            errors.append(f"{rule_id}: input_assessments must be a list")

    if errors:
        raise Guideline2GraphProgramError(f"{source}: " + "; ".join(errors))


def validate_with_original_models(program: Mapping[str, Any]) -> Dict[str, Any]:
    """Cross-check a program against the extraction layer's Pydantic models.

    The runtime is self-contained; this optional check proves that projected
    JSON also satisfies the extraction contract shipped in this package.
    """

    try:
        from guideline2action.graph.models import ClinicalRule, ConditionDAG, Predicates
    except Exception as exc:  # pragma: no cover - optional cross-check
        return {"available": False, "valid": None, "reason": f"extraction_models_unavailable: {exc}"}

    try:
        for predicate in program.get("predicates") or []:
            Predicates.model_validate(predicate)
        for rule in program.get("rules") or []:
            ConditionDAG.model_validate(rule.get("condition_dag"))
            ClinicalRule.model_validate(rule)
        return {
            "available": True,
            "valid": True,
            "predicate_count": len(program.get("predicates") or []),
            "rule_count": len(program.get("rules") or []),
        }
    except Exception as exc:
        return {
            "available": True,
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


@dataclass(frozen=True)
class _RuntimeValue:
    status: str
    value: Any = None
    unit: Optional[str] = None
    comparator: Optional[str] = None
    truth: Optional[Truth] = None
    fact_ids: tuple[str, ...] = ()
    issue: str = ""

    @classmethod
    def unknown(cls, issue: str, *, fact_ids: Sequence[str] = ()) -> "_RuntimeValue":
        return cls(
            status="unknown",
            truth=Truth.UNKNOWN,
            fact_ids=tuple(fact_ids),
            issue=issue,
        )

    @classmethod
    def boolean(
        cls,
        truth: Truth,
        *,
        value: Any = None,
        fact_ids: Sequence[str] = (),
        issue: str = "",
    ) -> "_RuntimeValue":
        return cls(
            status="known" if truth is not Truth.UNKNOWN else "unknown",
            value=value,
            truth=truth,
            fact_ids=tuple(fact_ids),
            issue=issue,
        )

    def as_dict(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "status": self.status,
            "value": copy.deepcopy(self.value),
            "unit": self.unit,
            "comparator": self.comparator,
            "fact_ids": list(self.fact_ids),
            "issue": self.issue,
        }
        if self.truth is not None:
            output["truth"] = self.truth.value
        return output


def _compare_text(value: Any, specification: Mapping[str, Any]) -> Truth:
    operator = str(specification.get("operator") or "").strip().lower()
    range_spec = specification.get("range")
    if isinstance(range_spec, Mapping):
        if not _is_number(value):
            return Truth.UNKNOWN
        lower = range_spec.get("lower")
        upper = range_spec.get("upper")
        if lower is not None and not _is_number(lower):
            return Truth.UNKNOWN
        if upper is not None and not _is_number(upper):
            return Truth.UNKNOWN
        inclusive = bool(range_spec.get("inclusive", True))
        numeric = float(value)
        lower_ok = lower is None or (numeric >= float(lower) if inclusive else numeric > float(lower))
        upper_ok = upper is None or (numeric <= float(upper) if inclusive else numeric < float(upper))
        return Truth.TRUE if lower_ok and upper_ok else Truth.FALSE
    normalized = _normalize_text(value)
    expected = specification.get("value")
    values = expected if isinstance(expected, list) else [expected]
    normalized_values = [_normalize_text(item) for item in values if item is not None]
    if operator in {"ne", "!="}:
        if len(normalized_values) != 1:
            return Truth.UNKNOWN
        if _is_number(value) and _is_number(values[0]):
            return Truth.TRUE if float(value) != float(values[0]) else Truth.FALSE
        return Truth.TRUE if normalized != normalized_values[0] else Truth.FALSE
    if operator in {"gt", ">", "ge", ">=", "lt", "<", "le", "<="}:
        if len(values) != 1 or not _is_number(value) or not _is_number(values[0]):
            return Truth.UNKNOWN
        left = float(value)
        right = float(values[0])
        operations = {
            "gt": left > right,
            ">": left > right,
            "ge": left >= right,
            ">=": left >= right,
            "lt": left < right,
            "<": left < right,
            "le": left <= right,
            "<=": left <= right,
        }
        return Truth.TRUE if operations[operator] else Truth.FALSE
    if operator == "contains_any":
        return Truth.TRUE if any(item and item in normalized for item in normalized_values) else Truth.FALSE
    if operator in {"eq", "equals", "=="}:
        if len(values) == 1 and _is_number(value) and _is_number(values[0]):
            return Truth.TRUE if float(value) == float(values[0]) else Truth.FALSE
        return Truth.TRUE if normalized in normalized_values else Truth.FALSE
    if operator == "in":
        return Truth.TRUE if normalized in normalized_values else Truth.FALSE
    raise Guideline2GraphProgramError(f"unsupported predicate text comparison {operator!r}")


def _compare_values(left: _RuntimeValue, right: _RuntimeValue, operator: str) -> Truth:
    if left.status != "known" or right.status != "known":
        return Truth.UNKNOWN
    operator = operator.strip().lower()
    if not _is_number(left.value) or not _is_number(right.value):
        if operator in {"eq", "=="}:
            return (
                Truth.TRUE
                if _normalize_text(left.value) == _normalize_text(right.value)
                else Truth.FALSE
            )
        if operator in {"ne", "!="}:
            return (
                Truth.TRUE
                if _normalize_text(left.value) != _normalize_text(right.value)
                else Truth.FALSE
            )
        return Truth.UNKNOWN
    left_unit = canonical_unit(left.unit)
    right_unit = canonical_unit(right.unit)
    if left_unit and right_unit and left_unit != right_unit:
        return Truth.UNKNOWN
    if (left_unit is None) != (right_unit is None):
        # Zero is invariant under positive scale conversions.  This covers the
        # common narrative shorthand "ANC 0" without inventing a non-zero unit.
        non_unit_value = left.value if left_unit is None else right.value
        if float(non_unit_value) != 0.0:
            return Truth.UNKNOWN

    left_value = float(left.value)
    right_value = float(right.value)
    comparator = left.comparator
    if comparator and comparator not in {"=", "=="}:
        # Interpret a bound such as ">5%" as an interval, not as the scalar 5.
        if comparator == ">" and operator in {"gt", ">", "ge", ">="}:
            if left_value >= right_value:
                return Truth.TRUE
            return Truth.UNKNOWN
        if comparator == ">=" and operator in {"gt", ">", "ge", ">="}:
            if left_value > right_value or (left_value == right_value and operator in {"ge", ">="}):
                return Truth.TRUE
            return Truth.UNKNOWN
        if comparator == "<" and operator in {"lt", "<", "le", "<="}:
            if left_value <= right_value:
                return Truth.TRUE
            return Truth.UNKNOWN
        if comparator == "<=" and operator in {"lt", "<", "le", "<="}:
            if left_value < right_value or (left_value == right_value and operator in {"le", "<="}):
                return Truth.TRUE
            return Truth.UNKNOWN
        return Truth.UNKNOWN

    operations = {
        "gt": left_value > right_value,
        ">": left_value > right_value,
        "ge": left_value >= right_value,
        ">=": left_value >= right_value,
        "lt": left_value < right_value,
        "<": left_value < right_value,
        "le": left_value <= right_value,
        "<=": left_value <= right_value,
        "eq": left_value == right_value,
        "==": left_value == right_value,
        "ne": left_value != right_value,
        "!=": left_value != right_value,
    }
    if operator not in operations:
        raise Guideline2GraphProgramError(f"unsupported numeric comparison {operator!r}")
    return Truth.TRUE if operations[operator] else Truth.FALSE


def _apply_conversion(
    value: _RuntimeValue,
    specification: Any,
) -> _RuntimeValue:
    if value.status != "known":
        return value
    if not isinstance(specification, Mapping):
        return _RuntimeValue.unknown("conversion_spec_missing", fact_ids=value.fact_ids)
    method = str(specification.get("method") or "").strip().casefold()
    factor = specification.get("factor")
    from_unit = canonical_unit(specification.get("from_unit"))
    to_unit = canonical_unit(specification.get("to_unit"))
    if method not in {"multiply", "multiplicative_factor", "multiply_by_factor"}:
        return _RuntimeValue.unknown("conversion_method_unsupported", fact_ids=value.fact_ids)
    if not _is_number(factor) or float(factor) <= 0:
        return _RuntimeValue.unknown("conversion_factor_invalid", fact_ids=value.fact_ids)
    if not from_unit or not to_unit:
        return _RuntimeValue.unknown("conversion_unit_missing", fact_ids=value.fact_ids)
    if canonical_unit(value.unit) != from_unit:
        return _RuntimeValue.unknown("conversion_unit_mismatch", fact_ids=value.fact_ids)
    if not _is_number(value.value):
        return _RuntimeValue.unknown("conversion_value_non_numeric", fact_ids=value.fact_ids)
    return _RuntimeValue(
        status="known",
        value=float(value.value) * float(factor),
        unit=to_unit,
        comparator=value.comparator,
        fact_ids=value.fact_ids,
    )


def _compare_runtime_value(
    value: _RuntimeValue,
    comparison: Mapping[str, Any],
) -> _RuntimeValue:
    if value.status != "known":
        return value
    expected_unit = canonical_unit(comparison.get("unit"))
    actual_unit = canonical_unit(value.unit)
    if expected_unit and actual_unit != expected_unit:
        return _RuntimeValue.unknown(
            "comparison_unit_mismatch",
            fact_ids=value.fact_ids,
        )
    operator = str(comparison.get("operator") or "").strip().lower()
    threshold = comparison.get("value")
    if value.comparator and _is_number(value.value) and _is_number(threshold):
        truth = _compare_values(
            value,
            _RuntimeValue(status="known", value=threshold, unit=actual_unit),
            operator,
        )
    else:
        truth = _compare_text(value.value, comparison)
    return _RuntimeValue.boolean(
        truth,
        value=copy.deepcopy(value.value),
        fact_ids=value.fact_ids,
        issue="" if truth is not Truth.UNKNOWN else "comparison_not_evaluable",
    )


def _compare_runtime_pair(
    left: _RuntimeValue,
    right: _RuntimeValue,
    comparison: Mapping[str, Any],
    *,
    left_time_scope: str = "",
    right_time_scope: str = "",
) -> _RuntimeValue:
    fact_ids = [*left.fact_ids, *right.fact_ids]
    if left.status != "known":
        return _RuntimeValue.unknown(left.issue or "left_input_unknown", fact_ids=fact_ids)
    if right.status != "known":
        return _RuntimeValue.unknown(right.issue or "right_input_unknown", fact_ids=fact_ids)
    expected_unit = canonical_unit(comparison.get("unit"))
    left_unit = canonical_unit(left.unit)
    right_unit = canonical_unit(right.unit)
    if expected_unit and (left_unit != expected_unit or right_unit != expected_unit):
        return _RuntimeValue.unknown("comparison_unit_mismatch", fact_ids=fact_ids)
    if left_unit != right_unit:
        return _RuntimeValue.unknown("comparison_unit_mismatch", fact_ids=fact_ids)
    truth = _compare_values(
        left,
        right,
        str(comparison.get("operator") or ""),
    )
    return _RuntimeValue.boolean(
        truth,
        value={
            "left": {
                "value": copy.deepcopy(left.value),
                "unit": left_unit,
                "time_scope": left_time_scope or None,
            },
            "right": {
                "value": copy.deepcopy(right.value),
                "unit": right_unit,
                "time_scope": right_time_scope or None,
            },
        },
        fact_ids=fact_ids,
        issue="" if truth is not Truth.UNKNOWN else "comparison_not_evaluable",
    )


class Guideline2GraphRuntimeV1:
    """Execute one migrated Guideline2Graph program over extracted facts."""

    def __init__(self, program: Mapping[str, Any]):
        validate_program(program)
        self.program = copy.deepcopy(dict(program))
        self.predicates = {
            str(item["id"]): dict(item) for item in self.program.get("predicates") or []
        }
        self.knowledge_relations = {
            (
                _normalize_text(item.get("subject")),
                _normalize_text(item.get("relation")),
                _normalize_text(item.get("object")),
            )
            for item in self.program.get("knowledge_relations") or []
            if isinstance(item, Mapping)
        }

    def _matching_facts(
        self,
        predicate: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        predicate_id = str(predicate.get("id") or "")
        accepted_concepts = {
            *_fact_aliases(predicate),
            _normalize_text(predicate.get("entity")),
        }
        accepted_concepts.discard("")
        time_scopes = _allowed_time_scopes(predicate)
        source_kind = _predicate_source_kind(predicate)
        retrieve = predicate.get("retrieve") or {}
        filters = retrieve.get("filters") if isinstance(retrieve, Mapping) else {}
        required_unit = canonical_unit(
            filters.get("unit") if isinstance(filters, Mapping) else None
        )
        matches = [
            fact
            for fact in facts
            if (
                _fact_source_kind(fact) == source_kind
                and (
                not str(fact.get("bound_predicate_id") or "")
                or str(fact.get("bound_predicate_id")) == predicate_id
                )
            )
            and _fact_identifiers(fact) & accepted_concepts
            and (
                not time_scopes
                or bool(_fact_time_scopes(fact) & time_scopes)
            )
            and (
                required_unit is None
                or canonical_unit(fact.get("unit")) == required_unit
            )
        ]
        time_rank = {"current": 0, "conditional": 1, "historical": 2, "baseline": 3}
        return sorted(
            matches,
            key=lambda fact: (
                time_rank.get(_fact_time_scope(fact), 9),
                str(fact.get("fact_id") or ""),
            ),
        )

    def _extract_fact_value(
        self,
        predicate: Mapping[str, Any],
        fact: Mapping[str, Any],
    ) -> _RuntimeValue:
        fact_id = str(fact.get("fact_id") or "")
        extract = predicate.get("extract") or {}
        if not isinstance(extract, Mapping):
            return _RuntimeValue.unknown("invalid_extract_spec", fact_ids=[fact_id])
        path = str(extract.get("path") or "value")
        parser = str(extract.get("parser") or "")
        value: Any = _assertion_truth(fact) if path == "assertion" else fact.get("value")
        if parser == "blood_pressure_systolic":
            value = _blood_pressure_component(value, 0)
        elif parser == "blood_pressure_diastolic":
            value = _blood_pressure_component(value, 1)
        elif parser:
            raise Guideline2GraphProgramError(
                f"{predicate.get('id')}: unsupported extract parser {parser!r}"
            )
        if value is None:
            return _RuntimeValue.unknown("extracted_value_missing", fact_ids=[fact_id])
        return _RuntimeValue(
            status="known",
            value=copy.deepcopy(value),
            unit=canonical_unit(fact.get("unit") or predicate.get("unit")),
            comparator=_quote_comparator(fact),
            fact_ids=(fact_id,),
        )

    def _finish_predicate_value(
        self,
        predicate: Mapping[str, Any],
        value: _RuntimeValue,
        *,
        source_fact: Optional[Mapping[str, Any]] = None,
    ) -> _RuntimeValue:
        if predicate.get("conversion") is not None:
            value = _apply_conversion(value, predicate.get("conversion"))
        comparison = _comparison_spec(predicate)
        if comparison is not None:
            return _compare_runtime_value(value, comparison)
        if value.status != "known":
            return value
        final_type = str(predicate.get("final_output_type") or "")
        if final_type in {"Bool", "TruthValue"}:
            if value.truth is not None:
                return _RuntimeValue.boolean(
                    value.truth,
                    value=copy.deepcopy(value.value),
                    fact_ids=value.fact_ids,
                    issue=value.issue,
                )
            if isinstance(value.value, bool):
                truth = Truth.TRUE if value.value else Truth.FALSE
            elif str(value.value).strip().lower() in {"true", "false", "unknown"}:
                truth = Truth(str(value.value).strip().lower())
            elif final_type == "TruthValue":
                return _RuntimeValue.unknown(
                    "non_boolean_truth_value",
                    fact_ids=value.fact_ids,
                )
            elif source_fact is not None:
                truth = Truth.TRUE if _assertion_truth(source_fact) else Truth.FALSE
            else:
                return _RuntimeValue.unknown(
                    "non_boolean_value",
                    fact_ids=value.fact_ids,
                )
            return _RuntimeValue.boolean(
                truth,
                value=copy.deepcopy(value.value),
                fact_ids=value.fact_ids,
            )
        if final_type in {"Quantity", "Integer"}:
            if not _is_number(value.value):
                return _RuntimeValue.unknown(
                    "non_numeric_quantity",
                    fact_ids=value.fact_ids,
                )
            return _RuntimeValue(
                status="known",
                value=float(value.value),
                unit=canonical_unit(value.unit),
                comparator=value.comparator,
                fact_ids=value.fact_ids,
            )
        return value

    def _right_input_matches(
        self,
        predicate: Mapping[str, Any],
        comparison: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        left_fact: Mapping[str, Any],
    ) -> tuple[List[Mapping[str, Any]], str]:
        binding = _binding_section(predicate, "QuestionObservation")
        source_payload = predicate.get("source_payload") or {}
        retrieve = predicate.get("retrieve") or {}
        explicit: List[Any] = [comparison.get("right_input_ref")]
        for field in ("right_input_id", "right_concept_id", "right_source_input_id"):
            if binding.get(field) is not None:
                explicit.append(binding[field])
            if isinstance(source_payload, Mapping) and source_payload.get(field) is not None:
                explicit.append(source_payload[field])
        for field in ("right_input_refs", "right_concept_ids"):
            if isinstance(binding.get(field), list):
                explicit.extend(binding[field])
        explicit_aliases = {
            _normalize_text(item) for item in explicit if str(item or "").strip()
        }
        dependency_aliases = {
            _normalize_text(item)
            for item in predicate.get("dependencies") or []
            if str(item or "").strip()
        }
        left_fact_id = str(left_fact.get("fact_id") or "")
        predicate_id = str(predicate.get("id") or "")

        def select(aliases: set[str]) -> List[Mapping[str, Any]]:
            matches = [
                fact
                for fact in facts
                if str(fact.get("fact_id") or "") != left_fact_id
                and (
                    not str(fact.get("bound_predicate_id") or "")
                    or str(fact.get("bound_predicate_id")) == predicate_id
                )
                and _fact_identifiers(fact) & aliases
            ]
            right_scopes: Any = None
            if isinstance(retrieve, Mapping):
                right_scopes = retrieve.get("right_time_scopes")
            if right_scopes is None:
                right_scopes = binding.get("right_time_scopes")
            if isinstance(right_scopes, list):
                allowed = {
                    _normalize_text(item) for item in right_scopes if str(item).strip()
                }
                matches = [
                    fact for fact in matches if _fact_time_scopes(fact) & allowed
                ]
            return sorted(matches, key=lambda fact: str(fact.get("fact_id") or ""))

        explicit_matches = select(explicit_aliases)
        if explicit_matches:
            return explicit_matches, ""
        dependency_matches = select(dependency_aliases)
        if dependency_matches:
            return dependency_matches, ""
        return [], "right_input_missing"

    def _evaluate_predicate_output(
        self,
        predicate: Mapping[str, Any],
        dependency_results: Mapping[str, _RuntimeValue],
    ) -> _RuntimeValue:
        dependency_ids = _predicate_dependency_ids(predicate)
        if not dependency_ids:
            return _RuntimeValue.unknown("predicate_dependency_missing")
        values = [
            dependency_results.get(
                dependency_id,
                _RuntimeValue.unknown("predicate_dependency_missing"),
            )
            for dependency_id in dependency_ids
        ]
        comparison = _comparison_spec(predicate)
        if comparison is not None and comparison.get("right_input_ref"):
            binding = _binding_section(predicate, "PredicateOutput")
            right_ref = str(
                binding.get("right_predicate_id")
                or comparison.get("right_input_ref")
                or ""
            )
            right_indexes = [
                index for index, dependency_id in enumerate(dependency_ids)
                if dependency_id == right_ref
            ]
            if len(right_indexes) != 1 or len(values) != 2:
                return _RuntimeValue.unknown(
                    "predicate_dependency_pair_ambiguous",
                    fact_ids=[fact_id for value in values for fact_id in value.fact_ids],
                )
            right_index = right_indexes[0]
            left_index = 1 - right_index
            left = values[left_index]
            right = values[right_index]
            if predicate.get("conversion") is not None:
                left = _apply_conversion(left, predicate.get("conversion"))
                right = _apply_conversion(right, predicate.get("conversion"))
            return _compare_runtime_pair(left, right, comparison)
        if len(values) != 1:
            return _RuntimeValue.unknown(
                "predicate_dependency_count_invalid",
                fact_ids=[fact_id for value in values for fact_id in value.fact_ids],
            )
        return self._finish_predicate_value(predicate, values[0])

    def _evaluate_quantity_difference(
        self,
        predicate: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        calculation: Mapping[str, Any],
    ) -> _RuntimeValue:
        predicate_id = str(predicate.get("id") or "")
        temporal_inputs = calculation.get("temporal_inputs") or {}
        if not isinstance(temporal_inputs, Mapping):
            temporal_inputs = {}

        def select(role: str) -> tuple[Optional[Mapping[str, Any]], str]:
            reference = str(calculation.get(f"{role}_input_ref") or "")
            aliases = {_normalize_text(reference)}
            expected_timepoint = _normalize_text(
                temporal_inputs.get(f"{role}_timepoint")
            )
            matches = [
                fact
                for fact in facts
                if (
                    not str(fact.get("bound_predicate_id") or "")
                    or str(fact.get("bound_predicate_id")) == predicate_id
                )
                and bool(_fact_identifiers(fact) & aliases)
                and (
                    not expected_timepoint
                    or expected_timepoint in _fact_time_scopes(fact)
                )
            ]
            if not matches:
                return None, f"{role}_input_missing"
            if len(matches) != 1:
                return None, f"{role}_input_ambiguous"
            return matches[0], ""

        minuend_fact, issue = select("minuend")
        if issue:
            return _RuntimeValue.unknown(issue)
        subtrahend_fact, issue = select("subtrahend")
        if issue:
            return _RuntimeValue.unknown(issue)
        assert minuend_fact is not None and subtrahend_fact is not None
        minuend = self._extract_fact_value(predicate, minuend_fact)
        subtrahend = self._extract_fact_value(predicate, subtrahend_fact)
        fact_ids = [*minuend.fact_ids, *subtrahend.fact_ids]
        if minuend.status != "known" or subtrahend.status != "known":
            return _RuntimeValue.unknown(
                minuend.issue or subtrahend.issue or "calculation_input_unknown",
                fact_ids=fact_ids,
            )
        if not _is_number(minuend.value) or not _is_number(subtrahend.value):
            return _RuntimeValue.unknown(
                "calculation_input_non_numeric", fact_ids=fact_ids
            )
        requirements = calculation.get("unit_requirements") or {}
        if not isinstance(requirements, Mapping):
            requirements = {}
        minuend_unit = canonical_unit(minuend.unit)
        subtrahend_unit = canonical_unit(subtrahend.unit)
        expected_minuend_unit = canonical_unit(requirements.get("minuend_unit"))
        expected_subtrahend_unit = canonical_unit(
            requirements.get("subtrahend_unit")
        )
        if expected_minuend_unit and minuend_unit != expected_minuend_unit:
            return _RuntimeValue.unknown(
                "minuend_unit_mismatch", fact_ids=fact_ids
            )
        if expected_subtrahend_unit and subtrahend_unit != expected_subtrahend_unit:
            return _RuntimeValue.unknown(
                "subtrahend_unit_mismatch", fact_ids=fact_ids
            )
        if requirements.get("require_matching_units") and minuend_unit != subtrahend_unit:
            return _RuntimeValue.unknown(
                "calculation_unit_mismatch", fact_ids=fact_ids
            )
        result = _RuntimeValue(
            status="known",
            value=float(minuend.value) - float(subtrahend.value),
            unit=canonical_unit(calculation.get("result_unit")),
            fact_ids=tuple(fact_ids),
        )
        comparison = calculation.get("comparison")
        if not isinstance(comparison, Mapping):
            comparison = _comparison_spec(predicate)
        if not isinstance(comparison, Mapping):
            return _RuntimeValue.unknown(
                "calculation_comparison_missing", fact_ids=fact_ids
            )
        comparison = copy.deepcopy(dict(comparison))
        if comparison.get("value") is None and comparison.get("threshold") is not None:
            comparison["value"] = comparison.pop("threshold")
        return _compare_runtime_value(result, comparison)

    def _evaluate_quantity_multiply_by_factor(
        self,
        predicate: Mapping[str, Any],
        calculation: Mapping[str, Any],
        dependency_results: Mapping[str, _RuntimeValue],
    ) -> _RuntimeValue:
        input_ref = str(calculation.get("input_ref") or "")
        value = dependency_results.get(
            input_ref,
            _RuntimeValue.unknown("predicate_dependency_missing"),
        )
        if value.status != "known":
            return value
        if not _is_number(value.value):
            return _RuntimeValue.unknown(
                "calculation_value_non_numeric",
                fact_ids=value.fact_ids,
            )
        factor = calculation.get("factor")
        if not _is_number(factor):
            return _RuntimeValue.unknown(
                "calculation_factor_invalid",
                fact_ids=value.fact_ids,
            )
        input_unit = canonical_unit(calculation.get("input_unit"))
        actual_unit = canonical_unit(value.unit)
        if input_unit and actual_unit != input_unit:
            return _RuntimeValue.unknown(
                "calculation_unit_mismatch",
                fact_ids=value.fact_ids,
            )
        result_unit = canonical_unit(calculation.get("result_unit"))
        result = _RuntimeValue(
            status="known",
            value=float(value.value) * float(factor),
            unit=result_unit,
            fact_ids=value.fact_ids,
        )
        comparison = calculation.get("comparison")
        if not isinstance(comparison, Mapping):
            comparison = _comparison_spec(predicate)
        if isinstance(comparison, Mapping):
            return _compare_runtime_value(result, comparison)
        return result

    def _evaluate_relative_percentage_change(
        self,
        predicate: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        calculation: Mapping[str, Any],
    ) -> _RuntimeValue:
        predicate_id = str(predicate.get("id") or "")
        temporal_inputs = calculation.get("temporal_inputs") or {}
        if not isinstance(temporal_inputs, Mapping):
            temporal_inputs = {}

        def select(role: str) -> tuple[Optional[Mapping[str, Any]], str]:
            reference = str(calculation.get(f"{role}_input_ref") or "")
            expected_timepoint = _normalize_text(
                temporal_inputs.get(f"{role}_timepoint")
            )
            matches = [
                fact
                for fact in facts
                if (
                    not str(fact.get("bound_predicate_id") or "")
                    or str(fact.get("bound_predicate_id")) == predicate_id
                )
                and _normalize_text(reference) in _fact_identifiers(fact)
                and (
                    not expected_timepoint
                    or expected_timepoint in _fact_time_scopes(fact)
                )
            ]
            if not matches:
                return None, f"{role}_input_missing"
            if len(matches) != 1:
                return None, f"{role}_input_ambiguous"
            return matches[0], ""

        baseline_fact, issue = select("baseline")
        if issue:
            return _RuntimeValue.unknown(issue)
        current_fact, issue = select("current")
        if issue:
            return _RuntimeValue.unknown(issue)
        assert baseline_fact is not None and current_fact is not None
        baseline = self._extract_fact_value(predicate, baseline_fact)
        current = self._extract_fact_value(predicate, current_fact)
        fact_ids = [*baseline.fact_ids, *current.fact_ids]
        if baseline.status != "known" or current.status != "known":
            return _RuntimeValue.unknown(
                baseline.issue or current.issue or "calculation_input_unknown",
                fact_ids=fact_ids,
            )
        if not _is_number(baseline.value) or not _is_number(current.value):
            return _RuntimeValue.unknown(
                "calculation_input_non_numeric", fact_ids=fact_ids
            )
        if float(baseline.value) <= 0:
            return _RuntimeValue.unknown(
                "baseline_input_not_positive", fact_ids=fact_ids
            )
        requirements = calculation.get("unit_requirements") or {}
        if not isinstance(requirements, Mapping):
            requirements = {}
        baseline_unit = canonical_unit(baseline.unit)
        current_unit = canonical_unit(current.unit)
        expected_baseline_unit = canonical_unit(
            requirements.get("baseline_unit")
        )
        expected_current_unit = canonical_unit(requirements.get("current_unit"))
        if expected_baseline_unit and baseline_unit != expected_baseline_unit:
            return _RuntimeValue.unknown(
                "baseline_unit_mismatch", fact_ids=fact_ids
            )
        if expected_current_unit and current_unit != expected_current_unit:
            return _RuntimeValue.unknown("current_unit_mismatch", fact_ids=fact_ids)
        if requirements.get("require_matching_units") and baseline_unit != current_unit:
            return _RuntimeValue.unknown(
                "calculation_unit_mismatch", fact_ids=fact_ids
            )
        result = _RuntimeValue(
            status="known",
            value=(
                (float(current.value) - float(baseline.value))
                / float(baseline.value)
                * 100.0
            ),
            unit="percent",
            fact_ids=tuple(fact_ids),
        )
        comparison = calculation.get("comparison")
        if not isinstance(comparison, Mapping):
            comparison = _comparison_spec(predicate)
        if not isinstance(comparison, Mapping):
            return _RuntimeValue.unknown(
                "calculation_comparison_missing", fact_ids=fact_ids
            )
        return _compare_runtime_value(result, comparison)

    def _evaluate_cockcroft_gault_normalized_creatinine_clearance(
        self,
        predicate: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        calculation: Mapping[str, Any],
    ) -> _RuntimeValue:
        predicate_id = str(predicate.get("id") or "")
        roles = {
            "age": "age_input_ref",
            "sex": "sex_input_ref",
            "body_weight": "body_weight_input_ref",
            "serum_creatinine": "serum_creatinine_input_ref",
        }
        selected: Dict[str, Mapping[str, Any]] = {}
        for role, field in roles.items():
            reference = _normalize_text(calculation.get(field))
            matches = [
                fact
                for fact in facts
                if (
                    not str(fact.get("bound_predicate_id") or "")
                    or str(fact.get("bound_predicate_id")) == predicate_id
                )
                and reference in _fact_identifiers(fact)
            ]
            if not matches:
                return _RuntimeValue.unknown(f"{role}_input_missing")
            if len(matches) != 1:
                return _RuntimeValue.unknown(f"{role}_input_ambiguous")
            selected[role] = matches[0]

        values = {
            role: self._extract_fact_value(predicate, fact)
            for role, fact in selected.items()
        }
        fact_ids = [
            fact_id
            for role in roles
            for fact_id in values[role].fact_ids
        ]
        unknown = next(
            (value for value in values.values() if value.status != "known"), None
        )
        if unknown is not None:
            return _RuntimeValue.unknown(
                unknown.issue or "calculation_input_unknown", fact_ids=fact_ids
            )
        age = values["age"]
        weight = values["body_weight"]
        creatinine = values["serum_creatinine"]
        if not all(_is_number(value.value) for value in (age, weight, creatinine)):
            return _RuntimeValue.unknown(
                "calculation_input_non_numeric", fact_ids=fact_ids
            )
        if (
            float(age.value) <= 0
            or float(age.value) >= 140
            or float(weight.value) <= 0
            or float(creatinine.value) <= 0
        ):
            return _RuntimeValue.unknown(
                "calculation_input_out_of_domain", fact_ids=fact_ids
            )
        for role, field in (
            ("age", "age_unit"),
            ("body_weight", "body_weight_unit"),
            ("serum_creatinine", "serum_creatinine_unit"),
        ):
            expected_unit = canonical_unit(calculation.get(field))
            if expected_unit and canonical_unit(values[role].unit) != expected_unit:
                return _RuntimeValue.unknown(
                    f"{role}_unit_mismatch", fact_ids=fact_ids
                )
        sex = _normalize_text(values["sex"].value)
        if sex in {"female", "f", "woman", "female at birth", "女"}:
            sex_factor = float(calculation.get("female_factor"))
        elif sex in {"male", "m", "man", "male at birth", "男"}:
            sex_factor = 1.0
        else:
            return _RuntimeValue.unknown("sex_input_unknown", fact_ids=fact_ids)
        constant = float(calculation.get("constant"))
        absolute_clearance = (
            (140.0 - float(age.value))
            * float(weight.value)
            * sex_factor
            / (constant * float(creatinine.value))
        )
        result = _RuntimeValue(
            status="known",
            value=absolute_clearance / float(weight.value),
            unit=canonical_unit(calculation.get("result_unit")),
            fact_ids=tuple(fact_ids),
        )
        comparison = calculation.get("comparison")
        if not isinstance(comparison, Mapping):
            comparison = _comparison_spec(predicate)
        if isinstance(comparison, Mapping):
            return _compare_runtime_value(result, comparison)
        return result

    def _evaluate_predicate(
        self,
        predicate: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        dependency_results: Optional[Mapping[str, _RuntimeValue]] = None,
    ) -> _RuntimeValue:
        calculation = _calculation_spec(predicate)
        if calculation is not None:
            operator = str(calculation.get("operator") or "")
            if operator == "quantity_difference_at_least":
                return self._evaluate_quantity_difference(
                    predicate, facts, calculation
                )
            if operator == "quantity_multiply_by_factor":
                return self._evaluate_quantity_multiply_by_factor(
                    predicate,
                    calculation,
                    dependency_results or {},
                )
            if operator == "relative_percentage_change":
                return self._evaluate_relative_percentage_change(
                    predicate, facts, calculation
                )
            if operator == "cockcroft_gault_normalized_creatinine_clearance":
                return self._evaluate_cockcroft_gault_normalized_creatinine_clearance(
                    predicate, facts, calculation
                )
        if _uses_predicate_outputs(predicate):
            return self._evaluate_predicate_output(predicate, dependency_results or {})
        matches = self._matching_facts(predicate, facts)
        if not matches:
            return _RuntimeValue.unknown("no_matching_question_fact")
        reduction = predicate.get("reduction") or {}
        operator = str(reduction.get("operator") or "none") if isinstance(reduction, Mapping) else "none"
        if operator == "exists":
            return _RuntimeValue.boolean(
                Truth.TRUE,
                value=True,
                fact_ids=[str(item.get("fact_id") or "") for item in matches],
            )
        if operator not in {"none", "most_recent"}:
            raise Guideline2GraphProgramError(
                f"{predicate.get('id')}: unsupported reduction {operator!r}"
            )
        left_fact = matches[0]
        left = self._extract_fact_value(predicate, left_fact)
        comparison = _comparison_spec(predicate)
        if comparison is not None and comparison.get("right_input_ref"):
            right_matches, issue = self._right_input_matches(
                predicate,
                comparison,
                facts,
                left_fact,
            )
            if issue:
                return _RuntimeValue.unknown(issue, fact_ids=left.fact_ids)
            if len(right_matches) != 1:
                return _RuntimeValue.unknown(
                    "right_input_ambiguous",
                    fact_ids=[
                        *left.fact_ids,
                        *(str(item.get("fact_id") or "") for item in right_matches),
                    ],
                )
            right_fact = right_matches[0]
            right = self._extract_fact_value(predicate, right_fact)
            if predicate.get("conversion") is not None:
                left = _apply_conversion(left, predicate.get("conversion"))
                right = _apply_conversion(right, predicate.get("conversion"))
            return _compare_runtime_pair(
                left,
                right,
                comparison,
                left_time_scope=_fact_time_scope(left_fact),
                right_time_scope=_fact_time_scope(right_fact),
            )
        return self._finish_predicate_value(
            predicate,
            left,
            source_fact=left_fact,
        )

    def _literal(self, value: Any) -> _RuntimeValue:
        if isinstance(value, Mapping):
            literal_value = value.get("value")
            unit = canonical_unit(value.get("unit"))
        else:
            literal_value = value
            unit = None
        if isinstance(literal_value, bool):
            return _RuntimeValue.boolean(
                Truth.TRUE if literal_value else Truth.FALSE,
                value=literal_value,
            )
        return _RuntimeValue(status="known", value=literal_value, unit=unit)

    def _evaluate_dag(
        self,
        rule: Mapping[str, Any],
        predicate_results: Mapping[str, _RuntimeValue],
        assessment_results: Mapping[str, _RuntimeValue],
    ) -> tuple[_RuntimeValue, Dict[str, Dict[str, Any]]]:
        dag = rule["condition_dag"]
        nodes = {str(item["id"]): dict(item) for item in dag["nodes"]}
        cache: Dict[str, _RuntimeValue] = {}
        trace: Dict[str, Dict[str, Any]] = {}

        def operand(value: Any, stack: set[str]) -> _RuntimeValue:
            if isinstance(value, str):
                return evaluate(value, stack)
            if isinstance(value, Mapping) and "id" in value:
                return evaluate(str(value["id"]), stack)
            return self._literal(value)

        def evaluate(node_id: str, stack: set[str]) -> _RuntimeValue:
            if node_id in cache:
                return cache[node_id]
            if node_id in stack:
                result = _RuntimeValue.unknown("condition_dag_cycle")
                cache[node_id] = result
                return result
            node = nodes.get(node_id)
            if node is None:
                result = _RuntimeValue.unknown("condition_dag_node_missing")
                cache[node_id] = result
                return result
            active = set(stack)
            active.add(node_id)
            node_type = str(node.get("type") or "")
            dependencies: List[str] = []
            if node_type == "predicate_ref":
                predicate_id = str(node.get("predicate_ref") or "")
                dependencies = [predicate_id]
                result = predicate_results.get(
                    predicate_id,
                    _RuntimeValue.unknown("referenced_predicate_missing"),
                )
            elif node_type == "clinical_assessment_ref":
                assessment_spec_id = str(node.get("assessment_spec_id") or "")
                dependencies = [assessment_spec_id]
                result = assessment_results.get(
                    assessment_spec_id,
                    _RuntimeValue.unknown("clinical_assessment_missing"),
                )
            elif node_type == "literal":
                result = self._literal(node.get("value"))
            elif node_type == "compare":
                left_spec = node.get("left")
                right_spec = node.get("right")
                dependencies = [
                    str(item)
                    for item in (left_spec, right_spec)
                    if isinstance(item, str)
                ]
                left = operand(left_spec, active)
                right = operand(right_spec, active)
                truth = _compare_values(left, right, str(node.get("operator") or ""))
                result = _RuntimeValue.boolean(
                    truth,
                    value={"left": left.as_dict(), "right": right.as_dict()},
                    fact_ids=[*left.fact_ids, *right.fact_ids],
                    issue="" if truth is not Truth.UNKNOWN else "comparison_not_evaluable",
                )
            elif node_type == "combine":
                inputs = [str(item) for item in node.get("inputs") or []]
                dependencies = inputs
                children = [evaluate(item, active) for item in inputs]
                truths = [child.truth or Truth.UNKNOWN for child in children]
                operator_name = str(node.get("operator") or "AND").upper()
                if operator_name in {"AND", "ALL"}:
                    truth = kleene_all(truths)
                elif operator_name in {"OR", "ANY"}:
                    truth = kleene_any(truths)
                elif operator_name == "AT_LEAST":
                    parameters = node.get("parameters") or {}
                    minimum = int(parameters.get("minimum") or 0)
                    if minimum < 1:
                        raise Guideline2GraphProgramError(
                            f"{node_id}: AT_LEAST requires positive parameters.minimum"
                        )
                    truth = kleene_at_least(truths, minimum)
                elif operator_name == "NOT":
                    if len(truths) != 1:
                        raise Guideline2GraphProgramError(
                            f"{node_id}: NOT requires exactly one input"
                        )
                    truth = kleene_not(truths[0])
                else:
                    raise Guideline2GraphProgramError(
                        f"{node_id}: unsupported combine operator {operator_name!r}"
                    )
                result = _RuntimeValue.boolean(
                    truth,
                    value=[child.truth.value if child.truth else None for child in children],
                    fact_ids=[fact_id for child in children for fact_id in child.fact_ids],
                )
            elif node_type == "exists":
                specification = node.get("input") or {}
                relation = specification.get("knowledge_relation") if isinstance(specification, Mapping) else None
                if not isinstance(relation, Mapping):
                    result = _RuntimeValue.unknown("unsupported_exists_input")
                else:
                    key = (
                        _normalize_text(relation.get("subject")),
                        _normalize_text(relation.get("relation")),
                        _normalize_text(relation.get("object")),
                    )
                    truth = Truth.TRUE if key in self.knowledge_relations else Truth.UNKNOWN
                    result = _RuntimeValue.boolean(
                        truth,
                        value=dict(relation),
                        issue="" if truth is Truth.TRUE else "knowledge_relation_not_found",
                    )
            else:
                result = _RuntimeValue.unknown(f"unsupported_dag_node_type:{node_type}")
            cache[node_id] = result
            trace[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "return_type": str(node.get("return_type") or ""),
                "dependencies": dependencies,
                **result.as_dict(),
            }
            return result

        root_id = str(dag["root"])
        root_result = evaluate(root_id, set())
        for node_id in nodes:
            evaluate(node_id, set())
        return root_result, trace

    def execute(
        self,
        facts: Sequence[Mapping[str, Any]],
        clinical_assessments: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        validate_facts(facts)
        validate_clinical_assessments(clinical_assessments)
        fact_records = [dict(item) for item in facts]
        assessment_specs = {
            str(item.get("assessment_spec_id") or ""): dict(item)
            for item in self.program.get("clinical_assessment_specs") or []
            if isinstance(item, Mapping) and str(item.get("assessment_spec_id") or "")
        }
        assessment_results: Dict[str, _RuntimeValue] = {}
        assessment_review_required: Dict[str, bool] = {}
        for assessment in clinical_assessments or []:
            spec_id = str(assessment.get("assessment_spec_id") or "")
            if spec_id not in assessment_specs:
                raise Guideline2GraphProgramError(
                    f"clinical assessment references unknown assessment_spec_id {spec_id!r}"
                )
            raw_value = str(assessment.get("value") or "").strip().lower()
            if raw_value == "true":
                truth = Truth.TRUE
                issue = ""
            elif raw_value == "false":
                truth = Truth.FALSE
                issue = ""
            else:
                truth = Truth.UNKNOWN
                issue = "clinical_assessment_unknown"
            assessment_results[spec_id] = _RuntimeValue.boolean(
                truth,
                value=raw_value,
                fact_ids=[
                    str(item)
                    for item in assessment.get("supporting_observation_ids") or []
                    if str(item)
                ],
                issue=issue,
            )
            assessment_review_required[spec_id] = bool(
                assessment.get("requires_human_review")
                or str(assessment_specs[spec_id].get("human_review_policy") or "")
                == "required_before_action"
            )
        predicate_values: Dict[str, _RuntimeValue] = {}

        def evaluate_predicate(
            predicate_id: str,
            active: tuple[str, ...] = (),
        ) -> _RuntimeValue:
            if predicate_id in predicate_values:
                return predicate_values[predicate_id]
            if predicate_id in active:
                return _RuntimeValue.unknown("predicate_dependency_cycle")
            predicate = self.predicates.get(predicate_id)
            if predicate is None:
                return _RuntimeValue.unknown("predicate_dependency_missing")
            dependencies: Dict[str, _RuntimeValue] = {}
            if _uses_predicate_outputs(predicate):
                next_active = (*active, predicate_id)
                dependencies = {
                    dependency_id: evaluate_predicate(dependency_id, next_active)
                    for dependency_id in _predicate_dependency_ids(predicate)
                }
            result = self._evaluate_predicate(
                predicate,
                fact_records,
                dependencies,
            )
            predicate_values[predicate_id] = result
            return result

        for predicate_id in self.predicates:
            evaluate_predicate(predicate_id)
        predicate_trace = {
            predicate_id: {
                "predicate_id": predicate_id,
                "entity": str(self.predicates[predicate_id].get("entity") or ""),
                "aspect": str(self.predicates[predicate_id].get("aspect") or ""),
                "final_output_type": str(
                    self.predicates[predicate_id].get("final_output_type") or ""
                ),
                "source_kind": str(
                    self.predicates[predicate_id].get("source_kind")
                    or "question_observation"
                ),
                **value.as_dict(),
            }
            for predicate_id, value in predicate_values.items()
        }

        native_nodes: Dict[str, Dict[str, Any]] = {}
        rule_results: List[Dict[str, Any]] = []
        action_status: MutableMapping[str, str] = {}
        action_activation_status: MutableMapping[str, str] = {}
        action_definitions: MutableMapping[str, Dict[str, Any]] = {}
        action_review_specs: MutableMapping[str, set[str]] = {}
        action_review_reasons: MutableMapping[str, set[str]] = {}
        status_rank = {
            "not_activated": 0,
            "unresolved": 1,
            "required": 2,
            "pending_human_review": 3,
        }
        activation_rank = {
            "not_activated": 0,
            "unresolved": 1,
            "required": 2,
        }
        for rule in self.program.get("rules") or []:
            rule_id = str(rule["id"])
            root, nodes = self._evaluate_dag(rule, predicate_values, assessment_results)
            for node_id, node_result in nodes.items():
                previous = native_nodes.get(node_id)
                if previous is not None:
                    comparable_previous = {
                        key: value for key, value in previous.items() if key != "rule_ids"
                    }
                    comparable_current = {
                        key: value for key, value in node_result.items() if key != "rule_ids"
                    }
                    if comparable_previous != comparable_current:
                        scoped_id = f"{rule_id}::{node_id}"
                        native_nodes[scoped_id] = {
                            **node_result,
                            "node_id": scoped_id,
                            "source_node_id": node_id,
                            "rule_ids": [rule_id],
                        }
                        continue
                    previous.setdefault("rule_ids", []).append(rule_id)
                else:
                    native_nodes[node_id] = {
                        **node_result,
                        "rule_ids": [rule_id],
                    }
            truth = root.truth or Truth.UNKNOWN
            base_status = {
                Truth.TRUE: "required",
                Truth.FALSE: "not_activated",
                Truth.UNKNOWN: "unresolved",
            }[truth]
            assessment_spec_ids = sorted(
                {
                    str(node.get("assessment_spec_id") or "")
                    for node in (rule.get("condition_dag") or {}).get("nodes") or []
                    if isinstance(node, Mapping)
                    and str(node.get("node_type") or node.get("type") or "")
                    == "clinical_assessment_ref"
                    and str(node.get("assessment_spec_id") or "")
                }
            )
            reviewed_assessment_spec_ids = [
                spec_id
                for spec_id in assessment_spec_ids
                if assessment_review_required.get(spec_id, False)
            ]
            action = dict(rule.get("action") or {})
            configured_review_reasons = {
                str(item)
                for item in action.get("human_review_reasons") or []
                if str(item)
            }
            if action.get("human_review_required") and not configured_review_reasons:
                configured_review_reasons.add("action_review_required")
            human_review_required = bool(
                base_status == "required"
                and (reviewed_assessment_spec_ids or configured_review_reasons)
            )
            status = "pending_human_review" if human_review_required else base_status
            output = dict(action.get("output") or {})
            emitted_ids = [str(item) for item in output.get("action_ids") or []]
            labels = output.get("labels") or {}
            action_metadata = output.get("action_metadata") or {}
            for action_id in emitted_ids:
                prior_activation = action_activation_status.get(action_id, "not_activated")
                action_activation_status[action_id] = (
                    base_status
                    if activation_rank[base_status] >= activation_rank[prior_activation]
                    else prior_activation
                )
                prior = action_status.get(action_id, "not_activated")
                action_status[action_id] = (
                    status if status_rank[status] >= status_rank[prior] else prior
                )
                if reviewed_assessment_spec_ids:
                    action_review_specs.setdefault(action_id, set()).update(
                        reviewed_assessment_spec_ids
                    )
                if configured_review_reasons:
                    action_review_reasons.setdefault(action_id, set()).update(
                        configured_review_reasons
                    )
                metadata = (
                    action_metadata.get(action_id) or {}
                    if isinstance(action_metadata, Mapping)
                    else {}
                )
                action_definitions[action_id] = {
                    "action_id": action_id,
                    "label": str(labels.get(action_id) or action_id)
                    if isinstance(labels, Mapping)
                    else action_id,
                    "permission": str(
                        metadata.get("permission") or action.get("permission") or ""
                    ),
                    "intent": str(metadata.get("intent") or action.get("intent") or ""),
                }
            rule_results.append(
                {
                    "rule_id": str(rule["id"]),
                    "label": str(rule.get("label") or ""),
                    "root_id": str(rule["condition_dag"]["root"]),
                    "truth": truth.value,
                    "missing_data_policy": str(rule.get("missing_data_policy") or ""),
                    "action_ids": emitted_ids,
                    "activation_status": base_status,
                    "action_status": status,
                    "human_review_required": human_review_required,
                    "human_review_spec_ids": reviewed_assessment_spec_ids,
                    "human_review_reasons": sorted(configured_review_reasons),
                }
            )

        actions = [
            {
                **action_definitions[action_id],
                "activation_status": action_activation_status[action_id],
                "execution_status": status,
                "human_review_required": status == "pending_human_review",
                "human_review_spec_ids": sorted(action_review_specs.get(action_id, set())),
                "human_review_reasons": sorted(action_review_reasons.get(action_id, set())),
                "execution_authorized": status == "required",
            }
            for action_id, status in action_status.items()
        ]
        return {
            "trace_version": TRACE_SCHEMA_VERSION,
            "qid": str(self.program["qid"]),
            "graph_id": str(self.program.get("graph_id") or ""),
            "input_contract": "question_facts_only_no_truth_bindings",
            "question_fact_count": len(fact_records),
            "bound_question_fact_ids": sorted(
                {
                    fact_id
                    for result in predicate_values.values()
                    for fact_id in result.fact_ids
                    if fact_id
                }
            ),
            "predicate_results": predicate_trace,
            "clinical_assessment_results": {
                spec_id: value.as_dict()
                for spec_id, value in sorted(assessment_results.items())
            },
            "condition_node_results": native_nodes,
            "rule_results": rule_results,
            "actions": actions,
            "oracle_used": False,
            "baseline_patient_state_used": False,
            "truth_labels_accepted_as_input": False,
            "clinical_assessments_accepted_as_input": bool(clinical_assessments),
        }


__all__ = [
    "OBSERVATION_DOCUMENT_SCHEMA_VERSION",
    "PROGRAM_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "Guideline2GraphProgramError",
    "Guideline2GraphRuntimeV1",
    "Truth",
    "canonical_unit",
    "kleene_all",
    "kleene_any",
    "kleene_at_least",
    "load_observation_document",
    "load_program",
    "validate_facts",
    "validate_program",
    "validate_with_original_models",
]
