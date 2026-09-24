"""Deterministic graph QA and repair helpers for v2 extraction outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .models import MedicationTerm, Term


def _item_id(item: Any) -> str | None:
    if hasattr(item, "id"):
        return getattr(item, "id")
    if isinstance(item, dict):
        return item.get("id")
    return None


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _humanize_id(identifier: str) -> str:
    leaf = identifier.split(".")[-1]
    return leaf.replace("_", " ").replace("-", " ").strip().title() or identifier


def _term_label(entity_id: str, entity_type: str | None = None) -> str:
    kind = (entity_type or "").lower()
    if entity_id.startswith("cond.") or kind == "condition":
        return "conditions"
    if entity_id.startswith("meas.") or kind in {"measurement", "measure"}:
        return "measures"
    if entity_id.startswith("proc.") or kind == "procedure":
        return "procedures"
    return "observations"


def _data_bindings_for(entity_id: str, entity_type: str | None = None) -> Dict[str, Dict[str, Any]]:
    kind = (entity_type or "").lower()
    if entity_id.startswith("cond.") or kind == "condition":
        return {"FHIR": {"resource": "Condition"}, "OMOP": {"table": "condition_occurrence"}, "QDM": {}}
    if entity_id.startswith("meas.") or kind in {"measurement", "measure"}:
        return {"FHIR": {"resource": "Observation"}, "OMOP": {"table": "measurement"}, "QDM": {}}
    if entity_id.startswith("proc.") or kind == "procedure":
        return {"FHIR": {"resource": "Procedure"}, "OMOP": {"table": "procedure_occurrence"}, "QDM": {}}
    return {"FHIR": {"resource": "Observation"}, "OMOP": {"table": "observation"}, "QDM": {}}


def _source_evidence_from_predicate(predicate: Any) -> List[Dict[str, Any]]:
    source_text = _get(predicate, "source_text") or ""
    source_span = _get(predicate, "source_span")
    if not source_text and not source_span:
        return []
    return [{
        "source_text": source_text or None,
        "source_span": source_span,
        "confidence": 0.5,
    }]


def _minimal_term_from_predicate(predicate: Any) -> Term | MedicationTerm | None:
    entity_id = _get(predicate, "entity")
    if not entity_id:
        return None
    entity_type = _get(predicate, "entity_type")
    name = _get(predicate, "name") or _get(predicate, "description") or _humanize_id(entity_id)
    evidence = _source_evidence_from_predicate(predicate)

    if entity_id.startswith("med."):
        return MedicationTerm(
            id=entity_id,
            name=name,
            clinical_entity=entity_id,
            concept=name,
            data_bindings={
                "FHIR": {"resource": "MedicationStatement"},
                "OMOP": {"table": "drug_exposure"},
                "QDM": {},
            },
            fhir_binding_hint={"resource": "MedicationStatement"},
            omop_binding_hint={"table": "drug_exposure"},
            binding_status="candidate",
            normalization_confidence=0.5,
            source_evidence=evidence,
        )

    bindings = _data_bindings_for(entity_id, entity_type)
    return Term(
        id=entity_id,
        name=name,
        label=_term_label(entity_id, entity_type),
        type="Code",
        clinical_entity=entity_id,
        concept=name,
        value_domain="Code",
        data_bindings=bindings,
        fhir_binding_hint=bindings.get("FHIR", {}),
        omop_binding_hint=bindings.get("OMOP", {}),
        binding_status="candidate",
        normalization_confidence=0.5,
        source_evidence=evidence,
    )


def _minimal_med_from_subject(subject_id: str, rule: Any) -> MedicationTerm:
    source_text = _get(rule, "source_text") or _get(rule, "label") or ""
    return MedicationTerm(
        id=subject_id,
        name=_humanize_id(subject_id),
        clinical_entity=subject_id,
        concept=_humanize_id(subject_id),
        data_bindings={
            "FHIR": {"resource": "MedicationStatement"},
            "OMOP": {"table": "drug_exposure"},
            "QDM": {},
        },
        fhir_binding_hint={"resource": "MedicationStatement"},
        omop_binding_hint={"table": "drug_exposure"},
        binding_status="candidate",
        normalization_confidence=0.5,
        source_evidence=[{"source_text": source_text, "confidence": 0.5}] if source_text else [],
    )


def ensure_graph_references(result: Dict[str, List[Any]]) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """Ensure predicate entities and action subjects have term/med-term records.

    This is a deterministic referential-integrity repair. It does not claim
    terminology verification; auto-created records are low-confidence candidates.
    """
    terms = list(result.get("terms", []) or [])
    med_terms = list(result.get("med_terms", []) or [])
    predicates = list(result.get("predicates", []) or [])
    rules = list(result.get("rules", []) or [])

    term_ids = {_item_id(item) for item in terms}
    med_ids = {_item_id(item) for item in med_terms}
    created_terms: List[str] = []
    created_meds: List[str] = []

    for predicate in predicates:
        entity_id = _get(predicate, "entity")
        if not entity_id or entity_id in term_ids or entity_id in med_ids:
            continue
        created = _minimal_term_from_predicate(predicate)
        if created is None:
            continue
        if isinstance(created, MedicationTerm):
            med_terms.append(created)
            med_ids.add(created.id)
            created_meds.append(created.id)
        else:
            terms.append(created)
            term_ids.add(created.id)
            created_terms.append(created.id)

    for rule in rules:
        action = _get(rule, "action", {}) or {}
        subjects = _get(action, "subjects", []) or []
        for subject_id in subjects:
            if not subject_id or subject_id in term_ids or subject_id in med_ids:
                continue
            if subject_id.startswith("med."):
                med = _minimal_med_from_subject(subject_id, rule)
                med_terms.append(med)
                med_ids.add(med.id)
                created_meds.append(med.id)

    repaired = {
        **result,
        "terms": terms,
        "med_terms": med_terms,
        "predicates": predicates,
        "rules": rules,
    }
    report = {
        "auto_created_terms": created_terms,
        "auto_created_med_terms": created_meds,
        "auto_created_term_count": len(created_terms),
        "auto_created_med_term_count": len(created_meds),
    }
    return repaired, report
