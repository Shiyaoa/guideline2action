"""Deterministic Missing Variable Derivation.

Scans each ClinicalRule's ConditionDAG for predicate_ref references, cross-references
them against the extracted Predicates list, and produces MissingVariableSpec lists
per rule. This is a deterministic post-processing pass -- no LLM calls.
"""

from typing import List, Dict, Any, Optional, Set, Tuple

from .models import (
    ClinicalRule,
    Predicates,
    ConditionDAG,
    DAGNode,
)

# ============ Variable Type Mapping ============

# Map Predicate aspect to ClinicalVariableType string values.
# These align with the patient-state ClinicalVariableType values used by the harness.
ASPECT_TO_VARIABLE_TYPE: Dict[str, str] = {
    "existence": "diagnosis",
    "diagnostic_criterion": "diagnosis",
    "molecular_marker": "molecular_marker",
    "risk_group": "risk_stratification",
    "quantity": "laboratory_value",
    "quantity_range": "laboratory_value",
    "status": "clinical_status",
    "treatment_eligibility": "treatment_history",
    "time_since_event": "temporal",
    "complication_grade": "clinical_status",
    "organ_function": "organ_function",
    "infection_marker": "laboratory_value",
    "drug_interaction": "medication",
    "contraindication": "contraindication",
    "dose_constraint": "medication",
    "stage": "clinical_status",
    "delta": "laboratory_value",
    "duration": "temporal",
    "risk": "risk_stratification",
}

# Map Predicate entity_type to ClinicalVariableType.
ENTITY_TYPE_TO_VARIABLE_TYPE: Dict[str, str] = {
    "condition": "diagnosis",
    "observation": "laboratory_value",
    "medication": "medication",
    "procedure": "procedure",
    "resource": "clinical_status",
}

# Fallback variable type when mapping cannot determine.
FALLBACK_VARIABLE_TYPE = "clinical_status"

# ============ DAG Walking ============


def _collect_predicate_refs(condition_dag: ConditionDAG) -> Set[str]:
    """Walk all DAG nodes and collect unique predicate_ref references.

    Args:
        condition_dag: The rule's ConditionDAG to walk.

    Returns:
        Set of predicate_ref string values found in the DAG nodes.
    """
    refs: Set[str] = set()
    for node in condition_dag.nodes:
        ref = getattr(node, "predicate_ref", None)
        if ref:
            refs.add(ref)
        # Also check inputs list for predicate_ref-style strings
        inputs = getattr(node, "inputs", None) or []
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, str) and inp.startswith("pred."):
                    refs.add(inp)
        # Check left/right for predicate refs
        if isinstance(getattr(node, "left", None), str) and getattr(node, "left", "").startswith("pred."):
            refs.add(node.left)
        if isinstance(getattr(node, "right", None), str) and getattr(node, "right", "").startswith("pred."):
            refs.add(node.right)
        # Check single input
        inp = getattr(node, "input", None)
        if isinstance(inp, str) and inp.startswith("pred."):
            refs.add(inp)
    return refs


def _resolve_variable_type(predicate: Predicates) -> str:
    """Determine the ClinicalVariableType for a predicate.

    Uses the predicate's entity_type first for medication (to avoid
    over-mapping existence->diagnosis for medication predicates),
    then aspect, then entity_type as fallback.

    Args:
        predicate: The Predicates object to map.

    Returns:
        ClinicalVariableType string value.
    """
    entity_type = getattr(predicate, "entity_type", None)

    # Medication entity_type takes priority to avoid "existence" -> "diagnosis" mapping
    if entity_type == "medication":
        return ENTITY_TYPE_TO_VARIABLE_TYPE.get(entity_type, "medication")

    aspect = getattr(predicate, "aspect", None)
    if aspect:
        var_type = ASPECT_TO_VARIABLE_TYPE.get(aspect)
        if var_type:
            return var_type

    if entity_type:
        var_type = ENTITY_TYPE_TO_VARIABLE_TYPE.get(entity_type)
        if var_type:
            return var_type

    return FALLBACK_VARIABLE_TYPE


def derive_missing_variables(
    rules: List[ClinicalRule],
    predicates: List[Predicates],
) -> Dict[str, List[Dict[str, Any]]]:
    """Derive MissingVariableSpec lists for each ClinicalRule.

    For each rule:
    1. Walk its ConditionDAG collecting predicate_ref references.
    2. Look up each referenced predicate in the extracted Predicates list.
    3. Map each predicate to a ClinicalVariableType.
    4. Produce a MissingVariableSpec dict per unique variable.

    Args:
        rules: List of extracted ClinicalRules.
        predicates: List of extracted Predicates objects.

    Returns:
        Dict keyed by rule_id, each value a list of MissingVariableSpec dicts.
        Rules with no predicate dependencies get empty lists.
    """
    # Build predicate lookup by id
    pred_by_id: Dict[str, Predicates] = {}
    for pred in predicates:
        pid = getattr(pred, "id", None)
        if pid:
            pred_by_id[pid] = pred

    result: Dict[str, List[Dict[str, Any]]] = {}

    for rule in rules:
        rule_id = getattr(rule, "id", "")
        missing_data_policy = getattr(rule, "missing_data_policy", "propagate_unknown")

        if not rule_id:
            continue

        cond_dag = getattr(rule, "condition_dag", None)
        if not cond_dag:
            result[rule_id] = []
            continue

        predicate_refs = _collect_predicate_refs(cond_dag)

        if not predicate_refs:
            result[rule_id] = []
            continue

        # Build variable specs -- one per unique predicate ref
        # Map null_policy from rule-level policy
        policy_map = {
            "suppress": "suppress",
            "warn": "warn",
            "assume_false": "assume_false",
            "propagate_unknown": "propagate_unknown",
        }
        null_policy = policy_map.get(missing_data_policy, "propagate_unknown")

        # Group predicate refs by variable (deduplicate by entity)
        seen_entities: Set[str] = set()
        specs: List[Dict[str, Any]] = []

        for pred_ref in sorted(predicate_refs):
            pred = pred_by_id.get(pred_ref)
            entity = getattr(pred, "entity", pred_ref) if pred else pred_ref

            # Deduplicate: one MissingVariableSpec per unique entity
            if entity in seen_entities:
                continue
            seen_entities.add(entity)

            var_type = _resolve_variable_type(pred) if pred else FALLBACK_VARIABLE_TYPE
            var_name = (
                getattr(pred, "name", entity)
                if pred
                else pred_ref
            )

            spec = {
                "variable_name": var_name,
                "variable_type": var_type,
                "required_by_node_ids": [pred_ref],
                "question_to_clinician": None,
                "null_policy": null_policy,
            }
            specs.append(spec)

            # Re-check all other refs that resolve to the same entity
            for other_ref in sorted(predicate_refs):
                if other_ref == pred_ref:
                    continue
                other_pred = pred_by_id.get(other_ref)
                if other_pred and getattr(other_pred, "entity", "") == entity:
                    # Add this node id to the existing spec
                    spec["required_by_node_ids"].append(other_ref)

        result[rule_id] = specs

    return result
