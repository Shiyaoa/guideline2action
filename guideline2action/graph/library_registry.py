"""Single source of truth for v2 registered library function ids."""
from __future__ import annotations

from typing import Dict, List, Optional


V2_LIBRARY_FUNCTION_SPECS: List[Dict] = [
    {
        "id": "lib.fhir.filter_by_status",
        "name": "Filter resources by status",
        "input_type": "List<Resource>",
        "output_type": "List<Resource>",
        "parameters": [
            {"name": "resource_list", "type": "List<Resource>"},
            {"name": "status_field", "type": "String"},
            {"name": "allowed_values", "type": "List<Code>"},
        ],
        "implementation_semantics": "Return resources whose status field is in allowed_values.",
        "cql_equivalent": "Verified, Confirmed, Completed, ActiveMedicationStatement status checks",
    },
    {
        "id": "lib.fhir.filter_by_lookback",
        "name": "Filter resources by lookback interval",
        "input_type": "List<Resource>",
        "output_type": "List<Resource>",
        "parameters": [
            {"name": "resource_list", "type": "List<Resource>"},
            {"name": "time_paths", "type": "List<String>"},
            {"name": "interval", "type": "Interval<DateTime>"},
        ],
        "implementation_semantics": "Return resources whose available time value falls within or overlaps the lookback interval.",
        "cql_equivalent": "ObservationLookBack, ConditionLookBack, ProcedureLookBack, MedicationStatementLookBack",
    },
    {
        "id": "lib.fhir.most_recent",
        "name": "Most recent resource",
        "input_type": "List<Resource>",
        "output_type": "Resource",
        "parameters": [
            {"name": "resource_list", "type": "List<Resource>"},
            {"name": "date_fallback_paths", "type": "List<String>"},
        ],
        "implementation_semantics": "Sort resources by the first non-null date fallback value and return the latest resource.",
        "cql_equivalent": "MostRecent, MostRecentCondition, MostRecentProcedure",
    },
    {
        "id": "lib.fhir.count",
        "name": "Count resources",
        "input_type": "List<Resource>",
        "output_type": "Integer",
        "parameters": [{"name": "resource_list", "type": "List<Resource>"}],
        "implementation_semantics": "Return the count of all resources in the input list.",
        "cql_equivalent": "Count([Resource: ValueSet])",
    },
    {
        "id": "lib.fhir.count_in_window",
        "name": "Count resources in lookback window",
        "input_type": "List<Resource>",
        "output_type": "Integer",
        "parameters": [
            {"name": "resource_list", "type": "List<Resource>"},
            {"name": "time_paths", "type": "List<String>"},
            {"name": "lookback", "type": "Quantity"},
        ],
        "implementation_semantics": "Apply filter_by_lookback and return the resulting count. Canonical rules should compare this Integer in a Rule DAG compare node.",
        "cql_equivalent": "Count(ObservationLookBack(...)) or Count(ConditionLookBack(...))",
    },
    {
        "id": "lib.fhir.count_all_time",
        "name": "Count resources across all history",
        "input_type": "List<Resource>",
        "output_type": "Integer",
        "parameters": [{"name": "resource_list", "type": "List<Resource>"}],
        "implementation_semantics": "Return the count of all resources in the input list. Canonical rules should compare this Integer in a Rule DAG compare node.",
        "cql_equivalent": "Count([Resource: ValueSet])",
    },
    {
        "id": "lib.fhir.extremum",
        "name": "Resource quantity extremum",
        "input_type": "List<Resource>",
        "output_type": "Quantity",
        "parameters": [
            {"name": "resource_list", "type": "List<Resource>"},
            {"name": "quantity_path", "type": "String"},
            {"name": "extremum_type", "type": "max|min"},
        ],
        "implementation_semantics": "Extract comparable quantities and return the maximum or minimum after unit normalization when possible.",
        "cql_equivalent": "HighestObservation and analogous min/max helpers",
    },
    {
        "id": "lib.fhir.concept_in_valueset",
        "name": "Concept value in value set",
        "input_type": "Concept",
        "output_type": "Bool",
        "parameters": [
            {"name": "concept", "type": "Concept"},
            {"name": "valueset", "type": "ValueSet"},
        ],
        "implementation_semantics": "Return true when any coding in the concept belongs to the referenced value set.",
        "cql_equivalent": "ConceptValue(...) in ValueSet",
    },
    {
        "id": "lib.fhir.condition.active",
        "name": "Active FHIR Condition",
        "input_type": "List<Condition>",
        "output_type": "List<Condition>",
        "parameters": [{"name": "conditions", "type": "List<Condition>"}],
        "implementation_semantics": "Return active conditions, including resource-specific clinicalStatus and abatement semantics.",
        "cql_equivalent": "ActiveCondition",
    },
    {
        "id": "lib.fhir.medication_statement.active",
        "name": "Active FHIR MedicationStatement",
        "input_type": "List<MedicationStatement>",
        "output_type": "List<MedicationStatement>",
        "parameters": [{"name": "medication_statements", "type": "List<MedicationStatement>"}],
        "implementation_semantics": "Return active medication statements, excluding wasNotTaken and ended administrations.",
        "cql_equivalent": "ActiveMedicationStatement",
    },
    {
        "id": "lib.fhir.medication_order.active",
        "name": "Active FHIR MedicationOrder",
        "input_type": "List<MedicationRequest>",
        "output_type": "List<MedicationRequest>",
        "parameters": [{"name": "medication_orders", "type": "List<MedicationRequest>"}],
        "implementation_semantics": "Return active or currently valid medication orders/requests.",
        "cql_equivalent": "ActiveMedicationOrder",
    },
    {
        "id": "lib.fhir.procedure.completed",
        "name": "Completed FHIR Procedure",
        "input_type": "List<Procedure>",
        "output_type": "List<Procedure>",
        "parameters": [{"name": "procedures", "type": "List<Procedure>"}],
        "implementation_semantics": "Return procedures whose status and timing indicate completion.",
        "cql_equivalent": "Completed",
    },
    {
        "id": "lib.fhir.observation.verified",
        "name": "Verified FHIR Observation",
        "input_type": "List<Observation>",
        "output_type": "List<Observation>",
        "parameters": [{"name": "observations", "type": "List<Observation>"}],
        "implementation_semantics": "Return observations whose status is final, amended, corrected, or otherwise clinically verified.",
        "cql_equivalent": "Verified",
    },
]


_SPECS_BY_ID: Dict[str, Dict] = {spec["id"]: spec for spec in V2_LIBRARY_FUNCTION_SPECS}


def get_v2_library_function_specs() -> List[Dict]:
    """Return copies of registered v2 library function specs."""
    return [dict(spec) for spec in V2_LIBRARY_FUNCTION_SPECS]


def get_registered_library_function_ids() -> List[str]:
    """Return registered function ids in stable declaration order."""
    return [spec["id"] for spec in V2_LIBRARY_FUNCTION_SPECS]


def is_registered_library_function(function_id: Optional[str]) -> bool:
    """Check whether a library function id is registered."""
    return bool(function_id) and function_id in _SPECS_BY_ID
