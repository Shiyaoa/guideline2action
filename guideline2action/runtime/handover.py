"""Build one clinically readable trace for augmentation and refinement."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


CLINICAL_TRACE_VERSION = "agentic_kg.clinical_trace.v2"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _unique(values: Iterable[Any]) -> List[Any]:
    output: List[Any] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, "", [], {}):
            continue
        marker = repr(value)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


def _trace_body(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = trace.get("trace")
    return _mapping(nested) if isinstance(nested, Mapping) else trace


def _graph_body(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = graph.get("graph")
    return _mapping(nested) if isinstance(nested, Mapping) else graph


def _truth(value: Any) -> str:
    item = _mapping(value)
    truth = str(item.get("truth") or "").strip().lower()
    if truth in {"true", "false", "unknown"}:
        return truth
    fallback = str(item.get("value") or "").strip().lower()
    return fallback if fallback in {"true", "false", "unknown"} else "unknown"


def _humanize(value: Any) -> str:
    text = str(value or "").strip()
    for prefix in (
        "predicate::",
        "action::",
        "clinical_assessment_spec::",
        "assessment_spec.",
        "dag.",
        "pred.",
        "root.",
        "rule.",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = re.sub(r"\bq\d+\b[._-]*", "", text)
    text = text.replace("::", " ").replace(".", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip(" -")


_CLINICAL_TERMS = {
    "pml rara": "PML-RARA",
    "vod sos": "VOD/SOS",
    "sos vod": "SOS/VOD",
    "atra ato": "ATRA/ATO",
    "ta tma": "TA-TMA",
    "tat tma": "TA-TMA",
    "g csf": "G-CSF",
    "7plus3": "7+3",
}

_ACRONYMS = {
    "agvhd",
    "aki",
    "aml",
    "apl",
    "bae",
    "bal",
    "balf",
    "bos",
    "cbc",
    "cdi",
    "cmv",
    "cni",
    "cop",
    "crkp",
    "crrt",
    "csa",
    "cyp3a",
    "cyp3a4",
    "dic",
    "dli",
    "dlco",
    "ecg",
    "flt3",
    "gvhd",
    "gvl",
    "hidac",
    "hlh",
    "hma",
    "hsct",
    "icp",
    "ipa",
    "ips",
    "iv",
    "ivc",
    "lp",
    "mrd",
    "nec",
    "pcp",
    "pjp",
    "ppfe",
    "pres",
    "qtc",
    "rasburicase",
    "rrt",
    "tblb",
    "tls",
    "tma",
    "tpe",
    "ttp",
    "wbc",
}


def _clinical_language(value: Any) -> str:
    text = _humanize(value).lower()
    for source, target in _CLINICAL_TERMS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)
    words = [word.upper() if word.lower() in _ACRONYMS else word for word in text.split()]
    return " ".join(words)


def clinical_decision_name(root_id: str) -> str:
    """Turn an internal Boolean-root name into a readable clinical proposition."""
    phrase = _clinical_language(root_id)
    for prefix in ("root ", "state ", "gate ", "assessment ", "knowledge ", "r "):
        if phrase.lower().startswith(prefix):
            phrase = phrase[len(prefix) :]
            break
    lower = phrase.lower()
    if lower.endswith(" criteria met"):
        sentence = f"{phrase[:-4].strip()} are met."
    elif lower.endswith(" indicated"):
        sentence = f"{phrase[:-10].strip()} is indicated."
    elif lower.endswith(" detected"):
        sentence = f"{phrase[:-9].strip()} is detected."
    elif lower.endswith(" present"):
        sentence = f"{phrase[:-8].strip()} is present."
    elif lower.endswith(" working diagnosis"):
        sentence = f"{phrase[:-18].strip()} is the working diagnosis."
    elif lower.endswith(" likelihood"):
        sentence = f"{phrase[:-11].strip()} is clinically likely."
    elif " eligible for " in lower:
        subject, target = re.split(r"\beligible for\b", phrase, maxsplit=1, flags=re.IGNORECASE)
        sentence = f"{subject.strip()} is eligible for {target.strip()}."
    elif " ineligible for " in lower:
        subject, target = re.split(r"\bineligible for\b", phrase, maxsplit=1, flags=re.IGNORECASE)
        sentence = f"{subject.strip()} is ineligible for {target.strip()}."
    elif lower.endswith(" eligible"):
        sentence = f"The patient is eligible for {phrase[:-9].strip()}."
    elif lower.startswith(
        (
            "avoid ",
            "consider ",
            "continue ",
            "defer ",
            "hold ",
            "initiate ",
            "interrupt ",
            "monitor ",
            "perform ",
            "reduce ",
            "repeat ",
            "require ",
            "reserve ",
            "resume ",
            "select ",
            "start ",
            "stop ",
            "urgently ",
        )
    ):
        sentence = f"The patient-specific conditions to {phrase} are met."
    elif lower.startswith(
        (
            "classic ",
            "clinical ",
            "highly suspected ",
            "leading ",
            "possible ",
            "probable ",
            "suspected ",
        )
    ) or " compatible " in f" {lower} ":
        sentence = f"The clinical pattern supports {phrase}."
    else:
        sentence = f"The patient meets the clinical conditions for {phrase}."
    return sentence[:1].upper() + sentence[1:]


def _predicate_label(
    predicate_id: str,
    predicates: Mapping[str, Mapping[str, Any]],
    graph_nodes: Mapping[str, Mapping[str, Any]],
) -> str:
    predicate = _mapping(predicates.get(predicate_id))
    node = _mapping(graph_nodes.get(f"predicate::{predicate_id}"))
    return _clinical_language(
        node.get("label") or predicate.get("name") or predicate.get("label") or predicate_id
    )


def _assessment_label(
    assessment_id: str,
    assessments: Mapping[str, Mapping[str, Any]],
) -> str:
    assessment = _mapping(assessments.get(assessment_id))
    return str(assessment.get("question") or assessment.get("clinical_label") or _clinical_language(assessment_id)).strip()


def _action_label(trace_action: Mapping[str, Any], graph_action: Mapping[str, Any]) -> str:
    output = _mapping(graph_action.get("output"))
    source = _mapping(graph_action.get("source_payload"))
    return str(
        output.get("text")
        or output.get("clinical_guidance")
        or source.get("value")
        or trace_action.get("label")
        or ""
    ).strip()


def _component_texts(graph_action: Mapping[str, Any]) -> List[str]:
    source = _mapping(graph_action.get("source_payload"))
    values: List[str] = []
    for component in _items(source.get("components")):
        value = str(
            component.get("clinical_label")
            or component.get("clinical_guidance")
            or component.get("text")
            or component.get("value")
            or ""
        ).strip()
        if value:
            values.append(value)
    for key in ("requirements", "monitoring"):
        for component in _items(graph_action.get(key)):
            value = str(
                component.get("clinical_label")
                or component.get("clinical_guidance")
                or component.get("text")
                or component.get("value")
                or ""
            ).strip()
            if value:
                values.append(value)
    return _unique(values)


def _boundary_text(graph_action: Mapping[str, Any]) -> str:
    source = _mapping(graph_action.get("source_payload"))
    if source.get("route_not_selected"):
        return "The represented route does not choose between the available treatment-adjustment options."
    if source.get("does_not_exclude_alternatives"):
        return "The represented diagnosis does not exclude alternative diagnoses."
    if source.get("dose_not_specified"):
        return "The represented action does not specify a dose."
    return ""


def _time_text(record: Mapping[str, Any]) -> str:
    temporal = _mapping(record.get("temporal"))
    values = [
        str(value).strip()
        for value in (temporal.get("relation"), temporal.get("offset"), record.get("timepoint"))
        if value not in (None, "")
    ]
    return " ".join(_unique(values))


def _observed_value(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    value = record.get("value")
    if value is None or value is True:
        return None
    output: Dict[str, Any] = {"value": value}
    if record.get("unit"):
        output["unit"] = record["unit"]
    return output


def _build_observations(trace: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    output: List[Dict[str, Any]] = []
    key_by_source: Dict[str, str] = {}
    group_by_evidence: Dict[tuple[Any, ...], Dict[str, Any]] = {}

    def add(record: Mapping[str, Any], source_id: str) -> None:
        span = _mapping(record.get("source_span"))
        text = str(span.get("text") or "").strip()
        time = _time_text(record)
        evidence_key = (
            text or _clinical_language(record.get("concept_id") or record.get("condition_concept_id")),
            span.get("start"),
            span.get("end"),
            time,
        )
        item = group_by_evidence.get(evidence_key)
        if item is None:
            item = {
                "key": f"O{len(output) + 1}",
                "clinical_observation": evidence_key[0],
            }
            if time:
                item["time"] = time
            polarity = str(record.get("polarity") or "present").strip()
            if polarity != "present":
                item["polarity"] = polarity
            output.append(item)
            group_by_evidence[evidence_key] = item
        observed = _observed_value(record)
        if observed is not None:
            item["values"] = _unique([*item.get("values", []), observed])
        key_by_source[source_id] = str(item["key"])

    for observation in _items(trace.get("observations")):
        source_id = str(observation.get("observation_id") or "")
        if source_id:
            add(observation, source_id)
    for assertion in _items(trace.get("clinical_assertions")):
        source_id = str(assertion.get("assertion_id") or "")
        if source_id:
            add(assertion, source_id)
    return output, key_by_source


def _logic_expression(
    node_id: str,
    nodes: Mapping[str, Mapping[str, Any]],
    fact_key_by_ref: Mapping[tuple[str, str], str],
) -> Any:
    node = _mapping(nodes.get(node_id))
    node_type = str(node.get("node_type") or node.get("type") or "")
    if node_type == "literal":
        # A rule pinned to an unresolved constant has no fact operands.
        return {"literal": node.get("value")}
    if node_type == "predicate_ref":
        return fact_key_by_ref[("predicate", str(node.get("predicate_ref") or ""))]
    if node_type == "clinical_assessment_ref":
        assessment_id = str(
            node.get("assessment_spec_id")
            or node.get("clinical_assessment_spec_ref")
            or ""
        )
        return fact_key_by_ref[("assessment", assessment_id)]
    if node_type != "combine":
        raise ValueError(f"unsupported clinical decision node {node_id}: {node_type}")
    children = [
        _logic_expression(str(child), nodes, fact_key_by_ref)
        for child in node.get("inputs") or []
    ]
    operator = str(node.get("operator") or "").upper()
    if operator == "AND":
        return {"all": children}
    if operator == "OR":
        return {"any": children}
    if operator == "NOT":
        return {"not": children[0]}
    if operator == "AT_LEAST":
        minimum = int(_mapping(node.get("parameters")).get("minimum"))
        return {"at_least": {"minimum": minimum, "of": children}}
    raise ValueError(f"unsupported clinical decision operator {operator} in {node_id}")


def evaluate_logic(expression: Any, fact_states: Mapping[str, str]) -> str:
    """Evaluate a projected Boolean expression with true/false/unknown values."""
    if isinstance(expression, str):
        return fact_states[expression]
    if not isinstance(expression, Mapping):
        raise ValueError(f"invalid clinical decision expression: {expression!r}")
    if "literal" in expression:
        value = expression["literal"]
        if value is None:
            return "unknown"
        return "true" if value else "false"
    if "all" in expression:
        values = [evaluate_logic(item, fact_states) for item in expression["all"]]
        if "false" in values:
            return "false"
        return "unknown" if "unknown" in values else "true"
    if "any" in expression:
        values = [evaluate_logic(item, fact_states) for item in expression["any"]]
        if "true" in values:
            return "true"
        return "unknown" if "unknown" in values else "false"
    if "not" in expression:
        value = evaluate_logic(expression["not"], fact_states)
        return {"true": "false", "false": "true", "unknown": "unknown"}[value]
    if "at_least" in expression:
        spec = _mapping(expression["at_least"])
        values = [evaluate_logic(item, fact_states) for item in spec.get("of") or []]
        minimum = int(spec.get("minimum"))
        true_count = values.count("true")
        unknown_count = values.count("unknown")
        if true_count >= minimum:
            return "true"
        if true_count + unknown_count < minimum:
            return "false"
        return "unknown"
    raise ValueError(f"invalid clinical decision expression: {expression!r}")


def project_clinical_trace(
    trace: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project a raw execution trace into observation -> fact -> decision -> action."""
    source_trace = _trace_body(trace)
    source_graph = _graph_body(graph)
    predicates = {
        str(item.get("id")): item
        for item in _items(source_graph.get("predicates"))
        if item.get("id")
    }
    graph_nodes = {
        str(item.get("id")): item
        for item in _items(source_graph.get("nodes"))
        if item.get("id")
    }
    assessments = {
        str(item.get("assessment_spec_id") or item.get("id")): item
        for item in _items(source_graph.get("clinical_assessment_specs"))
        if item.get("assessment_spec_id") or item.get("id")
    }
    graph_actions = {
        str(item.get("id")): item
        for item in _items(source_graph.get("actions"))
        if item.get("id")
    }
    graph_rules = {
        str(item.get("id")): item
        for item in _items(source_graph.get("rules"))
        if item.get("id")
    }
    predicate_results = _mapping(source_trace.get("predicate_results"))
    assessment_results = _mapping(source_trace.get("clinical_assessment_results"))
    condition_results = _mapping(source_trace.get("condition_node_results"))
    observations, observation_key_by_source = _build_observations(source_trace)

    facts: List[Dict[str, Any]] = []
    fact_key_by_ref: Dict[tuple[str, str], str] = {}
    for predicate_id, value in predicate_results.items():
        result = _mapping(value)
        key = f"F{len(facts) + 1}"
        fact_key_by_ref[("predicate", str(predicate_id))] = key
        fact: Dict[str, Any] = {
            "key": key,
            "clinical_fact": _predicate_label(str(predicate_id), predicates, graph_nodes),
            "state": _truth(result),
        }
        based_on = _unique(
            observation_key_by_source[str(source_id)]
            for source_id in result.get("fact_ids") or []
            if str(source_id) in observation_key_by_source
        )
        if based_on:
            fact["based_on"] = based_on
        facts.append(fact)

    for assessment_id, value in assessment_results.items():
        result = _mapping(value)
        key = f"F{len(facts) + 1}"
        fact_key_by_ref[("assessment", str(assessment_id))] = key
        fact = {
            "key": key,
            "clinical_fact": _assessment_label(str(assessment_id), assessments),
            "state": _truth(result),
        }
        based_on = _unique(
            observation_key_by_source[str(source_id)]
            for source_id in result.get("fact_ids") or []
            if str(source_id) in observation_key_by_source
        )
        if based_on:
            fact["based_on"] = based_on
        facts.append(fact)

    decisions_by_root: Dict[str, Dict[str, Any]] = {}
    decision_key_by_rule: Dict[str, str] = {}
    for rule_result in _items(source_trace.get("rule_results")):
        rule_id = str(rule_result.get("rule_id") or "")
        rule = _mapping(graph_rules.get(rule_id))
        dag = _mapping(rule.get("condition_dag"))
        root_id = str(
            dag.get("boolean_root")
            or dag.get("root")
            or rule_result.get("root_id")
            or ""
        )
        if root_id not in decisions_by_root:
            nodes = {
                str(item.get("id")): item
                for item in _items(dag.get("nodes"))
                if item.get("id")
            }
            root_result = _mapping(condition_results.get(root_id)) or rule_result
            # A graph may state the clinical proposition verbatim (projected
            # graphs carry the rule label); otherwise derive it from the root id.
            stated = str(
                dag.get("clinical_decision")
                or dag.get("label")
                or rule.get("clinical_decision")
                or rule.get("label")
                or ""
            ).strip()
            decisions_by_root[root_id] = {
                "key": f"D{len(decisions_by_root) + 1}",
                "clinical_decision": stated or clinical_decision_name(root_id),
                "state": _truth(root_result),
                "logic": _logic_expression(root_id, nodes, fact_key_by_ref),
            }
        decision_key_by_rule[rule_id] = str(decisions_by_root[root_id]["key"])

    rule_ids_by_action: Dict[str, List[str]] = {}
    for rule in _items(source_graph.get("rules")):
        rule_id = str(rule.get("id") or "")
        for action_ref in rule.get("action_refs") or []:
            action_id = str(
                action_ref.get("id") if isinstance(action_ref, Mapping) else action_ref
            )
            if action_id:
                rule_ids_by_action.setdefault(action_id, []).append(rule_id)

    actions: List[Dict[str, Any]] = []
    for trace_action in _items(source_trace.get("actions")):
        action_id = str(trace_action.get("action_id") or "")
        graph_action = _mapping(graph_actions.get(action_id))
        action: Dict[str, Any] = {
            "clinical_guidance": _action_label(trace_action, graph_action),
            "state": str(trace_action.get("activation_status") or "unresolved"),
            "because": _unique(
                decision_key_by_rule[rule_id]
                for rule_id in rule_ids_by_action.get(action_id, [])
                if rule_id in decision_key_by_rule
            ),
        }
        components = _component_texts(graph_action)
        if components:
            action["source_supported_components"] = components
        boundary = _boundary_text(graph_action)
        if boundary:
            action["information_boundary"] = boundary
        parameters = {
            key: graph_action.get(key)
            for key in ("dose", "duration", "timing")
            if graph_action.get(key) not in (None, "", [], {})
        }
        if parameters:
            action["parameters"] = parameters
        actions.append(action)

    return {
        "projection_version": CLINICAL_TRACE_VERSION,
        "qid": str(source_trace.get("qid") or trace.get("qid") or ""),
        "observations": observations,
        "facts": facts,
        "decisions": list(decisions_by_root.values()),
        "actions": actions,
        "information_boundaries": [
            "KG non-coverage does not make unlisted content incorrect, prohibited, or deletable.",
            "Unknown facts and unresolved actions must remain uncertain.",
        ],
    }


def get_clinical_trace(
    trace: Mapping[str, Any],
    graph: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Read an embedded clinical trace or build it from the raw trace and graph."""
    source_trace = _trace_body(trace)
    embedded = source_trace.get("clinical_trace")
    if isinstance(embedded, Mapping):
        return copy.deepcopy(dict(embedded))
    if graph is None:
        raise ValueError("raw trace requires its graph to build the clinical trace")
    return project_clinical_trace(trace, graph)


def attach_clinical_trace(
    trace: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach the shared downstream projection to a newly generated raw trace."""
    enriched = copy.deepcopy(dict(trace))
    enriched["clinical_trace"] = project_clinical_trace(trace, graph)
    return enriched


__all__ = [
    "CLINICAL_TRACE_VERSION",
    "attach_clinical_trace",
    "clinical_decision_name",
    "evaluate_logic",
    "get_clinical_trace",
    "project_clinical_trace",
]
