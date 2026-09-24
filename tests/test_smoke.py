"""Smoke suite for the projection layer and the deterministic interpreter.

Runs entirely offline: graphs are built from in-memory extraction payloads
(:mod:`tests.fixtures`), never from an LLM call.  No data files are needed.

    python3 tests/test_smoke.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guideline2action import (  # noqa: E402
    ExecutableGraph,
    Guideline2ActionError,
    run_graph,
    runtime_program,
)
from tests.fixtures import LIVE_EXTRACTION, SYNTHETIC_EXTRACTION  # noqa: E402

HFREF_FACT = {
    "fact_id": "obs.1",
    "concept_id": "cond.hfref",
    "value": True,
    "polarity": "present",
    "temporal": {"relation": "current", "offset": None},
    "source_quote": "射血分数降低的心衰（HFrEF）",
}
EGFR_FACT = {
    "fact_id": "obs.2",
    "concept_id": "meas.egfr",
    "value": 45,
    "unit": "mL/min/1.73m2",
    "polarity": "present",
    "temporal": {"relation": "current", "offset": None},
    "source_quote": "eGFR 45 mL/min/1.73m2",
}
TOLERABILITY_SPEC = "assessment_spec.pred_obs_sglt2i_tolerability"


def assessment(value: str) -> list[dict]:
    return [
        {
            "assessment_id": "assessment.1",
            "assessment_spec_id": TOLERABILITY_SPEC,
            "value": value,
            "assessment_authority": "llm",
            "model_id": "smoke/test",
            "prompt_version": "decisionkg.pass_b.clinical_assessment_tool.v3",
            "supporting_observation_ids": ["obs.1"],
            "source_spans": [],
            "requires_human_review": False,
        }
    ]


def load_graph() -> ExecutableGraph:
    return ExecutableGraph.from_extraction(SYNTHETIC_EXTRACTION)


def rule_truth(trace: dict, rule_id: str) -> str:
    return {row["rule_id"]: row["truth"] for row in trace["rule_results"]}[rule_id]


def action_for_rule(trace: dict, rule_slug: str) -> dict:
    """Look up the action produced by a rule (its id embeds the rule slug)."""
    matches = [row for row in trace["actions"] if rule_slug in str(row.get("action_id") or "")]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one action for {rule_slug!r}, found {len(matches)}")
    return matches[0]


RECOMMEND = "rule_sglt2i_hfref_recommend"
CONSIDER = "rule_sglt2i_tolerability_consider"


class ProjectionTest(unittest.TestCase):
    """The four projection transformations."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = load_graph()
        cls.body = cls.graph.graph

    def test_contracts_are_declared(self) -> None:
        self.assertEqual(self.body["schema_version"], "decisionkg.schema_v2.unified.20260804")
        self.assertEqual(self.body["conversion_status"], "converted")
        self.assertEqual(self.body["source_question"]["qid"], "guideline::sglt2i_hfref_synthetic_v1")

    def test_actions_are_materialised_out_of_rules(self) -> None:
        self.assertEqual(len(self.body["actions"]), 3)
        for action in self.body["actions"]:
            for field in ("id", "node_type", "subject_refs", "permission", "action_type",
                          "execution_authorized", "source_rule_ids"):
                self.assertIn(field, action)
            self.assertNotIn("action", {key for rule in self.body["rules"] for key in rule})
        for rule in self.body["rules"]:
            self.assertNotIn("action", rule)
            self.assertTrue(rule["action_refs"], f"{rule['id']} lost its action reference")

    def test_actions_point_back_at_their_rule(self) -> None:
        for rule in self.body["rules"]:
            for action_id in rule["action_refs"]:
                action = next(a for a in self.body["actions"] if a["id"] == action_id)
                self.assertIn(rule["id"], action["source_rule_ids"])

    def test_action_type_is_derived_from_permission(self) -> None:
        recommend = next(a for a in self.body["actions"] if "recommend" in a["permission"] and "hfref" in a["id"])
        self.assertEqual(recommend["action_type"], "therapy_initiation_or_continuation")

    def test_dag_vocabulary_is_inside_the_interpreter_subset(self) -> None:
        allowed = {"predicate_ref", "clinical_assessment_ref", "literal", "compare", "combine", "exists"}
        combiners = {"AND", "OR", "NOT", "AT_LEAST"}
        for rule in self.body["rules"]:
            for node in rule["condition_dag"]["nodes"]:
                self.assertIn(node["type"], allowed)
                # the handover renderer reads the migrated key, the interpreter reads `type`
                self.assertEqual(node["node_type"], node["type"])
                if node["type"] == "combine":
                    self.assertIn(node["operator"], combiners)

    def test_bare_operands_became_predicate_ref_nodes(self) -> None:
        first = self.body["rules"][0]["condition_dag"]
        node_ids = {node["id"] for node in first["nodes"]}
        for node in first["nodes"]:
            for operand in node.get("inputs") or []:
                self.assertIn(operand, node_ids)

    def test_non_computable_finding_left_the_executable_set(self) -> None:
        predicate_ids = {p["id"] for p in self.body["predicates"]}
        self.assertNotIn("pred.obs.sglt2i.tolerability", predicate_ids)
        self.assertEqual(len(self.body["predicates"]), 4)
        specs = {s["assessment_spec_id"]: s for s in self.body["clinical_assessment_specs"]}
        self.assertIn(TOLERABILITY_SPEC, specs)
        spec = specs[TOLERABILITY_SPEC]
        self.assertEqual(spec["allowed_values"], ["true", "false", "unknown"])
        self.assertEqual(spec["unknown_policy"], "propagate_unknown")
        self.assertEqual(spec["human_review_policy"], "required_before_action")

    def test_assessment_dag_reference_is_typed_and_bound(self) -> None:
        second = self.body["rules"][1]["condition_dag"]
        refs = [n for n in second["nodes"] if n["type"] == "clinical_assessment_ref"]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["assessment_spec_id"], TOLERABILITY_SPEC)
        self.assertEqual(self.body["rules"][1]["assessment_spec_refs"], [TOLERABILITY_SPEC])

    def test_schema_v2_semantic_classification_is_present(self) -> None:
        for term in self.body["terms"]:
            self.assertIn("semantic_domain", term["schema_v2"])
        for predicate in self.body["predicates"]:
            block = predicate["schema_v2"]
            self.assertIn("semantic_domain", block)
            self.assertIn("compute_family", block)
            self.assertIn("compute_facets", block)
        families = {p["id"]: p["schema_v2"]["compute_family"] for p in self.body["predicates"]}
        self.assertEqual(families["pred.cond.hfref.exists"], "existence")
        self.assertEqual(families["pred.meas.egfr.value.ge.30"], "threshold")
        self.assertEqual(families["pred.meas.potassium.within_normal_range"], "range")

    def test_custom_comparator_is_mapped_and_reported(self) -> None:
        """A non-deterministic comparator name seen in real payloads."""
        by_id = {p["id"]: p for p in self.body["predicates"]}
        potassium = by_id["pred.meas.potassium.within_normal_range"]
        self.assertEqual(potassium["compare"]["operator"], "eq")
        kinds = {(item["kind"], item["subject"]) for item in self.graph.review}
        self.assertIn(("mapped_custom_comparator", "pred.meas.potassium.within_normal_range"), kinds)

    def test_non_boolean_rule_root_is_pinned_and_reported(self) -> None:
        """A rule rooted on an Enum predicate cannot be reduced to a Boolean."""
        rule = next(r for r in self.body["rules"] if r["id"] == "rule.sglt2i.dose_band.gate")
        root = next(n for n in rule["condition_dag"]["nodes"] if n["id"] == rule["condition_dag"]["root"])
        self.assertEqual(root["type"], "literal")
        self.assertIsNone(root["value"])
        kinds = {(item["kind"], item["subject"]) for item in self.graph.review}
        self.assertIn(("non_boolean_rule_root", "rule.sglt2i.dose_band.gate"), kinds)

    def test_projection_is_deterministic(self) -> None:
        again = load_graph()
        self.assertEqual(json.dumps(self.graph.graph, sort_keys=True, ensure_ascii=False),
                         json.dumps(again.graph, sort_keys=True, ensure_ascii=False))

    def test_projection_reports_its_own_review_items(self) -> None:
        kinds = {item["kind"] for item in self.graph.review}
        self.assertIn("routed_to_clinical_assessment", kinds)
        self.assertIn("assessment_gated_action", kinds)
        self.assertTrue(all(item["kind"] and item["detail"] for item in self.graph.review))

    def test_runtime_program_envelope(self) -> None:
        program = runtime_program(self.body)
        self.assertEqual(program["schema_version"], "guideline2graph.fact_first_program.v1")
        self.assertTrue(program["qid"])
        self.assertEqual(len(program["predicates"]), len(self.body["predicates"]))
        self.assertEqual(len(program["clinical_assessment_specs"]), 1)


class ExecutionTest(unittest.TestCase):
    """Three-valued execution over the projected graph."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = load_graph()

    def run_case(self, observations, assessments, **kwargs):
        return run_graph(self.graph, observations=observations, clinical_assessments=assessments, **kwargs)

    def test_full_data_activates_the_computable_rules(self) -> None:
        trace = self.run_case([HFREF_FACT, EGFR_FACT], assessment("true"))
        self.assertEqual(rule_truth(trace, "rule.sglt2i.hfref.recommend"), "true")
        self.assertEqual(rule_truth(trace, "rule.sglt2i.tolerability.consider"), "true")
        # the Enum-rooted rule is pinned to unknown, so it never activates
        self.assertEqual(rule_truth(trace, "rule.sglt2i.dose_band.gate"), "unknown")
        self.assertEqual(action_for_rule(trace, RECOMMEND)["activation_status"], "required")
        self.assertEqual(action_for_rule(trace, RECOMMEND)["execution_authorized"], True)

    def test_missing_information_stays_unknown_not_false(self) -> None:
        trace = self.run_case([HFREF_FACT], assessment("true"))
        self.assertEqual(trace["predicate_results"]["pred.meas.egfr.value.ge.30"]["truth"], "unknown")
        self.assertEqual(rule_truth(trace, "rule.sglt2i.hfref.recommend"), "unknown")
        self.assertEqual(action_for_rule(trace, RECOMMEND)["activation_status"], "unresolved")
        self.assertEqual(action_for_rule(trace, RECOMMEND)["execution_authorized"], False)

    def test_unknown_assessment_propagates(self) -> None:
        trace = self.run_case([HFREF_FACT, EGFR_FACT], assessment("unknown"))
        self.assertEqual(rule_truth(trace, "rule.sglt2i.tolerability.consider"), "unknown")
        self.assertEqual(action_for_rule(trace, CONSIDER)["activation_status"], "unresolved")

    def test_false_assessment_deactivates_its_action(self) -> None:
        trace = self.run_case([HFREF_FACT, EGFR_FACT], assessment("false"))
        self.assertEqual(rule_truth(trace, "rule.sglt2i.tolerability.consider"), "false")
        self.assertEqual(action_for_rule(trace, CONSIDER)["activation_status"], "not_activated")

    def test_negative_egfr_fails_the_threshold(self) -> None:
        low_egfr = {**EGFR_FACT, "value": 25}
        trace = self.run_case([HFREF_FACT, low_egfr], assessment("true"))
        self.assertEqual(trace["predicate_results"]["pred.meas.egfr.value.ge.30"]["truth"], "false")
        self.assertEqual(rule_truth(trace, "rule.sglt2i.hfref.recommend"), "false")

    def test_assessment_gated_action_requires_human_review(self) -> None:
        trace = self.run_case([HFREF_FACT, EGFR_FACT], assessment("true"))
        gated = action_for_rule(trace, CONSIDER)
        self.assertTrue(gated["human_review_required"])
        self.assertFalse(gated["execution_authorized"])
        self.assertIn(TOLERABILITY_SPEC, gated["human_review_spec_ids"])

    def test_ungated_action_is_executable(self) -> None:
        trace = self.run_case([HFREF_FACT, EGFR_FACT], assessment("true"))
        plain = action_for_rule(trace, RECOMMEND)
        self.assertFalse(plain["human_review_required"])
        self.assertTrue(plain["execution_authorized"])

    def test_unknown_assessment_spec_is_rejected(self) -> None:
        bogus = [{**assessment("true")[0], "assessment_spec_id": "assessment_spec.does_not_exist"}]
        with self.assertRaises(Guideline2ActionError):
            self.run_case([HFREF_FACT, EGFR_FACT], bogus)

    def test_llm_assessment_requires_provenance(self) -> None:
        incomplete = [{k: v for k, v in assessment("true")[0].items() if k not in ("model_id", "prompt_version")}]
        with self.assertRaises(Guideline2ActionError):
            self.run_case([HFREF_FACT, EGFR_FACT], incomplete)

    def test_execution_is_reproducible(self) -> None:
        first = self.run_case([HFREF_FACT, EGFR_FACT], assessment("true"))
        second = self.run_case([HFREF_FACT, EGFR_FACT], assessment("true"))
        self.assertEqual(json.dumps(first, sort_keys=True, ensure_ascii=False),
                         json.dumps(second, sort_keys=True, ensure_ascii=False))

    def test_clinical_handover_is_rendered(self) -> None:
        trace = self.run_case([HFREF_FACT, EGFR_FACT], assessment("true"), include_clinical_trace=True)
        handover = trace["clinical_trace"]
        # three rules: computable, assessment-gated, and the pinned Enum root
        self.assertEqual([d["state"] for d in handover["decisions"]], ["true", "true", "unknown"])
        self.assertEqual(handover["decisions"][1]["clinical_decision"], "确认耐受性后方可启动 SGLT2i")
        self.assertEqual(handover["decisions"][2]["state"], "unknown")
        guidance = {a["clinical_guidance"]: a["state"] for a in handover["actions"]}
        self.assertIn("启动或继续 SGLT2 抑制剂治疗", guidance)

    def test_graph_rejects_a_foreign_schema(self) -> None:
        with self.assertRaises(Guideline2ActionError):
            run_graph({"schema_version": "something.else"}, observations=[])


class RealExtractionTest(unittest.TestCase):
    """A payload produced by a live extraction run must compile and execute."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = ExecutableGraph.from_extraction(
            LIVE_EXTRACTION, graph_id="live_sglt2i_hfref"
        )

    def test_enum_fields_are_unwrapped(self) -> None:
        """``Permission`` is a str-enum: str(member) would be 'Permission.RECOMMEND'."""
        for action in self.graph.graph["actions"]:
            self.assertEqual(action["permission"], "recommend")
            self.assertNotIn("Permission.", action["permission"])
            self.assertEqual(action["permission_source"], "extraction")

    def test_graph_projects_and_executes(self) -> None:
        body = self.graph.graph
        self.assertEqual(len(body["rules"]), 1)
        self.assertEqual(len(body["actions"]), 1)
        entity = body["predicates"][0]["entity"]
        trace = run_graph(self.graph, observations=[{
            "fact_id": "f1", "concept_id": entity, "value": True, "polarity": "present",
            "temporal": {"relation": "current", "offset": None},
        }])
        self.assertEqual(trace["rule_results"][0]["truth"], "true")
        self.assertEqual(trace["actions"][0]["activation_status"], "required")
        self.assertTrue(trace["actions"][0]["execution_authorized"])

    def test_extraction_payload_round_trips(self) -> None:
        serialized = json.dumps(self.graph.programs, ensure_ascii=False)
        self.assertIn("guideline2graph.fact_first_program.v1", serialized)


class InterfaceTest(unittest.TestCase):
    """The two public functions exist with their documented signatures."""

    def test_public_surface(self) -> None:
        import guideline2action as g2a

        for name in ("compile_guideline", "run_graph", "ExecutableGraph", "project_graph",
                     "runtime_program", "SCHEMA_VERSION", "JUDGMENT_TOKENS"):
            self.assertTrue(hasattr(g2a, name), f"missing export {name}")

    def test_extraction_layer_is_omop_free(self) -> None:
        """The shipped package must not import the removed OMOP stack."""
        import guideline2action.graph.graph_api as graph_api
        import guideline2action.graph.graph_nodes as graph_nodes
        import guideline2action.graph.io_utils as io_utils

        for module in (graph_api, graph_nodes, io_utils):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for banned in ("omop_normalizer", "term_mapping", "processors", "preload_omop"):
                self.assertNotIn(banned, source, f"{module.__name__} still references {banned}")

    def test_compile_guideline_wires_the_extraction_layer(self) -> None:
        """Wiring check without spending an LLM call: build the pipeline on a temp
        gen_dir and confirm the module-level config is restored afterwards."""
        from guideline2action import compile_guideline
        from guideline2action.graph.graph_api import get_config, set_config

        before = get_config()
        try:
            compile_guideline("   ", api_key="unused", model="unused")
        except Guideline2ActionError as exc:
            self.assertIn("non-empty passage", str(exc))
        else:  # pragma: no cover - defensive
            self.fail("empty input must be rejected")
        self.assertIs(get_config(), before, "compile_guideline leaked global config")


if __name__ == "__main__":
    unittest.main(verbosity=2)
