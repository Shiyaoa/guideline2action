# Guideline2Action

**English** · [中文](README.zh-CN.md)

Compile **clinical guideline text** into an **executable decision graph**, then run it deterministically against patient data.

```python
from guideline2action import compile_guideline, run_graph

graph = compile_guideline([
    "2023 Chinese Heart Failure Guideline\n"
    "In patients with heart failure with reduced ejection fraction (HFrEF), "
    "sodium-glucose cotransporter-2 inhibitors (SGLT2i) are recommended to reduce "
    "the risk of heart failure hospitalisation and cardiovascular death.",
    "Expert Consensus on the Clinical Use of SGLT2i\n"
    "When estimated glomerular filtration rate (eGFR) is below 30 mL/min/1.73m², "
    "SGLT2i should be used with caution and renal function monitored closely.",
])

result = run_graph(graph, observations=[
    {"fact_id": "f1", "concept_id": "cond.hfref", "value": True,
     "polarity": "present", "temporal": {"relation": "current", "offset": None}},
    {"fact_id": "f2", "concept_id": "meas.egfr", "value": 45, "unit": "mL/min/1.73m2",
     "polarity": "present", "temporal": {"relation": "current", "offset": None}},
])
```

`compile_guideline` makes one LLM call (it reads the guideline text). `run_graph` calls **no LLM at all**: the same graph plus the same patient data always yields the same result.

---

## Why this layer exists

Guideline text mixes three kinds of content, and they differ in computability:

| Kind | Example | Decided by |
|---|---|---|
| **Computable criterion** (predicate) | `eGFR < 30`, `abnormal promyelocytes present`, `age ≥ 60` | The interpreter, with deterministic operators |
| **Clinical judgement that cannot be computed** | overall fitness for intensive therapy, tolerability, whether the disease is under stable control | **The LLM answers** `true` / `false` / `unknown` |
| **A finding the case already states** | confirmed APL, stage 2 disease | A predicate **reads** it; nothing is re-derived |

The second kind is the crux: it is not a predicate and must not masquerade as a computed result. This library moves such findings **out of the executable predicate set** and turns each one into a `ClinicalAssessmentSpec` (declaring the question, the allowed values, the admissible evidence, and the review policy); the DAG then refers to it through a `clinical_assessment_ref` node. The boundary comes from the guideline specification and is implemented as written:

* `aspect == "treatment_eligibility"`, or
* the identifier/name carries a composite-judgement token such as `fitness` / `tolerab*` / `suitab*` / `feasib*` / `readiness` / `stable_control`

When information is missing the result stays `unknown` and **never degrades to `false`** (strong Kleene three-valued propagation). "The guideline does not cover this" and "the patient does not meet this" therefore remain distinguishable in the output.

---

## Architecture

```text
Guideline text
   │  graph/          ← extraction: recommendations → LSH clustering → terms → predicates → rules → safety
   ▼
Extraction payload {terms, med_terms, predicates, rules, contraindications, safety_constraints}
   │  projection.py   ← projection layer (new in this library, 1,297 lines): four deterministic transforms
   │     1. action materialisation: rules[].action → top-level actions[] + rules[].action_refs
   │     2. DAG vocabulary normalisation: 16 node types → the interpreter's 6; operator normalisation;
   │        bare predicate ids → predicate_ref nodes; non-Boolean roots → unresolved constants
   │     3. ClinicalAssessment routing: non-computable findings → ClinicalAssessmentSpec
   │     4. schema_v2 semantic classification: semantic_domain / compute_family / compute_facets
   ▼
Executable Schema-v2 graph (schema_version = decisionkg.schema_v2.unified.20260804)
   │  runtime/        ← interpreter: deterministic evaluation + three-valued propagation
   │                    + execution authorisation + clinical handover rendering
   ▼
{assessments, actions and their status, per-node trace, clinical handover text}
```

Action status is derived from *rule truth × review policy*:

| Status | Meaning |
|---|---|
| `required` | Condition holds and no review is required → executable (`execution_authorized: true`) |
| `pending_human_review` | Condition holds, but gated by an assessment carrying `human_review_policy: required_before_action` |
| `not_activated` | Condition is determinately false |
| `unresolved` | Information is insufficient; the decision stays open |

---

## Install and verify

```bash
pip install -e .

# Offline self-check: 32 cases, no API key, no data files
python3 tests/test_smoke.py
```

The two extraction payloads the self-check needs are **inlined in `tests/fixtures.py`**; the repository ships no data files:

| Fixture | Origin | Edge cases covered |
|---|---|---|
| `SYNTHETIC_EXTRACTION` | hand-written | computable threshold, a `treatment_eligibility` assessment, the custom comparator `within_normal_range`, and a value-typed (`Enum`) predicate used directly as a rule root |
| `LIVE_EXTRACTION` | a **real** LLM extraction (one Chinese HFrEF recommendation, 66.7 s) | a real shape in which `Permission` arrives as a `str` enum and the fact concept is indicated only by `retrieve.code_binding` |

Neither payload contains patient data, answers, or oracle labels.

LLM configuration follows the extraction layer's environment conventions:

```bash
LLM_PROVIDER=deepseek          # or dashscope (default)
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
```

---

## Public interface

### `compile_guideline(text, **kwargs) -> ExecutableGraph`

| Parameter | Description |
|---|---|
| `text` | One guideline passage, or a list of passages; a passage may carry its provenance as `"<source>\n<recommendation>"` |
| `api_key` / `base_url` / `model` / `temperature` | LLM overrides; defaults to `LLMConfig.from_env()` |
| `graph_id` / `qid` | Stable identifiers; `qid` defaults to a slug derived from `graph_id` |
| `gen_dir` | Stage-cache directory; defaults to a temporary directory, so the library never writes inside its own installation |
| `max_concurrency` | Number of parallel cluster extractions |

Returns an `ExecutableGraph`: it behaves like the graph dict (hand it straight to `run_graph` or `json.dump`) and additionally exposes `.graph`, `.review`, `.programs`, `.to_json()`.

### `run_graph(graph, observations, clinical_assessments=None, clinical_assertions=None, *, include_clinical_trace=False) -> dict`

| Input | Shape |
|---|---|
| `observations` | `{fact_id, concept_id, value, unit?, polarity?, temporal?, source_quote?}` — bound to predicates through `concept_id` |
| `clinical_assessments` | `{assessment_id, assessment_spec_id, value: true\|false\|unknown, assessment_authority: llm\|clinical_expert, model_id, prompt_version, supporting_observation_ids, source_spans, requires_human_review}` |
| `clinical_assertions` | `{assertion_id, concept_id, value, asserted_by, source_span, timepoint}` |

Returns a trace: `predicate_results`, `clinical_assessment_results`, `condition_node_results`, `rule_results`, `actions`, and optionally `clinical_trace` — the "clinical handover" text given to the LLM: decision propositions, their states, which facts support them, and the actions and unresolved conditions.

---

## Quality and boundaries

The projection layer is a **pure function**: it reads the extraction payload only — never patient data, never oracle or reference answers. Anything it cannot transform deterministically is written to `.review` instead of quietly altering semantics:

| `review.kind` | Meaning |
|---|---|
| `routed_to_clinical_assessment` | The finding is not deterministically computable; it became an assessment question |
| `mapped_custom_comparator` | A custom comparator was mapped to the nearest executable operator and needs human confirmation |
| `unsupported_comparison` | The comparator has no equivalent; the comparison was dropped (and recorded) |
| `non_executable_dag_node` | A DAG node type the extraction layer offers but the interpreter has no operator for |
| `non_boolean_rule_root` | The rule root is value-typed and cannot reduce to a Boolean condition → the rule is pinned to unresolved |
| `unresolved_dag_operand` | A DAG operand is neither a predicate nor a node |
| `orphan_predicates` | Declared but referenced by no rule |
| `assessment_gated_action` | The action is gated by an assessment; execution depends on the review policy |

**Verified scope** (three independent layers):

| Verification | Input | Result |
|---|---|---|
| Offline cases | synthetic + real extraction fixtures | 32/32 green |
| Real extraction payloads | 62 batches of a 106-question AML guideline corpus (4,559 rules, 10,966 predicates, 1,325 assessments) | projected 62/62, executed 62/62, zero failures; three-valued outcomes `true 3072 / false 590 / unknown 897` |
| Live end to end | one Chinese guideline recommendation (a real LLM call) | `compile_guideline` in 66.7 s → 1 term / 1 med_term / 1 predicate / 1 rule / 1 action → `run_graph` yields `required` with `execution_authorized: true` |

**Not included**: OMOP/FHIR vocabulary standardisation, the per-question graph migration chain, benchmark evaluation and scoring, and manuscript figures. Graphs are driven by guideline text alone; patient data is supplied by the caller.

## Layout

```text
README.md          English description (this file)
README.zh-CN.md    Chinese description
guideline2action/
├── graph/          extraction layer (guideline text → terms/predicates/rules)
├── projection.py   projection layer (extraction payload → executable Schema-v2 graph)
├── runtime/        interpreter + clinical handover rendering
└── __init__.py     compile_guideline / run_graph
tests/
├── fixtures.py     two inlined extraction payloads (synthetic + real)
└── test_smoke.py   32 offline cases
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
