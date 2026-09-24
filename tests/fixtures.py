"""Test fixtures: extraction payloads shaped exactly like the real ones.

Both payloads are in-memory only — no data files ship with this repository.

* ``SYNTHETIC_EXTRACTION`` is hand-written and covers the projection's edge
  cases: a computable threshold, a ``treatment_eligibility`` finding (which is
  not computable), a custom comparator seen in real payloads
  (``within_normal_range``), and a value-typed (``Enum``) predicate used
  directly as a rule root.
* ``LIVE_EXTRACTION`` is a **real** extraction payload: one Chinese HFrEF
  recommendation compiled end to end by ``compile_guideline`` in 66.7 s. It is
  kept because it exercises shapes the synthetic payload does not — notably
  ``Permission`` arriving as a ``str`` enum (whose ``str()`` renders as
  ``"Permission.RECOMMEND"``) and an extraction payload that points at its fact
  concepts through ``retrieve.code_binding``.

Neither payload contains patient data, answers, or oracle labels.
"""

from __future__ import annotations

import json
from typing import Any, Dict

SYNTHETIC_EXTRACTION: Dict[str, Any] = json.loads(
    r'''
{
 "guideline_id": "sglt2i_hfref_synthetic_v1",
 "metadata": {
  "note": "Synthetic extraction payload used for smoke tests; mirrors the shape returned by ClinicalGuidelinePipeline.run()."
 },
 "terms": [
  {
   "id": "cond.hfref",
   "name": "Heart failure with reduced ejection fraction",
   "type": "condition",
   "clinical_entity": "cond.hfref",
   "concept": "Heart failure with reduced ejection fraction",
   "value_domain": "Boolean",
   "binding_status": "candidate"
  },
  {
   "id": "meas.egfr",
   "name": "Estimated glomerular filtration rate",
   "type": "measurement",
   "clinical_entity": "meas.egfr",
   "concept": "eGFR",
   "value_domain": "Quantity",
   "unit": "mL/min/1.73m2",
   "binding_status": "candidate"
  },
  {
   "id": "meas.potassium",
   "name": "Serum potassium",
   "type": "measurement",
   "clinical_entity": "meas.potassium",
   "concept": "Potassium",
   "value_domain": "Quantity",
   "unit": "mmol/L",
   "binding_status": "candidate"
  }
 ],
 "med_terms": [
  {
   "id": "med.class.sglt2i",
   "name": "Sodium-glucose cotransporter-2 inhibitor",
   "drug_class": "SGLT2 inhibitor",
   "clinical_entity": "med.class.sglt2i",
   "binding_status": "candidate"
  }
 ],
 "predicates": [
  {
   "id": "pred.cond.hfref.exists",
   "name": "HFrEF present",
   "entity": "cond.hfref",
   "entity_type": "condition",
   "aspect": "existence",
   "input_shape": "single",
   "return_type": "Bool",
   "final_output_type": "TruthValue",
   "null_policy": "unknown",
   "temporal_scope": "all_time",
   "retrieve": {
    "resource": "QuestionObservation",
    "concepts": [
     "cond.hfref"
    ]
   },
   "extract": {
    "path": "value",
    "type": "TruthValue"
   },
   "reduction": {
    "operator": "most_recent",
    "output_type": "TruthValue"
   },
   "evidence": [
    {
     "source": "中国心力衰竭诊断和治疗指南2023"
    }
   ]
  },
  {
   "id": "pred.meas.egfr.value.ge.30",
   "name": "eGFR at least 30",
   "entity": "meas.egfr",
   "entity_type": "measurement",
   "aspect": "quantity",
   "input_shape": "single",
   "return_type": "Bool",
   "final_output_type": "TruthValue",
   "null_policy": "unknown",
   "temporal_scope": "all_time",
   "retrieve": {
    "resource": "QuestionObservation",
    "concepts": [
     "meas.egfr"
    ]
   },
   "extract": {
    "path": "value",
    "type": "Quantity"
   },
   "reduction": {
    "operator": "most_recent",
    "output_type": "TruthValue"
   },
   "compare": {
    "operator": "ge",
    "value": 30,
    "unit": "mL/min/1.73m2"
   },
   "unit": "mL/min/1.73m2"
  },
  {
   "id": "pred.obs.sglt2i.tolerability",
   "name": "Patient tolerates SGLT2 inhibitor therapy",
   "entity": "med.class.sglt2i",
   "entity_type": "clinical_assessment",
   "aspect": "treatment_eligibility",
   "input_shape": "single",
   "return_type": "Bool",
   "final_output_type": "TruthValue",
   "null_policy": "unknown",
   "temporal_scope": "all_time",
   "description": "根据当前资料，患者整体是否能够耐受 SGLT2 抑制剂治疗？"
  },
  {
   "id": "pred.meas.potassium.within_normal_range",
   "name": "Serum potassium within normal range",
   "entity": "meas.potassium",
   "entity_type": "measurement",
   "aspect": "quantity_range",
   "input_shape": "single",
   "return_type": "Bool",
   "final_output_type": "TruthValue",
   "null_policy": "unknown",
   "temporal_scope": "all_time",
   "unit": "mmol/L",
   "compare": {
    "operator": "within_normal_range",
    "value": "normal",
    "unit": "mmol/L"
   }
  },
  {
   "id": "pred.obs.sglt2i.dose_band",
   "name": "SGLT2 inhibitor dose band",
   "entity": "med.class.sglt2i",
   "entity_type": "medication",
   "aspect": "status",
   "input_shape": "single",
   "return_type": "Enum",
   "final_output_type": "Enum",
   "null_policy": "unknown",
   "temporal_scope": "all_time"
  }
 ],
 "rules": [
  {
   "id": "rule.sglt2i.hfref.recommend",
   "label": "HFrEF 患者推荐 SGLT2i 治疗",
   "source_text": "对于射血分数降低的心力衰竭患者，建议使用 SGLT2i 治疗，以降低心衰住院和心血管死亡风险。",
   "boolean_root": "ROOT",
   "missing_data_policy": "propagate_unknown",
   "condition_dag": {
    "nodes": [
     {
      "id": "pred.cond.hfref.exists",
      "type": "predicate_ref",
      "predicate_ref": "pred.cond.hfref.exists",
      "return_type": "Bool"
     },
     {
      "id": "pred.meas.egfr.value.ge.30",
      "type": "predicate_ref",
      "predicate_ref": "pred.meas.egfr.value.ge.30",
      "return_type": "Bool"
     },
     {
      "id": "N1",
      "type": "combine",
      "operator": "and",
      "inputs": [
       "pred.cond.hfref.exists",
       "pred.meas.egfr.value.ge.30"
      ],
      "return_type": "Bool"
     },
     {
      "id": "ROOT",
      "type": "combine",
      "operator": "and",
      "inputs": [
       "N1"
      ],
      "return_type": "Bool"
     }
    ],
    "root": "ROOT"
   },
   "action": {
    "subjects": [
     "med.class.sglt2i"
    ],
    "permission": "recommend",
    "intent": "initiate_or_continue",
    "monitoring": [
     "监测肾功能与容量状态"
    ],
    "output": {
     "text": "启动或继续 SGLT2 抑制剂治疗"
    }
   }
  },
  {
   "id": "rule.sglt2i.tolerability.consider",
   "label": "确认耐受性后方可启动 SGLT2i",
   "source_text": "启动 SGLT2 抑制剂前应评估患者整体耐受性。",
   "boolean_root": "ROOT",
   "missing_data_policy": "propagate_unknown",
   "condition_dag": {
    "nodes": [
     {
      "id": "pred.obs.sglt2i.tolerability",
      "type": "predicate_ref",
      "predicate_ref": "pred.obs.sglt2i.tolerability",
      "return_type": "Bool"
     },
     {
      "id": "ROOT",
      "type": "combine",
      "operator": "and",
      "inputs": [
       "pred.obs.sglt2i.tolerability"
      ],
      "return_type": "Bool"
     }
    ],
    "root": "ROOT"
   },
   "action": {
    "subjects": [
     "med.class.sglt2i"
    ],
    "permission": "consider",
    "intent": "initiate",
    "output": {
     "text": "在确认可耐受后考虑启动 SGLT2 抑制剂"
    }
   }
  },
  {
   "id": "rule.sglt2i.dose_band.gate",
   "label": "按剂量档位给药",
   "boolean_root": "ROOT",
   "missing_data_policy": "propagate_unknown",
   "condition_dag": {
    "nodes": [
     {
      "id": "pred.obs.sglt2i.dose_band",
      "type": "predicate_ref",
      "predicate_ref": "pred.obs.sglt2i.dose_band",
      "return_type": "Enum"
     },
     {
      "id": "ROOT",
      "type": "predicate_ref",
      "predicate_ref": "pred.obs.sglt2i.dose_band",
      "return_type": "Enum"
     }
    ],
    "root": "ROOT"
   },
   "action": {
    "subjects": [
     "med.class.sglt2i"
    ],
    "permission": "recommend",
    "intent": "adjust_dose",
    "output": {
     "text": "依据剂量档位调整 SGLT2i"
    }
   }
  }
 ]
}
'''
)

LIVE_EXTRACTION: Dict[str, Any] = json.loads(
    r'''
{
 "guideline_id": "中国心力衰竭诊断和治疗指南2023",
 "metadata": {
  "guideline_id": "中国心力衰竭诊断和治疗指南2023",
  "compiler_version": "0.3.0",
  "compilation_timestamp": "2026-09-24T12:20:53.479382+00:00",
  "total_rules": 1,
  "total_predicates": 1,
  "total_contraindications": 0,
  "total_safety_constraints": 0,
  "human_review_status": "unreviewed"
 },
 "terms": [
  {
   "id": "cond.heart_failure_reduced_ef",
   "name": "Heart failure with reduced ejection fraction",
   "label": "conditions",
   "type": "Code",
   "clinical_entity": "",
   "concept": "",
   "binding_status": "candidate",
   "clinical_entity_name": "Heart failure with reduced ejection fraction",
   "concept_name": "Heart failure with reduced ejection fraction",
   "value_domain_name": "Code",
   "value_set_binding_name": {
    "type_name": "Unknown",
    "name_name": "Heart failure with reduced ejection fraction",
    "confidence_name": 0.3
   },
   "data_bindings_name": {
    "FHIR_name": {
     "resource_name": "Condition"
    },
    "OMOP_name": {
     "table_name": "condition_occurrence"
    }
   },
   "fhir_binding_hint_name": {
    "resource_name": "Condition"
   },
   "omop_binding_hint_name": {
    "table_name": "condition_occurrence"
   },
   "binding_status_name": "candidate",
   "source_evidence_name": [
    {
     "source_text_name": "射血分数降低的心力衰竭（HFrEF）患者"
    }
   ],
   "normalization_confidence_name": 0.9
  }
 ],
 "predicates": [
  {
   "id": "pred.cond.hfref.exists",
   "name": "HFrEF诊断存在",
   "description": "患者存在射血分数降低的心力衰竭（HFrEF）诊断。",
   "source_text": "射血分数降低的心力衰竭（HFrEF）患者",
   "entity": "cond.heart_failure_reduced_ef",
   "entity_type": "condition",
   "aspect": "existence",
   "input_shape": "List<Condition>",
   "reduction": {
    "operator": "exists",
    "output_type": "Bool"
   },
   "return_type": "Bool",
   "final_output_type": "Bool",
   "temporal_scope": {
    "mode": "all_time"
   },
   "value_set_binding": {
    "type": "ValueSet",
    "name": "Heart failure with reduced ejection fraction",
    "confidence": 0.3
   },
   "null_policy": "unknown",
   "evidence": [
    {
     "source_text": "射血分数降低的心力衰竭（HFrEF）患者"
    }
   ],
   "source_span": {
    "source_text": "射血分数降低的心力衰竭（HFrEF）患者"
   },
   "dependencies": [
    "cond.heart_failure_reduced_ef"
   ]
  }
 ],
 "rules": [
  {
   "id": "rule.hfref_recommend_sglt2i",
   "label": "HFrEF患者推荐使用SGLT2i治疗",
   "source_text": "对于射血分数降低的心力衰竭（HFrEF）患者，建议使用SGLT2i治疗。",
   "source_span": {
    "source_text": "对于射血分数降低的心力衰竭（HFrEF）患者，建议使用SGLT2i治疗。"
   },
   "source_evidence": [
    {
     "source": "中国心力衰竭诊断和治疗指南2023",
     "quote": "对于射血分数降低的心力衰竭（HFrEF）患者，建议使用SGLT2i治疗。",
     "source_text": "对于射血分数降低的心力衰竭（HFrEF）患者，建议使用SGLT2i治疗。"
    }
   ],
   "input_predicates": [
    "pred.cond.hfref.exists"
   ],
   "condition_dag": {
    "nodes": [
     {
      "id": "ROOT",
      "type": "predicate_ref",
      "predicate_ref": "pred.cond.hfref.exists",
      "return_type": "Bool"
     }
    ],
    "root": "ROOT"
   },
   "boolean_root": "ROOT",
   "action": {
    "subjects": [
     "med.class.sglt2i"
    ],
    "permission": "recommend",
    "strength": "strong",
    "intent": "initiate_or_continue"
   },
   "scope": {
    "guideline_domain": [
     "cardiology",
     "heart_failure"
    ],
    "criteria": [
     {
      "criterion_id": "crit_diagnosis_hfref",
      "expression": "射血分数降低的心力衰竭（HFrEF）",
      "rationale": "对于射血分数降低的心力衰竭（HFrEF）患者",
      "type": "diagnosis"
     }
    ]
   },
   "missing_data_policy": "propagate_unknown",
   "provenance": [
    {
     "source": "中国心力衰竭诊断和治疗指南2023",
     "quote": "对于射血分数降低的心力衰竭（HFrEF）患者，建议使用SGLT2i治疗。"
    }
   ]
  }
 ],
 "missing_variables": {
  "rule.hfref_recommend_sglt2i": [
   {
    "variable_name": "HFrEF诊断存在",
    "variable_type": "diagnosis",
    "required_by_node_ids": [
     "pred.cond.hfref.exists"
    ],
    "null_policy": "propagate_unknown"
   }
  ]
 },
 "med_terms": [],
 "contraindications": [],
 "safety_constraints": []
}
'''
)
