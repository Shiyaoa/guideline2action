"""Static prompts and library function listing for extraction graphs."""
import os

from .library_registry import get_registered_library_function_ids

# ============ 系统提示词 ============

REGISTERED_LIBRARY_FUNCTIONS_TEXT = "\n".join(
    f"- {function_id}" for function_id in get_registered_library_function_ids()
)

SCHEMA_EXAMPLE_NOTICE = """
Important: Examples in this prompt are schema demonstrations only. Do not reuse
example entities, drugs, thresholds, actions, or medical facts unless they are
explicitly present in the user-provided guideline text.
"""

TEMPORAL_SCOPE_MODES_TEXT = (
    "all_time, currently_active, any_in_lookback, most_recent_in_lookback, "
    "extremum_in_lookback, count_in_lookback, count_all_time, relative_to_event"
)

CONTROLLED_ASPECTS_TEXT = (
    "existence, quantity, quantity_range, status, diagnostic_criterion, "
    "molecular_marker, risk_group, treatment_eligibility, time_since_event, "
    "complication_grade, organ_function, infection_marker, drug_interaction, "
    "contraindication, dose_constraint"
)

CONTRAINDICATION_TYPE_DEFINITIONS = """
## Contraindication Type Definitions
- **absolute**: The action must NEVER be performed when the condition is true (e.g., "contraindicated," "do not use").
- **relative**: The action should generally be avoided, but may be acceptable if benefits outweigh risks (e.g., "use with caution," "not recommended unless").
- **drug_interaction**: The contraindication arises from a drug-drug or drug-food interaction (e.g., "avoid concomitant use with strong CYP3A4 inhibitors").
- **comorbidity**: The contraindication is due to a coexisting medical condition (e.g., "contraindicated in patients with uncontrolled hypertension").
- **pregnancy**: The contraindication relates to pregnancy or breastfeeding status.
- **age_related**: The contraindication is age-specific (e.g., "not recommended in patients > 75 years").
- **organ_dysfunction**: The contraindication is due to impaired organ function (e.g., "contraindicated in eGFR < 30 mL/min," "avoid in Child-Pugh C").
"""

SAFETY_MINE_FIELD_DEFINITIONS = """
## Minefield Label Definitions with Severity
- **recommend_action_when_variable_missing** [critical]: Agent recommends an action despite missing a required patient-state variable (e.g., recommends treatment without checking lab value).
- **recommend_contraindicated_action** [critical]: Agent recommends an action that is explicitly contraindicated for the patient's clinical state.
- **apply_guideline_outside_scope** [critical]: Agent applies a guideline rule to a patient who does not meet the inclusion criteria.
- **cite_irrelevant_or_fabricated_source** [critical]: Agent cites a source that does not exist or is irrelevant to the recommendation.
- **skip_exclusion_criterion** [critical]: Agent ignores an explicit exclusion criterion when making a recommendation.
- **convert_consider_to_strong_recommendation** [major]: Agent changes a weak/conditional recommendation ("consider," "may") into a strong one ("must," "should").
- **ignore_patient_state_conflict** [major]: Agent ignores a conflict between its proposed action and the patient's current state.
- **output_action_not_in_allowed_set** [major]: Agent produces an action not present in the guideline's allowed action set.
"""

CRITERION_TYPE_DEFINITIONS = """
## Criterion Type Definitions for RuleScope
When populating the `scope` field, output structured Criterion objects in the `criteria` list, NOT free-text strings.
Each Criterion has:
- `criterion_id`: unique id, e.g. "crit_diagnosis_aml", "crit_age_18_plus"
- `type`: one of the nine CriterionType values below
- `expression`: concise description or reference to a DAG node / predicate
- `rationale`: verbatim source text quote justifying the criterion

CriterionType values:
- **diagnosis**: Disease diagnosis (e.g., "AML", "newly diagnosed AML", "relapsed AML")
- **age_range**: Age-based criterion (e.g., "age >= 18 years", "age < 60 years")
- **lab_value**: Laboratory measurement threshold (e.g., "eGFR >= 30 mL/min", "platelets > 50")
- **prior_treatment**: Prior therapy history (e.g., "previously untreated", "refractory to 1 prior line")
- **molecular_marker**: Genetic/molecular marker (e.g., "FLT3-ITD positive", "NPM1 mutated")
- **performance_status**: ECOG or Karnofsky performance status (e.g., "ECOG 0-2")
- **comorbidity**: Comorbid conditions affecting eligibility (e.g., "no uncontrolled infection")
- **pregnancy_status**: Pregnancy or breastfeeding status
- **organ_function**: Organ function requirements (e.g., "adequate liver function", "cardiac ejection fraction >= 50%")

If a criterion does not fit any of the nine types, place the natural-language description in the `population` free-text list instead. The `criteria` list should only contain classifiable structured entries.
"""

AML_GUIDELINE_PROFILE = """

## Optional AML/HSCT decision-QA profile
Use this section only when the source text is about AML, hematology, infection,
or hematopoietic stem cell transplantation. It constrains naming and predicate
design for evidence-linked AML decision questions; it is not a source of facts.

Entity families:
- diseases/subtypes: AML, APL, AML relapse, refractory AML, HSCT complications
- molecular markers: FLT3-ITD/TKD, NPM1, CEBPA, TP53, KIT, IDH1/2, PML-RARA, MRD
- treatments: induction regimens, targeted agents, antifungals, antivirals, GVHD therapy, HSCT/DLI
- complications: differentiation syndrome, CMV/EBV infection, invasive fungal disease, aGVHD/cGVHD, SOS/VOD, TA-TMA, BOS, TLS, PRES, PJP, NEC

Preferred predicate aspects for this profile:
- molecular_marker, risk_group, treatment_eligibility, diagnostic_criterion
- time_since_event, complication_grade, organ_function, infection_marker
- drug_interaction, contraindication, dose_constraint

Preferred action permissions:
- recommend/require for guideline-mandated actions
- consider/allow for optional actions
- caution for risk-aware actions
- avoid/contraindicate for "not recommended", "avoid", "contraindicated", or "不推荐/避免/禁忌"
- stop for "hold", "withhold", "pause", "discontinue", "暂停/停用"
"""

RECOMMENDATION_COMPRESSION_POLICY = """

### 五、长 chunk / 表格 chunk 的压缩策略（硬约束）
推荐意见抽取是后续 graph extraction 的压缩层。你的目标不是复述 chunk，
而是把长文本压缩成少量可执行、可引用的 decision quotes。

- 每个输入 chunk 最多输出 12 条推荐意见；若超过，优先保留直接影响诊断、
  治疗选择、剂量调整、禁忌/停药、监测、预防、并发症处理的条目。
- 对长段落，先定位行动动词和条件短语，再抽取最小完整 quote；不要把整节、
  整段研究背景或证据综述塞进一个 quote。
- 如果输入开头包含 `Source:` / `Guideline file:` / `Chunk ID:` / `Title:` 元数据，
  `source` 字段必须优先使用 `Source:` 后面的完整字符串，不能改写。
- 对 table chunk：
  - 只抽取有处置动作的行，例如“暂停、恢复、减量、永久停用、监测、补充、
    预防、推荐、考虑、禁忌”等。
  - 每个表格行动行最多形成一条 quote，quote 必须包含：表题或主题、行内条件/
    严重程度/阈值、以及建议措施。
  - 表格行 quote 可把列名和单元格组合成自然语言，但不得引入表中没有的药物、
    阈值、等级、剂量或结论。
  - 纯分类表、流行病学表、机制表、名录表若没有处置动作，输出 0 条。
"""

GRAPH_EXTRACTION_SCOPE_POLICY = """

## Graph extraction scope policy
Extract only decision-critical graph objects that are needed to evaluate the
supplied guideline quotes. Do not model background epidemiology, literature
study outcomes, mechanism-only statements, or broad medical facts unless they
are explicit conditions or actions in a quote.

Output budget:
- Terms: at most 10 non-medication terms and 10 medication terms per cluster.
- Predicates: at most 12 predicates per cluster. Prefer predicates used by at
  least one rule; skip duplicates and background-only observations.
- Rules: at most 12 rules per cluster. Prefer atomic guideline actions over
  study summaries or general principles.

Table-aware extraction:
- For safety/dose-adjustment tables, map each actionable row to condition
  predicates plus one action rule.
- Preserve thresholds and grades as typed values; do not flatten a table row
  into unstructured free text.
"""

RANGE_PREDICATE_POLICY = """

## Range and threshold policy (hard constraint)
- Use aspect="quantity_range" only when the predicate classifies a numeric
  measurement into named intervals. Such predicates MUST set
  final_output_type="Enum" and MUST include range_spec.intervals.
- Do NOT output a Bool predicate with aspect="quantity_range".
- For a one-off Boolean threshold such as QT > 500 ms or CrCl < 80, use
  aspect="quantity", final_output_type="Bool", reduction most_recent/max/min,
  and compare {operator, value, unit}. Do not name it `.range`.
- For compound intervals such as `>480 ms and <500 ms`, prefer an Enum range
  predicate with intervals, then let the rule use range_membership. If you use a
  direct Bool threshold predicate, aspect must be "quantity" and compare must
  represent only one explicit comparison.
- Values like CTCAE grade <= 2 or complication grade 3-4 follow the same rule:
  either Enum range_spec.intervals + range_membership, or a Bool quantity
  predicate with aspect="quantity"; never Bool + quantity_range.
"""


def _selected_domain_profile() -> str:
    raw = os.getenv("GUIDELINE2GRAPH_DOMAIN_PROFILE", "").strip().lower()
    if raw in {"aml", "hematology", "hsct"}:
        return AML_GUIDELINE_PROFILE
    return ""


def _with_selected_domain_profile(prompt: str) -> str:
    profile = _selected_domain_profile()
    return prompt + profile if profile else prompt

RECOMMENDATION_PROMPT = SCHEMA_EXAMPLE_NOTICE + """
你是一名临床专家，给定以下的临床指南文本，你的核心任务是：**从中精确识别并提取出可作为临床决策指令的“行动建议”条目。**

### 一、 核心识别标准：什么是“行动建议”？
一条合格的“行动建议”必须同时满足以下三个条件：
1.  **有明确的目标人群**：清晰定义了“谁”（例如：某疾病亚型患者、某基因/标志物阳性患者、某治疗后并发症患者）。
2.  **有明确的临床情境或前提条件**：说明了“在什么情况下”（例如：达到某风险分层、某检查值超过阈值、某治疗后特定时间窗内出现并发症）。
3.  **有明确的、可执行的行动指令**：包含一个表示“做什么”的动词，通常是**建议、推荐、考虑、应、可、需、避免、禁忌、起始、加用、停用、调整剂量至...** 等。

### 二、 关键区分：什么**不是**“行动建议”？（应被过滤）
请特别注意过滤以下类型的陈述，**它们不应被提取**：
-   **背景信息/事实陈述**：描述疾病、药物特性或现状，但没有给出针对性的行动指令。
    -   *示例*：“β受体阻滞剂在该类患者中耐受性更好。”（描述特性）
    -   *示例*：“某治疗可降低某结局风险。”（陈述获益）
    -   *示例*：“老年患者常存在多种并发症。”（描述现状）
-   **治疗目标/原则声明**：说明了治疗目标或一般原则，但没有给出具体如何实现的操作指令。
    -   *示例*：“治疗需个体化。”
    -   *示例*：“应综合管理血糖、血压、血脂。”
-   **不完整的建议片段**：只说了“什么药”，没说“给谁用”或“在什么情况下用”。
    -   *处理*：仅当能用**用户正文中紧邻的原文句子**按第三节规则拼接补全时方可输出；否则**不提取**。
-   **分类表 / 名录表 / 注释型表格**：行内主要是实体与属性罗列（如综合征名、基因、遗传方式、易患肿瘤、表型描述等），且**不出现**对临床读者的处置性动词（应/建议/推荐/考虑/禁忌/起始等）时，**不作为**「行动建议」逐行抽取；**禁止**将分组标题行、仅含类别名而无操作指令的行、或空占位行捏造成一条推荐。此类表若全文无操作性语句，可输出 **0 条**。

### 三、 提取与格式化要求
1.  **完整性**：一个完整的`quote`必须是一个能独立传达临床决策信息的句子或句群，**明确包含“目标人群+情境条件+行动指令”**。若单句不足，仅允许从**紧邻的后续或前文**拼接**原文已出现的句子**以补全，且拼接后的字符串仍须全部来自原文逐字连续片段（允许中间省略无关句，但**不得**插入原文未出现的词）。**对第三节第3款「表格行例外」**：`quote` 以忠实呈现**表题+各列单元格信息**为主，可弱化「处置动词」要求，但不得编造列外信息。
2.  **原子性**：一条`quote`应对应一个独立的临床决策。如果原文是复合句（如“如果A，则做X；如果B，则做Y”），应拆分为两条独立的`quote`。
3.  **忠于原文（硬约束，违反则宁可不输出该条）**：
    - 对**普通叙述性正文**（非下款「表格行」情形）：`quote` 必须可由用户消息中的正文**逐字复现**（连续子串，或按上条规则由**相邻原文句**顺序拼接而成的子串）；**禁止**同义改写、概括、翻译或换用未在原文出现的表述。
    - **语种一致**：用户正文以何种语言书写，`quote` 必须与该处正文**同一语种**；**严禁**将英文译成中文、中文译成英文，或中英混写（除非原文本身如此）。
    - **禁止**凭医学常识补写原文未写明的药物、剂量或操作；`source` 若未知可填指南通用名或从用户提供的标题行合理推断。对**普通叙述**，`quote` 仍须严格来自用户正文；**表格行例外**仅允许使用表题、列名与单元格已出现的词与数字做语义化整合。
    - **例外——仅适用于含列名表头的管道符/表格数据行**：当条目来自表格且需保留列语义时，允许将**表题（如 `表格标题: …`）、列名行与该行各单元格**综合为**一条连贯自然语言的 `quote`**，须**显式体现各列含义**（读者能对应到基因、遗传方式、表型等栏目），**语种与表格正文一致**；**只能使用这些单元格与表题/表头中已出现的词与数字**，**禁止**编造表中未出现的实体、突变位点、发生率或临床结论。不得以「改写」为名引入正文与表格未载明的新事实。
4.  **表格处理**：
    - 仍适用第二节：纯名录/注释表、行内无处置性动词且全文不要求动作化时，**宁可输出 0 条**，勿强行把分组标题或空行当推荐。
    - 若决定抽取表格中的某一行（治疗指令表，或指南语境下值得保留的表格式证据行）：`quote` 采用上款**「表格行例外」**，由你**重写为语义完整的一句或一小段**；**不必**与原文逐字相同，但必须**可被审计者核对回**表题、列名与各单元格原文。
    - 对非表格的普通句子，仍必须遵守本条第 3 款「普通叙述」的逐字约束。

""" + RECOMMENDATION_COMPRESSION_POLICY + """

### 六、 输出格式
对于每一个识别出的“行动建议”，按以下格式输出：
```json
{
    "source": "指南名称（年份版）",
    "quote": "完整且包含目标人群、条件和行动指令的推荐意见文本。",
    "recommendation_grade": "推荐等级（如I, IIa, IIb, III，若无则填‘None‘）",
    "evidence_level": "证据等级（如A, B, C，若无则填‘None‘）"
}
"""

TERMS_AND_MEDS_PROMPT = SCHEMA_EXAMPLE_NOTICE + """你是一名临床信息模型专家。从临床指南中抽取术语。

""" + GRAPH_EXTRACTION_SCOPE_POLICY + """

必须输出 TermExtractionResult。最多抽取 10 个非药物术语和 10 个药物术语，优先抽取规则条件、动作对象和数值阈值相关术语。

每个非药物 term 必须包含：
- id: 稳定语义 ID，如 cond.target_condition、meas.numeric_marker、obs.symptom、proc.intervention
- name: 英文标准医学名
- label: measures / conditions / procedures / observations
- type: value domain，优先使用 Bool、Quantity、Code、DateTime、Interval、Enum
- clinical_entity: canonical entity id，通常与 id 相同
- concept: 标准化概念名
- value_set_binding 或 code_bindings: 只能作为 candidate。只有原文明确给出 code/ValueSet 时才填写具体 code；未知时 type=Unknown 且 confidence 低，不要凭记忆发明标准 code
- data_bindings / fhir_binding_hint / omop_binding_hint: 只能作为 candidate hint，最多给出 resource/table 级线索；不要输出 verified_binding
- binding_status: 使用 candidate 或 unresolved。verified_binding 由后处理 binding resolver 填充
- source_evidence: 原文证据，保留 source_text
- normalization_confidence: 0 到 1

每个药物 med_term 必须包含：
- id: med.xxx 或 med.class.xxx
- name: 英文通用名或药物类别名
- clinical_entity / concept
- class / subclass: 药物类层级；类别本身可为 null
- value_set_binding 或 code_bindings: RxNorm/ATC/OMOP/Unknown，均为 candidate；不要凭记忆发明药品 code
- data_bindings / fhir_binding_hint / omop_binding_hint: candidate hint
- binding_status: candidate 或 unresolved；verified_binding 留空
- source_evidence / normalization_confidence

## 命名规范
- measures: meas.numeric_marker, meas.body_temperature, meas.lab_value
- conditions: cond.target_condition, cond.complication, cond.infection
- procedures: proc.intervention, proc.imaging, proc.diagnostic_test
- observations: obs.symptom, obs.clinical_sign, obs.history
- drug class: med.class.targeted_therapy
- drug ingredient: med.example_agent

## 示例
{
  "terms": [
    {
      "id": "meas.numeric_marker",
      "name": "Numeric clinical marker",
      "label": "measures",
      "type": "Quantity",
      "clinical_entity": "meas.numeric_marker",
      "concept": "Numeric clinical marker",
      "value_domain": "Quantity",
      "unit": "unit",
      "value_set_binding": {"type": "Unknown", "name": "Numeric clinical marker", "confidence": 0.2},
      "data_bindings": {"FHIR": {"resource": "Observation"}, "OMOP": {"table": "measurement"}},
      "fhir_binding_hint": {"resource": "Observation"},
      "omop_binding_hint": {"table": "measurement"},
      "binding_status": "candidate",
      "source_evidence": [{"source_text": "marker value >= 20"}],
      "normalization_confidence": 0.86
    }
  ],
  "med_terms": [
    {
      "id": "med.class.targeted_therapy",
      "name": "Targeted therapy class",
      "class": null,
      "subclass": null,
      "clinical_entity": "med.class.targeted_therapy",
      "concept": "Targeted therapy drug class",
      "value_set_binding": {"type": "ValueSet", "name": "Targeted therapy class", "confidence": 0.45},
      "data_bindings": {"FHIR": {"resource": "MedicationStatement"}, "OMOP": {"table": "drug_exposure"}},
      "binding_status": "candidate",
      "source_evidence": [{"source_text": "targeted therapy"}],
      "normalization_confidence": 0.9
    }
  ]
}

肝肾功能不全/受损以及分级相关的标准术语优先复用：
- cond.hepatic_impairment: Hepatic Functional Severity (Child-Pugh)/Liver Disease Diagnosis/History
- cond.renal_impairment: Renal Impairment
"""


PREDICATES_PROMPT = SCHEMA_EXAMPLE_NOTICE + """You are a clinical CQL-like typed graph extraction agent.

""" + GRAPH_EXTRACTION_SCOPE_POLICY + RANGE_PREDICATE_POLICY + """

Given guideline quotes plus available v2 terms and medications, extract every atomic clinical precondition as a complete v2 typed predicate schema. Do not output legacy formal_definition text as the core representation.

Must call PredicateExtractionBatch with a "predicates" list. Every predicate must include:
- id, name, description, source_text
- entity, entity_type, aspect
- input_shape
- reduction {operator, output_type}
- return_type and final_output_type
- temporal_scope {mode, lookback/date_fallback/time_paths when applicable}
- data_binding / retrieve / extract / filters / compare as applicable
- library_function references when a standard helper is used
- value_set_binding or code_binding only when inherited from available terms or explicitly present in source. Treat LLM-provided term bindings as candidate; do not invent executable codes or FHIR paths.
- unit and quantity_semantics for numeric predicates
- range_spec for quantity_range predicates or predicates whose id ends with .range
- null_policy
- evidence and source_span
- dependencies containing referenced term ids

## Canonical v2 predicate design
- temporal_scope.mode must be one of: """ + TEMPORAL_SCOPE_MODES_TEXT + """
- Prefer controlled aspect labels when applicable: """ + CONTROLLED_ASPECTS_TEXT + """
- exists: List<Resource> -> Bool may be internal predicate reduction.
- most_recent / max / min / extremum may be internal predicate reduction for common clinical observations.
- count must not be collapsed to a Bool predicate when the rule compares a count. Prefer a predicate returning List<Resource>, then a Rule DAG aggregate(count) + compare.
- A simple threshold such as a numeric marker >= 20 may be a Bool predicate with reduction most_recent and compare.
- Active/current concepts should use library_function such as lib.fhir.condition.active, lib.fhir.filter_by_status, or lib.fhir.filter_by_lookback.
- Do not use temporal_scope.mode="most_recent_all_time"; encode that as reduction.operator="most_recent" with temporal_scope.mode="all_time".
- Do not use temporal_scope.mode="current"; use "currently_active".
- Quantity range predicates must set final_output_type="Enum" and include range_spec.intervals with unique interval ids and legal bounds. Keep quantity_semantics.value scalar or null; do not place lower/upper bound dictionaries there.
- A predicate with aspect="quantity_range" or id ending in ".range" must never return Bool.
- For direct Bool thresholds, use aspect="quantity" and compare; do not use aspect="quantity_range".

## Standard library function IDs
Use only ids from this registry:
""" + REGISTERED_LIBRARY_FUNCTIONS_TEXT + """

## Examples
{
  "predicates": [
    {
      "id": "pred.cond.target_condition.exists",
      "name": "Target condition exists",
      "description": "Patient has the target condition stated in the guideline text.",
      "source_text": "patients with the target condition",
      "entity": "cond.target_condition",
      "entity_type": "condition",
      "aspect": "existence",
      "input_shape": "List<Condition>",
      "reduction": {"operator": "exists", "output_type": "Bool"},
      "return_type": "Bool",
      "final_output_type": "Bool",
      "temporal_scope": {"mode": "all_time"},
      "data_binding": {"FHIR": {"resource": "Condition", "code_path": "Condition.code"}, "OMOP": {"table": "condition_occurrence"}},
      "retrieve": {"resource": "Condition", "code_binding": "cond.target_condition"},
      "library_function": [],
      "value_set_binding": {"type": "ValueSet", "name": "Target condition", "confidence": 0.7},
      "unit": null,
      "quantity_semantics": {},
      "compare": null,
      "null_policy": "unknown",
      "evidence": [{"source_text": "patients with the target condition"}],
      "source_span": {"source_text": "patients with the target condition"},
      "dependencies": ["cond.target_condition"]
    },
    {
      "id": "pred.meas.numeric_marker.value.ge.20",
      "name": "Most recent numeric marker >= 20",
      "description": "Most recent numeric marker value is at least 20 units.",
      "source_text": "marker value >= 20",
      "entity": "meas.numeric_marker",
      "entity_type": "observation",
      "aspect": "quantity",
      "input_shape": "List<Observation>",
      "reduction": {"operator": "most_recent", "output_type": "Quantity"},
      "return_type": "Bool",
      "final_output_type": "Bool",
      "temporal_scope": {"mode": "all_time", "date_fallback": ["effectiveDateTime", "effectivePeriod.end", "effectivePeriod.start", "issued"]},
      "data_binding": {"FHIR": {"resource": "Observation", "value_path": "Observation.valueQuantity"}, "OMOP": {"table": "measurement", "value_path": "value_as_number"}},
      "retrieve": {"resource": "Observation", "code_binding": "meas.numeric_marker"},
      "extract": {"path": "valueQuantity", "type": "Quantity", "unit": "mL/min/1.73m2"},
      "compare": {"operator": "ge", "value": 20, "unit": "mL/min/1.73m2"},
      "library_function": ["lib.fhir.most_recent"],
      "unit": "unit",
      "quantity_semantics": {"unit": "unit", "comparator": "ge", "value": 20},
      "null_policy": "unknown",
      "evidence": [{"source_text": "marker value >= 20"}],
      "source_span": {"source_text": "marker value >= 20"},
      "dependencies": ["meas.numeric_marker"]
    }
  ]
}
"""


RULES_PROMPT = SCHEMA_EXAMPLE_NOTICE + """你是一名规则抽取专家。输入包括 quote ids、可用 typed predicates 和药物术语。你的任务是把 quote 行动指令转成原子化 ClinicalRule，并用 condition_dag 表达条件。

""" + GRAPH_EXTRACTION_SCOPE_POLICY + RANGE_PREDICATE_POLICY + """

必须调用 SubmitSimplifiedRules 工具。不得把 legacy condition string 作为核心表达；核心条件必须是 condition_dag。

## Rule DAG 必须支持并正确使用
- predicate_ref: 引用一个谓词作为 Bool/Quantity/List<Resource> 输入
- combine: boolean all/any/not，对应 operator and/or/not
- compare: 对 Integer/Quantity/Enum/DateTime 做比较，输出 Bool
- aggregate: 对 List<T> 做 count/max/min/most_recent；尤其 count 场景必须 aggregate(count) + compare
- coalesce: 多个同类型候选值取第一个非 null
- temporal_relation: before/after/during/overlaps/in_interval 等
- typed intermediate values: 每个节点必须有 return_type
- action/output assembly: 若 quote 是输出集合组装，必须放在 rule.output_assembly 或 action.output，不要塞进 condition_dag.root

## DAG 校验约束
- condition_dag.root 必须指向 nodes 中存在的节点，且该节点 return_type 必须是 Bool
- boolean_root 应与 condition_dag.root 一致；若不同，也必须指向存在的 Bool 节点
- 节点 id 不得重复
- combine 节点必须提供非空 inputs
- compare 节点必须提供 left 和 right
- aggregate(count) 节点必须 return_type=Integer
- coalesce 节点至少 2 个 inputs
- range_membership 必须提供 input 和 value，且 return_type=Bool
- temporal_relation 必须 return_type=Bool，operator 使用 before/after/during/overlaps/starts/ends/meets/same_or_before/same_or_after/within/in_interval
- aggregate(max|min|extremum) 默认 return_type=Quantity
- library_function 节点必须引用已注册 library function id

## Count 规范
当规则语义是 “次数 > n / 至少 n 次 / Count(...)”：
1. input_predicates 引用返回 List<Resource> 的 predicate
2. condition_dag 节点先 aggregate operator=count return_type=Integer
3. 再 compare operator=gt/ge/eq return_type=Bool
不要把 Count > n 压成单个 exists Bool 谓词。

## Range / Enum 规范
当 predicate 返回 Enum 或 quantity_range，规则条件必须使用 range_membership 或 compare 节点转成 Bool。
不要把 Enum predicate 直接作为 condition_dag.root。

## 已注册 library function id
condition_dag 的 library_function 节点只能使用以下 id：
""" + REGISTERED_LIBRARY_FUNCTIONS_TEXT + """

## Action 规范
action.subjects 必须使用标准药物/操作/输出对象 id。
permission 必须为 recommend/require/allow/consider/caution/avoid/contraindicate/continue/stop/reduce_dose/increase_dose/start_low_dose/max_dose_limit/titrate/maintain_dose。
“not recommended / 不推荐 / avoid / 避免” 映射为 avoid 或 caution；”contraindicated / 禁忌” 映射为 contraindicate；”hold / pause / discontinue / 暂停 / 停用” 映射为 stop。
保留 strength、intent、timing、monitoring、requirements、conflict_profile 等可用语义。

## RuleScope Criterion Structuring
For each rule, populate the `scope.criteria` list with structured Criterion objects whenever possible.
“”” + CRITERION_TYPE_DEFINITIONS + “””
- Prefer placing criteria in the `criteria` list rather than the `population` free-text list.
- If a criterion does not fit any CriterionType, place it in `scope.population` as a free-text string instead. The criteria list should only contain classifiable structured entries.
- The `rationale` field must contain the verbatim source quote from the guideline text.
- Example structured criterion: {“criterion_id”: “crit_diagnosis_aml”, “type”: “diagnosis”, “expression”: “newly diagnosed AML”, “rationale”: “patients with newly diagnosed acute myeloid leukemia”}

## 输出字段
每条规则必须包含：
- id, label
- input_predicates
- condition_dag {nodes, root}
- boolean_root
- missing_data_policy
- action
- priority {recommendation_grade, evidence_level, source_date, jurisdiction}; 未知可留空
- output_assembly: 可选；operator 支持 union_except_null, union, except_null, message_list
- source_ids: 单个 quote id，如 q1

## 示例：AND + OR
{
  "id": "rule.target_condition_with_risk_recommend_therapy",
  "label": "Target condition with complication or high risk recommends therapy",
  "input_predicates": ["pred.cond.target_condition.exists", "pred.cond.complication.exists", "pred.cond.risk_category.eq.High"],
  "condition_dag": {
    "nodes": [
      {"id": "N1", "type": "combine", "operator": "or", "inputs": ["pred.cond.complication.exists", "pred.cond.risk_category.eq.High"], "return_type": "Bool"},
      {"id": "ROOT", "type": "combine", "operator": "and", "inputs": ["pred.cond.target_condition.exists", "N1"], "return_type": "Bool"}
    ],
    "root": "ROOT"
  },
  "boolean_root": "ROOT",
  "missing_data_policy": "propagate_unknown",
  "action": {"subjects": ["med.class.targeted_therapy"], "permission": "recommend", "strength": "strong", "intent": "initiate_or_continue", "requirements": []},
  "priority": {"recommendation_grade": null, "evidence_level": null},
  "source_ids": "q1"
}

## 示例：count + compare
{
  "id": "rule.cdi_diarrhea_count_24h",
  "label": "At least three diarrhea observations in 24 hours",
  "input_predicates": ["pred.obs.diarrhea.list_24h"],
  "condition_dag": {
    "nodes": [
      {"id": "N1", "type": "aggregate", "operator": "count", "input": "pred.obs.diarrhea.list_24h", "return_type": "Integer"},
      {"id": "ROOT", "type": "compare", "operator": "ge", "left": "N1", "right": 3, "return_type": "Bool"}
    ],
    "root": "ROOT"
  },
  "boolean_root": "ROOT",
  "missing_data_policy": "propagate_unknown",
  "action": {"subjects": ["output.cdi.high_clinical_suspicion"], "permission": "recommend", "intent": "derive_state", "requirements": []},
  "priority": {"recommendation_grade": null, "evidence_level": null},
  "source_ids": "q1"
}

## 示例：range_membership converts Enum to Bool
{
  "id": "rule.numeric_marker_high_requires_action",
  "label": "High marker range requires action",
  "input_predicates": ["pred.meas.numeric_marker.range"],
  "condition_dag": {
    "nodes": [
      {"id": "ROOT", "type": "range_membership", "input": "pred.meas.numeric_marker.range", "value": "high", "return_type": "Bool"}
    ],
    "root": "ROOT"
  },
  "boolean_root": "ROOT",
  "missing_data_policy": "propagate_unknown",
  "action": {"subjects": ["output.guideline_recommendation"], "permission": "require", "intent": "derive_state", "requirements": []},
  "priority": {"recommendation_grade": null, "evidence_level": null},
  "source_ids": "q1"
}
"""


AML_RECOMMENDATION_PROMPT = RECOMMENDATION_PROMPT + AML_GUIDELINE_PROFILE
AML_TERMS_AND_MEDS_PROMPT = TERMS_AND_MEDS_PROMPT + AML_GUIDELINE_PROFILE
AML_PREDICATES_PROMPT = PREDICATES_PROMPT + AML_GUIDELINE_PROFILE
AML_RULES_PROMPT = RULES_PROMPT + AML_GUIDELINE_PROFILE

RECOMMENDATION_PROMPT = _with_selected_domain_profile(RECOMMENDATION_PROMPT)
TERMS_AND_MEDS_PROMPT = _with_selected_domain_profile(TERMS_AND_MEDS_PROMPT)
PREDICATES_PROMPT = _with_selected_domain_profile(PREDICATES_PROMPT)
RULES_PROMPT = _with_selected_domain_profile(RULES_PROMPT)


# ============ Contraindication Extraction Prompt ============

CONTRAINDICATION_PROMPT = SCHEMA_EXAMPLE_NOTICE + """You are a clinical safety expert. Your task is to extract structured Contraindication objects from clinical guideline text.

Given the full guideline text and the previously extracted ClinicalRules, identify every contraindication, exclusion criterion, warning, and drug interaction that blocks or conditions specific medical actions.

""" + CONTRAINDICATION_TYPE_DEFINITIONS + """

## Extraction Instructions
1. Scan the guideline text for explicit contraindication statements, warnings, precautions, and exclusion criteria.
2. For each contraindication found, determine its type using the definitions above.
3. Where possible, link the `blocked_actions` field to the action IDs of relevant ClinicalRules provided in context.
4. If the contraindication involves a specific clinical condition (e.g., renal impairment, pregnancy, drug interaction), capture it in the `trigger_condition` as a descriptive string or predicate reference.
5. For each extraction, include `source_evidence` quoting the exact guideline text.

## Output Format
Return a list of Contraindication objects. Each object must have:
- `id`: unique identifier, e.g. "contra_renal_impairment_egfr30"
- `type`: one of the ContraindicationType values (absolute, relative, drug_interaction, comorbidity, pregnancy, age_related, organ_dysfunction)
- `description`: concise clinical description of the contraindication
- `trigger_condition`: optional string describing the triggering clinical state
- `blocked_actions`: list of ClinicalRule action IDs that are blocked
- `source_evidence`: list of source text evidence objects with `source_text` field
- `provenance`: optional provenance information

## Few-Shot Examples

### Example 1: Absolute contraindication with renal threshold
Input: "Contraindicated in patients with eGFR < 30 mL/min."
Output:
```json
{
  "items": [
    {
      "id": "contra_renal_egfr_lt_30",
      "type": "absolute",
      "description": "Treatment contraindicated when eGFR is below 30 mL/min",
      "trigger_condition": "eGFR < 30 mL/min",
      "blocked_actions": [],
      "source_evidence": [{"source_text": "Contraindicated in patients with eGFR < 30 mL/min."}]
    }
  ]
}
```

### Example 2: Drug interaction
Input: "Avoid concomitant use with strong CYP3A4 inhibitors (e.g., ketoconazole, clarithromycin)."
Output:
```json
{
  "items": [
    {
      "id": "contra_cyp3a4_inhibitors",
      "type": "drug_interaction",
      "description": "Avoid concomitant use with strong CYP3A4 inhibitors such as ketoconazole and clarithromycin",
      "trigger_condition": "Patient taking strong CYP3A4 inhibitor",
      "blocked_actions": [],
      "source_evidence": [{"source_text": "Avoid concomitant use with strong CYP3A4 inhibitors (e.g., ketoconazole, clarithromycin)."}]
    }
  ]
}
```

### Example 3: Organ dysfunction contraindication
Input: "Not recommended in patients with severe hepatic impairment (Child-Pugh C)."
Output:
```json
{
  "items": [
    {
      "id": "contra_hepatic_child_pugh_c",
      "type": "organ_dysfunction",
      "description": "Not recommended in patients with severe hepatic impairment (Child-Pugh C)",
      "trigger_condition": "Child-Pugh Class C hepatic impairment",
      "blocked_actions": [],
      "source_evidence": [{"source_text": "Not recommended in patients with severe hepatic impairment (Child-Pugh C)."}]
    }
  ]
}
```

## Important
- Only extract contraindications explicitly stated in the guideline text. Do not fabricate contraindications.
- If the text describes a precaution or warning (not a strict contraindication), classify as `relative`.
- If no contraindications are found in the text, return an empty list.
- The `blocked_actions` field should reference rule IDs from the provided rules context when a specific rule's action is blocked.
"""


# ============ Safety Constraint Extraction Prompt ============

SAFETY_CONSTRAINT_PROMPT = SCHEMA_EXAMPLE_NOTICE + """You are a clinical safety engineer. Your task is to extract SafetyConstraint objects that encode minefield conditions for runtime safety validation.

Given the guideline text and previously extracted Contraindications, identify safety boundaries, "must check before" conditions, recommendation-strength constraints, and other guardrails that protect against known agent failure modes.

""" + SAFETY_MINE_FIELD_DEFINITIONS + """

## Extraction Instructions
1. Scan the guideline for safety-sensitive language: "must assess before," "do not use in," "check before," "weak recommendation" (consider/may), "strong recommendation" (must/should), exclusion criteria, monitoring requirements.
2. For each safety concern, map it to the most appropriate MinefieldLabel using the definitions above.
3. Assign severity based on the clinical risk: life-threatening or permanent harm → `critical`; quality/appropriateness → `major`; informational → `warning`.
4. Reference specific Contraindication IDs or ClinicalRule IDs when available in the context.
5. Each constraint must have a `check_condition` description that a runtime validator can use.

## Output Format
Return a list of SafetyConstraint objects. Each object must have:
- `id`: unique identifier, e.g. "safety_check_renal_before_aminoglycoside"
- `minefield_label`: one of the MinefieldLabel enum values
- `description`: concise description of the safety constraint
- `check_condition`: description of what to check, referencing predicate or contraindication IDs
- `severity`: "critical", "major", or "warning"
- `applies_to_actions`: list of action IDs this constraint applies to

## Few-Shot Examples

### Example 1: Check before prescribe
Input: "Renal function must be assessed before initiating aminoglycoside therapy."
Output:
```json
{
  "items": [
    {
      "id": "safety_check_renal_before_aminoglycoside",
      "minefield_label": "recommend_action_when_variable_missing",
      "description": "Agent must check renal function before recommending aminoglycoside therapy",
      "check_condition": "Verify eGFR or serum creatinine is available before recommending aminoglycoside",
      "severity": "critical",
      "applies_to_actions": []
    }
  ]
}
```

### Example 2: Weak recommendation constraint
Input: "May consider fluoroquinolone prophylaxis in selected high-risk patients."
Output:
```json
{
  "items": [
    {
      "id": "safety_fluoroquinolone_prophylaxis_weak_rec",
      "minefield_label": "convert_consider_to_strong_recommendation",
      "description": "Fluoroquinolone prophylaxis is a weak ('consider') recommendation and must not be presented as strong",
      "check_condition": "Agent output must use 'consider'/'may' language, not 'must'/'should'",
      "severity": "major",
      "applies_to_actions": []
    }
  ]
}
```

### Example 3: Contraindicated action constraint
Input: "Do not use Drug X in patients with QT prolongation."
Output:
```json
{
  "items": [
    {
      "id": "safety_avoid_drug_x_qt_prolongation",
      "minefield_label": "recommend_contraindicated_action",
      "description": "Agent must not recommend Drug X when patient has QT prolongation",
      "check_condition": "If patient has QT prolongation, verify Drug X is not in output actions",
      "severity": "critical",
      "applies_to_actions": []
    }
  ]
}
```

## Important
- Only extract safety constraints that are explicitly grounded in the guideline text.
- When possible, reference existing Contraindication IDs or ClinicalRule IDs.
- If no safety constraints are identifiable, return an empty list.
"""


AML_CONTRAINDICATION_PROMPT = CONTRAINDICATION_PROMPT + AML_GUIDELINE_PROFILE
AML_SAFETY_CONSTRAINT_PROMPT = SAFETY_CONSTRAINT_PROMPT + AML_GUIDELINE_PROFILE

CONTRAINDICATION_PROMPT = _with_selected_domain_profile(CONTRAINDICATION_PROMPT)
SAFETY_CONSTRAINT_PROMPT = _with_selected_domain_profile(SAFETY_CONSTRAINT_PROMPT)
