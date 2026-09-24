# Guideline2Action

[English](README.en.md) · **中文**

把**临床指南文本**编译成**可执行决策图**，并确定性地在患者资料上执行它。

```python
from guideline2action import compile_guideline, run_graph

graph = compile_guideline([
    "中国心力衰竭诊断和治疗指南2023\n"
    "对于射血分数降低的心力衰竭（HFrEF）患者，建议使用钠-葡萄糖共转运蛋白2抑制剂（SGLT2i）治疗，"
    "以降低心衰住院和心血管死亡风险。",
    "SGLT2i临床应用专家共识\n"
    "当患者估算肾小球滤过率（eGFR）低于30 mL/min/1.73m²时，应慎用SGLT2i并加强肾功能监测。",
])

result = run_graph(graph, observations=[
    {"fact_id": "f1", "concept_id": "cond.hfref", "value": True,
     "polarity": "present", "temporal": {"relation": "current", "offset": None}},
    {"fact_id": "f2", "concept_id": "meas.egfr", "value": 45, "unit": "mL/min/1.73m2",
     "polarity": "present", "temporal": {"relation": "current", "offset": None}},
])
```

`compile_guideline` 调用一次 LLM（读指南文本）。`run_graph` **不调用任何 LLM**：同一张图加同一组患者资料，永远得到同一个结果。

---

## 为什么需要这一层

指南文本里混杂着三类内容，它们的可计算性不同：

| 类型 | 例子 | 由谁决定 |
|---|---|---|
| **可计算的判定（predicate）** | `eGFR < 30`、`存在异常早幼粒细胞`、`年龄 ≥ 60` | 解释器按确定性算子求值 |
| **不能确定性计算的临床判断** | 整体是否适合强化治疗、能否耐受、是否已临床稳定控制 | **LLM 回答** `true`/`false`/`unknown` |
| **题干已明示的断言** | 已确诊 APL、分期 2 级 | predicate 确定性**读取**，不重新推导 |

第二类是关键：它不是谓词，不能伪装成计算结果。本库把这类发现**移出可执行谓词集合**，改为一则 `ClinicalAssessmentSpec`（声明问题、允许值、可用证据、复核策略），DAG 里改用 `clinical_assessment_ref` 节点引用它。判定依据写在指南规范里，本库原样实现：

* `aspect == "treatment_eligibility"`，或
* 标识符/名称里带 `fitness` / `tolerab*` / `suitab*` / `feasib*` / `readiness` / `stable_control` 等综合判断特征词

缺信息时结果保持 `unknown`，**不会退化成 `false`**（强 Kleene 三值传播）。因此"指南没覆盖"和"患者不符合"在输出里是可区分的。

---

## 架构

```text
指南文本
   │  graph/          ← 抽取层：推荐意见 → LSH 聚类 → terms → predicates → rules → safety
   ▼
抽取产物 {terms, med_terms, predicates, rules, contraindications, safety_constraints}
   │  projection.py   ← 投影层（本库新增，1 297 行）：四步确定性变换
   │     1. action 物化：rules[].action → 顶层 actions[] + rules[].action_refs
   │     2. DAG 词表归一：16 种节点类型 → 解释器支持的 6 种；operator 归一；
   │        裸谓词 id → predicate_ref 节点；非布尔根 → 未解析常量
   │     3. ClinicalAssessment 路由：不可计算发现 → ClinicalAssessmentSpec
   │     4. schema_v2 语义分类：semantic_domain / compute_family / compute_facets
   ▼
可执行 Schema-v2 图（schema_version = decisionkg.schema_v2.unified.20260804）
   │  runtime/        ← 解释器：确定性求值 + 三值传播 + 执行许可 + 临床交接渲染
   ▼
{评估, 动作及状态, 逐节点 trace, 临床交接文本}
```

动作状态由「规则真值 × 复核策略」推出：

| 状态 | 含义 |
|---|---|
| `required` | 条件成立且无复核要求 → 可执行（`execution_authorized: true`） |
| `pending_human_review` | 条件成立，但被 `human_review_policy: required_before_action` 的评估档位门控 |
| `not_activated` | 条件确定为假 |
| `unresolved` | 资料不足，保持未决 |

---

## 安装与验证

```bash
pip install -e .

# 离线自检：32 个用例，不需要 API key、不需要任何数据文件
python3 tests/test_smoke.py
```

自检用的两份抽取产物**内联在 `tests/fixtures.py`** 里，仓库不携带数据文件：

| 夹具 | 来源 | 覆盖边界 |
|---|---|---|
| `SYNTHETIC_EXTRACTION` | 手写合成 | 可计算阈值、`treatment_eligibility` 型评估、自定义比较符 `within_normal_range`、值类型（`Enum`）谓词直接当规则根 |
| `LIVE_EXTRACTION` | **真实** LLM 抽取（一条中文 HFrEF 推荐意见，66.7 s） | `Permission` 以 str-enum 形态到达、事实概念仅由 `retrieve.code_binding` 指出的真实形态 |

两者都不含患者资料、不含答案或 Oracle。

LLM 配置沿用抽取层的环境变量约定：

```bash
LLM_PROVIDER=deepseek          # 或 dashscope（默认）
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
```

---

## 公开接口

### `compile_guideline(text, **kwargs) -> ExecutableGraph`

| 参数 | 说明 |
|---|---|
| `text` | 单条指南段落，或段落列表；段落可用 `"<来源>\n<推荐意见>"` 携带出处 |
| `api_key` / `base_url` / `model` / `temperature` | LLM 覆盖项；缺省走 `LLMConfig.from_env()` |
| `graph_id` / `qid` | 稳定标识；`qid` 缺省由 `graph_id` 派生 |
| `gen_dir` | 阶段缓存目录；缺省临时目录，库不会往安装目录写文件 |
| `max_concurrency` | 并行聚类抽取数 |

返回 `ExecutableGraph`：行为等同图 dict（可直接喂给 `run_graph` 或 `json.dump`），另有 `.graph`、`.review`、`.programs`、`.to_json()`。

### `run_graph(graph, observations, clinical_assessments=None, clinical_assertions=None, *, include_clinical_trace=False) -> dict`

| 输入 | 形状 |
|---|---|
| `observations` | `{fact_id, concept_id, value, unit?, polarity?, temporal?, source_quote?}` —— 通过 `concept_id` 绑定到谓词 |
| `clinical_assessments` | `{assessment_id, assessment_spec_id, value: true\|false\|unknown, assessment_authority: llm\|clinical_expert, model_id, prompt_version, supporting_observation_ids, source_spans, requires_human_review}` |
| `clinical_assertions` | `{assertion_id, concept_id, value, asserted_by, source_span, timepoint}` |

返回 trace：`predicate_results`、`clinical_assessment_results`、`condition_node_results`、`rule_results`、`actions`，以及（可选）`clinical_trace` —— 即交给 LLM 的"临床交接"文本：决策命题、状态、由哪些事实支撑、动作与未决条件。

---

## 质量与边界

投影层是**纯函数**：只读抽取产物，不读患者数据、不读 Oracle/参考答案。凡是它无法确定地转换的内容，都会写进 `.review` 而不是悄悄改动语义：

| `review.kind` | 含义 |
|---|---|
| `routed_to_clinical_assessment` | 发现不可确定性计算，已改为评估问题 |
| `mapped_custom_comparator` | 自定义比较符映射到了最近的可执行算子，需人工确认 |
| `unsupported_comparison` | 比较符无等价物，比较被丢弃（已记录） |
| `non_executable_dag_node` | 抽取层有而解释器无对应算子的 DAG 节点类型 |
| `non_boolean_rule_root` | 规则根是值类型，无法约简为布尔条件 → 规则固定未决 |
| `unresolved_dag_operand` | DAG 操作数既不是谓词也不是节点 |
| `orphan_predicates` | 声明了但没有任何规则引用 |
| `assessment_gated_action` | 动作被评估档位门控，执行取决于复核策略 |

**已验证范围**（三层，逐层独立）：

| 验证 | 输入 | 结果 |
|---|---|---|
| 离线用例 | 合成 + 真实抽取样例 | 32/32 全绿 |
| 真实抽取产物 | 106 题 AML 指南语料的 62 份 batch（4 559 条规则、10 966 条谓词、1 325 条评估） | 投影 62/62、执行 62/62、零失败；三值 `true 3072 / false 590 / unknown 897` |
| 现场端到端 | 一条中文指南推荐意见（真实 LLM 调用） | `compile_guideline` 66.7 s → 1 term/1 med_term/1 predicate/1 rule/1 action → `run_graph` 命中 `required` 且 `execution_authorized: true` |

**未包含**：OMOP/FHIR 词汇标准化、逐题（per-question）图迁移链、基准评测与判分、论文出图。图只由指南文本驱动，患者资料由调用方提供。

## 目录

```text
README.md          中文说明（本文件）
README.en.md       English description
guideline2action/
├── graph/         抽取层（指南文本 → terms/predicates/rules）
├── projection.py  投影层（抽取产物 → 可执行 Schema-v2 图）
├── runtime/       解释器 + 临床交接渲染
└── __init__.py    compile_guideline / run_graph
tests/
├── fixtures.py    两份内联抽取产物（合成 + 真实）
└── test_smoke.py  32 个离线用例
```

## 许可

Apache License 2.0 —— 见 [`LICENSE`](LICENSE)。
