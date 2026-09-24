"""
数据模型定义 - v2 schema-first typed graph.

v2 keeps the three extraction stages (terms -> predicates -> rules), but the
stage outputs are no longer legacy condition strings or formal_definition text.
Predicates carry typed CQL-like execution slots, and rules carry a typed
condition_dag / expression graph.
"""
from enum import Enum
from typing import Annotated, ClassVar, Sequence, TypedDict, List, Optional, TypeVar, Dict, Literal, Any, Union
import ast
import json
import operator
import logging
import re
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from .library_registry import is_registered_library_function

# Logger for validation warnings
_logger = logging.getLogger(__name__)

# LangGraph 相关导入（延迟导入以避免依赖问题）
try:
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    AnyMessage = None
    add_messages = None


# ============ 自定义 Reducer（带去重） ============

T = TypeVar("T")


def merge_by_id(existing: List[T], new: List[T]) -> List[T]:
    """
    自定义 reducer：合并列表并按 id 去重。
    后来的项会覆盖先前的同 id 项。
    """
    merged = {}

    for item in existing:
        item_id = _get_item_id(item)
        if item_id is not None:
            merged[item_id] = item
        else:
            _logger.warning(f"merge_by_id: 跳过没有有效 id 的项: {type(item).__name__}")

    for item in new:
        item_id = _get_item_id(item)
        if item_id is not None:
            merged[item_id] = item
        else:
            _logger.warning(f"merge_by_id: 跳过没有有效 id 的项: {type(item).__name__}")

    return list(merged.values())


def _get_item_id(item: Any) -> Optional[str]:
    """Return an id from Pydantic-style objects or raw dict compiler outputs."""
    if hasattr(item, "id") and item.id is not None:
        return str(item.id)
    if isinstance(item, dict):
        item_id = item.get("id")
        if item_id is not None:
            return str(item_id)
    return None


def _to_models(items: Any, model_cls):
    """确保列表元素为指定 Pydantic 模型实例，便于 merge_by_id 去重。"""
    if not items:
        return []
    converted = []
    for item in items:
        if isinstance(item, model_cls):
            converted.append(item)
        elif isinstance(item, dict):
            try:
                converted.append(model_cls.model_validate(item))
            except Exception as e:
                _logger.warning(f"_to_models: 无法将字典验证为 {model_cls.__name__}: {e}")
                continue
        else:
            _logger.warning(f"_to_models: 跳过非预期类型 {type(item).__name__} (期望 {model_cls.__name__})")
            continue
    return converted


def merge_cluster_cache_updates(existing: Dict[int, Dict[str, Any]], new: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """自定义 reducer：合并 cluster 缓存更新。"""
    merged = existing.copy()
    for cluster_id, updates in new.items():
        if cluster_id not in merged:
            merged[cluster_id] = {}
        merged[cluster_id].update(updates)
    return merged


# ============ 枚举类型 ============

class TermLabel(str, Enum):
    """术语分类标签"""
    MEASURES = "measures"
    CONDITIONS = "conditions"
    PROCEDURES = "procedures"
    OBSERVATIONS = "observations"


class Permission(str, Enum):
    """规则许可类型 - 用于 v2 action algebra 的 permission 槽位。"""
    ALLOW = "allow"
    RECOMMEND = "recommend"
    REQUIRE = "require"
    CAUTION = "caution"
    AVOID = "avoid"
    CONTRAINDICATE = "contraindicate"
    CONTINUE = "continue"
    STOP = "stop"
    CONSIDER = "consider"
    REDUCE_DOSE = "reduce_dose"
    INCREASE_DOSE = "increase_dose"
    START_LOW_DOSE = "start_low_dose"
    MAX_DOSE_LIMIT = "max_dose_limit"
    TITRATE = "titrate"
    MAINTAIN_DOSE = "maintain_dose"

    @classmethod
    def usage_permissions(cls) -> list:
        return [
            cls.CONTRAINDICATE,
            cls.AVOID,
            cls.CAUTION,
            cls.ALLOW,
            cls.RECOMMEND,
            cls.REQUIRE,
            cls.CONSIDER,
            cls.CONTINUE,
            cls.STOP,
        ]

    @classmethod
    def dose_permissions(cls) -> list:
        return [
            cls.REDUCE_DOSE,
            cls.INCREASE_DOSE,
            cls.START_LOW_DOSE,
            cls.MAX_DOSE_LIMIT,
            cls.TITRATE,
            cls.MAINTAIN_DOSE,
        ]

    @classmethod
    def priority_order(cls) -> list:
        """冲突检测优先级；越靠前越限制。"""
        return [cls.CONTRAINDICATE, cls.AVOID, cls.STOP, cls.CAUTION, cls.CONSIDER, cls.ALLOW, cls.RECOMMEND, cls.REQUIRE]

    @classmethod
    def dose_priority_order(cls) -> list:
        return [cls.REDUCE_DOSE, cls.MAX_DOSE_LIMIT, cls.START_LOW_DOSE, cls.TITRATE, cls.MAINTAIN_DOSE, cls.INCREASE_DOSE]

    def priority(self) -> int:
        order = self.priority_order()
        return order.index(self) if self in order else -1

    def is_restrictive(self) -> bool:
        return self in [Permission.CONTRAINDICATE, Permission.AVOID, Permission.CAUTION, Permission.STOP]

    def is_permissive(self) -> bool:
        return self in [Permission.ALLOW, Permission.RECOMMEND, Permission.REQUIRE, Permission.CONSIDER, Permission.CONTINUE]

    def is_dose_adjustment(self) -> bool:
        return self in self.dose_permissions()


class FlexibleModel(BaseModel):
    """Base model for schema-first outputs that still tolerates forward fields."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")


def _canonical_type_name(value: Any) -> Any:
    """Normalize common LLM spellings while preserving unknown forward types."""
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    aliases = {
        "bool": "Bool",
        "boolean": "Bool",
        "int": "Integer",
        "integer": "Integer",
        "datetime": "DateTime",
        "date_time": "DateTime",
        "quantity": "Quantity",
        "number": "Quantity",
        "observation": "Observation",
        "condition": "Condition",
    }
    return aliases.get(normalized.lower(), normalized)


def _parse_serialized_model_list(value: Any) -> Any:
    """Repair a list that an LLM returned as a JSON/Python-ish string."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return []

    fence = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    def _coerce(parsed: Any) -> Any:
        if isinstance(parsed, tuple):
            parsed = list(parsed)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return list(parsed)
        return value

    for candidate in (text,):
        try:
            return _coerce(json.loads(candidate))
        except Exception:
            pass
        try:
            return _coerce(ast.literal_eval(candidate))
        except Exception:
            pass

    pythonish = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
    pythonish = re.sub(r"\btrue\b", "True", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    try:
        return _coerce(ast.literal_eval(pythonish))
    except Exception:
        return value


# ============ v2 shared schema ============

class QuantityValue(FlexibleModel):
    value: Optional[Union[int, float]] = None
    unit: Optional[str] = None
    value_min: Optional[Union[int, float]] = None
    value_max: Optional[Union[int, float]] = None
    comparator: Optional[str] = None


class CodeBinding(FlexibleModel):
    """ValueSet or single code binding."""
    type: str = Field(default="ValueSet", description="ValueSet, Code, CodeSystem, OMOPConcept, or Unknown")
    name: Optional[str] = None
    oid: Optional[str] = None
    system: Optional[str] = None
    code: Optional[Union[str, int]] = None
    display: Optional[str] = None
    url: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BindingEvidence(FlexibleModel):
    """Audit trail for deterministic or candidate binding decisions."""
    source: str = "llm"
    matched_by: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note: Optional[str] = None
    omop_concept_id: Optional[int] = None
    vocabulary_id: Optional[str] = None


class BindingResolution(FlexibleModel):
    """Resolved binding payload.

    Full terminology bindings should only be marked verified after a
    deterministic terminology registry/validator or manual review. FHIR
    resource/path bindings can be verified separately by the local registry.
    """
    status: Literal["candidate", "verified", "rejected", "unresolved"] = "unresolved"
    fhir_binding: Dict[str, Any] = Field(default_factory=dict)
    omop_binding: Dict[str, Any] = Field(default_factory=dict)
    code_bindings: List[CodeBinding] = Field(default_factory=list)
    value_set_binding: Optional[CodeBinding] = None
    evidence: List[BindingEvidence] = Field(default_factory=list)
    unresolved_reasons: List[str] = Field(default_factory=list)


class DataBinding(FlexibleModel):
    """FHIR/OMOP/QDM binding hints for execution-time retrieval."""
    fhir: Dict[str, Any] = Field(default_factory=dict, alias="FHIR")
    omop: Dict[str, Any] = Field(default_factory=dict, alias="OMOP")
    qdm: Dict[str, Any] = Field(default_factory=dict, alias="QDM")


class SourceSpan(FlexibleModel):
    source_text: Optional[str] = None
    source_chunk: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    page: Optional[Union[int, str]] = None
    section: Optional[str] = None


class SourceEvidence(FlexibleModel):
    source: Optional[str] = None
    quote: Optional[str] = None
    source_text: Optional[str] = None
    source_span: Optional[SourceSpan] = None
    source_chunk: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TemporalScope(FlexibleModel):
    mode: str = Field(
        default="all_time",
        description="Standard v2 temporal mode, e.g. all_time/currently_active/any_in_lookback/most_recent_in_lookback/count_in_lookback/count_all_time/relative_to_event.",
    )
    lookback: Optional[QuantityValue] = None
    interval: Optional[Dict[str, Any]] = None
    anchor: Optional[str] = None
    offset: Optional[QuantityValue] = None
    time_paths: List[str] = Field(default_factory=list)
    date_fallback: List[str] = Field(default_factory=list)
    relation: Optional[str] = None
    time_resolution: Optional[str] = None

    STANDARD_MODES: ClassVar[set[str]] = {
        "all_time",
        "currently_active",
        "any_in_lookback",
        "most_recent_in_lookback",
        "extremum_in_lookback",
        "count_in_lookback",
        "count_all_time",
        "relative_to_event",
    }
    MODE_ALIASES: ClassVar[Dict[str, str]] = {
        "most_recent_all_time": "all_time",
        "count_in_window": "count_in_lookback",
        "any_in_window": "any_in_lookback",
        "lookback": "any_in_lookback",
    }

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            return "all_time"
        mode = str(value).strip().lower()
        mode = cls.MODE_ALIASES.get(mode, mode)
        if mode not in cls.STANDARD_MODES:
            allowed = ", ".join(sorted(cls.STANDARD_MODES))
            raise ValueError(f"Unsupported temporal_scope.mode '{value}'. Expected one of: {allowed}")
        return mode


class ReductionSpec(FlexibleModel):
    operator: Literal["none", "exists", "count", "most_recent", "max", "min", "extremum"] = "none"
    output_type: str = "Bool"
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_type", mode="before")
    @classmethod
    def normalize_output_type(cls, value: Any) -> Any:
        return _canonical_type_name(value)


class QuantitySemantics(FlexibleModel):
    unit: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[Union[int, float, str]] = None
    normalization: Dict[str, Any] = Field(default_factory=dict)
    interval: Optional[Dict[str, Any]] = None
    z3_sort: Optional[str] = None


class RangeInterval(FlexibleModel):
    id: str
    label: Optional[str] = None
    lower: Optional[Union[int, float]] = None
    upper: Optional[Union[int, float]] = None
    # Optional[bool]: LLM 合理地对开区间端点（lower/upper 为 None）返回 null，
    # 严格 bool 校验会拒绝 null 并丢弃整个 fragment。放宽为 Optional[bool]，
    # 配合下方 validate_bounds 对 None 的兼容处理。
    include_lower: Optional[bool] = True
    include_upper: Optional[bool] = True
    unit: Optional[str] = None

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError(f"Range interval '{self.id}' lower bound must be <= upper bound")
            # 仅当两端 include 都明确为 False 时才算 empty bounds；
            # None（开区间端点无意义）按非排除处理，避免误判。
            if self.lower == self.upper and self.include_lower is False and self.include_upper is False:
                raise ValueError(f"Range interval '{self.id}' has empty bounds because equal endpoints are excluded")
        return self


class RangeSpec(FlexibleModel):
    id: str
    label: Optional[str] = None
    intervals: List[RangeInterval] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_intervals(self):
        interval_ids = [interval.id for interval in self.intervals]
        duplicates = sorted({interval_id for interval_id in interval_ids if interval_ids.count(interval_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate range interval id(s): {', '.join(duplicates)}")
        return self


class OutputAssembly(FlexibleModel):
    """Rule output assembly algebra for CQL-style message/action lists."""
    operator: Literal["union_except_null", "union", "except_null", "message_list"]
    inputs: List[Any] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    output_type: str = "List<ActionOrMessage>"

    @field_validator("output_type", mode="before")
    @classmethod
    def normalize_output_type(cls, value: Any) -> Any:
        return _canonical_type_name(value)

    @model_validator(mode="after")
    def validate_assembly(self):
        if self.operator in {"union_except_null", "union", "except_null"} and not self.inputs:
            raise ValueError(f"output_assembly operator '{self.operator}' requires inputs")
        if self.operator == "message_list" and not self.messages:
            raise ValueError("output_assembly operator 'message_list' requires messages")
        return self


# ============ 核心数据模型 ============

class Term(FlexibleModel):
    """v2 非药物术语：标准概念 + 数据绑定 + 证据。"""
    id: str
    name: str
    label: TermLabel = Field(..., description="术语分类")
    type: str = Field(description="value domain or legacy term type")
    clinical_entity: str = Field(default="", description="Canonical entity id, usually same as id")
    concept: str = Field(default="", description="Canonical clinical concept name")
    value_domain: Optional[str] = Field(default=None, description="Bool/Quantity/Code/DateTime/Interval/Enum/List<Resource>")
    unit: Optional[str] = None
    value_set_binding: Optional[CodeBinding] = None
    code_bindings: List[CodeBinding] = Field(default_factory=list)
    data_bindings: DataBinding = Field(default_factory=DataBinding)
    fhir_binding_hint: Dict[str, Any] = Field(default_factory=dict)
    omop_binding_hint: Dict[str, Any] = Field(default_factory=dict)
    binding_status: Literal["candidate", "verified", "rejected", "unresolved"] = "candidate"
    candidate_binding: Dict[str, Any] = Field(default_factory=dict)
    verified_binding: Optional[BindingResolution] = None
    binding_evidence: List[BindingEvidence] = Field(default_factory=list)
    normalization: Dict[str, Any] = Field(default_factory=dict)
    normalization_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_evidence: List[SourceEvidence] = Field(default_factory=list)


class MedicationTerm(FlexibleModel):
    """v2 药物术语：药物/药物类概念 + RxNorm/OMOP/FHIR hints。"""
    id: str
    name: str
    drug_class: Optional[str] = Field(
        None,
        alias="class",
        description="所属的药物大类 ID。如果是药物类别本身，则为 null 或缺失。",
    )
    subclass: Optional[str] = None
    clinical_entity: str = Field(default="", description="Canonical medication entity id")
    concept: str = Field(default="", description="Canonical medication concept name")
    value_set_binding: Optional[CodeBinding] = None
    code_bindings: List[CodeBinding] = Field(default_factory=list)
    data_bindings: DataBinding = Field(default_factory=DataBinding)
    fhir_binding_hint: Dict[str, Any] = Field(default_factory=dict)
    omop_binding_hint: Dict[str, Any] = Field(default_factory=dict)
    binding_status: Literal["candidate", "verified", "rejected", "unresolved"] = "candidate"
    candidate_binding: Dict[str, Any] = Field(default_factory=dict)
    verified_binding: Optional[BindingResolution] = None
    binding_evidence: List[BindingEvidence] = Field(default_factory=list)
    normalization_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_evidence: List[SourceEvidence] = Field(default_factory=list)


class Predicates(FlexibleModel):
    """v2 typed predicate schema.

    A predicate describes a reusable CQL-like observation pattern. It may
    internally retrieve, filter, reduce, extract, and compare data, but those
    slots are structured fields rather than a free-text formal_definition.
    """
    id: str
    name: str
    description: str = ""
    source_text: str = ""
    legacy_id: Optional[str] = None

    entity: str
    entity_type: str
    aspect: str
    input_shape: str
    reduction: ReductionSpec
    return_type: str
    final_output_type: str
    temporal_scope: TemporalScope
    data_binding: DataBinding = Field(default_factory=DataBinding)
    library_function: List[str] = Field(default_factory=list)
    value_set_binding: Optional[CodeBinding] = None
    code_binding: Optional[CodeBinding] = None
    unit: Optional[str] = None
    quantity_semantics: QuantitySemantics = Field(default_factory=QuantitySemantics)
    range_spec: Optional[RangeSpec] = None
    retrieve: Dict[str, Any] = Field(default_factory=dict)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    extract: Dict[str, Any] = Field(default_factory=dict)
    compare: Optional[Dict[str, Any]] = None
    null_policy: Literal["false", "unknown", "null", "assume_false", "propagate_unknown"] = "unknown"
    evidence: List[SourceEvidence] = Field(default_factory=list)
    source_span: Optional[SourceSpan] = None
    dependencies: List[str] = Field(default_factory=list)
    z3_mapping: Dict[str, Any] = Field(default_factory=dict)

    # Legacy-only optional field. Do not use as the core expression in v2.
    formal_definition: Optional[str] = None

    @field_validator("return_type", "final_output_type", mode="before")
    @classmethod
    def normalize_predicate_output_types(cls, value: Any) -> Any:
        return _canonical_type_name(value)

    @field_validator("library_function", mode="before")
    @classmethod
    def validate_registered_library_functions(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value
        unknown = [fn_id for fn_id in value if not is_registered_library_function(fn_id)]
        if unknown:
            raise ValueError(f"Unknown library_function id(s): {', '.join(unknown)}")
        return value

    @model_validator(mode="after")
    def validate_range_predicate(self):
        is_range_predicate = self.aspect == "quantity_range" or self.id.endswith(".range")
        if not is_range_predicate:
            return self
        if self.final_output_type != "Enum":
            raise ValueError("Range predicates must have final_output_type='Enum'")
        if self.range_spec is None or not self.range_spec.intervals:
            raise ValueError("Range predicates must declare non-empty range_spec.intervals")
        return self


class Action(FlexibleModel):
    """v2 action algebra object."""
    subjects: List[str] = Field(description="药物、检查、操作或输出对象 id 列表")
    permission: Permission = Field(description="许可/动作类型")
    strength: Optional[str] = None
    intent: Optional[str] = None
    dose: Optional[Dict[str, Any]] = None
    duration: Optional[Dict[str, Any]] = None
    monitoring: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    timing: Optional[Dict[str, Any]] = None
    conflict_profile: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None


class Provenance(FlexibleModel):
    """来源信息"""
    source: str
    quote: str
    recommendation_grade: Optional[str] = None
    evidence_level: Optional[str] = None
    type: Optional[str] = Field(None, description="数据类型：guideline（指南）或 expert（专家共识）")
    bucket_id: Optional[int] = Field(None, description="LSH聚类桶ID")
    source_span: Optional[SourceSpan] = None


class DAGNode(FlexibleModel):
    """A typed Rule DAG node. Extra fields are allowed for CQL-like operators."""
    TEMPORAL_RELATIONS: ClassVar[set[str]] = {
        "before",
        "after",
        "during",
        "overlaps",
        "starts",
        "ends",
        "meets",
        "same_or_before",
        "same_or_after",
        "within",
        "in_interval",
    }

    id: str
    type: Literal[
        "predicate_ref",
        "combine",
        "compare",
        "aggregate",
        "coalesce",
        "temporal_relation",
        "range_membership",
        "library_function",
        "output_assembly",
        "literal",
        "filter",
        "exists",
        "sort",
        "interval",
        "unit_convert",
        "extract",
    ]
    operator: Optional[str] = None
    predicate_ref: Optional[str] = None
    input: Optional[Union[str, List[str], Dict[str, Any]]] = None
    inputs: List[Any] = Field(default_factory=list)
    left: Optional[Any] = None
    right: Optional[Any] = None
    value: Optional[Any] = None
    return_type: str
    typed_value: Optional[Dict[str, Any]] = None
    library_function: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    source_span: Optional[SourceSpan] = None

    @field_validator("return_type", mode="before")
    @classmethod
    def normalize_return_type(cls, value: Any) -> Any:
        return _canonical_type_name(value)

    @field_validator("operator", mode="before")
    @classmethod
    def normalize_operator(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped.lower() if re.search(r"[A-Za-z]", stripped) else stripped

    @model_validator(mode="after")
    def validate_node_shape(self):
        if self.type == "combine" and not self.inputs:
            raise ValueError(f"combine DAG node '{self.id}' must declare non-empty inputs")
        if self.type == "compare" and (self.left is None or self.right is None):
            raise ValueError(f"compare DAG node '{self.id}' must declare both left and right")
        if self.type == "coalesce" and len(self.inputs) < 2:
            raise ValueError(f"coalesce DAG node '{self.id}' must declare at least two inputs")
        if self.type == "range_membership":
            if self.input is None or self.value is None:
                raise ValueError(f"range_membership DAG node '{self.id}' must declare input and value")
            if self.return_type != "Bool":
                raise ValueError(f"range_membership DAG node '{self.id}' must return Bool")
        if self.type == "temporal_relation":
            if self.return_type != "Bool":
                raise ValueError(f"temporal_relation DAG node '{self.id}' must return Bool")
            if self.operator not in self.TEMPORAL_RELATIONS:
                allowed = ", ".join(sorted(self.TEMPORAL_RELATIONS))
                raise ValueError(f"temporal_relation DAG node '{self.id}' must use operator in: {allowed}")
        if self.type == "aggregate":
            if self.operator == "count" and self.return_type != "Integer":
                raise ValueError(f"aggregate(count) DAG node '{self.id}' must return Integer")
            if self.operator in {"max", "min", "extremum"} and self.return_type != "Quantity":
                raise ValueError(f"aggregate({self.operator}) DAG node '{self.id}' must return Quantity")
        if self.library_function is not None and not is_registered_library_function(self.library_function):
            raise ValueError(f"DAG node '{self.id}' references unknown library_function id '{self.library_function}'")
        if self.type == "library_function" and not self.library_function:
            raise ValueError(f"library_function DAG node '{self.id}' must declare library_function")
        return self


class ConditionDAG(FlexibleModel):
    nodes: List[DAGNode]
    root: str
    typed_intermediates: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dag(self):
        node_ids = [node.id for node in self.nodes]
        duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate DAG node id(s): {', '.join(duplicates)}")

        by_id = {node.id: node for node in self.nodes}
        if self.root not in by_id:
            raise ValueError(f"condition_dag.root '{self.root}' must reference an existing node")

        root_node = by_id[self.root]
        if root_node.return_type != "Bool":
            raise ValueError(f"condition_dag.root '{self.root}' must return Bool, got {root_node.return_type}")
        return self


def _resolve_rule_boolean_root(condition_dag: ConditionDAG, boolean_root: Optional[str]) -> str:
    by_id = {node.id: node for node in condition_dag.nodes}
    resolved = (boolean_root or "").strip()
    if not resolved or (resolved == "ROOT" and resolved not in by_id):
        resolved = condition_dag.root

    if resolved not in by_id:
        raise ValueError(f"boolean_root '{resolved}' must reference an existing condition_dag node")
    if by_id[resolved].return_type != "Bool":
        raise ValueError(f"boolean_root '{resolved}' must return Bool, got {by_id[resolved].return_type}")
    return resolved


class RulePriority(FlexibleModel):
    recommendation_grade: Optional[str] = None
    evidence_level: Optional[str] = None
    source_date: Optional[str] = None
    jurisdiction: Optional[str] = None


class RuleScope(FlexibleModel):
    population: List[str] = Field(default_factory=list)
    setting: Optional[str] = None
    guideline_domain: List[str] = Field(default_factory=list)
    criteria: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured inclusion/exclusion Criterion objects with criterion_id, type, expression, rationale fields. "
        "Populated when the enhanced RULES_PROMPT produces typed criteria; falls back to population free-text for unclassifiable criteria.",
    )


class ClinicalRule(FlexibleModel):
    """v2 clinical rule with typed condition_dag."""
    id: str
    label: str
    source_text: Optional[str] = None
    source_span: Optional[SourceSpan] = None
    source_evidence: List[SourceEvidence] = Field(default_factory=list)
    input_predicates: List[str] = Field(default_factory=list)
    condition_dag: ConditionDAG
    boolean_root: str = "ROOT"
    action: Action
    scope: RuleScope = Field(default_factory=RuleScope)
    priority: RulePriority = Field(default_factory=RulePriority)
    missing_data_policy: Literal["suppress", "warn", "assume_false", "propagate_unknown"] = "propagate_unknown"
    provenance: List[Provenance] = Field(default_factory=list)
    output_assembly: Optional[OutputAssembly] = None

    # Legacy-only optional field. Do not use as the core expression in v2.
    condition: Optional[str] = None

    @model_validator(mode="after")
    def validate_boolean_root(self):
        self.boolean_root = _resolve_rule_boolean_root(self.condition_dag, self.boolean_root)
        return self


# ============ 标准库函数模型 ============

class LibraryFunction(FlexibleModel):
    id: str
    name: str
    input_type: str
    output_type: str
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    implementation_semantics: str
    cql_equivalent: Optional[str] = None


# ============ 聚类相关模型 ============

class ProvenanceCluster(FlexibleModel):
    """推荐意见聚类"""
    cluster_id: int
    provenances: List[Provenance]
    texts_formatted: List[str] = Field(default_factory=list)


# ============ 列表包装类（用于 LLM 结构化输出） ============

class ProvenanceList(BaseModel):
    """包装类，用于接收多个 Provenance 对象"""
    items: List[Provenance] = Field(description="Provenance 对象列表")


class PredicatesList(BaseModel):
    """包装类，用于接收多个 Predicates 对象"""
    items: List[Predicates] = Field(description="Predicates 对象列表")


class TermList(BaseModel):
    """包装类，用于接收多个 Term 对象"""
    items: List[Term] = Field(description="Term 对象列表")


class MedicationTermList(BaseModel):
    """包装类，用于接收多个 MedicationTerm 对象"""
    items: List[MedicationTerm] = Field(description="MedicationTerm 对象列表")


class ClinicalRuleList(BaseModel):
    """包装类，用于接收多个 ClinicalRule 对象"""
    items: List[ClinicalRule] = Field(description="ClinicalRule 对象列表")


class TermExtractionResult(BaseModel):
    """术语抽取结果 - 合并术语和药物抽取"""
    terms: List[Term] = Field(default_factory=list, description="v2 非药物术语列表")
    med_terms: List[MedicationTerm] = Field(default_factory=list, description="v2 药物术语列表")

    @field_validator("terms", "med_terms", mode="before")
    @classmethod
    def repair_serialized_lists(cls, value: Any) -> Any:
        return _parse_serialized_model_list(value)


# ============ AgentState 定义 ============

class AgentState(TypedDict):
    """Agent 状态定义 - 支持两阶段 map-reduce"""
    input_texts: List[str]
    messages: Annotated[Sequence[AnyMessage], add_messages]
    provenance_buffer: Annotated[List[Provenance], operator.add]
    clusters: List[ProvenanceCluster]
    terms: Annotated[List[Term], merge_by_id]
    med_terms: Annotated[List[MedicationTerm], merge_by_id]
    predicates: Annotated[List[Predicates], merge_by_id]
    rules: Annotated[List[ClinicalRule], merge_by_id]
    cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]
    cluster_cache: Dict[int, Dict[str, Any]]
    # New compiler-enhancement fields
    contraindications: Annotated[List, merge_by_id]
    safety_constraints: Annotated[List, merge_by_id]
    missing_variables: dict
    graph_metadata: Optional[dict]


class ClusterState(TypedDict):
    """子图状态 - 用于处理单个聚类"""
    cluster_id: int
    provenances: List[Provenance]
    texts_formatted: List[str]
    terms: Annotated[List[Term], merge_by_id]
    med_terms: Annotated[List[MedicationTerm], merge_by_id]
    predicates: Annotated[List[Predicates], merge_by_id]
    rules: Annotated[List[ClinicalRule], merge_by_id]


# ============ LLM 结构化输出模型（用于谓词和规则抽取） ============

class PredicateExtractionBatch(BaseModel):
    """谓词抽取批量结果：直接输出 typed predicate schemas。"""
    predicates: List[Predicates] = Field(
        ...,
        description="A flat list of v2 typed predicate schema objects.",
    )


class RuleAction(Action):
    """LLM structured-output action model."""
    permission: Literal[
        "recommend", "require", "allow", "consider", "caution", "avoid",
        "contraindicate", "continue", "stop",
        "reduce_dose", "increase_dose", "start_low_dose", "max_dose_limit",
        "titrate", "maintain_dose",
    ] = Field(description="行动的许可类型或调整方向。")


class SimplifiedRuleItem(BaseModel):
    id: str = Field(description="规则的唯一标识符，必须以 'rule.' 开头。")
    label: str = Field(description="规则的简短人类可读摘要。")
    input_predicates: List[str] = Field(description="该规则引用的 predicate id 列表。")
    condition_dag: ConditionDAG = Field(description="v2 typed expression graph; root must resolve to Bool.")
    boolean_root: str = Field(default="ROOT", description="condition_dag 中最终 Bool 节点 id。")
    missing_data_policy: Literal["suppress", "warn", "assume_false", "propagate_unknown"] = "propagate_unknown"
    source_ids: str = Field(description="该规则对应的 quote 编号 (如 'q1')。")
    action: RuleAction
    scope: RuleScope = Field(default_factory=RuleScope)
    priority: RulePriority = Field(default_factory=RulePriority)
    output_assembly: Optional[OutputAssembly] = None

    # Legacy field intentionally optional and non-core.
    condition: Optional[str] = None

    @model_validator(mode="after")
    def validate_boolean_root(self):
        self.boolean_root = _resolve_rule_boolean_root(self.condition_dag, self.boolean_root)
        return self


class SubmitSimplifiedRules(BaseModel):
    rules: List[SimplifiedRuleItem] = Field(description="提取的 v2 原子化规则列表。")


class ContraindicationList(BaseModel):
    """Wrapper for LLM structured output of Contraindication objects."""
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Contraindication 对象列表")


class SafetyConstraintList(BaseModel):
    """Wrapper for LLM structured output of SafetyConstraint objects."""
    items: List[Dict[str, Any]] = Field(default_factory=list, description="SafetyConstraint 对象列表")
