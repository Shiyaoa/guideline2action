"""
Clinical Knowledge Pipeline - 临床指南知识抽取流水线

一个模块化的临床指南解析工具包，用于从临床指南文本中抽取结构化知识。

主要组件:
- models: 数据模型定义
- config: 配置管理
- graph: LangGraph 两阶段 Map-Reduce 工作流
- io_utils: 输入输出工具

快速开始:
```python
from guideline2action.graph import create_pipeline, save_to_gen

# 创建流水线
pipeline = create_pipeline(
    api_key="your-api-key",
    model="deepseek-v4-pro"
)

# 运行（单文本或多文本）
result = pipeline.run("单个文本")
result = pipeline.run(["文本1", "文本2", "文本3"])

# 异步运行
result = await pipeline.run_async(["文本1", "文本2"])

# 保存结果
save_to_gen(result)
```

分阶段运行（每阶段独立运行并保存结果）:
```python
from guideline2action.graph import (
    run_recommendation_clustering_pipeline,
    run_graph_extraction_pipeline,
    run_standardization,
)

# Pipeline A: 阶段 1-2，抽取推荐意见并聚类
run_recommendation_clustering_pipeline(texts, gen_dir="gen", max_concurrency=3)
# 输出 gen/provenances.json, gen/clusters.json, gen/cluster_cache.json

# Pipeline B: 阶段 3-5，复用 clusters/cache 调试 typed graph 抽取
run_graph_extraction_pipeline(gen_dir="gen", max_concurrency=3)
# 输出 gen/terms.json, gen/med_terms.json, gen/predicates.json, gen/rules.json

# 阶段 6: 后处理标准化（OMOP 匹配 + 同义术语合并）
standardization_result = run_standardization(gen_dir="gen", enable_review=True)
# 调试 graph 抽取时可跳过慢速本地 OMOP matching
fast_standardization_result = run_standardization(
    gen_dir="gen",
    enable_review=False,
    skip_omop_matching=True,
)
# 结果保存到 gen/terms_standardized.json 等
```
"""

# 版本信息
__version__ = "1.0.0"

# Compiler version for metadata tracking (Experiment 1 reproducibility)
COMPILER_VERSION = "0.3.0"

# 核心数据模型
from .models import (
    # 枚举
    TermLabel,
    Permission,
    # 核心模型
    Term,
    MedicationTerm,
    Predicates,
    Action,
    CodeBinding,
    BindingEvidence,
    BindingResolution,
    DataBinding,
    SourceEvidence,
    SourceSpan,
    TemporalScope,
    ReductionSpec,
    QuantitySemantics,
    RangeInterval,
    RangeSpec,
    OutputAssembly,
    DAGNode,
    ConditionDAG,
    RulePriority,
    LibraryFunction,
    Provenance,
    ClinicalRule,
    ProvenanceCluster,
    # 列表包装类
    TermList,
    MedicationTermList,
    PredicatesList,
    ProvenanceList,
    ClinicalRuleList,
    # 合并抽取结果
    TermExtractionResult,
    # 状态
    AgentState,
    ClusterState,
)

# 配置
from .config import (
    LLMConfig,
    PathConfig,
    MatchConfig,
    PipelineConfig,
    get_config,
    set_config,
)

# 标准库
# 图/流水线（底层节点与图）
from .graph import (
    # 节点函数
    extract_recommendation,
    # 异步节点函数
    asubgraph_extract_all_terms,
    async_extract_predicates_subgraph,
    async_extract_rules_subgraph,
    aprocess_cluster,
    # 图构建
    build_extraction_subgraph,
    async_build_extraction_subgraph,
    build_pipeline_graph,
    # 预构建的子图
    CLUSTER_SUBGRAPH,
    ASYNC_CLUSTER_SUBGRAPH,
)

# 接口与组合（基于 fan-out/fan-in 的封装）
from .graph_api import (
    ClinicalGuidelinePipeline,
    create_pipeline,
    save_results_from_cache,
    extract_provenances_stage,
    cluster_provenances_stage,
    extract_terms_stage,
    extract_predicates_stage,
    extract_rules_stage,
    run_recommendation_clustering_pipeline,
    run_graph_extraction_pipeline,
    process_clusters_async,
)

# LSH聚类
from .lsh_cluster import (
    lsh_cluster,
    LSHResult,
)

# IO工具
from .io_utils import (
    save_to_gen,
    load_from_gen,
    export_state_to_json,
    update_standard_library,
    save_stage_results,
    FailedTaskLogger,
    get_failed_task_logger,
)


# 定义公开API
__all__ = [
    # 版本
    "__version__",
    "COMPILER_VERSION",

    # 模型
    "TermLabel",
    "Permission",
    "Term",
    "MedicationTerm",
    "Predicates",
    "Action",
    "CodeBinding",
    "BindingEvidence",
    "BindingResolution",
    "DataBinding",
    "SourceEvidence",
    "SourceSpan",
    "TemporalScope",
    "ReductionSpec",
    "QuantitySemantics",
    "RangeInterval",
    "RangeSpec",
    "OutputAssembly",
    "DAGNode",
    "ConditionDAG",
    "RulePriority",
    "LibraryFunction",
    "Provenance",
    "ClinicalRule",
    "ProvenanceCluster",
    "TermList",
    "MedicationTermList",
    "PredicatesList",
    "ProvenanceList",
    "ClinicalRuleList",
    "TermExtractionResult",
    "AgentState",
    "ClusterState",

    # 配置
    "LLMConfig",
    "PathConfig",
    "MatchConfig",
    "PipelineConfig",
    "get_config",
    "set_config",

    # 标准库

    # 处理器

    # 节点函数
    "extract_recommendation",

    # 异步节点函数
    "asubgraph_extract_all_terms",
    "async_extract_predicates_subgraph",
    "async_extract_rules_subgraph",
    "aprocess_cluster",

    # LSH聚类
    "lsh_cluster",
    "LSHResult",

    # 图/流水线
    "build_extraction_subgraph",
    "async_build_extraction_subgraph",
    "build_pipeline_graph",
    "CLUSTER_SUBGRAPH",
    "ASYNC_CLUSTER_SUBGRAPH",

    # 接口/阶段
    "ClinicalGuidelinePipeline",
    "create_pipeline",
    "save_results_from_cache",
    "extract_provenances_stage",
    "cluster_provenances_stage",
    "extract_terms_stage",
    "extract_predicates_stage",
    "extract_rules_stage",
    "run_recommendation_clustering_pipeline",
    "run_graph_extraction_pipeline",
    "process_clusters_async",

    # 后处理标准化

    # 失败任务管理
    "FailedTaskLogger",
    "get_failed_task_logger",

    # IO
    "save_to_gen",
    "save_stage_results",
    "load_from_gen",
    "export_state_to_json",
    "update_standard_library",
]
