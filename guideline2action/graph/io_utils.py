"""
输入输出工具 - 用于读写数据
"""
import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import AgentState, Provenance, ProvenanceCluster
from .config import get_config

logger = logging.getLogger(__name__)


def save_to_gen(
    state: AgentState,
    gen_dir: Optional[str] = None,
    append: bool = True,
    save_provenances: bool = True,
    fields: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    将AgentState中的数据写入gen文件夹下的JSON文件

    Args:
        state: 包含提取结果的AgentState
        gen_dir: 输出目录，默认使用配置中的路径
        append: 是否追加模式（按id去重合并）
        save_provenances: 是否保存推荐意见缓存
        fields: 仅保存指定字段，None表示保存全部

    Returns:
        写入统计信息 {filename: count}
    """
    gen_dir = gen_dir or get_config().paths.gen_dir
    Path(gen_dir).mkdir(exist_ok=True)

    # 定义文件映射：state字段 -> 文件名
    file_mapping = {
        "terms": "terms.json",
        "med_terms": "med_terms.json",
        "predicates": "predicates.json",
        "rules": "rules.json"
    }

    # 可选：保存推荐意见
    if save_provenances:
        file_mapping["provenance_buffer"] = "provenances.json"

    # 仅保存指定字段
    if fields:
        file_mapping = {k: v for k, v in file_mapping.items() if k in fields}

    stats = {}
    
    for field, filename in file_mapping.items():
        filepath = os.path.join(gen_dir, filename)
        
        # 获取当前state中的数据
        new_items = state.get(field, [])
        if not new_items:
            continue
        
        # 转换为dict列表（Pydantic模型 -> dict）
        new_data = _to_dict_list(new_items)
        
        if append:
            # 读取已有数据（追加模式）
            existing_data = _load_json_file(filepath)
            # Provenance 用 quote 去重，其他用 id 去重
            if field == "provenance_buffer":
                merged_data = _merge_by_key(existing_data, new_data, key="quote")
            else:
                merged_data = _merge_by_id(existing_data, new_data)
        else:
            merged_data = new_data
        
        # 写入文件
        _save_json_file(filepath, merged_data)

        stats[filename] = len(new_data)
        logger.info(f"[save_to_gen] {filename}: +{len(new_data)} -> total {len(merged_data)}")

    return stats


def load_from_gen(
    gen_dir: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> Dict[str, List[dict]]:
    """
    从gen文件夹加载数据
    
    Args:
        gen_dir: 输入目录，默认使用配置中的路径
        fields: 要加载的字段列表，默认加载全部
        
    Returns:
        加载的数据 {field_name: [items]}
    """
    gen_dir = gen_dir or get_config().paths.gen_dir
    
    file_mapping = {
        "terms": "terms.json",
        "med_terms": "med_terms.json",
        "predicates": "predicates.json",
        "rules": "rules.json",
        "provenances": "provenances.json"
    }
    
    if fields:
        file_mapping = {k: v for k, v in file_mapping.items() if k in fields}
    
    result = {}
    for field, filename in file_mapping.items():
        filepath = os.path.join(gen_dir, filename)
        result[field] = _load_json_file(filepath)
    
    return result


# ============ Cluster Cache 读写 ============

def save_cluster_cache(
    cache: Dict[int, Dict[str, Any]],
    filepath: str,
) -> None:
    """
    保存按 cluster_id 切分的抽取结果缓存，供分阶段复用。

    结构示例:
    {
        "1": {
          
            "terms": [...],
            "med_terms": [...],
            "predicates": [...],
            "rules": [...]
        },
        ...
    }
    """
    serializable = {}
    for cid, entry in (cache or {}).items():
        serializable[str(cid)] = {
            
            "terms": _to_dict_list(entry.get("terms", [])),
            "med_terms": _to_dict_list(entry.get("med_terms", [])),
            "predicates": _to_dict_list(entry.get("predicates", [])),
            "rules": _to_dict_list(entry.get("rules", [])),
        }
    _save_json_file(filepath, serializable)
    logger.info(f"[save_cluster_cache] saved {len(serializable)} clusters -> {filepath}")


def load_cluster_cache(
    filepath: str,
) -> Dict[int, Dict[str, Any]]:
    """
    读取按 cluster_id 存储的缓存。返回的值仍是原始 dict/list，
    调用方可按需转换为 Pydantic 模型。
    """
    data = _load_json_file(filepath)
    if not isinstance(data, dict):
        return {}
    cache: Dict[int, Dict[str, Any]] = {}
    for cid_str, entry in data.items():
        try:
            cid = int(cid_str)
        except (TypeError, ValueError):
            continue
        cache[cid] = entry or {}
    logger.info(f"[load_cluster_cache] loaded {len(cache)} clusters from {filepath}")
    return cache


def export_state_to_json(
    state: AgentState,
    output_path: str,
    include_messages: bool = False
) -> None:
    """
    将完整的AgentState导出为单个JSON文件
    
    Args:
        state: AgentState
        output_path: 输出文件路径
        include_messages: 是否包含messages字段
    """
    export_data = {}
    
    for field in ["texts", "texts_formatted", "predicates", "terms", "med_terms", "rules"]:
        items = state.get(field, [])
        export_data[field] = _to_dict_list(items)
    
    if include_messages:
        messages = state.get("messages", [])
        export_data["messages"] = [
            {"type": type(m).__name__, "content": m.content}
            for m in messages
        ]
    
    _save_json_file(output_path, export_data)
    logger.info(f"[export] Saved state to {output_path}")


def _to_dict_list(items: List[Any]) -> List[dict]:
    """将Pydantic模型列表转换为dict列表"""
    result = []
    for item in items:
        if hasattr(item, 'model_dump'):
            result.append(item.model_dump(by_alias=True, exclude_none=True))
        elif hasattr(item, 'dict'):
            result.append(item.dict(by_alias=True, exclude_none=True))
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append(str(item))
    return result


def _load_json_file(filepath: str) -> List[dict]:
    """加载JSON文件"""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_json_file(filepath: str, data: Any) -> None:
    """保存JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _merge_by_id(existing: List[dict], new: List[dict]) -> List[dict]:
    """按id合并数据（新数据覆盖旧数据）"""
    return _merge_by_key(existing, new, key="id")


def _merge_by_key(existing: List[dict], new: List[dict], key: str = "id") -> List[dict]:
    """按指定key合并数据（新数据覆盖旧数据）"""
    existing_keys = {item.get(key) for item in existing if item.get(key)}
    merged = existing.copy()
    
    for item in new:
        item_key = item.get(key)
        if item_key and item_key in existing_keys:
            # 更新已存在的项
            for i, ex in enumerate(merged):
                if ex.get(key) == item_key:
                    merged[i] = item
                    break
        else:
            # 追加新项
            merged.append(item)
            if item_key:
                existing_keys.add(item_key)
    
    return merged


# ============ 标准库更新工具 ============

def update_standard_library(
    state: AgentState,
    standard_dir: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    将AgentState中的数据更新到标准库
    
    Args:
        state: AgentState
        standard_dir: 标准库目录
        fields: 要更新的字段列表
        
    Returns:
        更新统计信息
    """
    standard_dir = standard_dir or get_config().paths.standard_dir
    
    file_mapping = {
        "terms": "terms.json",
        "med_terms": "meds.json",
        "predicates": "predicates.json",
    }
    
    if fields:
        file_mapping = {k: v for k, v in file_mapping.items() if k in fields}
    
    stats = {}
    
    for field, filename in file_mapping.items():
        filepath = os.path.join(standard_dir, filename)
        new_items = state.get(field, [])
        
        if not new_items:
            continue
        
        new_data = _to_dict_list(new_items)
        existing_data = _load_json_file(filepath)
        merged_data = _merge_by_id(existing_data, new_data)
        
        _save_json_file(filepath, merged_data)
        
        stats[field] = len(new_data)
        logger.info(f"[update_standard] {filename}: added/updated {len(new_data)} items")

    return stats


# ============ 中间结果保存和加载 ============

def save_provenances(
    provenances: List[Provenance],
    filepath: Optional[str] = None
) -> int:
    """
    保存推荐意见到JSON文件

    Args:
        provenances: 推荐意见列表
        filepath: 保存路径，默认使用gen_dir/provenances.json

    Returns:
        保存的数量
    """
    if filepath is None:
        filepath = os.path.join(get_config().paths.gen_dir, "provenances.json")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = _to_dict_list(provenances)
    _save_json_file(filepath, data)

    logger.info(f"[save_provenances] 保存了 {len(data)} 条推荐意见到 {filepath}")
    return len(data)


def load_provenances(
    filepath: Optional[str] = None
) -> List[Provenance]:
    """
    从JSON文件加载推荐意见

    Args:
        filepath: 文件路径，默认使用gen_dir/provenances.json

    Returns:
        Provenance对象列表
    """
    if filepath is None:
        filepath = os.path.join(get_config().paths.gen_dir, "provenances.json")

    data = _load_json_file(filepath)
    provenances = []
    for item in data:
        try:
            provenance = Provenance(**item)
            provenances.append(provenance)
        except Exception as e:
            logger.warning(f"[load_provenances] 跳过无效数据: {e}")

    logger.info(f"[load_provenances] 从 {filepath} 加载了 {len(provenances)} 条推荐意见")
    return provenances


def save_clusters(
    clusters: List[ProvenanceCluster],
    bucket_index: Optional[Dict[int, List[int]]] = None,
    filepath: Optional[str] = None
) -> Dict[str, int]:
    """
    保存聚类结果到JSON文件

    Args:
        clusters: 聚类结果列表
        bucket_index: 桶索引（可选）
        filepath: 保存路径，默认使用gen_dir/clusters.json

    Returns:
        保存统计信息
    """
    if filepath is None:
        filepath = os.path.join(get_config().paths.gen_dir, "clusters.json")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    data = {
        "clusters": _to_dict_list(clusters),
        "bucket_index": bucket_index or {},
        "metadata": {
            "total_clusters": len(clusters),
            "total_provenances": sum(len(c.provenances) for c in clusters),
            "timestamp": str(Path(filepath).stat().st_mtime) if Path(filepath).exists() else None
        }
    }

    _save_json_file(filepath, data)

    stats = {
        "clusters": len(clusters),
        "provenances": sum(len(c.provenances) for c in clusters)
    }

    logger.info(f"[save_clusters] 保存了 {stats['clusters']} 个聚类，包含 {stats['provenances']} 条推荐意见到 {filepath}")
    return stats


def load_clusters(
    filepath: Optional[str] = None
) -> tuple[List[ProvenanceCluster], Dict[int, List[int]]]:
    """
    从JSON文件加载聚类结果

    Args:
        filepath: 文件路径，默认使用gen_dir/clusters.json

    Returns:
        (clusters, bucket_index) 元组
    """
    if filepath is None:
        filepath = os.path.join(get_config().paths.gen_dir, "clusters.json")

    data = _load_json_file(filepath)
    if not data:
        return [], {}

    clusters_data = data.get("clusters", [])
    bucket_index = data.get("bucket_index", {})

    clusters = []
    for item in clusters_data:
        try:
            cluster = ProvenanceCluster(**item)
            clusters.append(cluster)
        except Exception as e:
            logger.warning(f"[load_clusters] 跳过无效聚类: {e}")

    logger.info(f"[load_clusters] 从 {filepath} 加载了 {len(clusters)} 个聚类")
    return clusters, bucket_index


# ============ 失败任务记录器 ============


class FailedTaskLogger:
    """
    失败任务记录器 - 将超时/失败的任务保存到本地文件
    支持后续重新处理失败的任务

    线程安全的单例实现
    """
    _instance = None
    _lock = threading.Lock()
    _file_lock = threading.RLock()
    _log_file = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._failed_tasks = []
        self.configure()

    def configure(self, log_dir: Optional[str] = None):
        """Point the singleton at the current run's failed-task log file."""
        self._log_dir = Path(log_dir or get_config().paths.gen_dir)
        self._log_file = self._log_dir / "failed_tasks.json"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_failed_extraction(self, text_idx: int, text: str, error: str, node_output: Optional[Any] = None):
        """记录抽取阶段的失败任务"""
        task = {
            "stage": "extraction",
            "text_idx": text_idx,
            "text_preview": text[:500] + "..." if len(text) > 500 else text,
            "text_full": text,
            "error": str(error),
            "node_output": node_output,
            "timestamp": datetime.now().isoformat()
        }
        with self._file_lock:
            self._failed_tasks.append(task)
            self._save()
        logger.warning(f"  [FAILED] 任务 {text_idx} 失败已记录: {str(error)[:100]}")

    def log_failed_cluster(self, cluster_id: int, texts: List, error: str, node_output: Optional[Any] = None):
        """记录聚类处理阶段的失败任务"""
        task = {
            "stage": "cluster_processing",
            "cluster_id": cluster_id,
            "texts_count": len(texts),
            "texts_preview": [
                t.quote[:100] + "..." if hasattr(t, "quote") and len(t.quote) > 100
                else (t.quote if hasattr(t, "quote") else str(t)[:100])
                for t in texts[:3]
            ],
            "texts_full": [t.model_dump() if hasattr(t, "model_dump") else str(t) for t in texts],
            "error": str(error),
            "node_output": node_output,
            "timestamp": datetime.now().isoformat()
        }
        with self._file_lock:
            self._failed_tasks.append(task)
            self._save()
        logger.warning(f"  [FAILED] 聚类 {cluster_id} 处理失败已记录: {str(error)[:100]}")

    def log_generic_failure(self, stage: str, error: str, task_info: Dict[str, Any], node_output: Optional[Any] = None):
        """记录通用的失败任务"""
        task = {
            "stage": stage,
            "error": str(error),
            "task_info": task_info,
            "node_output": node_output,
            "timestamp": datetime.now().isoformat()
        }
        with self._file_lock:
            self._failed_tasks.append(task)
            self._save()
        logger.warning(f"  [FAILED] {stage} 失败已记录: {str(error)[:100]}")

    def _save(self):
        """保存失败任务到文件"""
        try:
            with self._file_lock:
                existing = []
                if self._log_file.exists():
                    with open(self._log_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)

                all_tasks = existing + self._failed_tasks

                with open(self._log_file, "w", encoding="utf-8") as f:
                    # 使用 default=str 防止 AIMessage 等对象序列化失败
                    json.dump(all_tasks, f, ensure_ascii=False, indent=2, default=str)

                self._failed_tasks = []
        except Exception as e:
            logger.warning(f"  [WARNING] 保存失败任务日志时出错: {e}")

    def get_failed_count(self) -> int:
        """获取失败任务数量"""
        with self._file_lock:
            if self._log_file.exists():
                with open(self._log_file, "r", encoding="utf-8") as f:
                    return len(json.load(f))
            return 0

    def clear(self):
        """清空失败任务记录"""
        with self._file_lock:
            self._failed_tasks = []
            if self._log_file.exists():
                self._log_file.unlink()

    @classmethod
    def reset(cls):
        """重置单例（用于新的 pipeline 运行），并清空当前 run 的失败日志。"""
        if cls._instance:
            cls._instance.configure()
            cls._instance.clear()


def get_failed_task_logger() -> FailedTaskLogger:
    """获取失败任务记录器单例"""
    return FailedTaskLogger()


def save_stage_results(
    result: Dict[str, List],
    gen_dir: Optional[str] = None,
    stage_name: str = ""
) -> Dict[str, int]:
    """
    保存单个阶段的抽取结果到文件

    用于每一步抽取完成后立即保存，实现增量持久化。

    Args:
        result: 抽取结果字典，包含 terms, med_terms, predicates, rules 等字段
        gen_dir: 输出目录
        stage_name: 阶段名称（用于日志）

    Returns:
        保存统计信息
    """
    from .models import AgentState

    gen_dir = gen_dir or get_config().paths.gen_dir

    # 构造 AgentState 以复用 save_to_gen 的逻辑
    agent_state = AgentState(
        messages=[],
        provenance_buffer=[],
        clusters=[],
        terms=result.get("terms", []),
        med_terms=result.get("med_terms", []),
        predicates=result.get("predicates", []),
        rules=result.get("rules", []),
    )

    # 仅保存有数据的字段
    fields_to_save = [f for f in ["terms", "med_terms", "predicates", "rules"] if result.get(f)]

    if not fields_to_save:
        logger.info(f"[save_stage_results]{stage_name} 无数据需要保存")
        return {}

    stats = save_to_gen(
        agent_state,
        gen_dir=gen_dir,
        append=True,
        save_provenances=False,
        fields=fields_to_save
    )

    if stage_name:
        logger.info(f"[save_stage_results]{stage_name} 已保存: {stats}")

    return stats
