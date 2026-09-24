"""
配置管理模块 - 集中管理所有配置项
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DOTENV_LOADED = False


def _load_project_dotenv() -> None:
    """加载仓库根目录 `.env`（不覆盖已在进程环境中的变量）。"""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_LOADED = True
        return
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    _DOTENV_LOADED = True


def reset_dotenv_and_config_cache() -> None:
    """测试用：下次 `from_env` / `get_config` 会重新加载 `.env` 与默认配置单例。"""
    global _DOTENV_LOADED
    _DOTENV_LOADED = False
    global _default_config
    _default_config = None


def _parse_llm_provider(raw: Optional[str]) -> str:
    """返回 ``dashscope`` 或 ``deepseek``。"""
    if raw is None or not str(raw).strip():
        return "dashscope"
    v = str(raw).strip().lower()
    if v in ("dashscope", "aliyun", "bailian", "qwen", "dash"):
        return "dashscope"
    if v in ("deepseek", "deepseek-official", "deepseek_api"):
        return "deepseek"
    raise ValueError(
        f"无效的 LLM_PROVIDER={raw!r}；请使用 dashscope 或 deepseek（见 .env.example）。"
    )


def _resolve_deepseek_base_url(url: str) -> str:
    """DeepSeek 官方 API 根地址；未设置 ``DEEPSEEK_BASE_URL`` 时默认为 ``https://api.deepseek.com``（不追加 ``/v1``）。"""
    u = (url or "").strip().rstrip("/")
    if not u:
        return "https://api.deepseek.com"
    return u


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.05  # 略低以提高结构化抽取（术语/谓词/规则）稳定性
    max_tokens: int = 16384  # 增加输出长度限制
    timeout: float = 300.0  # 5分钟超时，DashScope 响应可能较慢
    max_retries: int = 3
    # 由 LLM_PROVIDER 解析：dashscope（百炼/灵积兼容）| deepseek（官方 API）
    provider: str = "dashscope"

    @classmethod
    def from_env(cls, *, temperature: Optional[float] = None) -> "LLMConfig":
        """从环境变量构建；未设置的字段使用本类默认值。

        先加载仓库根目录 ``.env``（若已安装 python-dotenv），再读取变量。

        **平台选择**：``LLM_PROVIDER`` = ``dashscope``（默认）或 ``deepseek``。

        - **dashscope**：``LLM_API_KEY`` 或 ``DASHSCOPE_API_KEY``，``LLM_BASE_URL``，``LLM_MODEL``。
        - **deepseek**：``DEEPSEEK_API_KEY``（或回退 ``LLM_API_KEY``），``DEEPSEEK_BASE_URL``（可选，
          默认 ``https://api.deepseek.com``，不做路径改写），
          模型名 ``DEEPSEEK_MODEL`` 或 ``LLM_MODEL``（均未设时默认 ``deepseek-v4-pro``）。

        另有 ``LLM_TEMPERATURE``（可选）。若传入 ``temperature``，则强制使用该值（评估脚本等场景）。
        """
        _load_project_dotenv()
        defaults = cls()
        provider = _parse_llm_provider(os.getenv("LLM_PROVIDER"))

        resolved_temp = defaults.temperature
        if temperature is not None:
            resolved_temp = temperature
        else:
            env_temp = os.getenv("LLM_TEMPERATURE")
            if env_temp:
                try:
                    resolved_temp = float(env_temp)
                except ValueError:
                    pass

        if provider == "deepseek":
            api_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
            base_url = _resolve_deepseek_base_url(os.getenv("DEEPSEEK_BASE_URL", ""))
            model = (
                os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL") or "deepseek-v4-pro"
            ).strip()
        else:
            api_key = (
                os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
            ).strip()
            base_url = (os.getenv("LLM_BASE_URL", defaults.base_url) or defaults.base_url).strip()
            model = (os.getenv("LLM_MODEL", defaults.model) or defaults.model).strip()

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=resolved_temp,
            max_tokens=defaults.max_tokens,
            timeout=defaults.timeout,
            max_retries=defaults.max_retries,
            provider=provider,
        )


@dataclass
class PathConfig:
    """路径配置"""
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    standard_dir: str = field(default="")
    gen_dir: str = field(default="")
    
    def __post_init__(self):
        if not self.standard_dir:
            self.standard_dir = os.path.join(self.base_dir, "standard")
        if not self.gen_dir:
            self.gen_dir = os.path.join(self.base_dir, "gen")


@dataclass
class MatchConfig:
    """匹配配置

    OMOP 匹配分数通常为 0–100。CombinedTermProcessor 中：
    - 分数 < term_threshold / med_threshold：视为无匹配，不注册 OMOP。
    - 分数 >= high_confidence_skip_review：信任匹配器，不调用 LLM 审核。
    - 介于两者之间：在 enable_review=True 时调用 LLM 审核。

    提高映射「准度」、减少误匹配：略提高 term/med_threshold。
    减少 LLM 审核次数：略降低 high_confidence_skip_review（更多分数区间直接采纳）。
    review_threshold 保留供未来扩展；当前主逻辑以 high_confidence_skip_review 为准。
    """
    term_threshold: float = 85.0
    med_threshold: float = 85.0
    predicate_fuzzy_threshold: float = 70.0
    predicate_high_confidence: float = 90.0
    # LLM 审核（仅 CombinedTermProcessor 标准化路径）
    enable_review: bool = True
    review_threshold: float = 95.0  # 预留：与 UI/报表说明用
    # 匹配分 >= 此值则跳过 LLM 审核（默认 92：比原 98 更易达到，显著减少审核调用）
    high_confidence_skip_review: float = 92.0


@dataclass
class PipelineConfig:
    """流水线总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """从环境变量创建配置（LLM 部分见 LLMConfig.from_env）。"""
        return cls(llm=LLMConfig.from_env())


# 默认配置单例
_default_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """获取默认配置（首次从环境变量加载 LLM 段，与显式 set_config 并存）。"""
    global _default_config
    if _default_config is None:
        _default_config = PipelineConfig.from_env()
    return _default_config


def set_config(config: PipelineConfig):
    """设置默认配置"""
    global _default_config
    _default_config = config

