"""对外/对内统一的数据结构（Pydantic v2）。

这些模型同时承担三个角色：
1. FastAPI 的请求体 / 响应体校验；
2. 内部模块之间传递数据的契约；
3. 评测脚本读取 JSON 时的结构约定。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 文档与切分
# --------------------------------------------------------------------------- #
class SourceDocument(BaseModel):
    """从磁盘读入的一份原始文档。"""

    doc_id: str = Field(description="文档稳定 ID（基于相对路径的短哈希）")
    source: str = Field(description="相对知识库根目录的路径，作为引用来源标识")
    title: str = Field(default="", description="文档标题（取首个一级标题或文件名）")
    text: str = Field(description="纯文本正文")
    checksum: str = Field(description="内容 SHA-256，用于增量更新判断")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """切分后的知识片段，是向量库与 BM25 索引的最小单位。"""

    chunk_id: str = Field(description="全局唯一片段 ID：{doc_id}::{序号}")
    doc_id: str
    source: str
    title: str = ""
    section: str = Field(default="", description="所属章节标题路径，如 '保养 > 机油'")
    position: int = Field(default=0, description="片段在文档中的顺序号，从 0 开始")
    text: str
    char_start: int = 0
    char_end: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """带检索分数的片段。"""

    chunk_id: str
    doc_id: str
    source: str
    title: str = ""
    section: str = ""
    position: int = 0
    text: str
    score: float = Field(default=0.0, description="融合/排序后的最终分数，越大越相关")
    vector_score: Optional[float] = Field(default=None, description="向量相似度（余弦，越大越像）")
    keyword_score: Optional[float] = Field(default=None, description="BM25 分数")
    rerank_score: Optional[float] = Field(
        default=None,
        description="cross-encoder 精排分数（sigmoid 后 0~1）；只有启用精排时才有值",
    )
    heuristic_score: Optional[float] = Field(
        default=None,
        description="启用精排前的启发式分数，便于对比两种排序的差异",
    )
    fusion_score: Optional[float] = Field(
        default=None,
        description="RRF 融合分：向量与 BM25 两路的加权名次分之和。它决定谁能进精排候选池。",
    )
    signals: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "重排各信号的原始值（融合分 / 词覆盖 / 高精度串 / 中文词组 / 章节命中 / 整句命中）。"
            "用于核对「名义权重」与「实际贡献」是否一致——两者量纲不同，容易严重不符。"
        ),
    )
    from_vector: bool = False
    from_keyword: bool = False

    @property
    def citation_label(self) -> str:
        return f"{self.source}#{self.position}"


# --------------------------------------------------------------------------- #
# 问答
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """回答中的一条引用，index 与正文里的 [1] [2] 对应。"""

    index: int
    chunk_id: str
    source: str
    title: str = ""
    section: str = ""
    position: int = 0
    score: float = 0.0
    snippet: str = Field(default="", description="引用片段的前若干字符，便于人工核对")


class RetrievedDebug(BaseModel):
    """调试用的召回详情（/api/v1/retrieve 与 /api/v1/chat?debug=true）。"""

    rank: int
    chunk_id: str
    source: str
    title: str = ""
    section: str = ""
    position: int = 0
    score: float = 0.0
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = Field(
        default=None, description="精排概率（0~1），仅启用精排时有值"
    )
    heuristic_score: Optional[float] = Field(
        default=None, description="精排前的启发式分数，便于对比两种排序"
    )
    fusion_score: Optional[float] = Field(
        default=None, description="RRF 融合分，用于诊断「为什么这条没进精排候选」"
    )
    signals: Dict[str, float] = Field(
        default_factory=dict, description="重排各信号的原始值，用于权重分析与调参"
    )
    from_vector: bool = False
    from_keyword: bool = False
    snippet: str = ""
    text: str = Field(
        default="",
        description=(
            "片段完整正文，仅在调试模式（/api/v1/retrieve、chat?debug=true）返回。"
            "评测判断「关键短语是否在该片段中」必须看全文，不能只看 snippet 前 160 字。"
        ),
    )


class Latency(BaseModel):
    """耗时拆解（均为本次请求真实测量的毫秒数）。"""

    retrieve_ms: int = 0
    generate_ms: int = 0
    total_ms: int = 0


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="覆盖默认召回条数")
    retrieval_mode: Optional[str] = Field(
        default=None, description="vector / keyword / hybrid，临时覆盖配置"
    )
    debug: bool = Field(default=False, description="是否返回召回明细")
    session_id: Optional[str] = Field(default=None, description="预留：多轮会话标识")


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved: List[RetrievedDebug] = Field(default_factory=list)
    refused: bool = Field(default=False, description="是否走了无依据兜底回答")
    session_id: str = Field(default="", description="会话 ID；未启用多轮时为空串")
    rewritten_query: str = Field(
        default="",
        description=(
            "多轮时用于检索的改写后问题（指代消解后的独立问题）；单轮或未改写时为空串。"
            "它让「检索用了什么」可观测——改写错了能立刻看出来。"
        ),
    )
    used_history: bool = Field(
        default=False, description="本次回答是否使用了会话历史"
    )
    cache_hit: bool = Field(
        default=False,
        description=(
            "是否命中查询缓存。命中时 latency 是查缓存的时间（不是生成时间），"
            "usage/答案内容是首次请求的结果；索引变更后缓存自动失效。"
        ),
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description=(
            "拒答原因。可能取值：检索结果为空 / 最高召回分低于证据下限 / "
            "逐句归因得分低于阈值。未拒答时为 null。"
            "它用来判断拒答是「模型自觉」还是「服务端拦截」——前者换模型可能失效。"
        ),
    )
    retrieval_mode: str = ""
    model: str = ""
    latency: Latency = Field(default_factory=Latency)
    usage: Optional[TokenUsage] = None
    warnings: List[str] = Field(default_factory=list)
    attribution: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "逐句归因报告：统计（supported/weak/unsupported/no_citation）+ "
            "全部被校验句子的明细（status 标注各自结论，前端按需过滤/高亮）；"
            "每句的 citations 编号已解析成 sources（source / section / position）。"
            "仅在 ATTRIBUTION_ENABLED=true 时有值。"
        ),
    )


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    retrieval_mode: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    retrieval_mode: str
    latency_ms: int = 0
    results: List[RetrievedDebug] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 入库
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    rebuild: bool = Field(default=False, description="true=先清空向量库再全量重建")
    reset_registry: bool = Field(default=False, description="true=忽略增量记录，全部重切")
    paths: Optional[List[str]] = Field(
        default=None, description="只处理 data/documents 下的指定相对路径（调试用）"
    )
    async_mode: bool = Field(
        default=False,
        description=(
            "true=立即返回 job_id，后台执行，进度查 GET /api/v1/ingest/status/{job_id}；"
            "false（默认）=同步执行并返回完整结果。入库是索引级变更，"
            "同一时刻只允许一个任务在跑，冲突时返回 409。"
        ),
    )


class IngestFileResult(BaseModel):
    source: str
    status: str = Field(description="indexed / skipped / failed / deleted")
    chunks: int = 0
    error: Optional[str] = None


class IngestResponse(BaseModel):
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    deleted: int = 0
    total_chunks_in_store: int = 0
    embedding_provider: str = ""
    embedding_dim: int = 0
    duration_ms: int = 0
    files: List[IngestFileResult] = Field(default_factory=list)


class UploadResponse(BaseModel):
    filename: str
    saved_to: str
    size_bytes: int
    message: str


# --------------------------------------------------------------------------- #
# 健康与统计
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    version: str
    llm_ready: bool
    embedding_ready: bool
    vector_store_ready: bool
    collection_count: int = 0
    detail: Dict[str, Any] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    collection_count: int = 0
    indexed_documents: int = 0
    total_chunks_expected: int = 0
    bm25_docs: int = 0
    retrieval_mode: str = ""
    embedding_provider: str = ""
    embedding_dim: int = 0
    llm_model: str = ""
    documents_dir: str = ""
    documents_on_disk: int = 0
    config_warnings: List[str] = Field(default_factory=list)
    last_ingest: Optional[Dict[str, Any]] = None


class DeleteResponse(BaseModel):
    collection: str
    deleted_vectors: int
    cleared_bm25: bool
    message: str


# --------------------------------------------------------------------------- #
# 评测集
# --------------------------------------------------------------------------- #
class EvalItem(BaseModel):
    """评测集单条样本。字段全部允许为空字符串，便于先写题干后补答案。"""

    id: str
    category: str = Field(default="", description="问题分类，如 保养/保修/故障码")
    question: str
    ground_truth: str = Field(default="[待补充]", description="标准答案")
    expected_sources: List[str] = Field(
        default_factory=list, description="期望命中的来源文件（相对 data/documents 路径）"
    )
    expected_keywords: List[str] = Field(
        default_factory=list, description="答案中应出现的关键词，用于自动打分"
    )
    expected_snippets: List[str] = Field(
        default_factory=list,
        description=(
            "期望命中的**关键原文短语**（片段级相关性标注）。"
            "填写后，检索指标会判定「召回的片段是否真的含有这句话」，"
            "而不是只看来源文件对不对。留空则退回文件级判定。"
        ),
    )
    must_refuse: bool = Field(
        default=False, description="知识库无依据、期望系统明确拒答的负样本"
    )
    notes: str = ""


class EvalDataset(BaseModel):
    name: str = "auto_aftersales_eval"
    version: str = "1.0"
    description: str = ""
    items: List[EvalItem] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        default=None, description="相对项目根目录的评测集路径，默认 eval/eval_set.json"
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    retrieval_mode: Optional[str] = None


class EvalItemResult(BaseModel):
    id: str
    question: str
    answer: str = ""
    refused: bool = Field(
        default=False,
        description=(
            "有效拒答（按样本极性判定，见 metrics.effective_refusal）："
            "负样本只要声明了证据不足即为 True（含「声明缺口 + 披露相关证据」）；"
            "正样本只有纯拒答才为 True。"
        ),
    )
    hit_source: bool = Field(default=False, description="期望来源是否出现在召回中")
    keyword_coverage: float = Field(default=0.0, description="期望关键词覆盖率 0~1")
    citation_count: int = 0
    top_score: float = 0.0
    total_ms: int = 0
    error: Optional[str] = None
    # --- L1 指标：检索侧（口径见 app/core/metrics.py 模块文档） ---
    recall_at_k: Optional[float] = Field(
        default=None, description="前 k 条召回覆盖的期望来源比例；无期望来源时为 null"
    )
    mrr: float = Field(default=0.0, description="首个相关结果排名的倒数")
    ndcg_at_k: Optional[float] = Field(
        default=None, description="二值 nDCG@k，理想排序仅在召回集内计算"
    )
    precision_at_k: Optional[float] = Field(default=None, description="前 k 条里相关结果占比")
    first_relevant_rank: Optional[int] = Field(
        default=None, description="首个相关结果的 1-based 排名；无相关结果时为 null"
    )
    # --- L1 指标：生成侧 ---
    answer_token_f1: Optional[float] = Field(
        default=None, description="与 ground_truth 的 token 级 F1（词面重合，非语义正确率）"
    )
    attribution_score: Optional[float] = Field(
        default=None, description="逐句归因得分（来自 attribution 模块）；未启用时为 null"
    )
    failure_label: str = Field(default="", description="失败分型主标签，见 metrics.FAILURE_LABELS")
    refused_raw: bool = Field(
        default=False, description="API 直接返回的 refused 字段（只代表硬拒答）"
    )
    refusal_declared: bool = Field(
        default=False,
        description=(
            "是否声明了证据不足（硬拒答 or 拒答话术）。它与 refused 的区别是："
            "负样本里「声明缺口 + 披露相关证据」也算声明，但不算纯拒答。"
        ),
    )
    refusal_path: str = Field(
        default="", description="拒答路径：hard（检索为空）/ soft（纯拒答）/ partial（说明缺口后仍给出信息）/ none"
    )
    substantive: bool = Field(
        default=False, description="答案是否含有实质信息（有则不算拒答）"
    )
    relevance_level: str = Field(
        default="", description="本次检索指标的相关性粒度：snippet（含关键片段）/ file（仅来源文件）"
    )
    refusal_reason: Optional[str] = Field(
        default=None, description="服务端给出的拒答原因（检索为空 / 证据分过低 / 归因不足）"
    )


class EvalRunResponse(BaseModel):
    dataset_path: str
    total: int
    completed: int
    source_hit_rate: float = Field(description="命中期望来源的样本占比（本次运行实测）")
    avg_keyword_coverage: float = Field(description="平均关键词覆盖率（本次运行实测）")
    avg_latency_ms: float = Field(description="平均端到端耗时毫秒（本次运行实测）")
    results: List[EvalItemResult] = Field(default_factory=list)
    note: str = "以上均为本次运行的实际统计值，未做任何基准对比。"
    # --- L1 汇总指标（全部为规则统计，分母不同，不可混读） ---
    mean_recall_at_k: Optional[float] = None
    mean_mrr: Optional[float] = None
    mean_ndcg_at_k: Optional[float] = None
    mean_precision_at_k: Optional[float] = None
    mean_faithfulness: Optional[float] = Field(
        default=None, description="平均归因得分；未启用归因时为 null"
    )
    refusal_recall: Optional[float] = Field(
        default=None, description="负样本中被正确拒答的比例；无负样本时为 null"
    )
    over_refusal_rate: Optional[float] = Field(
        default=None, description="可答样本中被误拒的比例；无可答样本时为 null"
    )
    failure_distribution: Dict[str, int] = Field(
        default_factory=dict, description="失败分型分布（中文可读名 -> 条数）"
    )
    refusal_sources: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "拒答来源分布：server_empty（检索为空）/ server_guard（服务端证据或归因拦截）/ "
            "model_soft（模型自觉拒答）。换模型后重点看 model_soft 是否骤降。"
        ),
    )
