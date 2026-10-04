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
    from_vector: bool = False
    from_keyword: bool = False
    snippet: str = ""


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
    retrieval_mode: str = ""
    model: str = ""
    latency: Latency = Field(default_factory=Latency)
    usage: Optional[TokenUsage] = None
    warnings: List[str] = Field(default_factory=list)


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
    refused: bool = False
    hit_source: bool = Field(default=False, description="期望来源是否出现在召回中")
    keyword_coverage: float = Field(default=0.0, description="期望关键词覆盖率 0~1")
    citation_count: int = 0
    top_score: float = 0.0
    total_ms: int = 0
    error: Optional[str] = None


class EvalRunResponse(BaseModel):
    dataset_path: str
    total: int
    completed: int
    source_hit_rate: float = Field(description="命中期望来源的样本占比（本次运行实测）")
    avg_keyword_coverage: float = Field(description="平均关键词覆盖率（本次运行实测）")
    avg_latency_ms: float = Field(description="平均端到端耗时毫秒（本次运行实测）")
    results: List[EvalItemResult] = Field(default_factory=list)
    note: str = "以上均为本次运行的实际统计值，未做任何基准对比。"
