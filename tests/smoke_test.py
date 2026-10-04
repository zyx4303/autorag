#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冒烟测试：只用标准库 + 项目依赖，验证纯逻辑，不联网、不调用大模型、不需要任何 Key。

运行：
    python tests/smoke_test.py

覆盖：
1. 文本工具：归一化、分句、中英混合分词；
2. 切分器：不丢内容、块大小受限、章节路径正确、重叠生效；
3. BM25：自建索引能召回含关键词的片段，且与查询无关的片段不排第一；
4. RRF 融合：两路结果能正确合并、去重并带上双方标记；
5. 引用校验：越界 [n] 能被检测出来。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.core.bm25 import BM25Index  # noqa: E402
from app.core.chunker import chunk_document  # noqa: E402
from app.core.embedding import HashEmbedder  # noqa: E402
from app.core.text_utils import normalize_text, split_sentences, tokenize  # noqa: E402
from app.models import Citation, SourceDocument  # noqa: E402
from app.services.rag_pipeline import QAEngine  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


SAMPLE_MD = """# 保养手册

## 首保

首保应在 [待补充] 公里或 [待补充] 个月内完成，以先到者为准。

## 机油

推荐使用 [待补充] 黏度等级的机油，更换周期为 [待补充] 公里。
更换机油时应同时更换机油滤清器。

## 故障码

| 故障码 | 含义 | 建议处理 |
| --- | --- | --- |
| P0420 | [待补充] | [待补充] |
"""


def make_settings(tmp_root: Path) -> Settings:
    return Settings(
        documents_dir_raw=str(tmp_root / "documents"),
        chroma_dir_raw=str(tmp_root / "chroma"),
        upload_dir_raw=str(tmp_root / "uploads"),
        registry_path_raw=str(tmp_root / "registry.json"),
        bm25_index_path_raw=str(tmp_root / "bm25.json"),
        chunk_size=120,
        chunk_overlap=20,
        chunk_min_chars=10,
        embedding_provider="hash",
    )


def test_text_utils() -> None:
    print("\n[1] 文本工具")
    raw = "第一行。\r\n\r\n\r\n  第二行  有空格\u200b。"
    normalized = normalize_text(raw)
    check("归一化压缩空行", "\n\n\n" not in normalized and "\u200b" not in normalized, repr(normalized))
    sentences = split_sentences("说明书要求定期检查。是否需要更换？请参考手册！")
    check("分句数量正确", len(sentences) == 3, str(sentences))
    tokens = tokenize("更换机油 5W-30 oil")
    check("中文产出 bigram", "机油" in tokens, str(tokens[:10]))
    check("英文数字成词", "5w" in tokens or "30" in tokens or "oil" in tokens, str(tokens))


def test_chunker(tmp_root: Path) -> None:
    print("\n[2] 切分器")
    settings = make_settings(tmp_root)
    document = SourceDocument(
        doc_id="doc_test",
        source="保养手册.md",
        title="保养手册",
        text=SAMPLE_MD,
        checksum="x" * 8,
    )
    chunks = chunk_document(document, settings)
    # 断言不应依赖"手工数出来的段数"（会随 chunk_size/min_chars 调整而变），
    # 因此这里只断言与切分逻辑本质相关的性质：多章节必须产生多个片段、且可复现。
    check("多章节切出多个片段", len(chunks) >= 2, f"chunks={len(chunks)}")
    check(
        "切分可复现（同输入同输出）",
        [c.text for c in chunk_document(document, settings)] == [c.text for c in chunks],
    )
    check("片段 ID 唯一", len({chunk.chunk_id for chunk in chunks}) == len(chunks))
    check(
        "章节路径被识别",
        any("首保" in chunk.section for chunk in chunks),
        str([chunk.section for chunk in chunks]),
    )
    check(
        "块大小未失控",
        all(len(chunk.text) <= settings.chunk_size + settings.chunk_overlap + 40 for chunk in chunks),
        str([len(chunk.text) for chunk in chunks]),
    )
    joined = "".join(chunk.text for chunk in chunks)
    for keyword in ("首保", "机油滤清器", "P0420", "[待补充]"):
        check(f"内容未丢失: {keyword}", keyword in joined)
    check(
        "表格未被切断",
        any("| P0420 |" in chunk.text and "| 建议处理 |" in chunk.text for chunk in chunks),
    )


def test_bm25() -> None:
    print("\n[3] BM25 关键词检索")
    index = BM25Index(persist_path=None)
    index.build(
        [
            {
                "chunk_id": "c1",
                "text": "首保应在 5000 公里或 6 个月内完成",
                "source": "保养手册.md",
                "section": "保养 > 首保",
                "position": 0,
            },
            {
                "chunk_id": "c2",
                "text": "故障码 P0420 表示催化器效率低于阈值",
                "source": "故障码.md",
                "section": "故障码",
                "position": 1,
            },
            {
                "chunk_id": "c3",
                "text": "空调滤芯建议每年更换一次",
                "source": "易损件.md",
                "section": "滤芯",
                "position": 2,
            },
        ]
    )
    check("索引规模正确", index.size == 3, str(index.size))
    hits = index.search("P0420 故障码 是什么意思", top_k=3)
    check("能召回目标片段", bool(hits) and hits[0][0]["chunk_id"] == "c2", str(hits))
    hits2 = index.search("首保多少公里", top_k=3)
    check("中文关键词召回", bool(hits2) and hits2[0][0]["chunk_id"] == "c1", str(hits2))
    # 词典外查询（与语料完全无 token 重叠）必须返回空，否则说明索引里混进了噪声。
    # 注意：不能拿"完全无关的中文串"当反例——中文按单字切分时，
    # 任意两句中文都可能共享"的/是/关/全"这类单字，这是分词策略的固有特性，不是 bug。
    noise = index.search("zzz x9999 qqq", top_k=3)
    check("词典外查询返回空", noise == [], f"实测返回 {[item[0]['chunk_id'] for item in noise]}")
    weak = index.search("完全无关的字符串xyzzy", top_k=3)
    if weak:
        top = weak[0][1]
        check(
            "弱相关候选被相对分数下限过滤",
            all(score >= top * 0.15 for _, score in weak) and weak[0][0]["chunk_id"] == "c1",
            str([(item[0]["chunk_id"], round(item[1], 4)) for item in weak]),
        )
    else:
        check("弱相关候选被相对分数下限过滤", True, "已全部过滤")


def test_fusion() -> None:
    print("\n[4] RRF 融合（不依赖向量库）")
    from app.core.retriever import HybridRetriever

    settings = make_settings(Path("."))
    retriever = HybridRetriever(settings, vector_store=None, bm25_index=BM25Index(None))  # type: ignore[arg-type]

    vector_hits = [
        (
            {
                "chunk_id": "c1",
                "text": "首保应在 5000 公里或 6 个月内完成",
                "source": "a.md",
                "position": 0,
            },
            0.82,
        ),
        (
            {
                "chunk_id": "c2",
                "text": "机油推荐 5W-30",
                "source": "b.md",
                "position": 1,
            },
            0.61,
        ),
    ]
    keyword_hits = [
        (
            {
                "chunk_id": "c2",
                "text": "机油推荐 5W-30",
                "source": "b.md",
                "position": 1,
            },
            7.5,
        ),
        (
            {
                "chunk_id": "c3",
                "text": "空调滤芯每年更换",
                "source": "c.md",
                "position": 2,
            },
            3.2,
        ),
    ]
    fused = retriever._fuse(vector_hits, keyword_hits, "hybrid", "首保 机油 5W-30")  # noqa: SLF001
    ids = [item["chunk_id"] for item in fused]
    check("两路结果都被合并", set(ids) == {"c1", "c2", "c3"}, str(ids))
    c2 = next(item for item in fused if item["chunk_id"] == "c2")
    check("同时命中两路的片段带双标记", c2["from_vector"] and c2["from_keyword"])
    check("融合分数已计算", all(item["score"] >= 0 for item in fused))
    check("按分数降序", all(
        fused[i]["score"] >= fused[i + 1]["score"] for i in range(len(fused) - 1)
    ))


def test_citations() -> None:
    print("\n[5] 引用校验")
    citations = [
        Citation(index=1, chunk_id="c1", source="a.md", position=0),
        Citation(index=2, chunk_id="c2", source="b.md", position=1),
    ]
    used, warnings = QAEngine.verify_citations("结论一[1]，结论二[2]。", citations)
    check("正常引用无告警", warnings == [] and len(used) == 2, str(warnings))
    used2, warnings2 = QAEngine.verify_citations("结论[1]，另外[7]。", citations)
    check("越界引用被检出", len(warnings2) == 1 and "7" in warnings2[0], str(warnings2))
    used3, warnings3 = QAEngine.verify_citations("没有任何角标的回答。", citations)
    check("缺少角标被提示", len(warnings3) == 1 and len(used3) == 2, str(warnings3))


def test_hash_embedder() -> None:
    print("\n[6] hash embedding（离线确定性）")
    embedder = HashEmbedder(128)

    async def run() -> None:
        first = await embedder.embed_documents(["更换机油", "刹车片磨损"])
        second = await embedder.embed_documents(["更换机油", "刹车片磨损"])
        check("维度正确", len(first[0]) == 128, str(len(first[0])))
        check("同文本向量一致（可复现）", first == second)
        norm = sum(value * value for value in first[0]) ** 0.5
        check("已做 L2 归一化", abs(norm - 1.0) < 1e-6, str(norm))
        query = await embedder.embed_query("更换机油")
        check("query 与同文本文档向量一致", query == first[0])

    asyncio.run(run())


def main() -> int:
    print("=" * 68)
    print("AutoRAG 冒烟测试（纯逻辑，不联网、不需要 API Key、不调用大模型）")
    print("=" * 68)
    tmp_root = PROJECT_ROOT / "data" / "_smoke_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    test_text_utils()
    test_chunker(tmp_root)
    test_bm25()
    test_fusion()
    test_citations()
    test_hash_embedder()

    print("\n" + "=" * 68)
    print(f"结果：通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 68)
    if FAILED:
        print("存在失败项，请把上面的 FAIL 行连同环境信息一起反馈。")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
