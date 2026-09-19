"""学习用对比实验：BM25 在「精确」与「语义」两类查询上的失败模式。

设计思路：
1. 把内置语料 + 12 条评测查询当作基线，得到 BM25 召回画像
2. 新增一个扩展查询集，专门针对 BM25 失效场景（同义词、改写、跨语言、长文档）
3. 提供 query 改写（simple query expansion）这一对照实现，
   看改写后的查询能不能把 BM25 召回率拉回来

运行：
    python study_experiment.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evaluate import (
    DEFAULT_CORPUS,
    DEFAULT_QUERIES,
    BM25Retriever,
    chunk_text,
    chunks_to_docs,
    tokenize,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# 扩展查询集：专门设计来「打脸」纯 BM25
# 同一 expected 但换成不同表述，看 BM25 还能不能命中
# ---------------------------------------------------------------------------
EXPANDED_QUERIES: List[Dict[str, Any]] = [
    # 精确代码簇：BM25 应该仍然全胜（验证 tokenize 对连字符处理正确）
    {"query": "XR-7003 智能手机", "expected": ["xr_7003"], "category": "code+lang"},
    {"query": "http 状态码 401", "expected": ["http_401"], "category": "code+lang"},
    # 精确人名：把专有名词换成全小写或错位，BM25 仍应命中
    {"query": "ALEXANDER HUMPHREY", "expected": ["name_alexander_humphrey"], "category": "name-case"},
    {"query": "humphrey alexander", "expected": ["name_alexander_humphrey"], "category": "name-order"},
    # 语义改写簇：BM25 几乎必然失败（用来演示「为什么需要稠密 / 上下文感知」）
    {"query": "which phone model number is XR-7003", "expected": ["xr_7003"], "category": "paraphrase"},
    {"query": "the api status code meaning 403", "expected": ["http_403"], "category": "paraphrase"},
    {"query": "garbage collector in programming", "expected": ["sem_gc"], "category": "paraphrase"},
    {"query": "plant converting sunlight to food", "expected": ["sem_photo"], "category": "paraphrase"},
    # 跨语言：英文文档 vs 中文查询（BM25 无重叠词）
    {"query": "HTTP 错误 403 是什么意思", "expected": ["http_403"], "category": "cross-lang"},
    {"query": "什么是自动垃圾回收", "expected": ["sem_gc"], "category": "cross-lang"},
    # 长文档语义
    {"query": "rain returning to ocean loop", "expected": ["doc_watercycle"], "category": "paraphrase"},
]


# ---------------------------------------------------------------------------
# 简单的 query 改写：用一份手工的同义词 / 术语扩展词典，把查询扩成多版本，
# 最后做「任一版本命中即视为命中」的 union 召回。
# ---------------------------------------------------------------------------
SYNONYM_LEXICON: Dict[str, List[str]] = {
    "smartphone": ["phone", "mobile"],
    "phone": ["smartphone", "mobile"],
    "model": ["型号"],
    "智能手机": ["手机", "移动设备"],
    "手机": ["智能手机"],
    "garbage": ["回收", "清理"],
    "collector": ["收集器"],
    "回收": ["garbage"],
    "自动": ["automatic", "auto"],
    "memory": ["内存", "heap"],
    "programmer": ["developer", "工程师"],
    "vegetation": ["植物", "plants"],
    "plant": ["植物"],
    "sunlight": ["阳光", "光合作用"],
    "food": ["食物", "营养"],
    "encrypt": ["加密", "scramble"],
    "scramble": ["加密", "encrypt"],
    "eavesdropper": ["窃听者"],
    "water": ["水", "雨水"],
    "rain": ["雨", "降水"],
    "ocean": ["海洋", "大海"],
    "molten": ["熔融", "lava"],
    "rock": ["岩石", "岩浆"],
}


def expand_query_tokens(query: str) -> List[str]:
    """为每个 token 加同义词 token，BM25 用「原文 + 同义词」一起匹配。"""
    tokens = tokenize(query)
    expanded: List[str] = []
    seen = set()
    for tok in tokens:
        for t in [tok] + SYNONYM_LEXICON.get(tok, []):
            if t not in seen:
                seen.add(t)
                expanded.append(t)
    return expanded


class ExpandedBM25Retriever(BM25Retriever):
    """BM25Retriever 的小补丁：把查询侧的 token 替换为「原文 + 同义词」扩展。"""

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        tokens = expand_query_tokens(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda kv: kv[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# 跑实验
# ---------------------------------------------------------------------------
def build_pipeline(corpus, args) -> Tuple[Any, List[str], Dict[str, str]]:
    """构造 chunk 列表 + BM25 索引，复用 evaluate.py::chunk_text。"""
    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
    chunk_to_doc: Dict[str, str] = {}
    doc_text: Dict[str, str] = {}
    for doc in corpus:
        doc_text[doc["doc_id"]] = doc["text"]
        chunks = chunk_text(doc["text"], args["chunk_size"], args["chunk_overlap"])
        for i, chunk in enumerate(chunks):
            cid = f"{doc['doc_id']}::c{i}" if len(chunks) > 1 else doc["doc_id"]
            chunk_ids.append(cid)
            chunk_texts.append(chunk)
            chunk_to_doc[cid] = doc["doc_id"]
    bm25 = BM25Retriever(chunk_ids, chunk_texts)
    return bm25, chunk_ids, chunk_to_doc


def run_method(retriever, queries: List[Dict[str, Any]], chunk_to_doc, k: int) -> Dict[str, float]:
    per_method: List[Tuple[List[str], Sequence[str]]] = []
    per_query_records: List[Dict[str, Any]] = []
    for spec in queries:
        ranked_chunks = retriever.search(spec["query"], top_k=10)
        ranked_docs = chunks_to_docs(ranked_chunks, chunk_to_doc)
        ranked_ids = [doc_id for doc_id, _ in ranked_docs]
        gold = spec.get("expected", [])
        per_method.append((ranked_ids, gold))
        per_query_records.append({
            "query": spec["query"],
            "expected": gold,
            "category": spec.get("category", "unspecified"),
            "top1": ranked_ids[0] if ranked_ids else None,
            "mrr": round(reciprocal_rank(ranked_ids, gold), 4),
            "recall@k": round(recall_at_k(ranked_ids, gold, k), 4),
        })
    summary = aggregate_metrics(per_method, k)
    return {"summary": summary, "per_query": per_query_records}


def print_comparison_table(reports: Dict[str, Dict[str, float]], k: int, queries_label: str) -> None:
    print()
    print("=" * 84)
    print(f"{queries_label}  ·  Recall@{k} / MRR / nDCG@{k}")
    print("=" * 84)
    header = f"{'Method':<28}{'Recall@'+str(k):>12}{'MRR':>12}{'nDCG@'+str(k):>12}"
    print(header)
    print("-" * 84)
    for name, r in reports.items():
        m = r["summary"]
        print(f"{name:<28}{m['recall@k']:>12.4f}{m['mrr']:>12.4f}{m['ndcg@k']:>12.4f}")
    print("=" * 84)


def print_per_query(reports: Dict[str, Dict[str, Any]], methods_order: List[str]) -> None:
    print()
    print("逐条查询对比（MRR 列：1.00=第 1 位命中，0.50=第 2 位命中，0.00=未进 top-3）")
    print("-" * 84)
    short = {m: m.replace("BM25 (default)", "BM25").replace("BM25 + query expansion", "BM25+QE")
             for m in methods_order}
    header = f"{'Query':<44}" + "".join(f"{short[m]:>16}" for m in methods_order)
    print(header)
    print("-" * 84)
    # 假设每份报告 per_query 顺序一致
    for idx in range(len(reports[methods_order[0]]["per_query"])):
        rec0 = reports[methods_order[0]]["per_query"][idx]
        cells = ""
        for m in methods_order:
            v = reports[m]["per_query"][idx]
            cells += f"{v['mrr']:>16.2f}"
        q = rec0["query"]
        q = q if len(q) <= 43 else q[:40] + "..."
        print(f"{q:<44}{cells}")
    print("=" * 84)


def main() -> int:
    args = {"chunk_size": 280, "chunk_overlap": 40}

    # 把扩展查询塞进默认语料一起索引
    bm25_default, chunk_ids, chunk_to_doc = build_pipeline(DEFAULT_CORPUS, args)
    bm25_expanded, _, _ = build_pipeline(DEFAULT_CORPUS, args)

    # 用一个 hack：扩展版查询时换检索器
    class _SwitchRetriever:
        def __init__(self, real): self.real = real
        def search(self, query, top_k):
            # 如果 BM25 索引是 default 类就走扩展；否则默认
            if isinstance(self.real, ExpandedBM25Retriever):
                return self.real.search(query, top_k)
            return self.real.search(query, top_k)

    # 构造两个独立的 retriever
    expanded = ExpandedBM25Retriever.__new__(ExpandedBM25Retriever)
    expanded.chunk_ids = chunk_ids
    expanded.tokenized = bm25_default.tokenized
    expanded.bm25 = bm25_default.bm25  # 共用索引，只是查询时扩展

    # ---- 第 1 轮：内置 12 条查询 ----
    reports_default = {
        "BM25 (default)": run_method(bm25_default, DEFAULT_QUERIES, chunk_to_doc, k=3),
    }
    reports_expanded = {
        "BM25 + query expansion": run_method(expanded, DEFAULT_QUERIES, chunk_to_doc, k=3),
    }
    print_comparison_table(reports_default, k=3, queries_label="内置 12 条查询")
    print_comparison_table(reports_expanded, k=3, queries_label="内置 12 条查询（query 扩展后）")
    print_per_query({"BM25 (default)": reports_default["BM25 (default)"],
                     "BM25 + query expansion": reports_expanded["BM25 + query expansion"]},
                    methods_order=["BM25 (default)", "BM25 + query expansion"])

    # ---- 第 2 轮：扩展 11 条「打脸」查询 ----
    reports_default2 = {
        "BM25 (default)": run_method(bm25_default, EXPANDED_QUERIES, chunk_to_doc, k=3),
    }
    reports_expanded2 = {
        "BM25 + query expansion": run_method(expanded, EXPANDED_QUERIES, chunk_to_doc, k=3),
    }
    print_comparison_table(reports_default2, k=3, queries_label="扩展 11 条「BM25 杀手」查询")
    print_comparison_table(reports_expanded2, k=3, queries_label="扩展 11 条「BM25 杀手」查询（query 扩展后）")
    print_per_query({"BM25 (default)": reports_default2["BM25 (default)"],
                     "BM25 + query expansion": reports_expanded2["BM25 + query expansion"]},
                    methods_order=["BM25 (default)", "BM25 + query expansion"])

    # ---- 把全部结果写到 JSON，方便后续画图 / 报告 ----
    out_path = Path(__file__).parent / "study_results.json"
    payload = {
        "default_queries": {
            "bm25_default": reports_default["BM25 (default)"],
            "bm25_expanded": reports_expanded["BM25 + query expansion"],
        },
        "expanded_queries": {
            "bm25_default": reports_default2["BM25 (default)"],
            "bm25_expanded": reports_expanded2["BM25 + query expansion"],
        },
        "lexicon_size": len(SYNONYM_LEXICON),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 全部明细写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())