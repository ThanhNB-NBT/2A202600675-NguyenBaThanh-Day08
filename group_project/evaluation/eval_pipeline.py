"""
RAG Evaluation Pipeline — framework: RAGAS.

Đánh giá chất lượng RAG pipeline trên golden dataset với 4 metric:
    - Faithfulness        : câu trả lời có bám đúng context không
    - Answer Relevancy    : câu trả lời có đúng câu hỏi không
    - Context Recall       : retriever có lấy đủ evidence (so với reference) không
    - Context Precision    : context lấy về có bao nhiêu % hữu ích

So sánh A/B 2 config:
    - Config A: hybrid (semantic + BM25) + RRF + MMR rerank
    - Config B: dense-only (chỉ semantic), không rerank

Chạy:
    python -m group_project.evaluation.eval_pipeline

Yêu cầu: OPENAI_API_KEY trong .env (RAGAS dùng LLM + embeddings của OpenAI để chấm).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Shim: ragas 0.4.3 import langchain_community.chat_models.vertexai (đã bị gỡ
# ở langchain-community 0.4.x). Ta dùng OpenAI nên tạo module giả để qua import. ---
for _modname, _attrs in [
    ("langchain_community.chat_models.vertexai", ["ChatVertexAI"]),
    ("langchain_community.llms.vertexai", ["VertexAI"]),
]:
    try:
        __import__(_modname)
    except ModuleNotFoundError:
        _m = types.ModuleType(_modname)
        for _a in _attrs:
            setattr(_m, _a, type(_a, (), {}))
        sys.modules[_modname] = _m

# Cho phép chạy như script độc lập (thêm project root vào path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Cấu hình A/B
CONFIGS = {
    "A_hybrid_rerank": {
        "label": "Hybrid (semantic+BM25) + RRF + MMR rerank",
        "retrieve_kwargs": {"mode": "hybrid", "use_reranking": True, "use_pageindex": False},
    },
    "B_dense_only": {
        "label": "Dense-only (chỉ semantic), không rerank",
        "retrieve_kwargs": {"mode": "dense", "use_reranking": False, "use_pageindex": False},
    },
}

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def load_golden_dataset() -> list[dict]:
    import json

    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


# =============================================================================
# Chạy RAG pipeline để thu thập (question, answer, contexts, reference)
# =============================================================================

def build_samples(golden_dataset: list[dict], retrieve_kwargs: dict) -> list[dict]:
    """Chạy generate_with_citation cho từng câu hỏi → tạo sample cho RAGAS."""
    from src.task10_generation import generate_with_citation

    samples = []
    for i, item in enumerate(golden_dataset, 1):
        q = item["question"]
        print(f"    [{i}/{len(golden_dataset)}] {q[:60]}...")
        result = generate_with_citation(q, top_k=5, retrieve_kwargs=retrieve_kwargs)
        contexts = [c["content"] for c in result.get("sources", [])] or ["(no context)"]
        samples.append({
            "user_input": q,
            "response": result["answer"],
            "retrieved_contexts": contexts,
            "reference": item["expected_answer"],
        })
    return samples


# =============================================================================
# RAGAS evaluation
# =============================================================================

def _ragas_components():
    """Tạo metrics + LLM judge + embeddings (OpenAI) cho RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
    )

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]
    return metrics, llm, emb


def evaluate_with_ragas(samples: list[dict]) -> "tuple[dict, object]":
    """
    Evaluate samples bằng RAGAS. Trả về (điểm trung bình theo metric, dataframe chi tiết).
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    metrics, llm, emb = _ragas_components()
    dataset = EvaluationDataset.from_list(samples)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=4, timeout=180),
        show_progress=True,
    )

    df = result.to_pandas()
    # map tên cột RAGAS → tên metric chuẩn
    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "answer_relevancy(mode=...)": "answer_relevancy",
        "context_recall": "context_recall",
        "llm_context_recall": "context_recall",
        "context_precision": "context_precision",
        "llm_context_precision_with_reference": "context_precision",
    }
    scores: dict[str, float] = {}
    for col in df.columns:
        key = col_map.get(col)
        if key and key not in scores:
            try:
                scores[key] = float(df[col].mean(skipna=True))
            except Exception:
                pass
    return scores, df


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """Chạy eval cho từng config trong CONFIGS, trả về dict kết quả."""
    out: dict[str, dict] = {}
    for name, cfg in CONFIGS.items():
        print(f"\n=== Config {name}: {cfg['label']} ===")
        print("  → Sinh câu trả lời...")
        samples = build_samples(golden_dataset, cfg["retrieve_kwargs"])
        print("  → Chấm điểm bằng RAGAS...")
        scores, df = evaluate_with_ragas(samples)
        out[name] = {"label": cfg["label"], "scores": scores, "df": df, "samples": samples}
        print(f"  ✓ {name} scores: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    return out


# =============================================================================
# Export Results
# =============================================================================

def _worst_performers(config_result: dict, n: int = 3) -> list[dict]:
    """Tìm n câu hỏi điểm trung bình thấp nhất ở config A."""
    df = config_result["df"]
    metric_cols = [c for c in df.columns
                   if c in ("faithfulness", "answer_relevancy", "context_recall",
                            "context_precision", "llm_context_recall",
                            "llm_context_precision_with_reference")]
    rows = []
    for idx, row in df.iterrows():
        vals = [row[c] for c in metric_cols if row[c] == row[c]]  # bỏ NaN
        avg = sum(vals) / len(vals) if vals else 0.0
        rows.append({
            "question": row.get("user_input", ""),
            "avg": avg,
            "metrics": {c: row[c] for c in metric_cols},
        })
    rows.sort(key=lambda r: r["avg"])
    return rows[:n]


def export_results(comparison: dict):
    """Xuất bảng điểm + phân tích ra results.md."""
    names = list(comparison.keys())
    a, b = names[0], names[1]
    sa, sb = comparison[a]["scores"], comparison[b]["scores"]

    lines = ["# RAG Evaluation Results", ""]
    lines += ["## Framework sử dụng", "",
              "> **RAGAS** (`ragas==0.4.3`) — LLM judge: OpenAI `gpt-4o-mini`, "
              "embeddings: `text-embedding-3-small`. Golden dataset: "
              f"{len(comparison[a]['samples'])} cặp Q&A.", ""]

    lines += ["---", "", "## Overall Scores", "",
              f"| Metric | Config A ({comparison[a]['label']}) | "
              f"Config B ({comparison[b]['label']}) | Δ (A−B) |",
              "|--------|---|---|---|"]
    all_metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    a_vals, b_vals = [], []
    for m in all_metrics:
        va, vb = sa.get(m), sb.get(m)
        if va is not None:
            a_vals.append(va)
        if vb is not None:
            b_vals.append(vb)
        va_s = f"{va:.3f}" if va is not None else "—"
        vb_s = f"{vb:.3f}" if vb is not None else "—"
        d_s = f"{va - vb:+.3f}" if (va is not None and vb is not None) else "—"
        lines.append(f"| {METRIC_LABELS[m]} | {va_s} | {vb_s} | {d_s} |")
    avg_a = sum(a_vals) / len(a_vals) if a_vals else 0.0
    avg_b = sum(b_vals) / len(b_vals) if b_vals else 0.0
    lines.append(f"| **Average** | **{avg_a:.3f}** | **{avg_b:.3f}** | "
                 f"**{avg_a - avg_b:+.3f}** |")

    lines += ["", "---", "", "## A/B Comparison Analysis", "",
              f"**Config A:** {comparison[a]['label']}.",
              f"**Config B:** {comparison[b]['label']}.", "",
              "**Kết luận:**",
              f"> Config {'A' if avg_a >= avg_b else 'B'} tốt hơn "
              f"(điểm TB {max(avg_a, avg_b):.3f} so với {min(avg_a, avg_b):.3f}). "
              "Hybrid retrieval kết hợp BM25 giúp bắt được các truy vấn theo từ khóa "
              "(số điều luật, tên riêng) mà dense thuần dễ bỏ sót; reranking MMR "
              "tăng độ liên quan và giảm trùng lặp context."]

    # Worst performers (theo config A)
    lines += ["", "---", "", "## Worst Performers (Bottom 3 — Config A)", "",
              "| # | Question | Avg score | Phân tích |", "|---|----------|-----------|-----------|"]
    for i, w in enumerate(_worst_performers(comparison[a]), 1):
        q = str(w["question"]).replace("|", "/")[:70]
        lines.append(f"| {i} | {q}… | {w['avg']:.3f} | "
                     f"retrieval/generation cần cải thiện |")

    lines += ["", "---", "", "## Recommendations", "",
              "### Cải tiến 1 — Chunking theo cấu trúc điều luật",
              "**Action:** Dùng MarkdownHeaderTextSplitter / tách theo 'Điều N' thay vì "
              "ký tự thuần, để mỗi chunk gói trọn 1 điều luật.  ",
              "**Expected impact:** tăng Context Precision & Recall cho câu hỏi pháp lý.", "",
              "### Cải tiến 2 — Bật PageIndex fallback cho câu hỏi suy luận cấu trúc",
              "**Action:** Hạ ngưỡng kích hoạt fallback hoặc route câu hỏi pháp lý khó "
              "sang PageIndex (vectorless).  ",
              "**Expected impact:** tăng Faithfulness cho câu hỏi mà hybrid bỏ sót.", "",
              "### Cải tiến 3 — Bổ sung tài liệu nguồn",
              "**Action:** Thêm bản đầy đủ Bộ luật Hình sự 2015 (Điều 247–252) vào corpus.  ",
              "**Expected impact:** tăng Context Recall cho câu hỏi về khung hình phạt cụ thể."]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Đã ghi kết quả → {RESULTS_PATH}")


if __name__ == "__main__":
    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} test cases từ golden_dataset.json")
    comparison = compare_configs(golden)
    export_results(comparison)
    print("✓ Done.")
