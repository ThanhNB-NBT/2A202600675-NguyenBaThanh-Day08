"""Task 10 - Generation with citations."""

import re

from .search_utils import tokenize
from .task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = "Trả lời bằng tiếng Việt, chỉ dùng ngữ cảnh được cung cấp và luôn trích dẫn nguồn."

SMALL_TALK_TERMS = {
    "chào",
    "chao",
    "hello",
    "hi",
    "xin chào",
    "xin chao",
    "cảm ơn",
    "cam on",
    "thanks",
    "thank",
}


def _is_small_talk(query: str) -> bool:
    normalized = " ".join(tokenize(query))
    compact = normalized.strip()
    if not compact:
        return True
    if compact in SMALL_TALK_TERMS:
        return True
    return len(compact.split()) <= 2 and any(term in compact for term in SMALL_TALK_TERMS)


def _clean_evidence(text: str) -> str:
    """Remove markdown metadata/header lines before composing the answer."""
    cleaned = text.split("---")[-1] if "---" in text else text
    cleaned = re.sub(r"\*\*Source:\*\*\s+\S+", " ", cleaned)
    cleaned = re.sub(r"\*\*Crawled:\*\*\s+\S+", " ", cleaned)
    cleaned = re.sub(r"\*\*Mode:\*\*\s+\S+", " ", cleaned)
    cleaned = re.sub(
        r"^#+\s+.*?(?=(Theo|Báo|Bài|Dân Việt|VnExpress|Tuổi Trẻ|Công Luận|Một|Nội dung|Tài liệu)\b)",
        "",
        cleaned.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^#+\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text.strip()


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Keep the best chunk first and move strong secondary evidence to the end."""
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = list(reversed(chunks[1::2]))
    return front + back


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"source-{idx}")
        doc_type = metadata.get("type", "unknown")
        parts.append(
            f"[Document {idx} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    if _is_small_talk(query):
        return {
            "answer": "Chào bạn! Bạn có thể hỏi mình về pháp luật ma túy, các vụ việc nghệ sĩ liên quan đến ma túy, hoặc yêu cầu xem nguồn trích dẫn.",
            "sources": [],
            "retrieval_source": "none",
            "context": "",
        }

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not reordered:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        retrieval_source = "none"
    else:
        first = reordered[0]
        source = first.get("metadata", {}).get("source", "nguồn dữ liệu")
        evidence = _clean_evidence(first.get("content", ""))
        if len(evidence) > 420:
            evidence = evidence[:417].rstrip() + "..."
        answer = (
            f"Dựa trên nguồn đã thu thập, nội dung liên quan nhất là: "
            f"{evidence} [{source}]."
        )
        retrieval_source = first.get("source", "hybrid")

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
        "context": context,
    }


if __name__ == "__main__":
    print(generate_with_citation("Hình phạt tàng trữ ma túy?")["answer"])
