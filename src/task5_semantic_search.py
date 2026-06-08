"""Task 5 - Semantic search module.

Uses a local hashing bag-of-words similarity as a deterministic dense-search
stand-in. The return shape matches a vector store result.
"""

from collections import Counter

from .search_utils import cosine_from_counters, tokenize
from .task4_chunking_indexing import chunk_documents, load_documents


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks sorted by cosine-style semantic score."""
    chunks = chunk_documents(load_documents())
    query_vec = Counter(tokenize(query))
    scored: list[dict] = []

    for chunk in chunks:
        doc_vec = Counter(tokenize(chunk["content"]))
        score = cosine_from_counters(query_vec, doc_vec)
        if score > 0:
            scored.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
