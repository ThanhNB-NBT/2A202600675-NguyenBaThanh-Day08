"""Task 7 - Reranking with local overlap scoring and RRF."""

from .search_utils import term_overlap_score


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Offline cross-encoder substitute based on query/document overlap."""
    reranked: list[dict] = []
    for candidate in candidates:
        base_score = float(candidate.get("score", 0.0))
        relevance = term_overlap_score(query, candidate.get("content", ""))
        item = candidate.copy()
        item["score"] = float(0.35 * base_score + 0.65 * relevance)
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """MMR-compatible fallback: keep the best scoring diverse-enough items."""
    del query_embedding, lambda_param
    seen: set[str] = set()
    selected: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item.get("score", 0), reverse=True):
        key = candidate.get("content", "")[:120]
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    fused: list[dict] = []
    for content, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        item = items[content].copy()
        item["score"] = float(score)
        fused.append(item)
    return fused[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k=top_k)
    return rerank_cross_encoder(query, candidates, top_k=top_k)


if __name__ == "__main__":
    sample = [{"content": "Dieu 248 ma tuy", "score": 0.8, "metadata": {}}]
    print(rerank("hinh phat ma tuy", sample))
