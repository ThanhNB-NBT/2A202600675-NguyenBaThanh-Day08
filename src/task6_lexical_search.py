"""Task 6 - Lexical search with a small BM25 implementation."""

import math
from collections import Counter

from .search_utils import tokenize
from .task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []


class SimpleBM25:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(doc["content"]) for doc in corpus]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        doc_freq: Counter = Counter()
        for tokens in self.tokens:
            doc_freq.update(set(tokens))
        n_docs = max(len(corpus), 1)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for idx, counts in enumerate(self.term_counts):
            doc_len = self.lengths[idx] or 1
            score = 0.0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    return SimpleBM25(corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks sorted by BM25 score."""
    corpus = _load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results: list[dict] = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(score),
                "metadata": corpus[idx].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
