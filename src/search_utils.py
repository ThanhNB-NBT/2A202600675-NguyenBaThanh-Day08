"""Small offline helpers shared by the Day 8 RAG tasks.

The graded tests run without external services, so these helpers keep the
pipeline deterministic while preserving the same public interfaces.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def _repair_mojibake(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    """Tokenize text and normalize Vietnamese accents plus common mojibake."""
    variants = {text.lower(), _repair_mojibake(text).lower()}
    normalized = " ".join(variants | {_strip_accents(value) for value in variants})
    return re.findall(r"[\wÀ-ỹ]+", normalized, flags=re.UNICODE)


def load_markdown_documents(base_dir: Path = STANDARDIZED_DIR) -> list[dict]:
    documents: list[dict] = []
    if not base_dir.exists():
        return documents

    for md_file in sorted(base_dir.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(base_dir)
        doc_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(rel_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )
    return documents


def cosine_from_counters(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[t] * right[t] for t in shared)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def term_overlap_score(query: str, text: str) -> float:
    query_terms = Counter(tokenize(query))
    text_terms = Counter(tokenize(text))
    cosine = cosine_from_counters(query_terms, text_terms)
    matched = sum(1 for term in query_terms if term in text_terms)
    coverage = matched / max(len(query_terms), 1)
    return 0.75 * cosine + 0.25 * coverage
