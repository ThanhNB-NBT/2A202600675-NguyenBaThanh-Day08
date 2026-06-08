"""Task 4 - Chunking and lightweight offline indexing."""

import hashlib
import json
from pathlib import Path

from .search_utils import load_markdown_documents, tokenize

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "drug_law_index.json"

# Recursive character chunking is robust for mixed legal/news markdown.
# 500 chars keeps citations focused; 50 chars preserves context across splits.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Deterministic local embedding fallback. In a production demo this can be
# replaced with BAAI/bge-m3 or OpenAI embeddings without changing callers.
EMBEDDING_MODEL = "hashing-bow-offline"
EMBEDDING_DIM = 256
VECTOR_STORE = "local_json"


def load_documents() -> list[dict]:
    """Read all markdown files from data/standardized/."""
    return load_markdown_documents(STANDARDIZED_DIR)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into overlapping character chunks."""
    chunks: list[dict] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for doc in documents:
        text = " ".join(doc.get("content", "").split())
        if not text:
            continue
        for start in range(0, len(text), step):
            chunk_text = text[start:start + CHUNK_SIZE].strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": len(chunks),
                    },
                }
            )
            if start + CHUNK_SIZE >= len(text):
                break
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach normalized hashing bag-of-words vectors to chunks."""
    for chunk in chunks:
        vector = [0.0] * EMBEDDING_DIM
        for token in tokenize(chunk.get("content", "")):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % EMBEDDING_DIM] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        chunk["embedding"] = [v / norm for v in vector]
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Persist chunks and vectors to a local JSON vector store."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "store": VECTOR_STORE,
        "path": str(VECTOR_STORE_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "count": len(chunks),
        "chunks": chunks,
    }
    VECTOR_STORE_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index


def load_vectorstore() -> dict:
    """Load the local vector store, building it first if needed."""
    if not VECTOR_STORE_PATH.exists():
        chunks = embed_chunks(chunk_documents(load_documents()))
        return index_to_vectorstore(chunks)
    return json.loads(VECTOR_STORE_PATH.read_text(encoding="utf-8"))


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} -> {VECTOR_STORE_PATH}")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = embed_chunks(chunk_documents(docs))
    print(f"Created and embedded {len(chunks)} chunks")
    index = index_to_vectorstore(chunks)
    print(f"Indexed to local vector store: {index['path']}")


if __name__ == "__main__":
    run_pipeline()
