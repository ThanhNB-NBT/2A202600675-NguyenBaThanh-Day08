"""Task 8 - PageIndex Vectorless RAG.

Triển khai đúng luồng PageIndex:
1. Upload PDF bằng PageIndexClient.submit_document().
2. Lưu doc_id vào manifest.
3. Kiểm tra is_retrieval_ready().
4. Truy vấn bằng submit_query() và đọc kết quả qua get_retrieval().

Khi chưa có PAGEINDEX_API_KEY hoặc chưa cài package pageindex, module tự chuyển
sang chế độ fallback dạng cây cục bộ để demo/test không bị vỡ. Fallback này
không thay thế PageIndex thật; nó chỉ giúp app chạy được trong lớp học.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .search_utils import PROJECT_DIR, term_overlap_score

LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
PAGEINDEX_DIR = PROJECT_DIR / "data" / "pageindex"
PAGEINDEX_MANIFEST_PATH = PAGEINDEX_DIR / "pageindex_manifest.json"

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_POLL_SECONDS = float(os.getenv("PAGEINDEX_POLL_SECONDS", "5"))
PAGEINDEX_TIMEOUT_SECONDS = float(os.getenv("PAGEINDEX_TIMEOUT_SECONDS", "120"))


def _get_pageindex_client():
    """Khởi tạo PageIndexClient thật từ SDK."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Chưa cấu hình PAGEINDEX_API_KEY")
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài PageIndex SDK. Cài bằng: python -m pip install pageindex"
        ) from exc
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_manifest() -> dict:
    if not PAGEINDEX_MANIFEST_PATH.exists():
        return {"documents": []}
    return json.loads(PAGEINDEX_MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict) -> None:
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_documents(wait_until_ready: bool = False) -> dict:
    """Upload các PDF pháp luật lên PageIndex và lưu doc_id vào manifest.

    PageIndex cloud hiện ưu tiên PDF. Các bài báo markdown vẫn được dùng ở
    pipeline hybrid; Task 8 tập trung vào legal PDFs để đúng yêu cầu SDK.
    """
    client = _get_pageindex_client()
    manifest = _load_manifest()
    known_files = {item["file_path"] for item in manifest.get("documents", [])}

    uploaded = 0
    for pdf_file in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        file_path = str(pdf_file)
        if file_path in known_files:
            continue
        result = client.submit_document(file_path)
        doc_id = result["doc_id"]
        manifest.setdefault("documents", []).append(
            {
                "doc_id": doc_id,
                "file_path": file_path,
                "source": pdf_file.name,
                "status": "submitted",
            }
        )
        uploaded += 1

    if wait_until_ready:
        deadline = time.time() + PAGEINDEX_TIMEOUT_SECONDS
        for item in manifest.get("documents", []):
            while time.time() < deadline:
                if client.is_retrieval_ready(item["doc_id"]):
                    item["status"] = "ready"
                    break
                time.sleep(PAGEINDEX_POLL_SECONDS)

    _save_manifest(manifest)
    return {
        "mode": "pageindex_sdk",
        "uploaded": uploaded,
        "total_documents": len(manifest.get("documents", [])),
        "manifest_path": str(PAGEINDEX_MANIFEST_PATH),
    }


def refresh_pageindex_status() -> dict:
    """Cập nhật trạng thái và tree metadata cho các tài liệu đã upload."""
    client = _get_pageindex_client()
    manifest = _load_manifest()

    for item in manifest.get("documents", []):
        doc_id = item["doc_id"]
        if client.is_retrieval_ready(doc_id):
            item["status"] = "ready"
            try:
                tree = client.get_tree(doc_id, node_summary=True)
                item["tree_status"] = tree.get("status", "unknown")
                item["node_count"] = len(tree.get("result", []) or [])
            except Exception as exc:  # PageIndex API can be eventually consistent.
                item["tree_error"] = str(exc)
        else:
            item["status"] = "processing"

    _save_manifest(manifest)
    return manifest


def _query_one_document(client: Any, doc: dict, query: str, timeout_seconds: float) -> list[dict]:
    if not client.is_retrieval_ready(doc["doc_id"]):
        return []

    retrieval = client.submit_query(doc["doc_id"], query, thinking=False)
    retrieval_id = retrieval["retrieval_id"]
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        if result.get("status") == "completed":
            rows: list[dict] = []
            for node in result.get("retrieved_nodes", []) or []:
                title = node.get("title", "")
                node_id = node.get("node_id", "")
                contents = node.get("relevant_contents", []) or []
                if not contents and node.get("text"):
                    contents = [{"relevant_content": node["text"], "page_index": node.get("page_index")}]
                for content_item in contents:
                    text = content_item.get("relevant_content", "")
                    if not text:
                        continue
                    rows.append(
                        {
                            "content": text,
                            "score": 1.0,
                            "metadata": {
                                "source": doc.get("source", doc["doc_id"]),
                                "doc_id": doc["doc_id"],
                                "node_id": node_id,
                                "title": title,
                                "page_index": content_item.get("page_index"),
                                "retrieval_id": retrieval_id,
                                "mode": "pageindex_sdk",
                            },
                            "source": "pageindex",
                        }
                    )
            return rows
        if result.get("status") == "failed":
            return []
        time.sleep(PAGEINDEX_POLL_SECONDS)
    return []


def pageindex_search_cloud(query: str, top_k: int = 5) -> list[dict]:
    """Truy vấn PageIndex cloud bằng doc_id đã lưu trong manifest."""
    client = _get_pageindex_client()
    manifest = _load_manifest()
    documents = [
        item for item in manifest.get("documents", [])
        if item.get("doc_id") and item.get("status") in {"ready", "submitted"}
    ]
    if not documents:
        raise RuntimeError(
            "Chưa có doc_id PageIndex. Chạy upload_documents(wait_until_ready=True) trước."
        )

    results: list[dict] = []
    per_doc_timeout = max(PAGEINDEX_TIMEOUT_SECONDS / max(len(documents), 1), 10)
    for doc in documents:
        results.extend(_query_one_document(client, doc, query, per_doc_timeout))
        if len(results) >= top_k:
            break
    return results[:top_k]


def _local_tree_nodes() -> list[dict]:
    """Tạo cây cục bộ từ heading markdown để fallback vectorless."""
    nodes: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        title = md_file.stem
        section_lines: list[str] = []
        section_title = title

        for line in text.splitlines():
            if line.startswith("#"):
                if section_lines:
                    nodes.append(
                        {
                            "title": section_title,
                            "content": "\n".join(section_lines).strip(),
                            "metadata": {
                                "source": md_file.name,
                                "node_id": f"local-{len(nodes) + 1:04d}",
                                "mode": "local_tree_fallback",
                            },
                        }
                    )
                    section_lines = []
                section_title = line.lstrip("#").strip() or title
            else:
                section_lines.append(line)

        if section_lines:
            nodes.append(
                {
                    "title": section_title,
                    "content": "\n".join(section_lines).strip(),
                    "metadata": {
                        "source": md_file.name,
                        "node_id": f"local-{len(nodes) + 1:04d}",
                        "mode": "local_tree_fallback",
                    },
                }
            )
    return [node for node in nodes if node["content"]]


def pageindex_search_local(query: str, top_k: int = 5) -> list[dict]:
    """Fallback vectorless: tìm trên cây heading markdown, không dùng vector DB."""
    scored: list[dict] = []
    for node in _local_tree_nodes():
        score = term_overlap_score(query, f"{node['title']}\n{node['content']}")
        if score <= 0:
            continue
        scored.append(
            {
                "content": node["content"],
                "score": float(score),
                "metadata": node["metadata"] | {"title": node["title"]},
                "source": "pageindex",
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def pageindex_search(query: str, top_k: int = 5, use_cloud: bool | None = None) -> list[dict]:
    """Tìm kiếm bằng PageIndex thật nếu có cấu hình, ngược lại dùng fallback cây."""
    should_use_cloud = use_cloud if use_cloud is not None else bool(PAGEINDEX_API_KEY)
    if should_use_cloud:
        try:
            return pageindex_search_cloud(query, top_k=top_k)
        except Exception:
            if use_cloud is True:
                raise
    return pageindex_search_local(query, top_k=top_k)


if __name__ == "__main__":
    print(pageindex_search("ma túy", top_k=3))
