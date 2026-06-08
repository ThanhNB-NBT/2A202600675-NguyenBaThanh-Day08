"""Task 3 - Convert toàn bộ file trong data/landing/ thành Markdown."""

from __future__ import annotations

import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_markitdown():
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None
    return MarkItDown()


def _write_markdown(output_path: Path, content: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")
    return output_path


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = _get_markitdown()
    converted: list[Path] = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        output_path = output_dir / f"{filepath.stem}.md"
        print(f"Converting legal: {filepath.name}")

        content = convert_office_or_pdf(filepath, converter)

        converted.append(_write_markdown(output_path, content))
        print(f"  Đã lưu: {output_path}")
    return converted


def convert_office_or_pdf(filepath: Path, converter=None) -> str:
    """Convert PDF/DOCX bằng MarkItDown nếu có, nếu không dùng parser cục bộ."""
    if converter is not None:
        try:
            result = converter.convert(str(filepath))
            text_content = getattr(result, "text_content", "") or ""
            if len(text_content.strip()) > 200:
                return _format_legal_markdown(filepath, text_content)
        except Exception:
            pass

    if filepath.suffix.lower() == ".pdf":
        return _format_legal_markdown(filepath, extract_pdf_text(filepath))
    if filepath.suffix.lower() == ".docx":
        return _format_legal_markdown(filepath, extract_docx_text(filepath))

    return _format_legal_markdown(filepath, "")


def extract_pdf_text(filepath: Path) -> str:
    """Trích xuất text PDF bằng pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Chưa cài pypdf. Cài bằng: python -m pip install pypdf") from exc

    reader = PdfReader(str(filepath))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"## Trang {index}\n\n{text}")
    return "\n\n".join(pages)


def extract_docx_text(filepath: Path) -> str:
    """Trích xuất text DOCX tối thiểu bằng zip/XML chuẩn Office Open XML."""
    with zipfile.ZipFile(filepath) as archive:
        xml_content = archive.read("word/document.xml")
    root = ET.fromstring(xml_content)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _format_legal_markdown(filepath: Path, text: str) -> str:
    """Đóng gói text legal thành markdown có metadata, không chèn lời hướng dẫn."""
    content = text.strip()
    if not content:
        content = "Không trích xuất được nội dung văn bản."
    return f"""# {filepath.stem}

**Nguồn gốc:** `{filepath.name}`
**Định dạng gốc:** `{filepath.suffix.lower()}`

---

{content}
"""


def convert_news_articles() -> list[Path]:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting news: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"
        header = (
            f"# {data.get('title', 'Không rõ tiêu đề')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
            f"**Mode:** {data.get('crawl_mode', 'unknown')}\n\n"
            "---\n\n"
        )
        content = header + data.get("content_markdown", "")
        converted.append(_write_markdown(output_path, content))
        print(f"  Đã lưu: {output_path}")
    return converted


def convert_all() -> dict:
    """Convert toàn bộ landing zone sang markdown."""
    print("=" * 50)
    print("Task 3 - Convert sang Markdown")
    print("=" * 50)

    legal_outputs = convert_legal_docs()
    news_outputs = convert_news_articles()
    result = {
        "legal_count": len(legal_outputs),
        "news_count": len(news_outputs),
        "output_dir": str(OUTPUT_DIR),
        "files": [str(path) for path in legal_outputs + news_outputs],
    }
    print(f"Hoàn tất. Output tại: {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    convert_all()
