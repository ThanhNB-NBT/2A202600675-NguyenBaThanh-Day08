"""Task 1 - Thu thập văn bản pháp luật về ma túy và các chất cấm.

Mục tiêu:
1. Có tối thiểu 3 văn bản PDF/DOC/DOCX trong data/landing/legal/.
2. File không rỗng và có tên rõ ràng.
3. Có script kiểm kê để demo được dữ liệu đã thu thập.

Trong repo này các PDF đã được đặt sẵn trong data/landing/legal/. Nếu có link
tải trực tiếp, có thể dùng hàm download_file() để bổ sung tài liệu mới.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}

SUGGESTED_DOCUMENTS = [
    "Luật Phòng, chống ma túy 2021 (Luật số 73/2021/QH15)",
    "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
    "Bộ luật Hình sự 2015, sửa đổi 2017 - các tội phạm về ma túy",
    "Thông tư liên tịch 17/2015 về xác định tình trạng nghiện ma túy",
]


def setup_directory() -> Path:
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _filename_from_url(url: str, fallback_name: str | None = None) -> str:
    path_name = Path(urlparse(url).path).name
    return fallback_name or path_name or "legal-document.pdf"


def download_file(url: str, filename: str | None = None, timeout: int = 60) -> Path:
    """Tải một văn bản pháp luật từ direct link về data/landing/legal/."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài requests. Cài bằng: python -m pip install requests"
        ) from exc

    setup_directory()
    output_name = _filename_from_url(url, filename)
    output_path = DATA_DIR / output_name

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def collect_existing_files() -> list[dict]:
    """Kiểm kê các file legal hiện có trong landing zone."""
    setup_directory()
    files: list[dict] = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        files.append(
            {
                "filename": path.name,
                "path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "is_valid": path.stat().st_size > 1024,
            }
        )
    return files


def validate_collection(min_files: int = 3) -> dict:
    """Trả về trạng thái đạt/chưa đạt yêu cầu Task 1."""
    files = collect_existing_files()
    valid_files = [item for item in files if item["is_valid"]]
    return {
        "directory": str(DATA_DIR),
        "required_min_files": min_files,
        "file_count": len(files),
        "valid_file_count": len(valid_files),
        "passed": len(valid_files) >= min_files,
        "files": files,
        "suggested_documents": SUGGESTED_DOCUMENTS,
    }


def main() -> dict:
    result = validate_collection()
    print("Task 1 - Thu thập văn bản pháp luật")
    print(f"Thư mục: {result['directory']}")
    print(f"Số file hợp lệ: {result['valid_file_count']}/{result['required_min_files']}")
    for item in result["files"]:
        status = "OK" if item["is_valid"] else "quá nhỏ"
        print(f"- {item['filename']} ({item['size_bytes']} bytes) [{status}]")
    return result


if __name__ == "__main__":
    main()
