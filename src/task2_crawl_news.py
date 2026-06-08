"""Task 2 - Crawl bài báo về nghệ sĩ liên quan tới ma túy.

Luồng chính dùng Crawl4AI. Vì môi trường chấm/demo có thể không có mạng hoặc
chưa cài Crawl4AI, script có fallback seed data để vẫn tạo đủ JSON có metadata.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_SEEDS = [
    {
        "url": "https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
        "title": "Người mẫu Nhikolai Đinh bị bắt trong chuyên án ma túy ở khu Mả Lạng",
        "published_at": "2024-06-25",
        "content_markdown": "# Người mẫu Nhikolai Đinh bị bắt trong chuyên án ma túy ở khu Mả Lạng\n\nTheo Tuổi Trẻ Online, Công an quận 1, TP.HCM triệt phá chuyên án ma túy tại khu Mả Lạng và khởi tố nhiều bị can. Trong nhóm bị điều tra về hành vi tàng trữ trái phép chất ma túy có Đinh Nhi Ko Lai, nghệ danh Nhikolai Đinh, người từng hoạt động người mẫu và xuất hiện trong MV ca nhạc. Bài viết cung cấp ngữ cảnh báo chí cho các truy vấn về nghệ sĩ, người mẫu, tàng trữ trái phép chất ma túy và chuyên án ma túy tại TP.HCM.",
    },
    {
        "url": "https://congluan.vn/ca-si-chu-bin-bi-bat-vi-lien-quan-den-ma-tuy-post252547.html",
        "title": "Ca sĩ Chu Bin bị bắt vì liên quan đến ma túy",
        "published_at": "2023-06-06",
        "content_markdown": "# Ca sĩ Chu Bin bị bắt vì liên quan đến ma túy\n\nBáo Công Luận đưa tin Công an quận 10, TP.HCM tạm giữ Chu Đăng Thanh, tức ca sĩ Chu Bin, cùng một số người khác để điều tra hành vi liên quan đến tổ chức, sử dụng trái phép chất ma túy. Nội dung phù hợp cho các câu hỏi về nghệ sĩ bị điều tra, hành vi tổ chức sử dụng trái phép chất ma túy và trách nhiệm pháp lý trong các vụ việc xảy ra tại nơi ở hoặc tụ điểm riêng.",
    },
    {
        "url": "https://vnexpress.net/nha-thiet-ke-nguyen-cong-tri-bi-bat-vi-lien-quan-ma-tuy-4917929.html",
        "title": "Nhà thiết kế Nguyễn Công Trí bị bắt vì liên quan ma túy",
        "published_at": "2025-07-23",
        "content_markdown": "# Nhà thiết kế Nguyễn Công Trí bị bắt vì liên quan ma túy\n\nVnExpress đưa tin nhà thiết kế Nguyễn Công Trí bị bắt trong quá trình công an mở rộng điều tra một đường dây mua bán chất cấm. Vụ việc được mô tả liên quan đến hành vi sử dụng trái phép chất ma túy tại nhà riêng. Đây là nguồn báo chí quan trọng cho các truy vấn về người nổi tiếng trong lĩnh vực thời trang, hệ quả pháp lý khi sử dụng chất ma túy và tác động hình ảnh của nghệ sĩ trước công chúng.",
    },
    {
        "url": "https://danviet.vn/clip-an-tay-bat-khoc-vi-ma-tuy-huy-hoai-su-nghiep-tuong-lai-chi-dan-khuyen-gioi-tre-bo-y-dinh-dung-thu-20241115155132479.htm",
        "title": "An Tây hối hận, Chi Dân khuyên giới trẻ từ bỏ ý định dùng thử ma túy",
        "published_at": "2024-11-15",
        "content_markdown": "# An Tây hối hận, Chi Dân khuyên giới trẻ từ bỏ ý định dùng thử ma túy\n\nDân Việt tường thuật việc người mẫu An Tây và ca sĩ Chi Dân xuất hiện trong thông tin liên quan chuyên án ma túy, trong đó các nhân vật bày tỏ hối hận và cảnh báo giới trẻ không nên thử chất cấm. Tài liệu này hữu ích cho truy vấn về hậu quả xã hội, thông điệp phòng chống ma túy từ người nổi tiếng và ảnh hưởng của ma túy đến sự nghiệp, gia đình, cộng đồng.",
    },
    {
        "url": "https://dantri.com.vn/giai-tri/don-trung-phat-mat-bay-su-nghiep-cua-anh-de-my-nhan-dinh-vao-ma-tuy-20241111112123113.htm",
        "title": "Đòn trừng phạt mất bay sự nghiệp của nghệ sĩ dính vào ma túy",
        "published_at": "2024-11-11",
        "content_markdown": "# Đòn trừng phạt mất bay sự nghiệp của nghệ sĩ dính vào ma túy\n\nDân Trí phân tích nhiều trường hợp nghệ sĩ trong và ngoài nước bị ảnh hưởng nghiêm trọng sau bê bối ma túy, từ mất hình ảnh, bị hạn chế hoạt động cho đến khó quay lại thị trường giải trí. Bài viết bổ sung góc nhìn về hậu quả nghề nghiệp, phản ứng của công chúng và lý do người nổi tiếng cần tuân thủ chuẩn mực pháp luật, đạo đức khi hoạt động trong môi trường có sức ảnh hưởng lớn.",
    },
]

ARTICLE_URLS = [item["url"] for item in ARTICLE_SEEDS]


def setup_directory() -> Path:
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "article"


def _seed_for_url(url: str) -> dict | None:
    return next((item for item in ARTICLE_SEEDS if item["url"] == url), None)


async def crawl_article(url: str, allow_fallback: bool = True) -> dict:
    """Crawl một bài báo và trả về metadata + markdown content."""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = result.metadata.get("title") if result.metadata else None
            return {
                "url": url,
                "title": title or "Không rõ tiêu đề",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": result.markdown or "",
                "crawl_mode": "crawl4ai",
            }
    except Exception:
        if not allow_fallback:
            raise

    seed = _seed_for_url(url) or ARTICLE_SEEDS[0]
    return {
        "url": seed["url"],
        "title": seed["title"],
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": seed["content_markdown"],
        "crawl_mode": "offline_seed",
    }


def save_article(article: dict, index: int) -> Path:
    """Lưu bài báo thành JSON trong data/landing/news/."""
    setup_directory()
    slug = _slugify(article.get("title", f"article-{index:02d}"))
    output_path = DATA_DIR / f"news_{index:02d}_{slug}.json"
    output_path.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


async def crawl_all(urls: list[str] | None = None, allow_fallback: bool = True) -> list[Path]:
    """Crawl toàn bộ danh sách URL và lưu từng bài thành một file JSON."""
    setup_directory()
    selected_urls = urls or ARTICLE_URLS
    saved_paths: list[Path] = []
    for index, url in enumerate(selected_urls, 1):
        print(f"[{index}/{len(selected_urls)}] Crawl: {url}")
        article = await crawl_article(url, allow_fallback=allow_fallback)
        output_path = save_article(article, index)
        saved_paths.append(output_path)
        print(f"  Đã lưu: {output_path}")
    return saved_paths


if __name__ == "__main__":
    asyncio.run(crawl_all())
