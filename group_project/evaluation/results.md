# RAG Evaluation Results

## Framework sử dụng

> **RAGAS** (`ragas==0.4.3`) — LLM judge: OpenAI `gpt-4o-mini`, embeddings: `text-embedding-3-small`. Golden dataset: 16 cặp Q&A.

---

## Overall Scores

| Metric | Config A (Hybrid (semantic+BM25) + RRF + MMR rerank) | Config B (Dense-only (chỉ semantic), không rerank) | Δ (A−B) |
|--------|---|---|---|
| Faithfulness | 0.708 | 0.460 | +0.249 |
| Answer Relevancy | 0.425 | 0.260 | +0.164 |
| Context Recall | 0.729 | 0.500 | +0.229 |
| Context Precision | 0.567 | 0.499 | +0.068 |
| **Average** | **0.607** | **0.430** | **+0.178** |

---

## A/B Comparison Analysis

**Config A:** Hybrid (semantic+BM25) + RRF + MMR rerank.
**Config B:** Dense-only (chỉ semantic), không rerank.

**Kết luận:**
> Config A tốt hơn (điểm TB 0.607 so với 0.430). Hybrid retrieval kết hợp BM25 giúp bắt được các truy vấn theo từ khóa (số điều luật, tên riêng) mà dense thuần dễ bỏ sót; reranking MMR tăng độ liên quan và giảm trùng lặp context.

---

## Worst Performers (Bottom 3 — Config A)

| # | Question | Avg score | Phân tích |
|---|----------|-----------|-----------|
| 1 | Danh mục II của Nghị định 28/2026/NĐ-CP quy định về nhóm chất nào?… | 0.050 | retrieval/generation cần cải thiện |
| 2 | Tội tàng trữ trái phép chất ma túy được quy định tại điều nào của Bộ l… | 0.125 | retrieval/generation cần cải thiện |
| 3 | Thông tư liên tịch 17/2015 quy định về vấn đề gì?… | 0.319 | retrieval/generation cần cải thiện |

---

## Recommendations

### Cải tiến 1 — Chunking theo cấu trúc điều luật
**Action:** Dùng MarkdownHeaderTextSplitter / tách theo 'Điều N' thay vì ký tự thuần, để mỗi chunk gói trọn 1 điều luật.  
**Expected impact:** tăng Context Precision & Recall cho câu hỏi pháp lý.

### Cải tiến 2 — Bật PageIndex fallback cho câu hỏi suy luận cấu trúc
**Action:** Hạ ngưỡng kích hoạt fallback hoặc route câu hỏi pháp lý khó sang PageIndex (vectorless).  
**Expected impact:** tăng Faithfulness cho câu hỏi mà hybrid bỏ sót.

### Cải tiến 3 — Bổ sung tài liệu nguồn
**Action:** Thêm bản đầy đủ Bộ luật Hình sự 2015 (Điều 247–252) vào corpus.  
**Expected impact:** tăng Context Recall cho câu hỏi về khung hình phạt cụ thể.

---

## Bonus Demo — 4 câu hỏi làm LLM không trả lời được

Kiểm thử thủ công trên pipeline RAG (Config A: hybrid + rerank, `top_k=5`, không PageIndex).  
Mục tiêu bonus: mỗi câu khiến hệ thống **không trả lời đúng** hoặc **từ chối xác minh** (5 điểm/câu).

| # | Câu hỏi | Kết quả | Đánh giá | Root cause |
|---|---------|---------|---------|------------|
| 1 | Luật Phòng chống ma túy 2021 quy định những hình thức cai nghiện nào? | *"Tôi không thể xác minh thông tin này từ nguồn hiện có."* | **Fail rõ** | Corpus không có Luật 73/2021/QH14; retrieval kéo NĐ 105 + Thông tư 17 (quản lý người nghiện, không liệt kê hình thức cai nghiện). |
| 2 | Tội tàng trữ trái phép chất ma túy được quy định tại điều nào của Bộ luật Hình sự? | *"Điều 249... Tuy nhiên, tôi không thể xác minh thông tin này từ nguồn hiện có."* | **Fail mơ hồ** | LLM đoán đúng số điều nhưng context không chứa Điều 249; top source là NĐ 105 (Điều 28/29), không phải BLHS. |
| 3 | Danh mục III trong Nghị định 28/2026/NĐ-CP khác Danh mục II ở điểm nào? | *"Tôi không thể xác minh thông tin này từ nguồn hiện có."* | **Fail rõ** | Retrieval lấy Điều 3 "Hiệu lực thi hành", không phải mô tả Danh mục II/III; các danh mục có embedding gần giống nhau. |
| 4 | So sánh khung hình phạt tội vận chuyển trái phép (Điều 250) và tội mua bán trái phép (Điều 251) — trường hợp nhẹ nhất mỗi tội bao nhiêu năm tù? | *"Tôi không thể xác minh thông tin này từ nguồn hiện có."* | **Fail rõ** | Cần 2 điều luật cùng lúc; chunk Điều 250/251 không lọt top-k hoặc bị cắt rời trong file BLHS sửa đổi. |

**Kết luận bonus:** 4/4 câu đều không trả lời đúng đầy đủ. Câu 1 và 3–4 fail do **thiếu/sai retrieval**; câu 2 cho thấy LLM có thể **đoán đúng** nhưng vẫn từ chối vì prompt yêu cầu bám context — minh họa giới hạn RAG khi chunking chưa đủ tốt.

**Gợi ý demo:** Hỏi 1 câu dễ trước (vd *"An Tây có tên thật là gì?"* → trả lời đúng), sau đó hỏi 1 trong 4 câu trên → mở Sources chỉ chunk sai → giải thích fail ở retrieval/chunking, không phải LLM.