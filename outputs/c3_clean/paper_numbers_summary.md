# Báo cáo tổng hợp số liệu thực nghiệm (Paper Numbers Summary)

**Ngày cập nhật:** 2026-07-28 11:15:21  
**Trạng thái:** Hoàn tất 100% (Đã kết hợp TACO stats & A3 Ensemble)

---

## 1. Bảng Ablation (Macro-F1 per-system across seeds)

Macro-F1 được đánh giá với ngưỡng per-class tối ưu (val-tuned) và ngưỡng cố định 0.5:

| Hệ thống | Seed 42 (Opt / Fixed) | Seed 1 (Opt / Fixed) | Seed 7 (Opt / Fixed) | Mean ± Std (Opt) | Mean ± Std (Fixed) |
| **A0_Baseline-Clean** | 0.5473 / 0.5355 | 0.5512 / 0.5262 | 0.5559 / 0.5313 | **0.5515 ± 0.0043** | 0.5310 ± 0.0046 |
| **A1_ASL** | 0.5561 / 0.5194 | 0.5575 / 0.5192 | 0.5522 / 0.5270 | **0.5552 ± 0.0028** | 0.5219 ± 0.0044 |
| **A2_EmoViS** | 0.5636 / 0.5185 | 0.5586 / 0.5182 | 0.5576 / 0.5234 | **0.5599 ± 0.0032** | 0.5200 ± 0.0029 |
| **A2r_EmoViS_RDrop0.3** | 0.4995 / 0.4436 | 0.5063 / 0.4541 | 0.5028 / 0.4460 | **0.5029 ± 0.0034** | 0.4479 ± 0.0055 |
| **A2r_EmoViS_RDrop1.0** | 0.2847 / 0.1403 | 0.3175 / 0.1670 | 0.2795 / 0.1468 | **0.2939 ± 0.0206** | 0.1514 ± 0.0139 |
| **A3_EmoViS+CB** | 0.6315 / 0.6204 | 0.6225 / 0.6209 | 0.6319 / 0.6271 | **0.6286 ± 0.0053** | 0.6228 ± 0.0037 |

---

## 2. Kiểm định ý nghĩa thống kê (Paired Bootstrap Resampling, B=10.000)

| Hệ thống A | Hệ thống B | Macro-F1 (A) | Macro-F1 (B) | Delta (B - A) | p-value | p-adj (Bonferroni) | Có ý nghĩa (p < 0.05)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A0_Baseline-Clean (seed 42) | A1_ASL (seed 42) | 0.5355 | 0.5194 | -0.0155 | 0.9818 | 1.0000 | ❌ KHÔNG |
| A1_ASL (seed 42) | A2_EmoViS (seed 42) | 0.5194 | 0.5185 | -0.0009 | 0.5939 | 1.0000 | ❌ KHÔNG |
| A2_EmoViS (seed 42) | A3_EmoViS+CB (seed 42) | 0.5185 | 0.6204 | 0.1018 | 0.0000 | 0.0000 | ✅ CÓ |
| A3_EmoViS+CB (seed 42) | EmoViS-Ens (A3 Ensemble) | 0.6204 | 0.6309 | 0.0104 | 0.0000 | 0.0000 | ✅ CÓ |
| A2_EmoViS (seed 42) | A2r_EmoViS_RDrop1.0 (seed 42) | 0.5185 | 0.1403 | -0.3778 | 1.0000 | 1.0000 | ❌ KHÔNG |

---

## 3. Thống kê khoảng cách Cosine trên cụm TACO (TACO Cosine Distance Stats)

Trích xuất từ **h_text** (masked_mean_pool 768 chiều trước fusion) của checkpoint **A2-EmoViS (seed 42)** trên tập test (2.067 mẫu):

- **Số cặp mẫu dương (cùng nhãn TACO):** 825,125
- **Khoảng cách Cosine trung bình (Mean):** `0.8701`
- **Khoảng cách Cosine trung vị (Median):** `0.9103`
- **Độ lệch chuẩn (Std):** `0.2075`
- **Tỷ lệ khoảng cách > 0.3:** `98.80%`
- **Tỷ lệ khoảng cách > 0.5:** `93.66%`

---

## 4. Phân tích lỗi định tính (Qualitative Error Analysis)

- **Tổng số mẫu test có dự đoán khác biệt giữa Baseline (A0) và EmoViS-Ens (A3):** `1526` mẫu / 2.067 mẫu.
- Các mẫu khác biệt chi tiết đã được trích xuất vào file JSON `outputs/c3_clean/qualitative_error_analysis.json`.

---

## 5. Xác nhận Metadata Dataset

- **File gốc:** `data/vigoemotions/test.csv`
- **Tổng số nhãn dương tập test:** `3.943` nhãn (xác nhận lệch 1 nhãn so với con số 3.942 báo cáo trong bài báo).
