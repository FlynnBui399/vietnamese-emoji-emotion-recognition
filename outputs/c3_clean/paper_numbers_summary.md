# Báo cáo tổng hợp số liệu thực nghiệm (Paper Numbers Summary - 4A Notebooks)

**Ngày cập nhật:** 2026-07-29 13:10:49  
**Trạng thái:** Hoàn tất 100% (Đã tổng hợp từ 4 Notebooks A0, A1, A2, A3)

---

## 1. Bảng Ablation (Macro-F1 per-system across seeds)

Macro-F1 được đánh giá với ngưỡng per-class tối ưu (val-tuned) và ngưỡng cố định 0.5:

| Hệ thống | Seed 42 (Opt / Fixed) | Seed 1 (Opt / Fixed) | Seed 7 (Opt / Fixed) | Mean ± Std (Opt) | Mean ± Std (Fixed) |
| **A0_Baseline_Clean** | 0.6133 / 0.6074 | 0.6231 / 0.6182 | 0.6155 / 0.6185 | **0.6173 ± 0.0052** | 0.6147 ± 0.0063 |
| **A1_ASL** | 0.6070 / 0.5674 | 0.6128 / 0.5796 | 0.6147 / 0.5759 | **0.6115 ± 0.0040** | 0.5743 ± 0.0062 |
| **A2_EmoViS** | 0.6165 / 0.5872 | 0.6126 / 0.5855 | 0.6122 / 0.5873 | **0.6138 ± 0.0023** | 0.5867 ± 0.0010 |
| **A3_EmoViS_CB** | 0.6190 / 0.5853 | 0.6188 / 0.5866 | 0.6224 / 0.5851 | **0.6201 ± 0.0020** | 0.5857 ± 0.0008 |

---

## 2. Kiểm định ý nghĩa thống kê (Paired Bootstrap Resampling, B=10.000)

| Hệ thống A | Hệ thống B | Macro-F1 (A) | Macro-F1 (B) | Delta (B - A) | p-value | p-adj (Bonferroni) | Có ý nghĩa (p < 0.05)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A0_Baseline_Clean (seed 42) | A1_ASL (seed 42) | 0.6074 | 0.5674 | -0.0399 | 1.0000 | 1.0000 | ❌ KHÔNG |
| A1_ASL (seed 42) | A2_EmoViS (seed 42) | 0.5674 | 0.5872 | 0.0198 | 0.0000 | 0.0000 | ✅ CÓ |
| A2_EmoViS (seed 42) | A3_EmoViS_CB (seed 42) | 0.5872 | 0.5853 | -0.0019 | 0.7567 | 1.0000 | ❌ KHÔNG |
| A3_EmoViS_CB (seed 42) | EmoViS-Ens (A3 Ensemble) | 0.5853 | 0.6105 | 0.0252 | 0.0000 | 0.0000 | ✅ CÓ |

---

## 3. Thống kê khoảng cách Cosine trên cụm TACO (TACO Cosine Distance Stats)

Trích xuất từ **h_text** (masked_mean_pool 768 chiều trước fusion) của checkpoint **A2-EmoViS (seed 42)** trên tập test (2,067 mẫu):

- **Số cặp mẫu dương (cùng nhãn TACO):** 825,125
- **Khoảng cách Cosine trung bình (Mean):** `0.8701`
- **Khoảng cách Cosine trung vị (Median):** `0.9103`
- **Độ lệch chuẩn (Std):** `0.2075`
- **Tỷ lệ khoảng cách > 0.3:** `98.80%`
- **Tỷ lệ khoảng cách > 0.5:** `93.66%`

---

## 4. Phân tích lỗi định tính (Qualitative Error Analysis)

- **Tổng số mẫu test có dự đoán khác biệt giữa Baseline (A0) và EmoViS-Ens (A3):** `1217` mẫu / 2,067 mẫu.
- Các mẫu khác biệt chi tiết đã được trích xuất vào file JSON `outputs/c3_clean/qualitative_error_analysis.json`.

---

## 5. Xác nhận Metadata Dataset

- **File gốc:** `data/vigoemotions/test.csv`
- **Tổng số nhãn dương tập test:** `3,942` nhãn.
