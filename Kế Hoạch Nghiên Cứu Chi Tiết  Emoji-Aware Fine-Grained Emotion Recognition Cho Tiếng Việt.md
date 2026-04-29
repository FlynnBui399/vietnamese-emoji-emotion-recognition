# Kế Hoạch Nghiên Cứu Chi Tiết: Emoji-Aware Fine-Grained Emotion Recognition Cho Tiếng Việt

> **Tên đề tài gợi ý:** *Enhancing Vietnamese Fine-Grained Emotion Recognition via Emoji-Aware Representation Learning on Social Media Text*

***

## 1. Tổng Quan Đề Tài

### Bài toán (Problem Statement)

Nhận dạng cảm xúc chi tiết (*fine-grained emotion recognition*) trên mạng xã hội tiếng Việt đang đối mặt với hai thách thức lớn đồng thời:

1. **Emoji bị bỏ qua hoặc xử lý sai:** Phần lớn mô hình hiện tại hoặc xóa bỏ emoji hoặc chuyển sang mô tả văn bản — cả hai đều làm mất tín hiệu cảm xúc quan trọng. Thực nghiệm trên ViGoEmotions (2026) cho thấy việc *giữ nguyên emoji* cho kết quả tốt nhất với ViSoBERT và CafeBERT, nhưng chưa có nghiên cứu nào khai thác *biểu diễn vector của emoji* (như Emoji2Vec) để tăng cường đặc trưng.[^1][^2][^3]

2. **Class imbalance trong 27 nhãn cảm xúc:** Bộ dữ liệu ViGoEmotions có 27 cảm xúc với phân phối lệch nặng (*long-tailed*). Phương pháp hiện tại chỉ dùng `pos_weight` đơn giản trong Binary Cross-Entropy. Các kỹ thuật contrastive learning tiên tiến (như TACO framework, ACL 2025) giải quyết cả hai vấn đề *confusable emotions* lẫn *long-tailed emotions* cùng lúc nhưng chưa được áp dụng cho tiếng Việt.[^4][^5][^6]

### Lý do lựa chọn

- **Dataset có sẵn và mới nhất:** ViGoEmotions (2026) được publish tại EACL 2026, UIT-VSMEC là corpus gốc — đều public, không cần thu thập thêm[^7][^8]
- **SOTA baseline thấp:** ViSoBERT đạt MF1 = 61.50% trên ViGoEmotions — còn nhiều dư địa cải thiện[^2]
- **Research gap rõ ràng:** Paper ViGoEmotions tự đề xuất hướng: *"Future work could explore richer emoji embeddings, context-aware normalization, and multimodal approaches"*[^4]
- **Nhóm VKU chưa làm:** Nhóm VKU chỉ làm 3-class sentiment với Emoji trên PhoBERT, chưa đụng đến fine-grained emotion (27 lớp) hay ViGoEmotions[^3]

***

## 2. Mục Tiêu Nghiên Cứu

### Mục tiêu tổng quát
Đề xuất và thực nghiệm một kiến trúc **Emoji-Aware** kết hợp **Contrastive Learning** cho bài toán nhận dạng cảm xúc chi tiết 27 nhãn trên văn bản mạng xã hội tiếng Việt, vượt qua baseline hiện tại trên ViGoEmotions.

### Mục tiêu cụ thể

| # | Mục tiêu | Đầu ra tương ứng |
|---|---------|-----------------|
| 1 | Tái hiện kết quả baseline (ViSoBERT, CafeBERT, PhoBERT) trên ViGoEmotions | Bảng so sánh baseline |
| 2 | Tích hợp Emoji2Vec / EmojiSAGE vào backbone ViSoBERT/CafeBERT | Mô hình EmoBERT-Vi đề xuất |
| 3 | Giải quyết class imbalance và confusable emotions | Cải thiện MF1 ≥ 3–5% so với baseline |
| 4 | Phân tích lỗi hệ thống (error analysis) | Insight về emotion pairs dễ nhầm lẫn |
| 5 | Viết và nộp bài báo | 1 paper conference (VLSP/PACLIC) |

***

## 3. Input / Output / Giải Pháp

### Input
Văn bản mạng xã hội tiếng Việt chứa hỗn hợp: chữ thường, teencode (kiểu chữ GenZ), emoji, dấu câu không chuẩn.

**Ví dụ:**
```
"bài hát này hay thật sự 🥺❤️ nghe mà muốn khóc luôn ấy"
"ăn cơm chưa mà than nghèo 😂😂 buồn cười vl"
```

### Output
Vector xác suất cho **28 nhãn** (27 cảm xúc + neutral), dạng multi-label — một câu có thể có nhiều cảm xúc đồng thời.[^4]

**Ví dụ output:**
```
{"love": 0.82, "sadness": 0.61, "admiration": 0.34, ...}
```

### Kiến Trúc Giải Pháp Đề Xuất

Mô hình đề xuất gồm 3 thành phần chính:

#### Thành phần 1 — Emoji-Enriched Encoder

Thay vì chỉ giữ emoji thô (Scenario 1 của ViGoEmotions), đề xuất **hai nhánh song song**:

- **Nhánh text:** Sử dụng ViSoBERT (hoặc CafeBERT) để encode toàn bộ chuỗi văn bản (kể cả emoji token)
- **Nhánh emoji:** Trích xuất emoji từ chuỗi → ánh xạ sang **Emoji2Vec** (300-dim) hoặc **EmojiSAGE** vectors → encode chuỗi emoji riêng

**Fusion strategy:** Intermediate fusion (được chứng minh tốt nhất trong nghiên cứu 2025):[^9]
\[h_{fused} = W_1 \cdot h_{text} + W_2 \cdot h_{emoji} + b\]

Hoặc dùng cross-attention giữa `h_text` và `h_emoji` để model học trọng số tự động.

#### Thành phần 2 — Contrastive Learning cho Confusable Emotions

Áp dụng ý tưởng từ TACO framework (ACL 2025) vào tiếng Việt:[^6][^10]

- **Clustering-guided Contrastive Loss (CCL):** Nhóm các cảm xúc tương tự (e.g., anger/annoyance/disgust) vào cùng cluster, sau đó contrastive loss kéo gần intra-cluster và đẩy xa inter-cluster
- **Label Description Loss (LDL):** Sử dụng định nghĩa ngắn của từng nhãn cảm xúc bằng tiếng Việt để refine label embeddings, đặc biệt giúp các cảm xúc ít xuất hiện (long-tailed) được biểu diễn tốt hơn

Hàm loss tổng hợp:
\[\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda_1 \cdot \mathcal{L}_{CCL} + \lambda_2 \cdot \mathcal{L}_{LDL}\]

#### Thành phần 3 — LoRA Fine-tuning (Tùy chọn nếu thiếu GPU)

Nếu nhóm chỉ có GPU Colab Pro (VRAM ~16GB), thay vì full fine-tuning toàn bộ ViSoBERT (XLM-R base, 270M params), sử dụng **LoRA** (rank=8 hoặc 16):
- Giảm trainable parameters ~10-20 lần
- Đã được kiểm chứng đạt hiệu suất tương đương full fine-tuning trên tiếng Việt[^11][^12]
- Cho phép thực nghiệm nhanh hơn → nhiều ablation study hơn

***

## 4. Novelty và Đóng Góp Học Thuật

### So sánh với các công trình liên quan

| Công trình | Dataset | Backbone | Emoji Xử Lý | Class Imbalance | Multi-label |
|---|---|---|---|---|---|
| VKU 2025[^3] | UIT-VSFC, AIVIVN | PhoBERT | Emoji2Vec (3-class) | Không | ❌ |
| ViGoEmotions 2026[^2] | ViGoEmotions | ViSoBERT | Giữ nguyên/convert | pos_weight | ✅ |
| **Đề tài này** | ViGoEmotions | ViSoBERT/CafeBERT | **Emoji2Vec fusion** | **Contrastive + pos_weight** | ✅ |

### Đóng góp chính (Contributions)

1. **C1 — Emoji-Aware Representation:** Lần đầu tiên tích hợp Emoji2Vec (hoặc learned emoji embeddings) vào bài toán fine-grained emotion recognition tiếng Việt 27 nhãn — mở rộng từ 3-class sang multi-label, không chỉ là text preprocessing đơn giản

2. **C2 — Adapted Contrastive Learning:** Lần đầu tiên áp dụng clustering-guided contrastive learning (từ TACO, ACL 2025) cho tiếng Việt — giải quyết đồng thời confusable emotions và long-tailed class imbalance[^6]

3. **C3 — Ablation Study toàn diện:** So sánh có hệ thống các fusion strategies (early/intermediate/late), các emoji encoding methods, và fine-tuning strategies (full vs. LoRA) trên ViGoEmotions — đây là benchmark analysis chưa có ai làm[^11]

4. **C4 — Error Analysis:** Phân tích cặp cảm xúc dễ nhầm lẫn trong tiếng Việt (joy↔amusement, sadness↔disappointment, anger↔annoyance) kèm linguistic insight — có giá trị độc lập với cộng đồng nghiên cứu[^4]

***

## 5. Related Works (Tài Liệu Liên Quan Phải Review)

### 5.1 Vietnamese Emotion/Sentiment Models

| Paper | Venue | Key Contribution | Liên quan |
|---|---|---|---|
| ViSoBERT (Nguyen et al., 2023)[^13][^14] | EMNLP 2023 | Pre-trained model cho social media Việt, xử lý emoji qua tokenizer | **Backbone chính** |
| CafeBERT (Do et al., 2024)[^15][^16] | NAACL 2024 Findings | XLM-R enhanced với Vietnamese corpus, SOTA trên VLUE | **Backbone thay thế** |
| ViGoEmotions (Pham et al., 2026)[^2][^4] | EACL 2026 | Dataset 27 nhãn, benchmark 8 models | **Dataset + Baseline chính** |
| UIT-VSMEC (Ho et al., 2019)[^7][^17] | PACLING 2019 | Dataset 6 nhãn cảm xúc tiếng Việt | **Dataset phụ / prior work** |
| VKU Emoji (2025)[^3] | Scopus | VED_PhoBERT, E2V_PhoBERT cho 3-class SA | **Directly related, phải cite** |

### 5.2 Emoji trong NLP

| Paper | Venue | Key Contribution | Liên quan |
|---|---|---|---|
| Emoji2Vec (Eisner et al., 2016) | EMNLP Workshop | Pre-trained 300-dim emoji embeddings | **Technique chính** |
| Emoji Hate Speech (ACL 2024)[^18][^19] | ICON 2024 | Emoji2Vec + mBERT cho hate speech detection | **Kỹ thuật tham khảo** |
| Emoji-Aware SA (Nature 2025)[^20] | Scientific Reports | BERT + emoji fusion cho airline tweets, +9% accuracy | **Benchmark quốc tế** |
| Sentiment Emoji Review (2025)[^9] | Int'l Journal | Survey fusion strategies 2015–2025, intermediate fusion tốt nhất | **Survey để cite** |

### 5.3 Fine-Grained Emotion & Contrastive Learning

| Paper | Venue | Key Contribution | Liên quan |
|---|---|---|---|
| TACO (Gong et al., 2025)[^6][^10] | ACL 2025 | Triple-View + Clustering Contrastive Loss cho FEC | **Kỹ thuật contrastive chính** |
| GoEmotions (Demszky et al., 2020) | ACL 2020 | 27-label emotion taxonomy (English, Google) | **Taxonomy gốc của ViGoEmotions** |
| Multi-label Emotion (SemEval 2025)[^21] | SemEval | Ensemble transformer cho multi-label, macro F1=69.18% | **Benchmark quốc tế so sánh** |
| LoRA Vietnamese (HUFLIT 2025)[^11] | HJS | LoRA vs. full fine-tuning trên Vietnamese SA | **Justification cho LoRA** |

***

## 6. Kế Hoạch Thực Hiện Chi Tiết

### Giai đoạn 1 — Literature Review & Setup (Tuần 1–3)

| Tuần | Nhiệm vụ | Người thực hiện | Output |
|------|----------|-----------------|--------|
| 1 | Đọc 10 paper core (ViGoEmotions, ViSoBERT, CafeBERT, TACO, VKU, Emoji2Vec, LoRA) | Cả nhóm | Summary notes |
| 2 | Cài đặt môi trường: HuggingFace, PyTorch, VnCoreNLP, Colab Pro | 1 người | Environment ready |
| 2 | Download ViGoEmotions dataset + EDA (phân tích phân phối nhãn) | 1 người | EDA report |
| 3 | Tái hiện baseline ViSoBERT trên ViGoEmotions (Scenario 1) | Cả nhóm | Reproduce results |
| 3 | Xác nhận kết quả reproduce ≈ MF1 61.50%[^2] | Cả nhóm | Baseline confirmed |

### Giai đoạn 2 — Phát Triển Mô Hình (Tuần 4–9)

| Tuần | Nhiệm vụ | Chi tiết kỹ thuật |
|------|----------|-------------------|
| 4 | Xây dựng Emoji Extractor module | Regex + unicodedata để tách emoji; map sang Emoji2Vec 300-dim |
| 5 | Xây dựng Dual-Encoder + Intermediate Fusion | Concat `[CLS]_text` với mean-pooled emoji vectors; FC layer |
| 6 | Ablation Study 1: Emoji encoding strategies | So sánh: no emoji / keep raw / Emoji2Vec / convert-to-text |
| 7 | Implement Contrastive Loss | CCL: cosine similarity + margin loss; LDL: label description embeddings |
| 8 | Ablation Study 2: Loss function components | BCE only / BCE+CCL / BCE+LDL / BCE+CCL+LDL |
| 9 | LoRA experiments (nếu cần) | rank ∈ {4, 8, 16}; compare với full fine-tuning |

### Giai đoạn 3 — Phân Tích & Viết Paper (Tuần 10–14)

| Tuần | Nhiệm vụ | Output |
|------|----------|--------|
| 10 | Error analysis chi tiết: confusion matrix 27×27, case studies | Error analysis section |
| 11 | So sánh với tất cả baseline trong ViGoEmotions paper[^2] | Results table |
| 12 | Viết paper draft: Introduction + Related Work + Method | Draft v1 |
| 13 | Viết Experiments + Results + Discussion + Conclusion | Draft v2 |
| 14 | Proofread, format theo template venue, nộp bài | Submission-ready paper |

***

## 7. Yêu Cầu Tài Nguyên

### Phần cứng

| Tài nguyên | Yêu cầu tối thiểu | Gợi ý |
|------------|------------------|-------|
| GPU | 1× T4 16GB (Colab Pro) | Đủ nếu dùng LoRA[^11] |
| GPU (ideal) | 1× A100 40GB (Colab Pro+) | Full fine-tuning nhanh hơn |
| RAM | ≥ 16GB system RAM | Cần cho VnCoreNLP |
| Storage | ~5GB | Dataset + model checkpoints |

### Phần mềm & Thư viện

```
Python 3.10+
PyTorch 2.x
transformers (HuggingFace) >= 4.40
peft (LoRA implementation)
vnCoreNLP (word segmentation)
emoji (Python library - emoji extraction)
scikit-learn (metrics: macro F1, confusion matrix)
wandb (experiment tracking)
```

### Chi phí ước tính

| Khoản | Chi phí |
|-------|---------|
| Colab Pro (3 tháng) | ~$30 USD (~750k VND) |
| OpenAI API (nếu cần GPT cho ablation) | ~$10–20 USD |
| **Tổng** | **~$50 USD** |

***

## 8. Metrics Đánh Giá

| Metric | Mô tả | Mục tiêu |
|--------|-------|---------|
| **Macro F1 (MF1)** | Metric chính — tính đều trọng số cho tất cả 27 nhãn | Vượt 61.50% của ViSoBERT[^2] |
| **Weighted F1 (WF1)** | Tính theo tần suất xuất hiện của nhãn | Vượt 63.26%[^2] |
| **Per-class F1** | F1 cho từng nhãn cảm xúc | Cải thiện trên long-tailed classes |
| **Hamming Loss** | Độ lỗi trung bình trên multi-label | Càng thấp càng tốt |

***

## 9. Venue Submit Gợi Ý

| Venue | Deadline tham khảo | Tỷ lệ phù hợp |
|-------|-------------------|---------------|
| **VLSP 2026 Workshop** | Tháng 9–10/2026 | ⭐⭐⭐⭐⭐ Tốt nhất cho sinh viên Việt |
| **PACLIC 2026** | Tháng 6–7/2026 | ⭐⭐⭐⭐ Nhiều paper ABSA/emotion Việt Nam |
| **EMNLP 2026 Findings** | Tháng 5/2026 | ⭐⭐⭐ Khó hơn, cần kết quả mạnh |
| **Scopus Q2/Q3 Journal** | Linh hoạt | ⭐⭐⭐ Phù hợp sau khi có conference paper |

> **Khuyến nghị thực tế:** Submit **VLSP Workshop** trước (deadline ~tháng 9) — đây là venue tier 1 cho sinh viên Việt làm Vietnamese NLP, peer review nhưng không quá khắt khe về novelty như ACL/EMNLP. Sau đó mở rộng thêm ablation + error analysis để submit journal.

***

## 10. Rủi Ro và Phương Án Dự Phòng

| Rủi ro | Khả năng | Phương án dự phòng |
|--------|----------|-------------------|
| Emoji2Vec không cải thiện nhiều | Trung bình | Chuyển sang learnable emoji embeddings (train từ đầu) hoặc dùng CLIP visual embedding của emoji[^9] |
| Contrastive loss không hội tụ | Thấp | Giảm về BCE + focal loss + label smoothing (baseline mạnh hơn) |
| GPU không đủ | Thấp–Trung | Dùng LoRA rank=4 hoặc gradient checkpointing[^11] |
| Kết quả không vượt baseline | Thấp | Bài paper vẫn có giá trị nếu ablation study và error analysis đủ chi tiết và insightful |

***

## 11. Tóm Tắt Điểm Mạnh Của Đề Tài

- ✅ **Không cần xây dựng dataset:** ViGoEmotions (2026) đã public, chất lượng cao (EACL 2026)[^8]
- ✅ **Baseline thấp, dư địa cải thiện lớn:** MF1 chỉ 61.50% — rất nhiều chỗ để cải tiến[^2]
- ✅ **Research gap được paper gốc tự đề xuất:** "richer emoji embeddings, contrastive learning" — tức là cộng đồng chờ ai làm[^4]
- ✅ **Nhóm VKU chưa chạm đến:** Họ chỉ làm 3-class SA, không phải 27-label multi-label emotion[^3]
- ✅ **Kỹ thuật proof-of-concept có sẵn:** Emoji2Vec đã được test cho hate speech (ICON 2024), TACO đã test cho English FEC (ACL 2025)[^18][^6]
- ✅ **Chi phí thấp:** Colab Pro đủ nếu dùng LoRA[^12][^11]
- ✅ **Thời gian hợp lý:** 14 tuần (~3.5 tháng) đủ cho 1 nhóm 3–4 sinh viên

---

## References

1. [A Benchmark Dataset For Fine-grained Emotion Detection on ... - arXiv](https://arxiv.org/html/2602.08371v1)

2. [ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion ...](https://www.arxiv.org/abs/2602.08371) - Emotion classification plays a significant role in emotion prediction and harmful content detection....

3. [nghiên cứu giải pháp nâng cao chất lượng phân tích](https://elib.vku.udn.vn/bitstream/123456789/5013/1/B3.%2020-27.pdf)

4. [[Literature Review] ViGoEmotions: A Benchmark Dataset For Fine ...](https://www.themoonlight.io/en/review/vigoemotions-a-benchmark-dataset-for-fine-grained-emotion-detection-on-vietnamese-texts) - This paper introduces ViGoEmotions, a novel benchmark dataset for fine-grained emotion detection in ...

5. [A Benchmark Dataset For Fine-grained Emotion Detection on ... - arXiv](https://arxiv.org/html/2602.08371v2) - ... sentiment classification tasks involving Vietnamese social media content. Across all models, ret...

6. [A Triple-View Framework for Fine-Grained Emotion Classification ...](https://aclanthology.org/2025.acl-long.247/) - Fine-grained emotion classification (FEC) aims to analyze speakers' utterances and distinguish dozen...

7. [[1911.09339] Emotion Recognition for Vietnamese Social Media Text](https://arxiv.org/abs/1911.09339) - Emotion recognition plays a critical role in measuring the brand value of a product by recognizing s...

8. [[PDF] ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion ...](https://aclanthology.org/2026.eacl-long.129.pdf) - Accurate emo- tion recognition in text is essential for tasks such as sentiment analysis, emotion pr...

9. [[PDF] Sentiment Classification of Emoji and Text: An Analysis of Model ...](https://internationalpubls.com/index.php/cana/article/download/6072/3413/10813) - By synthesizing findings from 2015 to 2025, this review provides a comprehensive overview of how int...

10. [[PDF] A Triple-View Framework for Fine-Grained Emotion Classification ...](https://aclanthology.org/anthology-files/pdf/acl/2025.acl-long.247.pdf)

11. [[PDF] A COMPARATIVE STUDY OF FULL FINE-TUNING AND LORA](https://hjs.huflit.edu.vn/index.php/hjs/article/download/275/172/1708) - This paper presents a comparative study between full fine-tuning and Low-. Rank Adaptation – a param...

12. [Efficient Finetuning Large Language Models For Vietnamese Chatbot](https://huggingface.co/papers/2309.04646) - Parameter-efficient tuning using LoRA on open LLMs for Vietnamese language improves instruction-foll...

13. [[PDF] ViSoBERT: A Pre-Trained Language Model for Vietnamese Social ...](https://aclanthology.org/2023.emnlp-main.315.pdf)

14. [ViSoBERT: A Pre-Trained Language Model for Vietnamese ...](https://ar5iv.labs.arxiv.org/html/2310.11166) - English and Chinese, known as resource-rich languages, have witnessed the strong development of tran...

15. [[PDF] VLUE: A New Benchmark and Multi-task Knowledge Transfer ...](https://aclanthology.org/2024.findings-naacl.15.pdf)

16. [uitnlp/CafeBERT](https://huggingface.co/uitnlp/CafeBERT) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

17. [Emotion Recognition for Vietnamese Social Media Text](https://www.academia.edu/118078217/Emotion_Recognition_for_Vietnamese_Social_Media_Text) - The six basic emotions identified are enjoyment, sadness, anger, fear, disgust, and surprise. An int...

18. [[PDF] Utilizing Emoji to Aid Hate Speech Detection - ACL Anthology](https://aclanthology.org/2024.icon-1.64.pdf) - By integrating Emoji2Vec embeddings into the model, it gains access to the emotional nuances that of...

19. [[PDF] Utilizing Emoji to Aid Hate Speech Detection - ACL Anthology](https://aclanthology.org/anthology-files/pdf/icon/2024.icon-1.81.pdf)

20. [Sentiment analysis of emoji fused reviews using machine learning ...](https://www.nature.com/articles/s41598-025-92286-0) - We present a novel approach that performs sentiment analysis, with effective utilization of emojis a...

21. [[PDF] Multi-Label Emotion Detection Using Ensemble Transformer Models ...](https://aclanthology.org/2025.semeval-1.27.pdf) - Since each emotion category is treated as a binary classification problem (0 or 1), and the dataset ...

