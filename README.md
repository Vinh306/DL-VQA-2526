Hệ thống **Visual Question Answering** cho tiếng Việt — so sánh 5 kiến trúc mô hình khác nhau trên cùng một ảnh và câu hỏi đầu vào. Dữ liệu tập trung vào **ẩm thực Việt Nam** (15 món ăn, ~30 ảnh/món).

---

## Tổng quan

| Hướng | Mô hình                | Chiến lược                                             |
| ----- | ---------------------- | ------------------------------------------------------ |
| B1    | BLIP Zero-Shot         | Dịch VI→EN → BLIP → dịch EN→VI                         |
| B2    | BLIP Fine-Tuned (LoRA) | Fine-tune BLIP với adapter LoRA                        |
| B3    | BLIP DPO               | DPO (Direct Preference Optimization) trên BLIP LoRA    |
| C1    | Custom LSTM            | ResNet18 + PhoBERT + CoAttention + LSTM Decoder        |
| C2    | Custom Transformer     | ResNet18 + PhoBERT + CoAttention + Transformer Decoder |

Tất cả mô hình đều nhận **câu hỏi tiếng Việt** và trả về **câu trả lời tiếng Việt**.

---

## Cấu trúc dự án

```
.
├── data/
│   ├── images/             # Ảnh theo từng món ăn
│   ├── train.json          # Dữ liệu huấn luyện
│   └── test.json           # Dữ liệu kiểm tra
├── checkpoints/
│   ├── best_lstm.pt        # Checkpoint LSTM tốt nhất
│   └── best_transformer.pt # Checkpoint Transformer tốt nhất
│
├── # === Core Modules ===
├── VQA.py                  # VQAModel — lớp tổng hợp các encoder/decoder
├── ImageEncoder.py         # ResNet18 encoder (trích xuất đặc trưng ảnh)
├── QuestionEncoder.py      # PhoBERT encoder (mã hóa câu hỏi tiếng Việt)
├── CoAttention.py          # Co-Attention fusion (kết hợp ảnh + câu hỏi)
├── LSTMDecoder.py          # LSTM Decoder với dual attention
├── TransformerDecoder.py   # Transformer Decoder với causal masking
├── Dataset.py              # ViVQADataset + VQACollate
├── text_utils.py           # TextProcessor + Vocabulary (underthesea)
│
├── # === Pipelines (Notebooks) ===
├── prepare_data.ipynb      # Thu thập & tiền xử lý dữ liệu, tạo QA pairs
├── train_eval.ipynb        # Huấn luyện & đánh giá mô hình Custom (C1/C2)
├── B2_BLIP_FineTuned.ipynb # Fine-tune BLIP với LoRA
├── B3_BLIP_DPO.ipynb       # DPO training cho BLIP
├── eval.ipynb              # Đánh giá tổng hợp tất cả mô hình
│
├── # === Scripts ===
├── b1_zero_shot.py         # BLIP Zero-Shot pipeline (B1)
├── scrape_images.py        # Thu thập ảnh từ Bing bằng icrawler
├── llm_judge_gemini.py     # Đánh giá bằng Gemini (LLM-as-a-Judge)
├── VQAEvaluator.py         # Đánh giá tự động: BLEU, ROUGE-L, METEOR, BERTScore
└── web.py                  # Giao diện demo Streamlit
```

---

## Kiến trúc mô hình Custom (C1/C2)

```
Ảnh (384×384)
    └─► ResNet18 → (B, 144, 512)
                                  ┐
Câu hỏi (VI)                      ├─► CoAttention → img_ctx, q_out, q_vec
    └─► PhoBERT-base → (B, L, 768)┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                        LSTM Decoder        Transformer Decoder
                        (với dual attn)     (causal + cross-attn)
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                                  Câu trả lời (VI)
```

**CoAttention** cho phép ảnh và câu hỏi "chú ý" lẫn nhau hai chiều (bidirectional), sau đó kết hợp bằng gating + residual connection.

---

## Cài đặt

```bash
# 1. Clone repo
git clone <repo-url>
cd vivqa

# 2. Cài đặt thư viện
pip install torch torchvision transformers peft
pip install underthesea deep_translator streamlit
pip install sacrebleu rouge_score nltk bert_score
pip install icrawler Pillow google-genai pandas
```

### Chuẩn bị dữ liệu

```bash
# Thu thập ảnh (~15 món, 30 ảnh/món)
python scrape_images.py

# Xử lý dữ liệu thủ công để loại bỏ ảnh sai(quảng cáo, ảnh không rõ món ăn,...) để còn lại đúng 30 ảnh/món. Sau đó chạy prepare_data.ipynb để tạo QA pairs và split train/test
```

---

## Huấn luyện

### Mô hình Custom (LSTM / Transformer)

Chạy `train_eval.ipynb` — notebook hướng dẫn từng bước:

1. Load dữ liệu từ `data/train.json`
2. Khởi tạo `VQAModel` với `decoder_type="lstm"` hoặc `"transformer"`
3. Huấn luyện với cross-entropy loss + teacher forcing
4. Lưu checkpoint tốt nhất vào `checkpoints/`

### BLIP Fine-Tuned (B2)

Chạy `B2_BLIP_FineTuned.ipynb`:

- Fine-tune `Salesforce/blip-vqa-base` với LoRA adapter
- Pipeline: câu hỏi VI được dịch sang EN trước khi đưa vào BLIP
- Checkpoint được đẩy lên HuggingFace Hub (`lpv30/DL-CK`)

### BLIP DPO (B3)

Chạy `B3_BLIP_DPO.ipynb`:

- DPO (Direct Preference Optimization) trên checkpoint B2
- Cần chuẩn bị **preference pairs** (chosen/rejected)
- Checkpoint: `lpv30/DL-CK-DPO`

---

## Đánh giá

### Đánh giá tự động

```python
from VQAEvaluator import VQAEvaluator

evaluator = VQAEvaluator(device="cuda", bertscore_model="vinai/phobert-base")

# Đánh giá một mô hình
metrics, preds, refs = evaluator.evaluate(
    model, test_loader, model_name="LSTM", decode_fn=my_decode_fn
)

# So sánh tất cả mô hình
evaluator.compare()
```

Các metrics được tính: **Exact Match**, **BLEU-1/2/3/4**, **ROUGE-L**, **METEOR**, **BERTScore (P/R/F)**.

### LLM-as-a-Judge (Gemini)

```python
from llm_judge_gemini import LLMJudge

judge = LLMJudge(api_key="YOUR_GEMINI_KEY", model_name="BLIP-FT", max_samples=100)
results, summary = judge.evaluate(test_samples, predictions)

judge.plot_summary(save_path="judge_results.png")
judge.save("outputs/lstm_judge")
```

Thang điểm 0–10, phân tích phân bố điểm theo các dải: 10 / 7–9 / 4–6 / 1–3 / 0.

Chạy `eval.ipynb` để đánh giá tổng hợp cả 5 mô hình.

---

## Demo Web

```bash
streamlit run web.py
```

Giao diện so sánh kết quả dự đoán của **5 mô hình** cùng lúc trên một ảnh và câu hỏi bất kỳ.

![Demo screenshot](assets/demo.png)

> **Lưu ý:** Demo tải đồng thời cả 5 mô hình lên bộ nhớ. Cần tối thiểu **8GB VRAM** (GPU) hoặc **16GB RAM** (CPU).

---

## HuggingFace Hub

| Model                  | Hub ID                     |
| ---------------------- | -------------------------- |
| BLIP Fine-Tuned (LoRA) | `lpv30/DL-CK`              |
| BLIP DPO (LoRA)        | `lpv30/DL-CK-DPO`          |
| Question Encoder       | `vinai/phobert-base`       |
| Image-VQA Base         | `Salesforce/blip-vqa-base` |

---

## Yêu cầu hệ thống

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA 11.8+ (khuyến nghị; CPU cũng chạy được nhưng chậm hơn)
- Tối thiểu 8GB VRAM để chạy đầy đủ 5 mô hình cùng lúc trong demo

---

## Ghi chú

- **Dữ liệu**: Ảnh được thu thập tự động từ Bing bằng `icrawler`, resize về 384×384. QA pairs được tạo bán tự động và kiểm tra thủ công.
- **Tokenizer tiếng Việt**: Dùng `underthesea` cho word segmentation (phân đoạn từ), `vinai/phobert-base` để mã hóa câu hỏi.
- **Dịch máy**: Sử dụng `deep_translator` (Google Translate API) để chuyển đổi VI↔EN cho các mô hình BLIP.
- **Freeze layers**: PhoBERT được đóng băng 10 layer đầu để tiết kiệm bộ nhớ và tránh overfitting trên tập dữ liệu nhỏ.
