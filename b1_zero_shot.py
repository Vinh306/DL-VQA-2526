"""
Hướng B1 — BLIP Zero-Shot VQA (Tiếng Việt)
=============================================
Chiến lược tiếng Việt:
    Câu hỏi tiếng Việt được dịch sang tiếng Anh bằng deep_translator (GoogleTranslator)
    trước khi đưa vào BLIP. BLIP sinh câu trả lời tiếng Anh, sau đó dịch ngược về
    tiếng Việt để so sánh với ground-truth.

Lý do chọn dịch (translate) thay vì dùng trực tiếp:
    - BLIP-base được pretrain hoàn toàn bằng tiếng Anh → zero-shot trực tiếp
      với tiếng Việt cho kết quả rất kém.
    - Translate giúp tận dụng tối đa khả năng zero-shot của BLIP mà không
      cần fine-tune (phù hợp B1).
    - Với 4GB VRAM, tải thêm một mô hình dịch nhẹ (API-based) không tốn VRAM.

Yêu cầu:
    pip install transformers torch Pillow deep_translator underthesea
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from deep_translator import GoogleTranslator
from text_utils import TextProcessor
import json
import os
from tqdm import tqdm


# Dịch Việt sang Anh dùng GoogleTranslator
class ViEnTranslator:
    def __init__(self):
        self.vi2en = GoogleTranslator(source="vi", target="en")
        self.en2vi = GoogleTranslator(source="en", target="vi")

    def to_english(self, text):
        try:
            return self.vi2en.translate(text)
        except Exception:
            return text  # fallback: giữ nguyên nếu lỗi mạng

    def to_vietnamese(self, text):
        try:
            return self.en2vi.translate(text)
        except Exception:
            return text


# B1 Model (BLIP-base zero-shot)
class BLIPB1ZeroShot:
    def __init__(self, model_name = "Salesforce/blip-vqa-base", device = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Đang tải BLIP trên {self.device} ...")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.translator = ViEnTranslator()
        print("Model sẵn sàng.")


    def predict(self, image, question_vi):
        question_en = self.translator.to_english(question_vi)

        inputs = self.processor(image, question_en, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)

        answer_en = self.processor.decode(out[0], skip_special_tokens=True)
        answer_vi = self.translator.to_vietnamese(answer_en)

        return {
            "question_vi": question_vi,
            "question_en": question_en,
            "answer_en": answer_en,
            "answer_vi": answer_vi,
        }


    def evaluate(self, samples, image_root):
        predictions = []
        exact_matches = 0

        for item in tqdm(samples, desc="[B1] Zero-shot evaluation"):
            img_path = os.path.join(image_root, item["image_path"].replace("\\", os.sep))
            image = Image.open(img_path).convert("RGB")

            result = self.predict(image, item["question"])
            result["ground_truth"] = item["answer"]

            # Exact match (normalize)
            pred_norm = TextProcessor.preprocess(result["answer_vi"])
            gt_norm   = TextProcessor.preprocess(item["answer"])
            result["exact_match"] = int(pred_norm == gt_norm)
            exact_matches += result["exact_match"]

            predictions.append(result)

        accuracy = exact_matches / len(samples) if samples else 0.0
        print(f"Zero-shot Accuracy: {accuracy:.4f} ({exact_matches}/{len(samples)})")

        return {"accuracy": accuracy, "predictions": predictions}



if __name__ == "__main__":
    model = BLIPB1ZeroShot()

    # Demo với 1 ảnh
    # demo_image = Image.open("data/images/bánh_mì_Việt_Nam/000001.jpg").convert("RGB") 
    # question = "Bức ảnh này chụp những loại rau gì?"

    # result = model.predict(demo_image, question)
    # print("\n=== KẾT QUẢ DEMO ===")
    # for k, v in result.items():
    #     print(f"  {k}: {v}")

    # Test ngẫu nhiên trên tập test
    print("\n" + "="*60)
    print("  TEST NGẪU NHIÊN TRÊN TẬP TEST")
    print("="*60)

    import random
    with open("data/test.json", "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    # Lấy 5 sample ngẫu nhiên
    num_samples = min(5, len(test_samples))
    random_samples = random.sample(test_samples, num_samples)

    for idx, item in enumerate(random_samples, 1):
        print(f"\n[Mẫu {idx}]")
        print(f"  Câu hỏi: {item['question']}")
        
        img_path = os.path.join("", item["image_path"].replace("\\", os.sep))
        image = Image.open(img_path).convert("RGB")
        result = model.predict(image, item["question"])
        
        print(f"  Câu trả lời dự đoán: {result['answer_vi']}")
        print(f"  Câu trả lời tham khảo: {item['answer']}")
        print(f"  Đúng: {'✓' if TextProcessor.preprocess(result['answer_vi']) == TextProcessor.preprocess(item['answer']) else '✗'}")

