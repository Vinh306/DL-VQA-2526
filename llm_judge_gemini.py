from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google import genai
from google.genai import types as genai_types


# Data class — kết quả 1 sample 
@dataclass
class JudgeResult:
    question   : str
    ground_truth: str
    model_name : str = ""
    prediction : str = ""
    score: float | None = None
    explanation: str | None = None
    raw_response: str | None = None
    error      : str | None = None

    @property
    def is_valid(self):
        return self.error is None and self.score is not None

    def to_dict(self):
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "model_name": self.model_name,
            "prediction": self.prediction,
            "score": self.score,
            "explanation": self.explanation,
            "error": self.error,
        }


# LLMJudge — đánh giá độc lập 1 model 
class LLMJudge:

    def __init__(
        self,
        api_key = "",
        model = "gemini-2.5-flash",
        model_name = "Model",
        delay = 1.0,
        max_samples = 50,
        verbose = True,
    ):
        self.model       = model
        self.model_name  = model_name
        self.delay       = delay
        self.max_samples = max_samples
        self.verbose     = verbose

        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Cần API key. Truyền api_key=... hoặc set env GEMINI_API_KEY."
            )

        try:
            self._client     = genai.Client(api_key=self._api_key)
            self._genai_types = genai_types
        except ImportError as e:
            raise ImportError(e)

        self._results : list[JudgeResult] = []
        self._summary : dict = {}

    # Private: system prompt cho 1 model
    def _build_system_prompt(self):
        return (
            "Bạn là chuyên gia khó tính trong việc đánh giá hệ thống trả lời câu hỏi hình ảnh (VQA) tiếng Việt.\n"
            "Nhiệm vụ: Đánh giá câu trả lời của mô hình AI so với đáp án đúng.\n\n"
            "THANG ĐIỂM (0–10):\n"
            "10  — Hoàn toàn chính xác, tự nhiên\n"
            "7–9 — Đúng ý nghĩa, sai một số từ\n"
            "4–6 — Đúng khoảng 1 nửa số từ so với đáp án hoặc diễn đạt chưa tốt\n"
            "1–3 — Sai phần lớn các từ so với đáp án hoặc mơ hồ\n"
            "0   — Hoàn toàn sai / không liên quan\n\n"
            'Chỉ trả về DUY NHẤT một dòng JSON sau, không thêm gì khác:\n'
            '{"score": 7.5}\n'
            'Thay 7.5 bằng điểm thực tế. Không markdown, không giải thích.'
        )

    # Private: build user prompt 
    def _build_prompt(self, question, ground_truth, prediction):
        return "\n".join([
            f"Câu hỏi     : {question},",
            f"Đáp án đúng : {ground_truth},",
            f"Câu trả lời : {prediction},",
            "",
            "Hãy đánh giá và trả về JSON.",
        ])

    # Private: gọi Gemini 1 sample 
    def _call_once(self, question, ground_truth, prediction):
        result = JudgeResult(
            question = question,
            ground_truth = ground_truth,
            model_name = self.model_name,
            prediction = prediction,
        )
        try:
            resp = self._client.models.generate_content(
                model = self.model,
                config = self._genai_types.GenerateContentConfig(
                    system_instruction = self._build_system_prompt(),
                    temperature = 0.0,
                    max_output_tokens = 256,
                ),
                contents = self._build_prompt(question, ground_truth, prediction),
            )
            candidate = resp.candidates[0]
            finish = candidate.finish_reason.name  # "STOP", "MAX_TOKENS", "SAFETY", ...
            raw = candidate.content.parts[0].text.strip() if candidate.content.parts else ""
            result.raw_response = raw

            if finish != "STOP":
                raise ValueError(f"finish_reason={finish}, response bị cắt hoặc bị block")

            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # fallback: xoá trailing comma
                cleaned = re.sub(r",\s*([}\]])", r"\1", text)
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    # fallback cuối: regex tìm số
                    m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
                    if m:
                        parsed = {"score": float(m.group(1))}
                    else:
                        raise ValueError(f"Không parse được JSON: {raw!r}")

            result.score = float(parsed["score"])
            result.explanation = parsed.get("explanation", "")

        except Exception as exc:
            result.error = str(exc)
            print(f"\n    [raw_response] {result.raw_response!r}")

        return result

    # Public: evaluate 
    def evaluate(
        self,
        test_samples,
        predictions,
        max_samples = None,
    ) :
        n = min(
            max_samples or self.max_samples,
            len(test_samples),
            len(predictions),
        )

        if self.verbose:
            print(f"\n{'═'*62}")
            print(f"  LLM-AS-A-JUDGE  │  {self.model}  │  {n} samples")
            print(f"  Đánh giá model : {self.model_name}")
            print(f"{'═'*62}")

        results: list[JudgeResult] = []

        for i in range(n):
            sample = test_samples[i]
            q = sample.get("question", sample.get("q", ""))
            gt = sample.get("answer",   sample.get("a", ""))
            pred = predictions[i]

            if self.verbose:
                print(f"  [{i+1:>3}/{n}]  {q[:55]}...", end="  ")

            res = self._call_once(q, gt, pred)
            results.append(res)

            if self.verbose:
                if res.is_valid:
                    print(f"score={res.score:.1f}")
                else:
                    print(f"Lỗi: {res.error}")

            if i < n - 1:
                time.sleep(self.delay)

        self._results = results
        self._summary = self._compute_summary(results)

        if self.verbose:
            self._print_summary(self._summary)

        return results, self._summary

    # Public: save
    def save(self, prefix: str = "llm_judge_results"):
        if not self._results:
            print("Chưa có kết quả. Chạy .evaluate() trước.")
            return

        path = Path(prefix)
        path.parent.mkdir(parents=True, exist_ok=True)

        json_path = path.with_suffix(".json")
        payload = {
            "summary": self._summary,
            "details": [r.to_dict() for r in self._results],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        csv_path = path.with_suffix(".csv")
        pd.DataFrame([r.to_dict() for r in self._results]).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )

        print(f"Đã lưu JSON → {json_path}")
        print(f"Đã lưu CSV  → {csv_path}")

    # Public: plot_summary
    def plot_summary(self, save_path=None):
        if not self._summary:
            print("Chưa có kết quả. Chạy .evaluate() trước.")
            return

        s = self._summary
        valid = [r for r in self._results if r.is_valid]
        scores = [r.score for r in valid]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        # Histogram phân phối điểm
        ax1 = axes[0]
        ax1.hist(scores, bins=11, range=(0, 10.5), color="#4C72B0",
                 alpha=0.85, edgecolor="white", linewidth=1.2)
        ax1.axvline(s["avg_score"], color="#DD4444", linewidth=2,
                    linestyle="--", label=f"Trung bình: {s['avg_score']:.2f}")
        ax1.set_xlabel("Điểm (0–10)", fontsize=11)
        ax1.set_ylabel("Số sample", fontsize=11)
        ax1.set_title(f"Phân phối điểm — {self.model_name}", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", alpha=0.3)
        ax1.spines[["top", "right"]].set_visible(False)

        # Bar thống kê tóm tắt 
        ax2 = axes[1]
        labels = ["Điểm TB", "Điểm Min", "Điểm Max", "Median"]
        values = [
            s["avg_score"],
            s["min_score"],
            s["max_score"],
            s["median_score"],
        ]
        colors = ["#4C72B0", "#DD8844", "#44AA66", "#AA44BB"]
        bars = ax2.bar(labels, values, color=colors, width=0.5,
                       alpha=0.88, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
        ax2.set_ylim(0, 11)
        ax2.set_ylabel("Điểm (/ 10)", fontsize=11)
        ax2.set_title("Thống kê tổng hợp", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"LLM as a Judge — {self.model}  │  {s['valid_samples']} samples hợp lệ",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"  Đã lưu biểu đồ → {save_path}")
        plt.show()

    # Public: to_dataframe
    def to_dataframe(self):
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._results])

    # Properties
    @property
    def summary(self):
        return self._summary

    @property
    def results(self):
        return self._results

    # Private helpers
    def _compute_summary(self, results):
        valid  = [r for r in results if r.is_valid]
        n      = len(valid)
        if n == 0:
            return {"valid_samples": 0, "total_samples": len(results)}

        scores = [r.score for r in valid]

        # Phân phối theo dải điểm
        bands = {
            "perfect_10": sum(1 for s in scores if s == 10),
            "good_7_9": sum(1 for s in scores if 7 <= s < 10),
            "partial_4_6": sum(1 for s in scores if 4 <= s < 7),
            "poor_1_3": sum(1 for s in scores if 1 <= s < 4),
            "wrong_0": sum(1 for s in scores if s < 1),
        }

        import statistics
        return {
            "model_name": self.model_name,
            "total_samples": len(results),
            "valid_samples": n,
            "errors": len(results) - n,
            "avg_score": round(sum(scores) / n, 4),
            "median_score": round(statistics.median(scores), 4),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_bands": bands,
        }

    def _print_summary(self, s):
        n = s.get("valid_samples", 0)
        print(f"\n{'═'*62}")
        print(f"  KẾT QUẢ — {s.get('model_name', self.model_name)}")
        print(f"{'═'*62}")
        print(f"  Mẫu hợp lệ  : {n} / {s.get('total_samples', 0)}"
              f"  (lỗi: {s.get('errors', 0)})")
        print(f"  Điểm TB     : {s.get('avg_score', 0):.4f} / 10")
        print(f"  Median      : {s.get('median_score', 0):.4f} / 10")
        print(f"  Min / Max   : {s.get('min_score', 0):.1f} / {s.get('max_score', 0):.1f}")
        bands = s.get("score_bands", {})
        print(f"\n  Phân bố điểm:")
        print(f"    10 điểm   : {bands.get('perfect_10', 0):>3} sample")
        print(f"    7–9 điểm  : {bands.get('good_7_9', 0):>3} sample")
        print(f"    4–6 điểm  : {bands.get('partial_4_6', 0):>3} sample")
        print(f"    1–3 điểm  : {bands.get('poor_1_3', 0):>3} sample")
        print(f"    0 điểm    : {bands.get('wrong_0', 0):>3} sample")
        print(f"{'═'*62}\n")

    def __repr__(self):
        return (
            f"LLMJudge(model={self.model!r}, "
            f"model_name={self.model_name!r}, "
            f"evaluated={len(self._results)}, "
            f"max_samples={self.max_samples})"
        )