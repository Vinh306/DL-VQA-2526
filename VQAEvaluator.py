import re
import numpy as np
from typing import List, Dict, Tuple, Optional

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from bert_score import score as bert_score_fn


class VQAEvaluator:
    def __init__(
        self,
        device = "cuda",
        bertscore_model = "vinai/phobert-base",
        bertscore_layers = 12,
        bleu_tokenize = "char",       # "char" phù hợp tiếng Việt
        padding_idx = 0,
    ):
        self.device           = device
        self.bertscore_model  = bertscore_model
        self.bertscore_layers = bertscore_layers
        self.bleu_tokenize    = bleu_tokenize
        self.padding_idx      = padding_idx

        # Lưu kết quả của từng model để so sánh sau
        self._results: Dict[str, dict] = {}


    #  Các hàm metric riêng lẻ (private)                                 
    @staticmethod
    def _norm(s):
        return re.sub(r'\s+', ' ', s.lower().strip())

    def _exact_match(self, preds, refs) -> float:
        correct = sum(self._norm(p) == self._norm(r) for p, r in zip(preds, refs))
        return correct / len(refs) if refs else 0.0

    def _bleu(self, preds, refs) -> Dict[str, float]:
        bleu   = BLEU(tokenize=self.bleu_tokenize)
        result = bleu.corpus_score(preds, [refs])
        return {
            "bleu"  : result.score,
            "bleu_1": result.precisions[0],
            "bleu_2": result.precisions[1],
            "bleu_3": result.precisions[2],
            "bleu_4": result.precisions[3],
        }

    def _rouge_l(self, preds, refs):
        scorer_fn = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [
            scorer_fn.score(r, p)["rougeL"].fmeasure
            for p, r in zip(preds, refs)
        ]
        return float(np.mean(scores))

    def _meteor(self, preds, refs):
        scores = [
            nltk_meteor([r.split()], p.split())
            for p, r in zip(preds, refs)
        ]
        return float(np.mean(scores))

    def _bertscore(self, preds, refs):
        P, R, F = bert_score_fn(
            preds, refs,
            model_type  = self.bertscore_model,
            lang        = "vi",
            num_layers  = self.bertscore_layers,
            verbose     = False,
            device      = self.device,
        )
        return {
            "bertscore_P": float(P.mean()),
            "bertscore_R": float(R.mean()),
            "bertscore_F": float(F.mean()),
        }


    #  collect_preds_refs: tách riêng để dễ override theo từng model  
    def collect_preds_refs(
        self,
        model,
        loader,
        decode_fn=None,
    ):
        import torch
        model.eval()
        all_preds, all_refs = [], []

        with torch.no_grad():
            for batch in loader:
                if decode_fn is not None:
                    preds, refs = decode_fn(model, batch)
                else:
                    # Fallback: loader trả về pred/ref string trực tiếp
                    preds = batch["pred"] if isinstance(batch["pred"], list) else [batch["pred"]]
                    refs  = batch["ref"]  if isinstance(batch["ref"],  list) else [batch["ref"]]
                all_preds.extend(preds)
                all_refs.extend(refs)

        return all_preds, all_refs

    #  Hàm chính: evaluate một model                                     
    def evaluate(
        self,
        model,
        loader,
        model_name,
        decode_fn=None,
        split_name = "test",
        preds = None,
        refs = None,
        verbose: bool = True,
    ):

        if verbose:
            print(f"\n{'='*45}")
            print(f"  Evaluating [{model_name}] on [{split_name}]")
            print(f"{'='*45}")

        # Lấy predictions
        if preds is None or refs is None:
            preds, refs = self.collect_preds_refs(model, loader, decode_fn)

        # Tính metrics
        em      = self._exact_match(preds, refs)
        bleu    = self._bleu(preds, refs)
        rouge_l = self._rouge_l(preds, refs)
        meteor  = self._meteor(preds, refs)
        bscore  = self._bertscore(preds, refs)

        metrics = {
            "exact_match"  : round(em * 100, 2),
            "bleu"         : round(bleu["bleu"],   2),
            "bleu_1"       : round(bleu["bleu_1"], 2),
            "bleu_2"       : round(bleu["bleu_2"], 2),
            "bleu_3"       : round(bleu["bleu_3"], 2),
            "bleu_4"       : round(bleu["bleu_4"], 2),
            "rouge_l"      : round(rouge_l * 100, 2),
            "meteor"       : round(meteor  * 100, 2),
            "bertscore_P"  : round(bscore["bertscore_P"] * 100, 2),
            "bertscore_R"  : round(bscore["bertscore_R"] * 100, 2),
            "bertscore_F"  : round(bscore["bertscore_F"] * 100, 2),
        }

        # Lưu để compare sau
        self._results[model_name] = metrics

        if verbose:
            self._print_metrics(metrics)

        return metrics, preds, refs

    
    #  So sánh tất cả model đã evaluate                               
    def compare(self, highlight_best = True):
        if not self._results:
            print("Chưa có kết quả nào. Hãy chạy evaluate() trước.")
            return

        models   = list(self._results.keys())
        metrics_ = list(next(iter(self._results.values())).keys())
        col_w    = 16

        # Header
        header = f"  {'Metric':<22}" + "".join(f"{m:>{col_w}}" for m in models)
        print(f"\n{'='*( 22 + col_w * len(models) + 2)}")
        print(header)
        print(f"  {'-'*(20 + col_w * len(models))}")

        metric_labels = {
            "exact_match" : "Exact Match (%)",
            "bleu"        : "BLEU (corpus)",
            "bleu_1"      : "BLEU-1",
            "bleu_2"      : "BLEU-2",
            "bleu_3"      : "BLEU-3",
            "bleu_4"      : "BLEU-4",
            "rouge_l"     : "ROUGE-L (%)",
            "meteor"      : "METEOR (%)",
            "bertscore_P" : "BERTScore-P (%)",
            "bertscore_R" : "BERTScore-R (%)",
            "bertscore_F" : "BERTScore-F (%)",
        }

        for key in metrics_:
            label  = metric_labels.get(key, key)
            values = [self._results[m][key] for m in models]
            best   = max(values)
            row    = f"  {label:<22}"
            for v in values:
                cell = f"{v:.2f}"
                if highlight_best and v == best:
                    cell += "*"
                row += f"{cell:>{col_w}}"
            print(row)

        print(f"  {'*  = best':>{ 22 + col_w * len(models) - 2}}")
        print(f"{'='*(22 + col_w * len(models) + 2)}\n")

    def get_results(self) -> Dict[str, Dict[str, float]]:
        return self._results

    def reset(self) -> None:
        self._results.clear()


    #  Helper in bảng                                                      
    @staticmethod
    def _print_metrics(metrics):
        rows = [
            ("Exact Match (%)", metrics["exact_match"]),
            ("BLEU (corpus)",   metrics["bleu"]),
            ("BLEU-1",          metrics["bleu_1"]),
            ("BLEU-2",          metrics["bleu_2"]),
            ("BLEU-3",          metrics["bleu_3"]),
            ("BLEU-4",          metrics["bleu_4"]),
            ("ROUGE-L (%)",     metrics["rouge_l"]),
            ("METEOR (%)",      metrics["meteor"]),
            ("BERTScore-P (%)", metrics["bertscore_P"]),
            ("BERTScore-R (%)", metrics["bertscore_R"]),
            ("BERTScore-F (%)", metrics["bertscore_F"]),
        ]
        print(f"\n  {'Metric':<22} {'Value':>8}")
        print(f"  {'-'*32}")
        for label, val in rows:
            print(f"  {label:<22} {val:>8.2f}")
        print()