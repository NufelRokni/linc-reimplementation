import os, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
from .datasets import Sample

class MajorityVoter:
    def vote(self, labels: List[str]) -> str:
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        # Tie-breaker: first occurrence
        max_count = max(counts.values())
        winners = [l for l, c in counts.items() if c == max_count]
        if len(winners) == 1:
            return winners[0]
        for l in labels:
            if l in winners:
                return l
        return labels[0]

@dataclass
class Metrics:
    total: int = 0
    correct: int = 0
    by_class: Dict[str, Dict[str,int]] = field(default_factory=lambda: {
        "True": {"gold":0, "correct":0},
        "False": {"gold":0, "correct":0},
        "Uncertain": {"gold":0, "correct":0},
    })

    def add(self, gold: str, pred: str):
        self.total += 1
        self.by_class.setdefault(gold, {"gold":0,"correct":0})
        self.by_class[gold]["gold"] += 1
        if gold == pred:
            self.correct += 1
            self.by_class[gold]["correct"] += 1

    def summary(self):
        overall = {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.correct / self.total, 4) if self.total else 0.0
        }
        per_class = {}
        for k, v in self.by_class.items():
            g = v["gold"]
            c = v["correct"]
            per_class[k] = {
                "gold_count": g,
                "correct": c,
                "accuracy": round(c / g, 4) if g else 0.0
            }
        return {"overall": overall, "per_class": per_class}

def init_run_dir(output_root: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

def save_trace(fh, sample: Sample, attempts: List[dict], final_label: str, correct: bool):
    print("="*80, file=fh)
    print(f"ID: {sample.id}", file=fh)
    print("Premises:", file=fh)
    for p in sample.premises:
        print(f"  - {p}", file=fh)
    print(f"Conclusion: {sample.conclusion}", file=fh)
    print(f"Gold: {sample.label} | Final: {final_label} | {'CORRECT' if correct else 'WRONG'}", file=fh)
    for i, att in enumerate(attempts, 1):
        print("-"*40, file=fh)
        print(f"Attempt {i}:", file=fh)
        if "fol_theory" in att:
            print("[FOL_THEORY]", file=fh)
            print(att["fol_theory"], file=fh)
        if "fol_query" in att:
            print("[FOL_QUERY]", file=fh)
            print(att["fol_query"], file=fh)
        print("[RAW OUTPUT]", file=fh)
        print(att.get("raw","").strip(), file=fh)
        print(f"Predicted: {att.get('pred_label')}", file=fh)
        if "error" in att:
            print(f"[ERROR] {att['error']}", file=fh)
    print("\n", file=fh)
