"""Entry point for evaluating language models on logical inference tasks.

This script ties together all components of the LINC OOP re‑implementation.
It parses command line arguments, loads a dataset, initialises a model
and a prompting strategy, iterates over examples and records
predictions.  Results are written to disk along with a simple metrics
summary.  See ``README.md`` for usage examples.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from linc_oop.arguments import build_arg_parser
from linc_oop.datasets import FolioDataset, ProofWriterDataset
from linc_oop.models import LMModel
from linc_oop.prompts import PromptBuilder
from linc_oop.modes import BaselineMode, CotMode, ScratchpadMode, LincMode
from linc_oop.evaluation import MajorityVoter, Metrics, init_run_dir, save_trace


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    # Ensure reproducibility for sampling
    random.seed(args.seed)

    # Load dataset
    if args.task.lower() == "folio":
        dataset = FolioDataset(split=args.split, limit=args.limit)
    elif args.task.lower() == "proofwriter":
        dataset = ProofWriterDataset(split=args.split, limit=args.limit)
    else:
        raise ValueError(f"Unknown task '{args.task}'")

    # Map user precision (e.g. "16") to LMModel precision (e.g. "fp16")
    precision_map: Dict[str, str] = {"16": "fp16", "32": "fp32", "bf16": "bf16"}
    precision = precision_map.get(str(args.precision), str(args.precision))

    # Load model and prepare prompt builder
    model = LMModel(args.model, precision=precision, device=args.device)
    prompt_builder = PromptBuilder(shots=0, dataset=args.task)

    # Select mode class
    mode_map = {
        "baseline": BaselineMode,
        "cot": CotMode,
        "scratchpad": ScratchpadMode,
        "linc": LincMode,
    }
    mode_key = args.mode.lower()
    if mode_key not in mode_map:
        raise ValueError(f"Unknown mode '{args.mode}'")
    mode_cls = mode_map[mode_key]
    runner = mode_cls(model, prompt_builder, args)

    # Prepare output directory
    run_dir = init_run_dir(args.output_dir)
    preds_path = os.path.join(run_dir, "predictions.jsonl")
    traces_path = os.path.join(run_dir, "traces.txt")
    metrics = Metrics()
    voter = MajorityVoter()

    with open(preds_path, "w", encoding="utf-8") as fout, open(traces_path, "w", encoding="utf-8") as ftrace:
        for idx, sample in enumerate(dataset):
            attempts: List[Dict] = []
            for _ in range(max(1, args.n_samples)):
                pred = runner.predict(sample)
                attempts.append(pred)
            final_label = voter.vote([a.get("pred_label", "Uncertain") for a in attempts])
            metrics.add(sample.label, final_label)
            rec = {
                "id": sample.id,
                "gold": sample.label,
                "final_pred": final_label,
                "attempts": attempts,
                "premises": sample.premises,
                "conclusion": sample.conclusion,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            save_trace(ftrace, sample, attempts, final_label, final_label == sample.label)
            # Provide progress feedback every 25 examples
            if (idx + 1) % 25 == 0:
                print(f"[INFO] processed {idx + 1} / {len(dataset)}")

    # Write metrics summary
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fm:
        json.dump(metrics.summary(), fm, indent=2)
    print("[RESULTS]")
    print(json.dumps(metrics.summary(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
