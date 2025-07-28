#!/usr/bin/env python
import argparse, os, json, time, random, math, shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from linc_lib.arguments import build_arg_parser
from linc_lib.datasets import load_folio, Sample
from linc_lib.prompts import build_prompt
from linc_lib.generation import load_model_and_tokenizer, generate_once
from linc_lib.fol_parser import extract_fol_blocks, normalize_for_prover9
from linc_lib.prover import prove_label
from linc_lib.evaluation import MajorityVoter, Metrics, init_run_dir, save_trace

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def predict_once(args, model, tokenizer, sample: Sample, method: str) -> Dict[str, Any]:
    """
    Returns a dict with 'raw', 'pred_label', optional 'fol_theory', 'fol_query'
    """
    prompt = build_prompt(sample, method=method, shots=args.shots, dataset=args.dataset)

    gen_cfg = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=(args.temperature > 0.0),
        stop_tag="</FOL_QUERY>" if method in ("scratchpad","linc") else None
    )
    raw = generate_once(model, tokenizer, prompt, **gen_cfg)

    if method in ("scratchpad", "linc"):
        try:
            fol_theory, fol_query = extract_fol_blocks(raw)
        except Exception as e:
            return {"raw": raw, "pred_label": "Uncertain", "error": f"parse_error: {e}"}

        if method == "scratchpad":
            # Predict directly from the model's final answer tag if present, else fallback to consistency heuristic
            # Try to read an explicit ANSWER tag; otherwise, guess via simple lexical heuristic
            if "<ANSWER>" in raw:
                tail = raw.split("<ANSWER>", 1)[1].split("</ANSWER>")[0]
                guess = tail.strip().split()[0].strip().strip(".").capitalize()
            else:
                # Heuristic: if fol_query is present, keep Uncertain unless explicit True/False detected
                guess = "Uncertain"
            return {"raw": raw, "pred_label": guess, "fol_theory": fol_theory, "fol_query": fol_query}

        # LINC: call Prover9
        norm_theory = normalize_for_prover9(fol_theory)
        norm_query  = normalize_for_prover9(fol_query)
        label = prove_label(norm_theory, norm_query, prover9_path=args.prover9_path, timeout=args.prover9_timeout)
        return {"raw": raw, "pred_label": label, "fol_theory": norm_theory, "fol_query": norm_query}

    elif method == "cot":
        # Expect an ANSWER tag
        if "<ANSWER>" in raw:
            ans = raw.split("<ANSWER>",1)[1].split("</ANSWER>")[0]
            label = ans.strip().split()[0].strip().strip(".").capitalize()
        else:
            label = "Uncertain"
        return {"raw": raw, "pred_label": label}

    else:  # naive
        if "<ANSWER>" in raw:
            ans = raw.split("<ANSWER>",1)[1].split("</ANSWER>")[0]
            label = ans.strip().split()[0].strip().strip(".").capitalize()
        else:
            label = "Uncertain"
        return {"raw": raw, "pred_label": label}

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    run_dir = init_run_dir(args.output_dir)
    print(f"[INFO] Saving outputs under: {run_dir}")

    # Dataset
    if args.dataset.lower() == "folio":
        data = load_folio(split=args.split, source=args.folio_source)
    elif args.dataset.lower() == "proofwriter":
        raise SystemExit("ProofWriter is stubbed for now. Please use --dataset folio.")
    else:
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    # Model
    model, tokenizer = load_model_and_tokenizer(args.model, args.precision, args.device)

    voter = MajorityVoter()
    metrics = Metrics()

    preds_path = Path(run_dir) / "predictions.jsonl"
    traces_path = Path(run_dir) / "traces.txt"
    with open(preds_path, "w", encoding="utf-8") as fout, open(traces_path, "w", encoding="utf-8") as ftrace:
        for idx, sample in enumerate(data):
            per_attempt = []
            for s in range(args.n_samples):
                out = predict_once(args, model, tokenizer, sample, method=args.method)
                per_attempt.append(out)

            final_label = voter.vote([x["pred_label"] for x in per_attempt])
            correct = (final_label == sample.label)

            # metrics
            metrics.add(sample.label, final_label)

            rec = {
                "id": sample.id,
                "gold": sample.label,
                "final_pred": final_label,
                "attempts": per_attempt,
                "premises": sample.premises,
                "conclusion": sample.conclusion
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            save_trace(ftrace, sample, per_attempt, final_label, correct)

            if (idx+1) % 25 == 0:
                print(f"[INFO] processed {idx+1} / {len(data)}")

    # Save metrics
    metrics_path = Path(run_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fm:
        json.dump(metrics.summary(), fm, indent=2)
    print("[RESULTS]")
    print(json.dumps(metrics.summary(), indent=2))

if __name__ == "__main__":
    main()
