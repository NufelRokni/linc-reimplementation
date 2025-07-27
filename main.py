"""Entry point for the FOLIO baseline evaluator.

The ``main`` function defined in this module reproduces the behaviour
of the original ``baseline_linc.py`` script while delegating most
behaviour to reusable modules.  It loads the specified HuggingFace
model and dataset, constructs prompts with optional few‑shot
demonstrations, performs inference (with optional sampling and
majority voting) and writes the resulting predictions to disk.

Because the logic is now decomposed into smaller components the
function itself remains short and easy to follow.  Additional tasks
or inference modes can be integrated by modifying or extending this
module rather than rewriting from scratch.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

# Avoid importing HuggingFace transformers at module import time.  These
# heavy dependencies are imported lazily inside ``main`` so that the
# module can be imported in environments where transformers is not
# installed (e.g. for documentation generation).

from arguments import build_argparser
from data_loading import load_folio, SimpleLoader
from model_utils import torch_dtype_from_precision
from prompt import PromptBuilder, INSTRUCTION
from inference import infer_once_or_vote

__all__ = ["main"]


def main() -> None:
    """Execute the FOLIO baseline evaluation based on command line arguments."""
    args = build_argparser().parse_args()
    # Ensure reproducibility for few‑shot sampling
    random.seed(args.seed)
    # Resolve generation length: prefer max_length_generation if set
    if args.max_length_generation is not None:
        args.max_new_tokens = args.max_length_generation
    # 1) Load dataset (targets + train for few‑shot pool)
    hf_tgt_split, hf_train_split = load_folio(args)
    loader = SimpleLoader(hf_tgt_split)
    loader_train = SimpleLoader(hf_train_split) if hf_train_split is not None else None
    data = loader.data
    n_total = len(data)
    # 2) Load model and tokenizer
    dtype = torch_dtype_from_precision(args.precision)
    # Import transformers lazily here to avoid dependency at module import time
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=getattr(args, "trust_remote_code", True),
    )
    # Ensure a pad token exists; fall back to eos if necessary
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Import torch lazily to avoid mandatory dependency at import time
    import torch  # type: ignore
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=getattr(args, "trust_remote_code", True),
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # 3) Prompt builder
    prompt_builder = PromptBuilder(INSTRUCTION)
    # 4) Determine evaluation size
    n = args.limit if args.limit is not None else args.num_examples
    n = max(1, min(n, n_total))
    indices = list(range(n))  # deterministic evaluation on first N examples
    # 5) Sample a fixed k‑shot context once per run
    fewshot_pool_loader = loader_train if (args.shots > 0 and loader_train is not None) else loader
    fewshot_pool_size = len(fewshot_pool_loader.data)
    fewshot_indices: List[int] = []
    if args.shots > 0 and fewshot_pool_size > 0:
        pool = list(range(fewshot_pool_size))
        random.shuffle(pool)
        fewshot_indices = pool[: args.shots]
        # Note: original LINC does not enforce label diversity; we follow suit.
    correct = 0
    rows: List[Dict] = []
    # Iterate over evaluation examples
    for i in indices:
        target = loader.get_example(i)
        # Build demos: fixed set, avoid self‑demo if pool == target split
        demos: Optional[List[Dict]] = None
        if args.shots > 0 and fewshot_indices:
            demos = []
            for idx in fewshot_indices:
                # if we drew demos from the same split and one equals the target idx, swap to next
                if fewshot_pool_loader is loader and idx == i:
                    alt = (idx + 1) % n_total
                    if alt == i:
                        alt = (alt + 1) % n_total
                    demos.append(loader.get_example(alt))
                else:
                    demos.append(fewshot_pool_loader.get_example(idx))
        prompt = prompt_builder.build(demos, target)
        pred, votes = infer_once_or_vote(model, tokenizer, prompt, args)
        gold = target["label"]
        correct += int(str(pred) == str(gold))
        print(f"[{i+1}/{n}] pred={pred} gold={gold} votes={votes}")
        rows.append({"idx": i, "pred": pred, "gold": gold, "votes": dict(votes)})
    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "preds.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = correct / n if n else 0.0
    print(f"\nSaved {n} predictions to {out_path}")
    print(f"Accuracy on these {n} examples: {acc:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
