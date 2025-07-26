import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------
#  Dataset loading (LINC-aligned & robust)
# ------------------------------

def load_folio(args):
    """Return (target_split, train_split_or_None) from the FOLIO dataset.
    Prefer public ID; fall back to alternatives for compatibility.
    This mirrors LINC's practice of drawing few-shot demos from the *train* split
    while evaluating on validation/test when available.
    """
    last_err = None
    for ds_name in ("tasksource/folio", "yale-nlp/FOLIO", "folio"):
        try:
            ds_all = load_dataset(ds_name)
            # choose target split (default to validation if present)
            tgt_split_name = args.split if args.split in ds_all else (
                "validation" if "validation" in ds_all else list(ds_all.keys())[0]
            )
            tgt_split = ds_all[tgt_split_name]
            tr_split = ds_all["train"] if "train" in ds_all else None
            return tgt_split, tr_split
        except Exception as e:
            last_err = e
    raise FileNotFoundError(
        "Could not load FOLIO dataset. Tried tasksource/folio, yale-nlp/FOLIO, folio. "
        f"Last error: {last_err}"
    )


# ------------------------------
#  Utilities
# ------------------------------

_LABEL_CANON = {"true": "True", "false": "False", "uncertain": "Uncertain"}


def _canon_label(x) -> str:
    """Map various label spellings/encodings to {True, False, Uncertain}."""
    if x is None:
        return "Uncertain"
    if isinstance(x, (int, float)):
        # common numeric encodings: 1/0/2; keep conservative defaults
        m = {1: "True", 0: "False", 2: "Uncertain", 3: "Uncertain"}
        return m.get(int(x), "Uncertain")
    s = str(x).strip().lower()
    if s in _LABEL_CANON:
        return _LABEL_CANON[s]
    if s in {"entailed", "entails", "yes", "correct"}:
        return "True"
    if s in {"contradiction", "contradictory", "no", "incorrect"}:
        return "False"
    if s in {"unknown", "indeterminate", "cannot be determined", "cannot_be_determined"}:
        return "Uncertain"
    # last resort: preferentially match whole words
    for k in ["true", "false", "uncertain"]:
        if re.search(rf"\b{k}\b", s):
            return _LABEL_CANON[k]
    return "Uncertain"


def _extract_premises(sample) -> str:
    # typical FOLIO fields
    if "premises" in sample:
        prs = sample["premises"]
        if isinstance(prs, list):
            return "\n".join(str(p).strip() for p in prs)
        return str(prs)
    if "context" in sample:
        return str(sample["context"]).strip()
    return ""


def _extract_conclusion(sample) -> str:
    for k in ("hypothesis", "conclusion", "query", "statement"):
        if k in sample:
            return str(sample[k]).strip()
    return ""


class SimpleLoader:
    def __init__(self, hf_split):
        self.data = [self._normalize_row(r) for r in hf_split] if hf_split is not None else []

    @staticmethod
    def _normalize_row(row: Dict) -> Dict:
        premises = _extract_premises(row)
        conclusion = _extract_conclusion(row)
        raw_label = row.get("label", row.get("answer"))
        label = _canon_label(raw_label)
        return {"premises": premises, "conclusion": conclusion, "label": label}

    def get_example(self, idx: int) -> Dict:
        return self.data[idx]


# ------------------------------
#  Prompting (LINC-like template)
# ------------------------------

INSTRUCTION = (
    "Determine if the conclusion follows logically from the premises. "
    "Answer with exactly one of: True, False, or Uncertain."
)


class PromptBuilder:
    def __init__(self, instruction: str = INSTRUCTION):
        self.instruction = instruction

    def format_example(self, ex: Dict, with_label: bool) -> str:
        base = (
            f"Premises: {ex['premises']}\n"
            f"Conclusion: {ex['conclusion']}\n"
        )
        if with_label:
            base += f"Label: {ex['label']}\n\n"
        else:
            base += "Label: "  # model will complete
        return base

    def build(self, demos: Optional[List[Dict]], target: Dict) -> str:
        parts = [self.instruction, ""]
        if demos:
            for d in demos:
                parts.append(self.format_example(d, with_label=True))
        parts.append(self.format_example(target, with_label=False))
        return "\n".join(parts)


# ------------------------------
#  Inference (LINC-style voting over n_samples)
# ------------------------------

_LABEL_PATTERN = re.compile(r"\b(True|False|Uncertain)\b", re.IGNORECASE)


essential_labels = {"True", "False", "Uncertain"}

def parse_label_from_text(text: str) -> Optional[str]:
    m = _LABEL_PATTERN.search(text)
    if not m:
        return None
    return _canon_label(m.group(1))


def generate_once(model, tokenizer, prompt: str, args) -> str:
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = args.temperature and args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=float(args.temperature) if do_sample else None,
        top_p=float(args.top_p) if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(**model_inputs, **gen_kwargs)

    # slice only the continuation tokens (safer than string slicing)
    prompt_len = model_inputs["input_ids"].shape[-1]
    continuation_ids = out[0][prompt_len:]
    gen_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return gen_text.strip()


def infer_once_or_vote(model, tokenizer, prompt: str, args) -> Tuple[str, Counter]:
    votes = Counter()
    for _ in range(max(1, args.n_samples)):
        gen = generate_once(model, tokenizer, prompt, args)
        lab = parse_label_from_text(gen) or "Uncertain"
        votes[lab] += 1
        if args.verbose:
            print(f"  sample -> {gen!r} => {lab}")
    pred = votes.most_common(1)[0][0]
    return pred, votes


# ------------------------------
#  Args (align with LINC where possible, while keeping backward compat)
# ------------------------------


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="bigcode/starcoderplus", help="HF model id")
    p.add_argument("--split", type=str, default="validation",
                   choices=["train", "validation", "test", "validation_matched", "validation_mismatched", "dev", "val", "validation"])

    # LINC names and backward-compatible aliases
    p.add_argument("--n_samples", type=int, default=1, help="samples per example for majority vote (LINC-style)")
    p.add_argument("--batch_size", type=int, default=1)  # unused in this minimal script
    p.add_argument("--limit", type=int, default=None, help="max #examples (LINC-style)")
    p.add_argument("--num_examples", type=int, default=10, help="alias; used if --limit not set")
    p.add_argument("--max_length_generation", type=int, default=None, help="LINC name for new tokens; if set, overrides --max_new_tokens")
    p.add_argument("--max_new_tokens", type=int, default=8, help="alias; used when --max_length_generation not set")

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--shots", type=int, default=1, help="#demo examples in-context; 0 = zero-shot")

    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="LINC-style")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--use_auth_token", action="store_true")  # accepted for parity; not required if you've logged in
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def _torch_dtype_from_precision(arg: str):
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp32":
        return torch.float32
    return None


# ------------------------------
#  Main
# ------------------------------


def main():
    args = build_argparser().parse_args()
    random.seed(args.seed)

    # resolve generation length following LINC naming first
    args.max_new_tokens = args.max_length_generation if args.max_length_generation is not None else args.max_new_tokens

    # 1) Data (targets + train for few-shot pool)
    hf_tgt_split, hf_train_split = load_folio(args)
    loader = SimpleLoader(hf_tgt_split)
    loader_train = SimpleLoader(hf_train_split) if hf_train_split is not None else None

    data = loader.data
    n_total = len(data)

    # 2) Model
    torch_dtype = _torch_dtype_from_precision(args.precision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=getattr(args, "trust_remote_code", True),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=getattr(args, "trust_remote_code", True),
    )
    model.eval()

    # 3) Prompt builder
    prompt_builder = PromptBuilder(INSTRUCTION)

    # 4) Determine eval size (LINC uses --limit)
    n = args.limit if args.limit is not None else args.num_examples
    n = max(1, min(n, n_total))
    indices = list(range(n))  # first N examples; deterministic & simple

    # 5) Sample a *fixed* k-shot context once per run (LINC-style)
    fewshot_pool_loader = loader_train if (args.shots > 0 and loader_train is not None) else loader
    fewshot_pool_size = len(fewshot_pool_loader.data)
    fewshot_indices: List[int] = []
    if args.shots > 0 and fewshot_pool_size > 0:
        pool = list(range(fewshot_pool_size))
        random.shuffle(pool)
        fewshot_indices = pool[: args.shots]
        # Note: LINC does not enforce label diversity; we follow suit.

    correct = 0
    rows = []

    for i in indices:
        target = loader.get_example(i)

        # Build demos: fixed set, avoid self-demo if pool == target split
        demos: Optional[List[Dict]] = None
        if args.shots > 0 and fewshot_indices:
            demos = []
            for idx in fewshot_indices:
                # if we drew demos from the same split and one equals target idx, swap to next
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

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "preds.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / n if n else 0.0
    print(f"\nSaved {n} predictions to {out_path}")
    print(f"Accuracy on these {n} examples: {acc:.3f}")


if __name__ == "__main__":
    main()
