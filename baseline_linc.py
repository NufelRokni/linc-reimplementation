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
#  Dataset loading (robust)
# ------------------------------

def load_folio(args):
    """Return the requested split of the FOLIO dataset.
    Tries public ID first; falls back to alternatives for compatibility.
    """
    last_err = None
    for ds_name in ("tasksource/folio", "yale-nlp/FOLIO", "folio"):
        try:
            ds = load_dataset(ds_name)[args.split]
            return ds
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
        # common numeric encodings: 1/0/2 or similar; be conservative
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
    # try typical FOLIO fields
    if "premises" in sample:
        prs = sample["premises"]
        if isinstance(prs, list):
            return "\n".join(str(p).strip() for p in prs)
        return str(prs)
    if "context" in sample:
        return str(sample["context"]).strip()
    return ""  # fallback


def _extract_conclusion(sample) -> str:
    for k in ("hypothesis", "conclusion", "query", "statement"):
        if k in sample:
            return str(sample[k]).strip()
    return ""


class SimpleLoader:
    def __init__(self, hf_split):
        self.data = [self._normalize_row(r) for r in hf_split]

    @staticmethod
    def _normalize_row(row: Dict) -> Dict:
        premises = _extract_premises(row)
        conclusion = _extract_conclusion(row)
        # common label fields: label / answer
        raw_label = row.get("label", row.get("answer"))
        label = _canon_label(raw_label)
        return {
            "premises": premises,
            "conclusion": conclusion,
            "label": label,
        }

    def get_example(self, idx: int) -> Dict:
        return self.data[idx]


# ------------------------------
#  Prompting
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
            f"Premises: {ex['premises']}\n" \
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
#  Inference
# ------------------------------

_LABEL_PATTERN = re.compile(r"\b(True|False|Uncertain)\b", re.IGNORECASE)


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
    # remove None values to avoid HF warnings
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(**model_inputs, **gen_kwargs)
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # return only the continuation part after the prompt
    gen_text = full_text[len(tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)) :]
    return gen_text.strip()


def infer_once_or_vote(model, tokenizer, prompt: str, args) -> Tuple[str, Counter]:
    votes = Counter()
    for _ in range(max(1, args.n_samples)):
        gen = generate_once(model, tokenizer, prompt, args)
        lab = parse_label_from_text(gen) or "Uncertain"
        votes[lab] += 1
        if args.verbose:
            print(f"  sample -> {gen!r} => {lab}")
    # majority
    pred = votes.most_common(1)[0][0]
    return pred, votes


# ------------------------------
#  Main
# ------------------------------


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="bigcode/starcoderplus", help="HF model id")
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test", "validation_matched", "validation_mismatched", "dev", "val", "validation"])
    p.add_argument("--num_examples", type=int, default=10)
    p.add_argument("--shots", type=int, default=1, help="number of demo examples in-context; 0 = zero-shot")
    p.add_argument("--n_samples", type=int, default=1, help="samples per example for voting")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--verbose", action="store_true")
    return p


def _torch_dtype_from_arg(arg: str):
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp32":
        return torch.float32
    return None


def main():
    args = build_argparser().parse_args()
    random.seed(args.seed)

    # 1) Data
    hf_split = load_folio(args)
    loader = SimpleLoader(hf_split)
    data = loader.data

    # 2) Model
    torch_dtype = _torch_dtype_from_arg(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    # 3) Prompt builder
    prompt_builder = PromptBuilder(INSTRUCTION)

    # 4) Evaluation loop
    n = min(max(1, args.num_examples), len(data))
    indices = list(range(n))  # first N examples; customize if needed

    correct = 0
    rows = []

    for i in indices:
        target = loader.get_example(i)

        # Minimal change to avoid fixed demo bias: rotate demos relative to target
        demos: Optional[List[Dict]] = None
        if args.shots > 0:
            demos = []
            for s in range(args.shots):
                demo_idx = (i + 1 + s) % len(data)
                if demo_idx == i:
                    demo_idx = (demo_idx + 1) % len(data)
                demos.append(loader.get_example(demo_idx))

        prompt = prompt_builder.build(demos, target)

        pred, votes = infer_once_or_vote(model, tokenizer, prompt, args)
        gold = target["label"]
        is_correct = int(str(pred) == str(gold))
        correct += is_correct

        print(f"[{i+1}/{n}] pred={pred} gold={gold} votes={votes}")

        rows.append({
            "idx": i,
            "pred": pred,
            "gold": gold,
            "votes": dict(votes),
        })

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
