import argparse
import random, re
from dataclasses import dataclass, fields
from collections import Counter, defaultdict
import os
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Config:
    """Configuration for the baseline run."""

    model: str = "bigcode/starcoder"
    dataset: str = "folio"
    split: str = "validation"
    shots: int = 1
    seed: int = 42
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 0
    vote_k: int = 10
    do_sample: bool = False
    num_examples: int = 10
    shuffle: bool = False
    save_path: str = "outputs/predictions.jsonl"

    @staticmethod
    def parse_args() -> "Config":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="bigcode/starcoder")
        parser.add_argument("--dataset", type=str, default="folio", choices=["folio"])
        parser.add_argument("--split", type=str, default="validation")
        parser.add_argument("--shots", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max_new_tokens", type=int, default=64)
        parser.add_argument("--temperature", type=float, default=0.8)
        parser.add_argument("--top_p", type=float, default=1.0)
        parser.add_argument("--top_k", type=int, default=0)
        parser.add_argument("--vote_k", type=int, default=10)
        parser.add_argument("--do_sample", action="store_true")
        parser.add_argument("--num_examples", type=int, default=10)
        parser.add_argument("--shuffle", action="store_true")
        parser.add_argument("--save_path", type=str, default="outputs/predictions.jsonl")
        
        args = parser.parse_args()
        # The 'index' field was unused in the batch loop, so it's removed.
        # We create a dictionary from args and remove any keys not in Config.
        config_fields = {f.name for f in fields(Config)}
        args_dict = {k: v for k, v in vars(args).items() if k in config_fields}
        
        return Config(**args_dict)


class DatasetLoader:
    """Loads the FOLIO dataset and retrieves examples by index."""

    def __init__(self, dataset_name: str, split: str):
        self.data = load_dataset(dataset_name, split=split)

    def get_example(self, idx: int):
        ex = self.data[idx]
        premises = ex["premises"]
        conclusion = ex["conclusion"]
        label = ex["label"]
        return premises, conclusion, label


class FewShotPromptBuilder:
    """Builds a prompt with one demonstration (1-shot) and the current example."""

    def __init__(self, instruction: str):
        self.instruction = instruction

    def build(self, demos, example):
        prompt = f"{self.instruction}\n\n"
        for demo_prem, demo_conclusion, demo_label in demos:
            prompt += (
                f"Premises: {demo_prem}\n"
                f"Conclusion: {demo_conclusion}\n"
                f"Label: {demo_label}\n\n"
            )
        ex_prem, ex_conclusion, _ = example
        prompt += (
            f"Premises: {ex_prem}\n"
            f"Conclusion: {ex_conclusion}\n"
            f"Label: "
        )
        return prompt


class ModelClient:
    """Loads the language model and handles text generation."""

    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,           # Use bf16 precision
            device_map="auto"                     # Efficiently use available GPU(s)
        )
        self.model.eval()  # set to inference mode

    def generate(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ):
        # Encode prompt and generate continuation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
        )
        # Slice out only the newly generated tokens
        gen_ids = output_ids[0][input_ids.size(1):]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

class Evaluator:
    """Parses the model output and checks it against the gold label."""

    def normalize(self, text: str) -> str:
        # Strip whitespace and punctuation, take first word as prediction
        pred = text.strip()
        # Use regex to find a standalone True/False/Uncertain (case-insensitive)
        m = re.search(r"\b(true|false|uncertain)\b", pred, flags=re.IGNORECASE)
        if m:
            pred_label = m.group(1)  # get the matched word
        else:
            pred_label = pred
        return pred_label.strip().capitalize()

    def evaluate(self, output: str, gold_label: str):
        pred_label = self.normalize(output)
        correct = pred_label.lower() == gold_label.lower()
        return pred_label, correct

def load_folio(args):
    ds = load_dataset("folio")[args.split]
    data = list(ds)
    if args.shuffle:
        random.Random(args.seed).shuffle(data)
    if args.num_examples > 0:
        data = data[:args.num_examples]
    return data

def build_inputs_from_example(ex, tokenizer, device):
    premises_text = "\n".join(f"- {p}" for p in ex["premises"])
    prompt = (
        "Determine if the conclusion follows logically from the premises. "
        "Answer with True, False, or Uncertain.\n\n"
        f"Premises:\n{premises_text}\n\n"
        f"Conclusion: {ex['conclusion']}\n"
        "Label: "
    )
    return tokenizer(prompt, return_tensors="pt").to(device)

def parse_label(text):
    m = re.search(r"\b(true|false|uncertain)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return text.strip().capitalize()

def majority_vote(labels):
    counts = Counter(labels)
    max_c = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == max_c]
    if len(winners) == 1:
        return winners[0]
    first_pos = {lab: labels.index(lab) for lab in winners}
    return min(first_pos, key=first_pos.get)

def vote_once(model, tokenizer, inputs, args):
    out = model.generate(
        **inputs,
        do_sample=True if args.do_sample or args.vote_k > 1 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.vote_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    labels = [parse_label(t) for t in texts]
    return labels, majority_vote(labels)

def main():
    cfg = Config.parse_args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    if cfg.dataset == "folio":
        data = load_folio(cfg)
    else:
        raise ValueError("Only FOLIO is supported by this script right now.")

    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
    correct = 0
    written = 0

    # Initialize loader and prompt builder for few-shot
    loader = DatasetLoader("folio", cfg.split)
    instruction = "Determine if the conclusion follows logically from the premises. Answer with True, False, or Uncertain."
    prompt_builder = FewShotPromptBuilder(instruction)

    with open(cfg.save_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(data):
            inputs = build_inputs_from_example(ex, tokenizer, model.device)
            k_labels, pred = vote_once(model, tokenizer, inputs, cfg)
            gold = ex.get("label", None)
            correct += int(gold is not None and pred.lower() == str(gold).lower())
            rec = {
                "index": idx,
                "premises": ex["premises"],
                "conclusion": ex["conclusion"],
                "gold": gold,
                "pred": pred,
                "k_labels": k_labels,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{idx+1}/{len(data)}] pred={pred} gold={gold} votes={Counter(k_labels)}")

            # New code for few-shot sampling
            all_examples = data  # assuming 'data' is your loaded dataset
            target_idx = idx

            # Partition indices by label, skipping the target
            label_buckets = defaultdict(list)
            for i, ex2 in enumerate(all_examples):
                if i == target_idx:
                    continue
                label_buckets[ex2["label"]].append(i)

            # Sample one from each label (if available)
            picked = []
            for label in ["True", "False", "Uncertain"]:
                bucket = label_buckets[label]
                if bucket:
                    picked.append(random.choice(bucket))

            # Fill up to cfg.shots with random others (excluding target and already picked)
            remaining = set(range(len(all_examples))) - set(picked) - {target_idx}
            n_extra = max(0, cfg.shots - len(picked))
            if n_extra > 0 and remaining:
                picked.extend(random.sample(list(remaining), k=n_extra))

            # Build demos and target
            demos = [loader.get_example(i) for i in picked]
            target = loader.get_example(target_idx)
            # Build the prompt (not used further in this script, but constructed for completeness)
            _ = prompt_builder.build(demos, target)

    acc = correct / max(1, written)
    print(f"\nSaved {written} predictions to {cfg.save_path}")
    print(f"Accuracy on these {written} examples: {acc:.3f}")
DatasetLoaderloader, 
if __name__ == "__main__":
    main()


