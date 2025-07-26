import argparse
import random, re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Config:
    """Configuration for the baseline run."""

    model: str = "bigcode/starcoderplus"  # Hugging Face model name (StarCoder+)
    dataset: str = "tasksource/folio"  # Dataset name or path (FOLIO on HF)
    split: str = "validation"  # Dataset split to use (e.g., train/validation)
    index: int = 0  # Index of the example to test
    shots: int = 1  # Number of in-context examples (fixed to 1 in practice)
    seed: int = 42  # Random seed for reproducibility
    max_new_tokens: int = 64  # Max tokens to generate for the answer
    temperature: float = 0.8  # Sampling temperature (0 for deterministic)
    top_p: float = 1.0  # Top-p sampling (1.0 for no filtering)

    @staticmethod
    def parse_args() -> "Config":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="bigcode/starcoderplus")
        parser.add_argument("--dataset", default="folio")
        parser.add_argument("--split", default="validation")
        parser.add_argument("--index", type=int, default=0)
        parser.add_argument("--shots", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max_new_tokens", type=int, default=64)
        parser.add_argument("--temperature", type=float, default=0.8)
        parser.add_argument("--top_p", type=float, default=1.0)
        args = parser.parse_args()
        cfg = Config(**vars(args))
        # Resolve dataset name if shorthand is given:
        if cfg.dataset.lower() == "folio":
            cfg.dataset = "tasksource/folio"
        return cfg


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

    def build(self, demo, example):
        demo_prem, demo_concl, demo_label = demo
        ex_prem, ex_concl, _ = example
        prompt = (
            f"{self.instruction}\n\n"
            f"Premises: {demo_prem}\n"
            f"Conclusion: {demo_concl}\n"
            f"Label: {demo_label}\n\n"
            f"Premises: {ex_prem}\n"
            f"Conclusion: {ex_concl}\n"
            f"Label: "
        )
        return prompt


class ModelClient:
    """Loads the language model and handles text generation."""

    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # set to inference mode
        self.model.to(self.device)

    def generate(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ):
        # Encode prompt and generate continuation
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
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
        gen_ids = output_ids[0][input_ids.size(1) :]
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


def main():
    # Parse configuration
    cfg = Config.parse_args()
    # Set random seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # Load dataset and select examples
    loader = DatasetLoader(cfg.dataset, cfg.split)
    demo_idx = 0 if cfg.index != 0 else 1  # ensure demo and target are different
    demo_example = loader.get_example(demo_idx)
    target_example = loader.get_example(cfg.index)
    # Build prompt with one demonstration
    instruction = (
        "Determine if the conclusion follows logically from the premises. "
        "Answer with True, False, or Uncertain."
    )
    prompt_builder = FewShotPromptBuilder(instruction)
    prompt = prompt_builder.build(demo_example, target_example)
    # Load model and generate prediction
    model_client = ModelClient(cfg.model)
    output_text = model_client.generate(
        prompt, cfg.max_new_tokens, cfg.temperature, cfg.top_p
    )
    # Evaluate prediction
    evaluator = Evaluator()
    pred_label, is_correct = evaluator.evaluate(output_text, target_example[2])
    # Print results
    print("Prompt:\n" + prompt)
    print("Model output:", output_text.strip())
    print("Parsed prediction:", pred_label)
    print("Gold label:", target_example[2])
    print("Correct:", is_correct)


if __name__ == "__main__":
    main()
