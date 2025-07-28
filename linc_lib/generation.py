import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

def _dtype_from_precision(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32

def load_model_and_tokenizer(model_name: str, precision: str, device: str):
    dtype = _dtype_from_precision(precision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    return model, tokenizer

def generate_once(model, tokenizer, prompt: str, 
                  max_new_tokens: int = 384, temperature: float = 0.7,
                  top_p: float = 0.95, top_k: int = 50, do_sample: bool = True,
                  stop_tag: Optional[str] = None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if stop_tag and stop_tag in text:
        text = text.split(stop_tag)[0] + stop_tag
    return text
