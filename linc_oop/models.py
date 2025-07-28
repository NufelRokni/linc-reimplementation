"""Model loading and generation utilities.

This module defines a thin wrapper around HuggingFace causal language
models.  The wrapper hides away device and precision handling and
exposes a single ``generate`` method that takes a prompt and
generation hyperparameters.  A ``stop_tag`` can optionally be
provided; if present the generated text is truncated when the tag
occurs.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype_from_precision(precision: str) -> torch.dtype:
    """Map a user provided precision string onto a torch dtype.

    :param precision: One of ``fp32``, ``fp16`` or ``bf16``.
    :returns: Corresponding ``torch.dtype``.
    :raises ValueError: for unknown precision strings.
    """
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")


class LMModel:
    """Wrapper around a HuggingFace causal language model and its tokenizer."""

    def __init__(self, model_name: str, precision: str = "fp16", device: str = "cuda") -> None:
        """Initialise and load a model and its tokenizer.

        :param model_name: HuggingFace model identifier (e.g. ``bigcode/starcoderplus``).
        :param precision: Numerical precision for the model weights.
        :param device: Device to place the model on (``cuda`` or ``cpu``).
        """
        dtype = _dtype_from_precision(precision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Ensure the tokenizer has an end of sequence token; fall back to '</s>'.
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        # For generation we need a pad token; use eos if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Use device_map="auto" when using CUDA to automatically place model on available GPUs
        device_map = "auto" if device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        # If device_map is None and user requested cpu, move model explicitly
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: Optional[bool] = None,
        stop_tag: Optional[str] = None,
    ) -> str:
        """Generate a continuation from the model given a prompt.

        :param prompt: The input string to condition on.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param temperature: Sampling temperature (>0 enables sampling).
        :param top_p: Nucleus sampling parameter.
        :param top_k: Top‑k sampling parameter.
        :param do_sample: If ``None`` uses sampling when temperature > 0; otherwise overrides.
        :param stop_tag: Optional substring on which to truncate the output.
        :returns: The generated text (including the prompt) or truncated at ``stop_tag``.
        """
        # Encode prompt and move inputs to model device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Determine sampling behaviour
        use_sampling = bool(temperature and temperature > 0) if do_sample is None else do_sample
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=use_sampling,
            temperature=temperature if use_sampling else None,
            top_p=top_p if use_sampling else None,
            top_k=top_k if use_sampling else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Remove None values to placate transformers warnings
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if stop_tag and stop_tag in text:
            # Truncate at the first occurrence of the stop tag and append the tag
            text = text.split(stop_tag)[0] + stop_tag
        return text