"""Argument parsing for the FOLIO baseline runner.

The command line interface for the baseline script exposes a number
of parameters controlling the dataset split, few‑shot settings,
generation hyperparameters and output options.  Consolidating the
construction of this parser into a separate module makes the
definition easy to read and maintain, and also allows other modules
to reuse the same parser when necessary.
"""

from __future__ import annotations

import argparse

__all__ = ["build_argparser"]


def build_argparser() -> argparse.ArgumentParser:
    """Construct and return a populated argument parser.

    The returned parser largely mirrors the options available in the
    original LINC baseline script.  Additional arguments can be added
    here to enable future functionality (e.g. selecting different
    datasets or inference modes).
    """
    p = argparse.ArgumentParser()
    # Model configuration
    p.add_argument(
        "--model",
        type=str,
        default="bigcode/starcoderplus",
        help="HuggingFace model identifier to load",
    )
    # Dataset split selection
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=[
            "train",
            "validation",
            "test",
            "validation_matched",
            "validation_mismatched",
            "dev",
            "val",
        ],
        help="Name of the dataset split to evaluate on",
    )
    # Few shot sampling
    p.add_argument(
        "--shots",
        type=int,
        default=1,
        help="# of demonstration examples in context; 0 = zero‑shot",
    )
    # Voting settings
    p.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples per example for majority vote",
    )
    # Legacy alias: ``num_examples`` mirrors LINC's naming
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (LINC style --limit)",
    )
    p.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Alias for --limit when not specified",
    )
    # Generation hyperparameters
    p.add_argument(
        "--max_length_generation",
        type=int,
        default=None,
        help="LINC name for generation length; overrides --max_new_tokens",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum number of new tokens to generate",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation (0 for greedy)",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold",
    )
    # Precision / device handling
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Model precision to use when loading",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping for model loading (e.g. 'auto', 'cpu', 'cuda:0')",
    )
    # Output configuration
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to write prediction files",
    )
    # Token control flags (accepted for parity with LINC)
    p.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Pass through authentication token when loading models",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading remote model code (dangerous if untrusted)",
    )
    # Miscellaneous
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate generation results and votes",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for few‑shot sampling",
    )
    return p
