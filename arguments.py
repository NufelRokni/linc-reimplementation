"""Command line argument parsing for the LINC OOP re‑implementation.

This module defines a function to build an ``argparse`` parser
with sensible defaults for the logical inference tasks supported
by this project.  The resulting parser exposes options for the
dataset, model, evaluation mode, sampling parameters and more.

While the original LINC implementation spread argument parsing
across several scripts, here we centralise it to provide a
single, consistent interface.  The parser is intentionally
modular: new options can be added without modifying the rest of
the codebase.  See ``README.md`` for example usage.
"""

from __future__ import annotations

import argparse
from typing import Sequence, Optional


def build_arg_parser(argv: Optional[Sequence[str]] = None) -> argparse.ArgumentParser:
    """Construct and return an argument parser.

    The parser returned by this function defines a number of flags and
    options controlling how logical inference is performed.  The
    resulting object can be used to parse command line arguments and
    will populate an ``argparse.Namespace`` with attributes corresponding
    to each argument.

    :param argv: Optional sequence of argument strings (e.g. for unit testing).
    :returns: An ``ArgumentParser`` instance with all options defined.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a language model on logical inference tasks.  "
            "Select a dataset (folio or proofwriter), a model name or path, "
            "and an inference mode (baseline, scratchpad, cot, linc).  "
            "Additional options control sampling, evaluation size and output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset/task selection
    parser.add_argument(
        "--task",
        type=str,
        default="folio",
        choices=["folio", "proofwriter"],
        help="Which dataset to evaluate on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Split of the dataset to evaluate (e.g. train/validation/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples to process."
        " If omitted, all examples from the specified split are used.",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="bigcode/starcoderplus",
        help=(
            "Name or path of the HuggingFace model to load.  "
            "Supports StarCoder+, mistralai/Mistral-7B-v0.1 and Qwen models."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16",
        choices=["32", "16", "bf16"],
        help="Floating point precision to use when loading the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device to use (e.g. cuda or cpu).",
    )

    # Inference mode
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "scratchpad", "cot", "linc"],
        help="Inference mode controlling how prompts are constructed."
        " Baseline uses only the question; scratchpad uses FOL translation; "
        "cot uses chain‑of‑thought; linc uses FOL and external prover.",
    )

    # Sampling and generation parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate per example (for majority voting).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero means deterministic (greedy) decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top‑p nucleus sampling cutoff.  Only used when temperature > 0.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top‑k sampling cutoff.  Only used when temperature > 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation.  Currently unused; reserved for future use.",
    )

    # Miscellaneous options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to which predictions and traces will be saved.",
    )
    parser.add_argument(
        "--prover9_path",
        type=str,
        default="prover9",
        help="Path to the Prover9 executable (used only in LINC mode).",
    )
    parser.add_argument(
        "--prover9_timeout",
        type=int,
        default=30,
        help="Timeout (in seconds) for each Prover9 call in LINC mode.",
    )

    return parser
