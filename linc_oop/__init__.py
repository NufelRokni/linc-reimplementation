"""Top level package for the LINC OOP re‑implementation.

This package exposes the primary entry points and aggregates
important classes so they can be imported directly from the package.
"""

from .datasets import Sample, FolioDataset, ProofWriterDataset  # noqa: F401
from .models import LMModel  # noqa: F401
from .prompts import PromptBuilder  # noqa: F401
from .modes import BaselineMode, CotMode, ScratchpadMode, LincMode  # noqa: F401
from .evaluation import MajorityVoter, Metrics  # noqa: F401
from .fol import extract_fol_blocks, normalize_for_prover9  # noqa: F401
from .prover import prove_label  # noqa: F401

__all__ = [
    "Sample",
    "FolioDataset",
    "ProofWriterDataset",
    "LMModel",
    "PromptBuilder",
    "BaselineMode",
    "CotMode",
    "ScratchpadMode",
    "LincMode",
    "MajorityVoter",
    "Metrics",
    "extract_fol_blocks",
    "normalize_for_prover9",
    "prove_label",
]