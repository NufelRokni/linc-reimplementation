"""Dataset loading utilities.

This module defines simple classes for reading examples from the
FOLIO and ProofWriter benchmarks.  Each example is normalised
into a :class:`Sample` dataclass containing the premises, conclusion
and gold label.  A small amount of sanitisation and canonicalisation
is performed to smooth over differences between datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    # HuggingFace datasets can be a heavy dependency; import lazily.
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    load_dataset = None  # type: ignore


@dataclass
class Sample:
    """A single logical inference example.

    :param id:       A unique identifier for the example if available.
    :param premises: A list of premise sentences.
    :param conclusion: The hypothesis/conclusion sentence.
    :param label:    The gold truth value (``True``, ``False`` or ``Uncertain``).
    """

    id: str
    premises: List[str]
    conclusion: str
    label: str


def _canonicalise_label(raw: Optional[str]) -> str:
    """Map various label encodings onto the canonical set {True, False, Uncertain}.

    Different datasets spell truth values in different ways (e.g. ``1``/``0`` for
    True/False, ``Unknown`` instead of ``Uncertain``).  This helper collapses
    these variants into a consistent representation.  Unrecognised values
    default to ``Uncertain``.

    :param raw: Raw label value from the dataset.
    :returns: A canonicalised label string.
    """
    if raw is None:
        return "Uncertain"
    # Handle numeric encodings
    if isinstance(raw, (int, float)):
        mapping = {1: "True", 0: "False", 2: "Uncertain", 3: "Uncertain"}
        return mapping.get(int(raw), "Uncertain")
    s = str(raw).strip().lower()
    if s in {"true", "t", "yes", "entails", "entailed"}:
        return "True"
    if s in {"false", "f", "no", "contradiction", "contradictory"}:
        return "False"
    if s in {"uncertain", "unknown", "indeterminate", "cannot be determined", "not enough info", "undetermined"}:
        return "Uncertain"
    # Fallback: try to match a label in the string
    for key in ["true", "false", "uncertain"]:
        if key in s:
            return key.capitalize()
    return "Uncertain"


def _split_sentences(text: str) -> List[str]:
    """Conservative sentence splitter for premises.

    The FOLIO dataset stores multiple premises in a single string.  This
    helper splits on periods while preserving the trailing period and
    stripping whitespace.  Empty segments are dropped.

    :param text: Raw premises string.
    :returns: List of trimmed sentences ending with a period.
    """
    parts = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return [p + "." for p in parts]


class BaseDataset:
    """Abstract base class for datasets.

    Subclasses must populate ``self.samples`` with a list of
    :class:`Sample` objects.  A ``limit`` can be provided to only
    evaluate the first ``limit`` examples.
    """

    def __init__(self, split: str = "validation", limit: Optional[int] = None) -> None:
        self.split = split
        self.limit = limit
        self.samples: List[Sample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[Sample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class FolioDataset(BaseDataset):
    """Loader for the FOLIO logical reasoning benchmark.

    The loader first attempts to read the tasksource version of the
    dataset; if that fails it falls back to the yale version.  Only
    the specified split is loaded.  Each row is normalised into a
    :class:`Sample` with sentence‑level premises.
    """

    DS_NAMES = ("tasksource/folio", "yale-nlp/FOLIO", "folio")

    def __init__(self, split: str = "validation", limit: Optional[int] = None) -> None:
        super().__init__(split, limit)
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' library is required to load FOLIO. Please install it via pip."
            )
        last_err = None
        for name in self.DS_NAMES:
            try:
                ds = load_dataset(name, split=split)
                break
            except Exception as e:  # pragma: no cover - dataset loading errors propagate at runtime
                last_err = e
                ds = None
        if ds is None:
            raise FileNotFoundError(
                f"Could not load FOLIO dataset split '{split}'. Last error: {last_err}"
            )
        for idx, row in enumerate(ds):
            # Stop early if a limit is specified
            if self.limit is not None and idx >= self.limit:
                break
            prem = row.get("premises")
            conclusion = row.get("conclusion") or row.get("hypothesis")
            label = row.get("label")
            # Some variants use 'context' for premises
            if prem is None:
                prem = row.get("context", "")
            # Convert premises to list of sentences
            if isinstance(prem, list):
                premises = [str(p).strip() if str(p).strip().endswith(".") else str(p).strip() + "." for p in prem]
            else:
                premises = _split_sentences(str(prem)) if prem else []
            conclusion_str = str(conclusion).strip() if conclusion else ""
            self.samples.append(
                Sample(
                    id=str(row.get("example_id") or row.get("story_id") or idx),
                    premises=premises,
                    conclusion=conclusion_str,
                    label=_canonicalise_label(label),
                )
            )


class ProofWriterDataset(BaseDataset):
    """Loader for the ProofWriter dataset.

    ProofWriter examples consist of a theory (set of facts and rules),
    a question and an answer.  The theory is split on periods and
    newlines to form the premises.  Answers of ``Unknown`` or similar
    are mapped to ``Uncertain``.
    """

    DS_NAMES = ("tasksource/proofwriter", "proofwriter", "renma/ProofWriter")

    def __init__(self, split: str = "validation", limit: Optional[int] = None) -> None:
        super().__init__(split, limit)
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' library is required to load ProofWriter. Please install it via pip."
            )
        last_err = None
        ds = None
        for name in self.DS_NAMES:
            try:
                ds = load_dataset(name, split=split)
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                ds = None
        if ds is None:
            raise FileNotFoundError(
                f"Could not load ProofWriter dataset split '{split}'. Last error: {last_err}"
            )
        for idx, row in enumerate(ds):
            if self.limit is not None and idx >= self.limit:
                break
            theory = row.get("theory") or row.get("context") or ""
            # Some variants encode the theory as a list
            if isinstance(theory, list):
                theory_text = " ".join(str(x) for x in theory)
            else:
                theory_text = str(theory)
            # Split the theory into sentences; preserve trailing period
            premises = []
            for sent in theory_text.replace("\n", " ").split("."):
                s = sent.strip()
                if not s:
                    continue
                # Add back the full stop if it was removed
                if not s.endswith("."):
                    s = s + "."
                premises.append(s)
            question = row.get("question") or row.get("query") or ""
            answer = row.get("answer") or row.get("label") or row.get("target")
            self.samples.append(
                Sample(
                    id=str(row.get("id") or idx),
                    premises=premises,
                    conclusion=str(question).strip(),
                    label=_canonicalise_label(answer),
                )
            )