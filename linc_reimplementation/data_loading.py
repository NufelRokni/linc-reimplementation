"""Dataset loading utilities for the FOLIO benchmark.

The functions and classes defined in this module encapsulate all logic
related to reading examples from the FOLIO dataset and preparing them
for prompting a language model.  Splitting this out of the main
script allows the remainder of the codebase to remain agnostic to the
source of the data and makes it easier to add support for additional
datasets in the future (e.g. ProofWriter).  Should you wish to
integrate a new dataset, define a function analogous to
``load_folio`` and register it within any higher level task
dispatcher.

In addition to the baseline loader, this module defines a
``ScratchpadLoader`` which preserves first‑order logic translations of
the premises and conclusion where available.  When such translations
are missing the corresponding fields are set to ``None``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "load_folio",
    "SimpleLoader",
    "ScratchpadLoader",
    "canon_label",
]


def load_folio(args) -> Tuple[object, Optional[object]]:
    """Return a tuple ``(target_split, train_split)`` from the FOLIO dataset.

    The returned objects are HuggingFace ``Dataset`` objects corresponding
    to the requested evaluation split and (optionally) the training
    split.  This mirrors the behaviour of the original LINC code: the
    evaluation examples are drawn from the specified split (default
    ``validation``) and few‑shot demonstrations, if enabled, are
    sampled from the training split when available.

    :param args: Parsed arguments exposing the ``split`` attribute.
    :returns: tuple of (target split dataset, training split dataset or None)
    :raises FileNotFoundError: if no compatible dataset can be loaded.
    """
    # Avoid import of datasets at module import time; import lazily here.
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:  # pragma: no cover - handled at call time
        raise ImportError(
            "The 'datasets' library is required to load the FOLIO dataset. "
            "Please install it via pip (e.g. 'pip install datasets')."
        ) from e

    last_err: Optional[Exception] = None
    # Try a set of known dataset identifiers to maximise robustness.
    for ds_name in ("tasksource/folio", "yale-nlp/FOLIO", "folio"):
        try:
            ds_all = load_dataset(ds_name)
            # choose target split (default to validation if present)
            tgt_split_name = args.split if args.split in ds_all else (
                "validation" if "validation" in ds_all else list(ds_all.keys())[0]
            )
            tgt_split = ds_all[tgt_split_name]
            tr_split = ds_all.get("train")
            return tgt_split, tr_split
        except Exception as e:  # pragma: no cover - dataset loading errors propagate
            last_err = e
    raise FileNotFoundError(
        "Could not load FOLIO dataset. Tried tasksource/folio, yale-nlp/FOLIO, folio. "
        f"Last error: {last_err}"
    )


_LABEL_CANON = {"true": "True", "false": "False", "uncertain": "Uncertain"}


def canon_label(x) -> str:
    """Map various label spellings/encodings to the canonical set.

    This helper coalesces different spellings and encodings of the
    truth values used in the dataset into the standard labels
    {``True``, ``False``, ``Uncertain``}.  Numerical encodings and
    synonyms are also supported.  Any unrecognised value defaults to
    ``Uncertain``.

    :param x: Raw label value from the dataset.
    :returns: Canonical label string.
    """
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
    if s in {
        "unknown",
        "indeterminate",
        "cannot be determined",
        "cannot_be_determined",
    }:
        return "Uncertain"
    # last resort: preferentially match whole words
    for k in ["true", "false", "uncertain"]:
        if re.search(rf"\b{k}\b", s):
            return _LABEL_CANON[k]
    return "Uncertain"


def _extract_premises(sample: Dict) -> str:
    """Extract a string of premises from a dataset sample.

    The FOLIO dataset has several possible keys storing the premises;
    this helper attempts to standardise them.  If no premises field is
    present, an empty string is returned.

    :param sample: A dataset record containing premises.
    :returns: Concatenated premises separated by newlines.
    """
    if "premises" in sample and sample["premises"]:
        prs = sample["premises"]
        if isinstance(prs, list):
            return "\n".join(str(p).strip() for p in prs)
        return str(prs)
    if "context" in sample and sample["context"]:
        return str(sample["context"]).strip()
    return ""


def _extract_conclusion(sample: Dict) -> str:
    """Extract the conclusion/hypothesis field from a dataset sample.

    The FOLIO dataset and related benchmarks may use different keys to
    store the hypothesis/conclusion.  This helper checks a number of
    possible field names and returns the first that is present.

    :param sample: A dataset record.
    :returns: The conclusion string or an empty string if none exist.
    """
    for k in ("hypothesis", "conclusion", "query", "statement"):
        if k in sample and sample[k]:
            return str(sample[k]).strip()
    return ""


def _extract_fol(sample: Dict, keys: List[str]) -> Optional[str]:
    """Extract a FOL translation from a dataset sample.

    FOLIO examples may store first‑order logic translations under a
    variety of keys.  This helper searches for a list of candidate
    keys and returns a normalised string if found.  When the value is a
    list, its elements are concatenated with newlines.  If no key is
    present the function returns ``None``.

    :param sample: A dataset record.
    :param keys: Ordered list of potential field names.
    :returns: Normalised FOL translation or ``None``.
    """
    for k in keys:
        if k in sample and sample[k]:
            val = sample[k]
            if isinstance(val, list):
                return "\n".join(str(v).strip() for v in val)
            return str(val).strip()
    return None


class SimpleLoader:
    """A lightweight wrapper for accessing dataset examples.

    Instances of this class precompute a normalised list of data
    examples upon construction.  Each example is represented as a
    dictionary with three fields: ``premises``, ``conclusion`` and
    ``label``.  The normalisation abstracts away dataset quirks (such
    as nested lists of premises) and canonicalises labels.  The
    resulting ``get_example`` method can be used without worrying
    about dataset specificities.
    """

    def __init__(self, hf_split) -> None:
        self.data: List[Dict] = [self._normalize_row(r) for r in hf_split] if hf_split is not None else []

    @staticmethod
    def _normalize_row(row: Dict) -> Dict:
        premises = _extract_premises(row)
        conclusion = _extract_conclusion(row)
        raw_label = row.get("label", row.get("answer"))
        label = canon_label(raw_label)
        return {"premises": premises, "conclusion": conclusion, "label": label}

    def get_example(self, idx: int) -> Dict:
        """Return the normalised example at a given index.

        :param idx: Index into the underlying dataset split.
        :returns: A dictionary with keys ``premises``, ``conclusion`` and ``label``.
        """
        return self.data[idx]


class ScratchpadLoader:
    """Loader that preserves FOL translations when available.

    The scratchpad mode requires access to both the natural language
    statements and their first‑order logic translations.  This loader
    extends the baseline ``SimpleLoader`` by capturing additional
    ``premises_FOL`` and ``conclusion_FOL`` fields.  When a translation
    is missing the corresponding value is set to ``None``.  The same
    canonicalisation of labels is applied as in the baseline.
    """

    # Candidate keys for FOL translations.  These lists reflect common
    # variations observed in FOLIO and related datasets.  New keys can
    # easily be appended as needed.
    PREMISE_FOL_KEYS: List[str] = [
        "premises-FOL",
        "premise-FOL",
        "premises_fol",
        "premises_fol_text",
        "premisesFOL",
    ]
    CONCLUSION_FOL_KEYS: List[str] = [
        "conclusion-FOL",
        "hypothesis-FOL",
        "query-FOL",
        "conclusion_fol",
        "hypothesis_fol",
        "query_fol",
        "conclusionFOL",
    ]

    def __init__(self, hf_split) -> None:
        self.data: List[Dict] = [self._normalize_row(r) for r in hf_split] if hf_split is not None else []

    @classmethod
    def _normalize_row(cls, row: Dict) -> Dict:
        premises = _extract_premises(row)
        conclusion = _extract_conclusion(row)
        raw_label = row.get("label", row.get("answer"))
        label = canon_label(raw_label)
        premises_fol = _extract_fol(row, cls.PREMISE_FOL_KEYS)
        conclusion_fol = _extract_fol(row, cls.CONCLUSION_FOL_KEYS)
        return {
            "premises": premises,
            "conclusion": conclusion,
            "label": label,
            "premises_FOL": premises_fol,
            "conclusion_FOL": conclusion_fol,
        }

    def get_example(self, idx: int) -> Dict:
        """Return the normalised example at a given index, including FOL.

        :param idx: Index into the underlying dataset split.
        :returns: A dictionary with keys ``premises``, ``conclusion``,
            ``label``, ``premises_FOL`` and ``conclusion_FOL``.
        """
        return self.data[idx]