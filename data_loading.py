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
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ``torch`` is not required in this module.  It was imported in the
# original baseline for dtype handling but is unnecessary here.  We
# deliberately avoid importing heavy dependencies unless needed.
# HuggingFace's ``datasets`` module can be heavy and is not always available
# in all execution environments.  We therefore delay importing it until
# inside ``load_folio`` where it is actually needed.  This allows the
# rest of the module (e.g. unit tests or documentation tools) to be
# imported without requiring the dependency.

__all__ = [
    "load_folio",
    "SimpleLoader",
    "canon_label",
        "ScratchpadLoader",
]


def load_folio(args) -> Tuple[object, Optional[object]]:
    """Return a tuple ``(target_split, train_split)`` from the FOLIO dataset.

    The returned objects are HuggingFace Dataset objects corresponding
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

    last_err = None
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
    if "premises" in sample:
        prs = sample["premises"]
        if isinstance(prs, list):
            return "\n".join(str(p).strip() for p in prs)
        return str(prs)
    if "context" in sample:
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
        if k in sample:
            return str(sample[k]).strip()
    return ""


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
    """Loader that preserves per‑premise FOL annotations for scratchpad mode.

    This loader normalises FOLIO dataset records such that each returned example
    contains a list of premise strings, a corresponding list of FOL translations,
    a conclusion string and its FOL translation, and a canonicalised label.  It
    is intended for use with the scratchpad prompt, where demonstrations show
    TEXT/FOL pairs for every premise and conclusion and the answer is provided
    after an explicit ``ANSWER:`` tag.  If FOL translations are absent in the
    dataset, empty strings are supplied as placeholders so that the prompt
    structure remains consistent.
    """

    def __init__(self, hf_split) -> None:
        # Convert the HuggingFace split into a list of normalised examples.  If
        # ``hf_split`` is ``None`` (e.g. when no training set is available),
        # initialise an empty list.  The normalisation retains lists of
        # premises and FOL translations rather than flattening them into a
        # single string as done in ``SimpleLoader``.
        self.data: List[Dict] = [self._normalize_row(r) for r in hf_split] if hf_split is not None else []

    @staticmethod
    def _normalize_row(row: Dict) -> Dict:
        # Extract premises as a list.  The FOLIO dataset stores premises under
        # the key ``premises`` as a list of strings, but some variants may
        # provide a single string field (e.g. ``context``).  Normalise both
        # cases into a list of strings.
        prs = []
        if "premises" in row and row["premises"] is not None:
            if isinstance(row["premises"], list):
                prs = [str(p).strip() for p in row["premises"]]
            else:
                # single string; wrap in list
                prs = [str(row["premises"]).strip()]
        elif "context" in row and row["context"] is not None:
            prs = [str(row["context"]).strip()]
        else:
            prs = []
        # Extract FOL translations for premises.  Use whichever key is present
        # (``premises-FOL`` or ``premises_fol``).  If absent, generate a
        # placeholder list of empty strings matching the number of premises.
        pfol = None
        for key in ("premises-FOL", "premises_fol", "premises_folios", "premises_fols"):
            if key in row and row[key] is not None:
                pfol = row[key]
                break
        if pfol is None:
            pfol_list = ["" for _ in prs]
        else:
            if isinstance(pfol, list):
                pfol_list = [str(f).strip() for f in pfol]
            else:
                # single string or other type; wrap and duplicate if necessary
                pfol_list = [str(pfol).strip()] * max(1, len(prs))
        # Extract the conclusion and its FOL translation.  Several keys might
        # encode the conclusion/hypothesis in different dataset variants.
        concl = ""
        for k in ("conclusion", "hypothesis", "query", "statement"):
            if k in row and row[k] is not None:
                concl = str(row[k]).strip()
                break
        cfol = None
        for key in ("conclusion-FOL", "conclusion_fol", "conclusion-FOLIO", "conclusion_fols"):
            if key in row and row[key] is not None:
                cfol = row[key]
                break
        if cfol is None:
            concl_fol = ""
        else:
            # FOL for conclusion is a single string
            if isinstance(cfol, list):
                # join list if erroneously provided as list
                concl_fol = " ".join(str(x).strip() for x in cfol)
            else:
                concl_fol = str(cfol).strip()
        # Canonicalise the label
        raw_label = row.get("label", row.get("answer"))
        label = canon_label(raw_label)
        return {
            "premises": prs,
            "premises_fol": pfol_list,
            "conclusion": concl,
            "conclusion_fol": concl_fol,
            "label": label,
        }

    def get_example(self, idx: int) -> Dict:
        """Return the normalised scratchpad example at a given index.

        Each example includes a list of premises and corresponding FOL
        translations, a conclusion and its FOL, and the label.

        :param idx: Index into the underlying dataset split.
        :returns: Normalised dictionary as described in ``_normalize_row``.
        """
        return self.data[idx]
