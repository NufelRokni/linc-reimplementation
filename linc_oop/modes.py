"""Inference mode classes for LINC OOP.

This module defines a set of small classes encapsulating the logic
required to perform inference under different prompting strategies.
Each mode subclass exposes a ``predict`` method that takes a
:class:`Sample` and returns a dictionary with at least a ``pred_label``
field.  Additional fields (e.g. raw generation, FOL theory) may be
returned to aid downstream analysis or debugging.

By organising the modes into classes we separate the prompt
construction and post‑processing logic from the high‑level evaluation
loop.  Adding a new mode requires creating a new subclass and
implementing its ``predict`` method.  See ``BaselineMode`` for a
minimal example.
"""

from __future__ import annotations

from typing import Dict, Optional

from .datasets import Sample
from .prompts import PromptBuilder
from .models import LMModel
from .fol import extract_fol_blocks, normalize_for_prover9
from .prover import prove_label

__all__ = ["BaseMode", "BaselineMode", "CotMode", "ScratchpadMode", "LincMode"]


class BaseMode:
    """Abstract base class for inference modes.

    Concrete subclasses must implement ``predict`` which takes a
    :class:`Sample` and returns a dictionary with at least a
    ``pred_label`` string.  Other fields (such as ``raw``) are
    preserved for analysis.  Subclasses have access to the model and
    prompt builder via ``self.model`` and ``self.prompt_builder``.
    """

    def __init__(self, model: LMModel, prompt_builder: PromptBuilder, args) -> None:
        self.model = model
        self.prompt_builder = prompt_builder
        self.args = args

    def _generate(self, prompt: str, stop_tag: Optional[str] = None) -> str:
        """Helper to generate a continuation from the model.

        This method wraps the underlying :meth:`LMModel.generate` call and
        passes through hyperparameters from ``self.args``.  The
        ``stop_tag`` is forwarded to ``generate`` to optionally truncate
        the output when a particular substring is encountered.
        """
        return self.model.generate(
            prompt,
            max_new_tokens=getattr(self.args, "max_new_tokens", 256),
            temperature=getattr(self.args, "temperature", 0.0),
            top_p=getattr(self.args, "top_p", 1.0),
            top_k=getattr(self.args, "top_k", 50),
            stop_tag=stop_tag,
        )

    def predict(self, sample: Sample) -> Dict[str, Optional[str]]:
        """Generate a prediction for a single sample.

        Subclasses should override this method.  The returned dictionary
        must contain a ``pred_label`` key with one of the values
        {``True``, ``False``, ``Uncertain``}.  Additional keys may be
        present.  If an error occurs, the default implementation
        returns an ``Uncertain`` label.
        """
        raise NotImplementedError


class BaselineMode(BaseMode):
    """Naive inference mode without explicit reasoning.

    The baseline mode constructs a simple prompt containing the
    premises and conclusion and asks the model to output just the
    answer in an ``<ANSWER>`` tag.  The prediction is obtained by
    reading the first recognised truth value from the generated text.
    If no label can be parsed, ``Uncertain`` is returned.
    """

    def predict(self, sample: Sample) -> Dict[str, Optional[str]]:
        prompt = self.prompt_builder.build(sample, mode="baseline")
        # Stop generation at </ANSWER> if present to limit output length
        raw = self._generate(prompt, stop_tag="</ANSWER>")
        # Attempt to find the answer between <ANSWER> and </ANSWER>
        label = "Uncertain"
        if "<ANSWER>" in raw:
            tail = raw.split("<ANSWER>", 1)[1]
            # If a closing tag exists, restrict to its contents
            if "</ANSWER>" in tail:
                ans = tail.split("</ANSWER>", 1)[0]
            else:
                ans = tail
            # Canonicalise the first token of the answer
            token = ans.strip().split()[0] if ans.strip() else ""
            token = token.strip().strip(".").capitalize()
            if token in {"True", "False", "Uncertain"}:
                label = token
        return {"raw": raw, "pred_label": label}


class CotMode(BaseMode):
    """Chain‑of‑thought reasoning mode.

    This mode instructs the model to think step by step before
    answering.  The final answer is expected to appear between
    ``<ANSWER>`` tags.  Parsing logic mirrors that of the baseline.
    """

    def predict(self, sample: Sample) -> Dict[str, Optional[str]]:
        prompt = self.prompt_builder.build(sample, mode="cot")
        raw = self._generate(prompt, stop_tag="</ANSWER>")
        label = "Uncertain"
        if "<ANSWER>" in raw:
            ans = raw.split("<ANSWER>", 1)[1]
            ans = ans.split("</ANSWER>", 1)[0] if "</ANSWER>" in ans else ans
            token = ans.strip().split()[0] if ans.strip() else ""
            token = token.strip().strip(".").capitalize()
            if token in {"True", "False", "Uncertain"}:
                label = token
        return {"raw": raw, "pred_label": label}


class ScratchpadMode(BaseMode):
    """Scratchpad mode with FOL translation.

    The scratchpad mode asks the model to first translate the premises
    and conclusion into FOL, then provide the answer.  The FOL theory
    and query are extracted for potential downstream use, but the
    prediction itself is derived directly from the model's declared
    answer.  If no explicit answer can be parsed, ``Uncertain`` is
    returned.  Errors in parsing the FOL blocks do not prevent a
    prediction but the ``fol_theory`` and ``fol_query`` fields may be
    omitted.
    """

    def predict(self, sample: Sample) -> Dict[str, Optional[str]]:
        prompt = self.prompt_builder.build(sample, mode="scratchpad")
        # Stop after FOL blocks and answer tag to avoid runaway generation
        raw = self._generate(prompt, stop_tag="</ANSWER>")
        result: Dict[str, Optional[str]] = {"raw": raw, "pred_label": "Uncertain"}
        # Extract answer
        if "<ANSWER>" in raw:
            ans = raw.split("<ANSWER>", 1)[1]
            ans = ans.split("</ANSWER>", 1)[0] if "</ANSWER>" in ans else ans
            token = ans.strip().split()[0] if ans.strip() else ""
            token = token.strip().strip(".").capitalize()
            if token in {"True", "False", "Uncertain"}:
                result["pred_label"] = token
        # Extract FOL blocks for logging
        try:
            fol_theory, fol_query = extract_fol_blocks(raw)
            result["fol_theory"] = fol_theory
            result["fol_query"] = fol_query
        except Exception:
            # Parsing errors are silently ignored; FOL fields remain absent
            pass
        return result


class LincMode(BaseMode):
    """LINC mode combining FOL translation with symbolic proving.

    This mode mirrors the scratchpad prompt but instead of trusting
    the model's answer it invokes an external prover on the FOL
    translation.  After generating the FOL theory and query the logic
    is normalised for Prover9, which is then asked to determine
    entailment and contradiction.  The resulting truth value is
    returned as the prediction.  If proving fails (e.g. due to time‑out
    or malformed FOL) the prediction defaults to ``Uncertain``.
    """

    def predict(self, sample: Sample) -> Dict[str, Optional[str]]:
        prompt = self.prompt_builder.build(sample, mode="linc")
        raw = self._generate(prompt, stop_tag="</ANSWER>")
        result: Dict[str, Optional[str]] = {"raw": raw, "pred_label": "Uncertain"}
        # Attempt to extract FOL and run prover
        try:
            fol_theory, fol_query = extract_fol_blocks(raw)
            norm_theory = normalize_for_prover9(fol_theory)
            norm_query = normalize_for_prover9(fol_query)
            label = prove_label(
                norm_theory,
                norm_query,
                prover9_path=getattr(self.args, "prover9_path", "prover9"),
                timeout=getattr(self.args, "prover9_timeout", 30),
            )
            result["pred_label"] = label
            result["fol_theory"] = norm_theory
            result["fol_query"] = norm_query
        except Exception:
            # On any error, leave the prediction as Uncertain
            pass
        return result
