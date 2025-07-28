"""Prompt construction utilities.

This module defines a small class responsible for assembling input
prompts for the various inference modes supported by this project.
Each mode (baseline, chain‑of‑thought, scratchpad and LINC) has a
slightly different template guiding the language model to produce
useful intermediate reasoning.  The ``PromptBuilder`` hides these
differences behind a single ``build`` method.

The default templates draw inspiration from the original LINC
implementation but have been simplified for clarity.  Feel free to
customise the instruction strings to suit your own needs.
"""

from __future__ import annotations

from typing import List
from textwrap import dedent

from .datasets import Sample

__all__ = ["PromptBuilder"]


class PromptBuilder:
    """Construct prompts for logical inference tasks.

    A ``PromptBuilder`` holds the instruction and guidance strings for
    different modes of inference.  It can be reused across many
    examples.  At construction time you may optionally specify the
    number of few‑shot demonstrations to include in each prompt, as
    well as the dataset name (unused in the default implementation).
    """

    def __init__(self, shots: int = 0, dataset: str = "") -> None:
        self.shots = shots
        self.dataset = dataset
        # Base instruction shared by all modes
        self._base = dedent(
            """
            You are a careful logician. Decide whether the CONCLUSION logically
            follows from the PREMISES.  You must answer with exactly one of
            the three truth values: True, False or Uncertain.
            """
        ).strip()
        # Guidance for each inference mode
        self._naive_guide = dedent(
            """
            Output only <ANSWER>True|False|Uncertain</ANSWER>.
            """
        ).strip()
        self._cot_guide = dedent(
            """
            Think step by step.  After your reasoning, conclude with
            <ANSWER>True|False|Uncertain</ANSWER>.
            """
        ).strip()
        self._scratchpad_guide = dedent(
            """
            You may think in natural language, translating the problem into First‑Order
            Logic (FOL).  After reasoning, produce exactly two blocks and then an answer:

            <FOL_THEORY>
            <one or more FOL sentences, each ending with a period>
            </FOL_THEORY>

            <FOL_QUERY>
            <a single FOL sentence ending with a period>
            </FOL_QUERY>

            Finally, state your prediction in <ANSWER>True|False|Uncertain</ANSWER>.
            Do not put natural language inside the FOL blocks; use symbols such as all, exists,
            ->, &, |, -, <->.
            """
        ).strip()
        # LINC uses the same scratchpad guide since the prover operates on the FOL.
        self._linc_guide = self._scratchpad_guide

    def _format_premises(self, premises: List[str]) -> str:
        """Format a list of premise sentences for inclusion in a prompt.

        Each sentence is prefaced with a dash to aid readability.  An
        empty list yields an empty string.
        """
        if not premises:
            return ""
        return "\n".join(f"- {p.strip()}" for p in premises)

    def _build_body(self, sample: Sample) -> str:
        """Assemble the <TASK> section of the prompt.

        This includes the premises, conclusion and closes the task tag.
        """
        lines = ["<TASK>"]
        lines.append("<PREMISES>")
        lines.append(self._format_premises(sample.premises))
        lines.append("</PREMISES>")
        lines.append("<CONCLUSION> " + sample.conclusion.strip() + " </CONCLUSION>")
        lines.append("</TASK>")
        return "\n".join(lines)

    def build(self, sample: Sample, mode: str) -> str:
        """Construct a complete prompt for a given sample and mode.

        :param sample: The logical inference example to prompt about.
        :param mode: One of ``baseline``, ``cot``, ``scratchpad`` or ``linc``.
        :returns: A single string to feed into a language model.
        :raises ValueError: If an unknown mode is supplied.
        """
        mode = mode.lower()
        # Select guide string
        if mode == "baseline":
            guide = self._naive_guide
        elif mode == "cot":
            guide = self._cot_guide
        elif mode == "scratchpad":
            guide = self._scratchpad_guide
        elif mode == "linc":
            guide = self._linc_guide
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # Few‑shot section (currently unused): placeholder for potential examples.
        shots_section = ""
        # Build full prompt
        parts = [self._base, guide, shots_section, self._build_body(sample)]
        # Ensure exactly one blank line separates major sections
        prompt = "\n\n".join(part for part in parts if part).strip() + "\n"
        return prompt
