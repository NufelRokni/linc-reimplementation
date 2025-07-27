"""Utilities for constructing prompts for logical inference tasks.

The prompt construction logic is encapsulated within the
``PromptBuilder`` class.  This allows a caller to specify an
instruction (the task description) and supply zero or more
demonstrations (few‑shot examples).  The builder will assemble a
complete prompt string that can be passed directly into a language
model.  Separating prompt formatting from inference makes it easy to
swap in alternative templates for different modes (e.g. scratchpad or
chain‑of‑thought) in future.
"""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = [
    "INSTRUCTION",
    "PromptBuilder",
    "INSTRUCTION_SCRATCHPAD",
    "ScratchpadPromptBuilder",
]

# Scratchpad instruction: this describes the task for the FOL scratchpad
# setting.  It mirrors the original LINC instructions used for scratchpad mode
# where the model is asked to translate premises and conclusions into FOL
# before answering.
INSTRUCTION_SCRATCHPAD: str = (
    "The following is a first-order logic (FOL) problem.\n"
    "The problem is to determine whether the conclusion follows from the premises.\n"
    "The premises are given in the form of a set of first-order logic sentences.\n"
    "The conclusion is given in the form of a single first-order logic sentence.\n"
    "The task is to translate each of the premises and conclusions into FOL expressions, "
    "and then to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
)



# Default instruction given to the language model.  This mirrors the
# baseline prompt used in the original LINC baseline implementation
# which asks the model to determine whether the conclusion logically
# follows from the premises and to answer with one of three
# designated labels.
INSTRUCTION: str = (
    "Determine if the conclusion follows logically from the premises. "
    "Answer with exactly one of: True, False, or Uncertain."
)


class PromptBuilder:
    """Construct prompts from few‑shot demonstrations and a target example."""

    def __init__(self, instruction: str = INSTRUCTION) -> None:
        self.instruction = instruction

    def format_example(self, ex: Dict, with_label: bool) -> str:
        """Format a single example for inclusion in a prompt.

        :param ex: Dictionary containing ``premises``, ``conclusion`` and ``label``.
        :param with_label: If ``True`` include the ground truth label, otherwise
            leave the label field blank for the model to fill in.
        :returns: A formatted string representation of the example.
        """
        base = f"Premises: {ex['premises']}\n" f"Conclusion: {ex['conclusion']}\n"
        if with_label:
            base += f"Label: {ex['label']}\n\n"
        else:
            base += "Label: "  # model will complete
        return base

    def build(self, demos: Optional[List[Dict]], target: Dict) -> str:
        """Assemble the full prompt for the language model.

        :param demos: A list of demonstration examples or ``None`` if zero‑shot.
        :param target: The target example for which a prediction is required.
        :returns: A complete prompt string.
        """
        parts: List[str] = [self.instruction, ""]
        if demos:
            for d in demos:
                parts.append(self.format_example(d, with_label=True))
        parts.append(self.format_example(target, with_label=False))
        return "\n".join(parts)


class ScratchpadPromptBuilder:
    """Construct prompts for scratchpad mode.

    In scratchpad mode each premise and its first-order logic (FOL) translation
    are presented separately, followed by the conclusion and its FOL.  The
    model is then expected to produce an answer after an explicit "ANSWER:"
    prompt.  Few-shot demonstrations include the ground truth label after
    "ANSWER:" whereas the target example leaves the answer blank for the model
    to fill in.
    """

    def __init__(self, instruction: str = INSTRUCTION_SCRATCHPAD) -> None:
        self.instruction = instruction

    def format_example(self, ex: Dict, with_label: bool) -> str:
        """Format a single scratchpad example.

        :param ex: Dictionary containing ``premises`` (list of str),
                   ``premises_fol`` (list of str), ``conclusion`` (str),
                   ``conclusion_fol`` (str) and ``label`` (str).
        :param with_label: If ``True`` include the ground truth label after
                           ``ANSWER:``, otherwise leave it blank for the model.
        :returns: A formatted string representation suitable for inclusion in a
                  scratchpad prompt.
        """
        parts: List[str] = []
        # Show each premise and its FOL translation on separate lines
        premises = ex.get("premises", [])
        premises_fol = ex.get("premises_fol", [])
        # Pad FOL list if lengths mismatch
        if len(premises_fol) < len(premises):
            premises_fol = premises_fol + ["" for _ in range(len(premises) - len(premises_fol))]
        for prem, fol in zip(premises, premises_fol):
            parts.append(f"TEXT:\t{prem.strip()}")
            parts.append(f"FOL:\t{fol.strip()}")
        # Add the conclusion and its FOL translation
        concl = ex.get("conclusion", "").strip()
        concl_fol = ex.get("conclusion_fol", "").strip()
        parts.append(f"TEXT:\t{concl}")
        parts.append(f"FOL:\t{concl_fol}")
        # Append the answer
        if with_label:
            parts.append(f"ANSWER:\t{ex.get('label', '').strip()}")
        else:
            parts.append("ANSWER:\t")
        return "\n".join(parts) + "\n"

    def build(self, demos: Optional[List[Dict]], target: Dict) -> str:
        """Assemble the full scratchpad prompt.

        :param demos: List of demonstration examples or ``None`` for zero‑shot.
        :param target: Target example for which the model must produce an answer.
        :returns: Complete prompt string combining instructions, demonstrations and
                  the target example.
        """
        parts: List[str] = [self.instruction, ""]
        if demos:
            for d in demos:
                parts.append(self.format_example(d, with_label=True))
        parts.append(self.format_example(target, with_label=False))
        return "\n".join(parts)
