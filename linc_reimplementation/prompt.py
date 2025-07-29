"""Utilities for constructing prompts for logical inference tasks.

This module defines two prompt builders.  The baseline
``PromptBuilder`` produces a simple natural language instruction asking
the model to decide whether a conclusion follows from a set of
premises.  The ``ScratchpadPromptBuilder`` extends this by including
first‑order logic (FOL) translations of the premises and conclusion
when available and by explicitly indicating where the model should
write its answer.

Separating prompt formatting from inference makes it easy to swap in
alternative templates for different modes (e.g. chain‑of‑thought or
neurosymbolic reasoning) in future.
"""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = [
    "INSTRUCTION",
    "PromptBuilder",
    "SCRATCHPAD_INSTRUCTION",
    "ScratchpadPromptBuilder",
]

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
    """Construct prompts from few‑shot demonstrations and a target example.

    Parameters
    ----------
    instruction : str
        Natural language instruction to prepend to every prompt.  If
        omitted, the default baseline instruction is used.
    """

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


# Scratchpad mode instruction.  This explicitly tells the model to
# translate the natural language statements into FOL before
# determining whether the conclusion follows.  The answer must be one
# of three canonical labels.
SCRATCHPAD_INSTRUCTION: str = (
    "The following is a first-order logic (FOL) problem. Translate each premise "
    "and the conclusion into FOL expressions, then evaluate whether the conclusion "
    "follows. Answer with one of: True, False, Uncertain."
)


class ScratchpadPromptBuilder(PromptBuilder):
    """Prompt builder for scratchpad mode using FOL translations.

    This builder produces prompts that include both the original
    natural language statements (prefixed with ``TEXT:``) and their
    first‑order logic translations (prefixed with ``FOL:``) when
    available.  For demonstration examples the label is provided after
    ``ANSWER:``, whereas for the target example the ``ANSWER:`` tag is
    left blank for the model to fill in.
    """

    def __init__(self, instruction: str = SCRATCHPAD_INSTRUCTION) -> None:
        super().__init__(instruction=instruction)

    def format_example(self, ex: Dict, with_label: bool) -> str:
        """Format a scratchpad example including FOL translations.

        The input dictionary is expected to contain the keys ``premises``
        and ``conclusion``.  It may also contain ``premises_FOL`` and
        ``conclusion_FOL`` which, if present and non‑empty, are used
        alongside their natural language counterparts.  When FOL
        translations are missing, the corresponding ``FOL:`` lines are
        omitted to avoid confusing the model.

        :param ex: Example dictionary with optional FOL fields.
        :param with_label: If ``True`` include the ground truth label; otherwise
            leave the answer blank.
        :returns: Formatted scratchpad string.
        """
        lines: List[str] = []
        # Split premises into individual statements
        premise_texts = ex.get("premises", "").split("\n") if ex.get("premises") else []
        premise_fol = ex.get("premises_FOL")
        fol_texts: List[str] = []
        if premise_fol:
            fol_texts = premise_fol.split("\n")
        # Pair each premise with its FOL translation if available
        for idx, text in enumerate(premise_texts):
            if text:
                lines.append(f"TEXT: {text}")
                if fol_texts:
                    # Use the corresponding FOL line if it exists; otherwise skip
                    fol_line = fol_texts[idx] if idx < len(fol_texts) else ""
                    if fol_line:
                        lines.append(f"FOL:  {fol_line}")
        # Append the conclusion and its FOL translation
        conc_text = ex.get("conclusion", "")
        if conc_text:
            lines.append(f"TEXT: {conc_text}")
        conc_fol = ex.get("conclusion_FOL")
        if conc_fol:
            lines.append(f"FOL:  {conc_fol}")
        # Add the answer line
        if with_label:
            lines.append(f"ANSWER: {ex['label']}")
            lines.append("")  # extra newline after demo
        else:
            lines.append("ANSWER:")
        return "\n".join(lines)

    def build(self, demos: Optional[List[Dict]], target: Dict) -> str:
        """Assemble the scratchpad prompt for the language model.

        The prompt consists of an instruction, zero or more scratchpad
        formatted demonstrations with their answers, followed by the
        target example with a blank answer field.

        :param demos: Demonstration examples or ``None``.
        :param target: Target example requiring a prediction.
        :returns: A complete prompt string.
        """
        parts: List[str] = [self.instruction, ""]
        if demos:
            for d in demos:
                parts.append(self.format_example(d, with_label=True))
        parts.append(self.format_example(target, with_label=False))
        return "\n".join(parts)