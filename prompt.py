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

__all__ = ["INSTRUCTION", "PromptBuilder"]


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
