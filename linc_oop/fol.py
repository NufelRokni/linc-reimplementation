"""Utilities for handling first‑order logic (FOL) embedded in model outputs.

Certain inference modes (scratchpad and LINC) ask the language model to
produce structured FOL inside `<FOL_THEORY>` and `<FOL_QUERY>` tags.
This module defines helpers to extract those blocks, sanitise the
content and prepare it for consumption by Prover9.  The logic here is
adapted from the original LINC implementation but presented as
standalone functions without any hidden state.
"""

from __future__ import annotations

import re
from typing import Tuple, List

# Regular expressions to capture theory and query blocks.
FOL_THEORY_RE = re.compile(r"<FOL_THEORY>\s*(.*?)\s*</FOL_THEORY>", re.DOTALL | re.IGNORECASE)
FOL_QUERY_RE = re.compile(r"<FOL_QUERY>\s*(.*?)\s*</FOL_QUERY>", re.DOTALL | re.IGNORECASE)

# Characters permitted in sanitised FOL.  Any others are stripped.
ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_(),.;!&|<>-=\n\r\t~'"
)


def extract_fol_blocks(text: str) -> Tuple[str, str]:
    """Extract and sanitise the theory and query blocks from a string.

    :param text: Raw model output.
    :returns: Tuple of (theory, query) strings, each ending with a period.
    :raises ValueError: if either tag is missing or malformed.
    """
    m1 = FOL_THEORY_RE.search(text)
    m2 = FOL_QUERY_RE.search(text)
    if not m1 or not m2:
        raise ValueError("Missing FOL_THEORY or FOL_QUERY tags")
    theory = _sanitize(m1.group(1))
    query = _sanitize(m2.group(1))
    # Ensure trailing periods for prover safety
    if theory and not theory.strip().endswith("."):
        theory = theory.strip() + "."
    if query and not query.strip().endswith("."):
        query = query.strip() + "."
    return theory, query


def _sanitize(s: str) -> str:
    """Remove stray markdown and invalid characters from a FOL block.

    Only characters appearing in ``ALLOWED`` or known unicode logical
    symbols are kept.  Backticks and zero‑width spaces are dropped.
    """
    # Strip code fences and zero width joiners
    s = s.replace("```", "").replace("\u200b", "")
    return "".join(ch for ch in s if ch in ALLOWED or ch in "∀∃↔→¬∧∨⊕")


def _top_level_split(expr: str, sep: str) -> List[str]:
    """Split an expression on a separator only at top‑level parentheses.

    This helper allows us to expand XOR without splitting inside nested
    parentheses.  It is not a full parser but suffices for simple
    expressions produced by the model.
    """
    parts: List[str] = []
    depth = 0
    buff: List[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        # When at top level and the separator matches, split
        if depth == 0 and expr.startswith(sep, i):
            parts.append("".join(buff).strip())
            buff = []
            i += len(sep)
            continue
        buff.append(ch)
        i += 1
    parts.append("".join(buff).strip())
    return parts


def expand_xor(expr: str) -> str:
    """Expand top‑level XOR (⊕) occurrences into conjunction/disjunction form.

    The rule used is ``A ⊕ B = (A & -B) | (-A & B)``.  The expansion is
    applied repeatedly until no top‑level XORs remain.
    """
    while True:
        parts = _top_level_split(expr, "⊕")
        if len(parts) == 1:
            return expr
        a = parts[0]
        for b in parts[1:]:
            a = f"( ({a}) & -({b}) ) | ( -({a}) & ({b}) )"
        expr = a


def replace_unicode_symbols(s: str) -> str:
    """Replace unicode logical symbols with Prover9 friendly ASCII."""
    s = s.replace("∀", "all ")
    s = s.replace("∃", "exists ")
    s = s.replace("↔", "<->")
    s = s.replace("→", "->")
    s = s.replace("¬", "-")
    s = s.replace("∧", "&")
    s = s.replace("∨", "|")
    return s


def normalize_for_prover9(block: str) -> str:
    """Normalise a FOL theory or query for Prover9.

    The normalisation performs the following steps per sentence:

    * Expand top‑level XOR (⊕) operators.
    * Replace unicode logical symbols with ASCII equivalents.
    * Collapse multiple whitespace characters.
    * Ensure each sentence ends with a single period.

    :param block: Multi‑sentence theory or query separated by periods.
    :returns: Normalised block suitable for Prover9.
    """
    sentences = [t.strip() for t in block.split(".") if t.strip()]
    out: List[str] = []
    for sent in sentences:
        sent = expand_xor(sent)
        sent = replace_unicode_symbols(sent)
        # Collapse runs of whitespace
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent.endswith("."):
            sent += "."
        out.append(sent)
    return "\n".join(out)