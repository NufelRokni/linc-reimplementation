"""Interface to the Prover9 first‑order logic prover.

This module wraps the invocation of an external Prover9 binary.  Given a
theory and a query it constructs an input file, calls Prover9 via
``subprocess.run`` and parses the resulting output to decide whether the
query is provable, its negation is provable, or neither (in which case
the label is ``Uncertain``).  Timeouts are handled gracefully.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Dict


def _wrap_prover9_input(theory: str, query: str) -> str:
    """Wrap a theory and query into a Prover9 input format."""
    return f"""formulas(assumptions).
{theory}
end_of_list.

formulas(goals).
{query}
end_of_list.
"""


def call_prover9(theory: str, query: str, prover9_path: str = "prover9", timeout: int = 10) -> str:
    """Call Prover9 with the given theory and query and return its output.

    :param theory: Normalised theory (multiple sentences separated by newlines).
    :param query: Normalised query (single sentence ending with a period).
    :param prover9_path: Path to the Prover9 binary.  If the binary is
        in your PATH this can just be ``"prover9"``.
    :param timeout: Maximum number of seconds to allow Prover9 to run.
    :returns: Combined stdout and stderr from Prover9.
    """
    content = _wrap_prover9_input(theory, query)
    with tempfile.TemporaryDirectory() as td:
        infile = os.path.join(td, "input.in")
        with open(infile, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            r = subprocess.run(
                [prover9_path, "-f", infile],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout"
        return (r.stdout or "") + "\n" + (r.stderr or "")


def parse_prover9_output(out: str) -> Dict[str, bool]:
    """Parse Prover9 output and return a dictionary of result flags."""
    return {
        "proved": bool(re.search(r"THEOREM PROVED", out)),
        "timeout": "User CPU time limit exceeded" in out or "timeout" in out.lower(),
    }


def negate(formula: str) -> str:
    """Return the negation of a single FOL sentence.

    :param formula: Sentence ending with a period.
    :returns: Negated sentence ending with a period.
    """
    f = formula.strip()
    if not f.endswith("."):
        f += "."
    inner = f[:-1].strip()
    return f"-({inner})."


def prove_label(theory: str, query: str, prover9_path: str = "prover9", timeout: int = 10) -> str:
    """Determine whether a query follows from a theory using Prover9.

    Prove the query directly.  If that fails, prove its negation.  If
    neither is provable within the timeout, return ``"Uncertain"``.

    :param theory: Normalised FOL theory.
    :param query: Normalised FOL query.
    :param prover9_path: Path to Prover9 binary.
    :param timeout: Time limit in seconds for each prover call.
    :returns: ``"True"``, ``"False"`` or ``"Uncertain"``.
    """
    out1 = call_prover9(theory, query, prover9_path, timeout)
    res1 = parse_prover9_output(out1)
    if res1["proved"]:
        return "True"
    # Try negated query
    out2 = call_prover9(theory, negate(query), prover9_path, timeout)
    res2 = parse_prover9_output(out2)
    if res2["proved"]:
        return "False"
    return "Uncertain"