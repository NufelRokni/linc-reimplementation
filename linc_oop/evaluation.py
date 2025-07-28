"""Evaluation helpers for LINC OOP.

This module contains simple utilities to aggregate predictions across
multiple sampling runs, compute basic metrics and write out
predictions and traces.  The structures here are deliberately
lightweight; more sophisticated evaluation (such as confidence
intervals or per‑label breakdowns) can be layered on top.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .datasets import Sample

__all__ = ["MajorityVoter", "Metrics", "init_run_dir", "save_trace"]


class MajorityVoter:
    """Aggregate labels via majority vote.

    Given a sequence of predictions this helper returns the label that
    occurs most frequently.  Ties are broken arbitrarily by taking
    the first element of the most common subset.
    """

    def vote(self, labels: Iterable[str]) -> str:
        counts = Counter(labels)
        if not counts:
            return "Uncertain"
        return counts.most_common(1)[0][0]


class Metrics:
    """Track simple evaluation metrics.

    This class records the number of correct predictions and the
    total number of examples processed.  Additional metrics such as
    per‑label precision/recall could be added easily.
    """

    def __init__(self) -> None:
        self.total = 0
        self.correct = 0

    def add(self, gold: str, pred: str) -> None:
        self.total += 1
        if str(gold) == str(pred):
            self.correct += 1

    def summary(self) -> Dict[str, float]:
        acc = self.correct / self.total if self.total else 0.0
        return {"accuracy": acc, "n": self.total}


def init_run_dir(base_dir: str) -> str:
    """Create a unique output directory for this run.

    A timestamp is appended to the base directory name to avoid
    collisions.  The directory is created if it does not already
    exist.

    :param base_dir: Parent directory under which the run directory will be created.
    :returns: The path to the newly created run directory.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_trace(
    fhandle,
    sample: Sample,
    attempts: List[Dict[str, Optional[str]]],
    final_label: str,
    correct: bool,
) -> None:
    """Write a human readable trace of a prediction to disk.

    Each trace shows the premises, conclusion, the generated output
    for each sample, the predicted labels and whether the final
    prediction matched the gold label.  This is useful for manual
    inspection and debugging.

    :param fhandle: Open file handle to which the trace will be appended.
    :param sample: The input example.
    :param attempts: List of dictionaries returned by a mode's ``predict`` method.
    :param final_label: The majority voted label.
    :param correct: Whether ``final_label`` equals the gold label.
    """
    lines = []
    lines.append(f"ID: {sample.id}")
    lines.append(f"Gold: {sample.label}")
    lines.append(f"Premises:")
    for p in sample.premises:
        lines.append(f"  {p}")
    lines.append(f"Conclusion: {sample.conclusion}")
    for i, attempt in enumerate(attempts):
        raw = attempt.get("raw", "").replace("\n", " ")
        pred = attempt.get("pred_label", "Uncertain")
        lines.append(f"Attempt {i+1}: pred={pred} raw={raw[:200]}")
    lines.append(f"Final prediction: {final_label}  Correct={correct}")
    lines.append("".join(["-" * 40, "\n"]))
    fhandle.write("\n".join(lines) + "\n")
