"""Top level package for the re‑implemented FOLIO baseline and scratchpad modes.

This package contains all of the modules necessary to evaluate logical
reasoning on the FOLIO dataset.  The code has been deliberately
structured to separate concerns such as argument parsing, dataset
normalisation, prompt construction and inference so that new modes or
datasets can be added with minimal changes.  Two modes are currently
available:

* **baseline** – replicate the original LINC baseline prompting and
  prediction extraction.  Each example is reduced to its premises,
  conclusion and label, and the model is asked directly whether the
  conclusion follows.

* **scratchpad** – translate premises and conclusion into first‑order
  logic (FOL) where available and ask the model to reason over the
  translation.  Demonstrations include both the natural language
  statements and their FOL equivalents followed by an explicit
  ``ANSWER:`` tag for the label.

By keeping these concerns separate the implementation remains easy to
extend.  Additional modes (e.g. chain‑of‑thought or neurosymbolic
reasoning) can be introduced by adding new loader and prompt builder
classes and wiring them up in ``main.py``.
"""

from . import data_loading  # noqa: F401
from . import prompt  # noqa: F401
from . import inference  # noqa: F401
from . import arguments  # noqa: F401
from . import model_utils  # noqa: F401
from . import main  # noqa: F401

__all__ = [
    "data_loading",
    "prompt",
    "inference",
    "arguments",
    "model_utils",
    "main",
]