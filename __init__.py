"""Top level package for the FOLIO baseline implementation.

This package re-implements the baseline evaluation logic from the
original LINC repository in a more modular fashion. It is designed to
mirror the structure of the upstream codebase while remaining simple
enough to extend.  In particular, dataset loading, prompt
construction, inference and argument parsing are all separated into
independent modules.  As a result, adding support for new models,
datasets or evaluation modes can be achieved by implementing a small
interface rather than modifying a monolithic script.

Currently only the FOLIO dataset and baseline inference mode are
supported, but the code has been organised to make it straightforward
to incorporate additional tasks (e.g. ProofWriter) and alternative
reasoning strategies (e.g. scratch‑pad, chain‑of‑thought or
neurosymbolic methods) in the future.
"""

from . import data_loading  # noqa: F401
from . import prompt        # noqa: F401
from . import inference     # noqa: F401
from . import arguments     # noqa: F401
from . import model_utils   # noqa: F401
from . import main          # noqa: F401

__all__ = [
    "data_loading",
    "prompt",
    "inference",
    "arguments",
    "model_utils",
    "main",
]