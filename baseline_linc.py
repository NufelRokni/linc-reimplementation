"""Legacy entry point for the FOLIO evaluator.

This module exists solely to preserve backwards compatibility with
scripts or workflows that invoke ``baseline_linc.py`` directly.  It
delegates execution to the ``main`` function defined in
``linc_reimplementation.main``.
"""

from __future__ import annotations

from linc_reimplementation.main import main


if __name__ == "__main__":  # pragma: no cover
    main()