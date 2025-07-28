# LINC‑OOP: A clean, object oriented re‑implementation of LINC

This project contains a ground‑up re‑implementation of the core ideas
from the [LINC](https://arxiv.org/abs/2310.15164) paper.  The goal of
this repository is to provide a minimal, easy to understand and
extensible codebase for logical inference with large language models
and an optional first‑order logic prover.  Unlike previous
implementations, the code here is organised around a handful of small
classes so that adding new datasets, inference modes or models does
not require touching unrelated parts of the system.

At a high level the system works as follows:

* A **dataset loader** reads examples from FOLIO or ProofWriter and
  produces simple `Sample` objects.  Each sample contains a list of
  premises, a conclusion and a gold label.
* A **prompt builder** constructs a natural language prompt given a
  sample and an optional number of few‑shot demonstrations.  The
  prompt format varies depending on the chosen inference mode
  (baseline, chain‑of‑thought, scratchpad or full neurosymbolic).
* A **language model wrapper** loads a HuggingFace model and exposes
  a `generate` method that hides away device and dtype handling.
* A **mode** object encapsulates the logic for predicting a label from
  a sample.  The baseline and chain‑of‑thought modes simply parse
  labels from the model output.  The scratchpad and LINC modes also
  extract first‑order logic from the generated scratchpad.  The LINC
  mode calls out to an external Prover9 binary to determine truth of
  the conclusion under the generated theory.
* An **evaluation loop** runs over the dataset, obtains multiple
  generations per example if requested, performs majority voting and
  records metrics and per‑example traces.

## Quick start

Install the required packages into a fresh virtual environment (see
`requirements.txt` for exact versions):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the evaluator on 10 validation examples from the FOLIO
dataset with the scratchpad mode:

```bash
python -m linc_oop.main --model bigcode/starcoderplus \
  --task folio --mode scratchpad --n_samples 3 --limit 10
```

To evaluate on ProofWriter instead, specify `--task proofwriter`.  To
use the neurosymbolic prover, select `--mode linc` and ensure
Prover9 is installed and available on your PATH (see the original
paper for installation instructions).

This repository is intentionally simple; if you wish to extend it
further please read through the well commented source files under
`linc_oop/`.