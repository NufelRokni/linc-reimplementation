# LINC-Style Neurosymbolic Reasoning (Scratchpad + Prover) — Ready-to-Run

This project reimplements the core ideas from **LINC** with a robust scratchpad pipeline and an external **Prover9** step, updated for modern libraries. It targets **StarCoder+** and **Mistral-7B-v0.1** on CUDA and evaluates on **FOLIO**. A ProofWriter scaffold is included but not yet implemented.

## Key Features
- **Methods**: `naive`, `scratchpad`, `cot`, `linc` (neurosymbolic; uses Prover9).
- **Models**: `bigcode/starcoderplus` and `mistralai/Mistral-7B-v0.1`.
- **Dataset**: FOLIO via `datasets` (`tasksource/folio`). ProofWriter stub present.
- **Outputs**: Detailed traces and metrics under `outputs/run_*` (JSON + formatted text). Includes accuracy **by class** (True/False/Uncertain).
- **Scratchpad safety**: Tagged prompts `<REASONING>...</REASONING>`, `<FOL_THEORY>...</FOL_THEORY>`, `<FOL_QUERY>...</FOL_QUERY>`; strict parser and sanitization; early stop on `</FOL_QUERY>`.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU**: Set `CUDA_VISIBLE_DEVICES=0` (or as appropriate). Use `--precision fp16` to save VRAM.

### Example commands
**Scratchpad baseline** (10 samples, majority vote):
```bash
python linc_eval.py --model bigcode/starcoderplus --dataset folio \
  --method scratchpad --shots 1 --n_samples 10 --precision fp16
```

**Full LINC (LLM + Prover9)**:
```bash
python linc_eval.py --model mistralai/Mistral-7B-v0.1 --dataset folio \
  --method linc --shots 5 --n_samples 5 --precision fp16 --prover9_path prover9
```

## Prover9 install
- **Ubuntu/Debian**: `sudo apt-get install prover9` (or install from source)
- **macOS (Homebrew)**: `brew install prover9`
- Ensure the binary name/path is reachable (e.g., `prover9`) or pass `--prover9_path /path/to/prover9`

## Notes
- FOLIO loader uses `tasksource/folio` (free). If you have access to `yale-nlp/FOLIO`, you can switch by `--folio_source yale`.
- XOR (`⊕`) is expanded into `(A & -B) | (-A & B)` before sending to Prover9.
- If Prover9 fails or times out, the sample is marked `Uncertain` for that attempt (never crashes evaluation).

## Outputs
```
outputs/run_YYYYMMDD_HHMMSS/
  predictions.jsonl    # one record per example (all attempts)
  metrics.json         # overall + per-class
  traces.txt           # human-friendly, decorated traces
```
