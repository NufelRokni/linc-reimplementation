import argparse

def build_arg_parser():
    p = argparse.ArgumentParser(description="LINC-style evaluation with scratchpad and Prover9")
    p.add_argument("--model", type=str, default="bigcode/starcoderplus",
                   help="HF model id (e.g., bigcode/starcoderplus, mistralai/Mistral-7B-v0.1)")
    p.add_argument("--dataset", type=str, default="folio", choices=["folio","proofwriter"])
    p.add_argument("--split", type=str, default="validation", choices=["train","validation","test","val","dev"])
    p.add_argument("--folio_source", type=str, default="tasksource", choices=["tasksource","yale"],
                   help="Which HF dataset source to use for FOLIO")

    p.add_argument("--method", type=str, default="linc",
                   choices=["naive","scratchpad","cot","linc"],
                   help="Evaluation method")
    p.add_argument("--shots", type=int, default=1, help="Few-shot examples to include")
    p.add_argument("--n_samples", type=int, default=5, help="Number of generations per example for majority voting")

    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)

    p.add_argument("--precision", type=str, default="fp16", choices=["fp32","fp16","bf16"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])

    p.add_argument("--prover9_path", type=str, default="prover9",
                   help="Path to prover9 binary (if in PATH, just 'prover9')")
    p.add_argument("--prover9_timeout", type=int, default=10)

    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p
