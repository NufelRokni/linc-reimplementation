import tempfile, subprocess, os, re

def _wrap_prover9_input(theory: str, query: str) -> str:
    return f"""formulas(assumptions).
{theory}
end_of_list.

formulas(goals).
{query}
end_of_list.
"""

def call_prover9(theory: str, query: str, prover9_path: str = "prover9", timeout: int = 10) -> str:
    content = _wrap_prover9_input(theory, query)
    with tempfile.TemporaryDirectory() as td:
        infile = os.path.join(td, "input.in")
        with open(infile, "w") as f:
            f.write(content)
        try:
            r = subprocess.run([prover9_path, "-f", infile], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout"
    out = r.stdout + "\n" + r.stderr
    return out

def parse_prover9_output(out: str) -> dict:
    d = {
        "proved": bool(re.search(r"THEOREM PROVED", out)),
        "timeout": "User CPU time limit exceeded" in out or "timeout" in out.lower(),
    }
    return d

def negate(formula: str) -> str:
    f = formula.strip()
    if not f.endswith("."):
        f += "."
    inner = f[:-1].strip()
    return f"-({inner})."

def prove_label(theory: str, query: str, prover9_path: str = "prover9", timeout: int = 10) -> str:
    # Try prove(query)
    out1 = call_prover9(theory, query, prover9_path, timeout)
    res1 = parse_prover9_output(out1)
    if res1["proved"]:
        return "True"
    # Try prove(negated query)
    out2 = call_prover9(theory, negate(query), prover9_path, timeout)
    res2 = parse_prover9_output(out2)
    if res2["proved"]:
        return "False"
    # Otherwise unknown
    return "Uncertain"
