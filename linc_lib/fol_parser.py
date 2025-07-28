import re

FOL_THEORY_RE = re.compile(r"<FOL_THEORY>\s*(.*?)\s*</FOL_THEORY>", re.DOTALL | re.IGNORECASE)
FOL_QUERY_RE  = re.compile(r"<FOL_QUERY>\s*(.*?)\s*</FOL_QUERY>",   re.DOTALL | re.IGNORECASE)

def extract_fol_blocks(text: str):
    m1 = FOL_THEORY_RE.search(text)
    m2 = FOL_QUERY_RE.search(text)
    if not (m1 and m2):
        raise ValueError("Missing FOL tags")
    theory = sanitize(m1.group(1))
    query = sanitize(m2.group(1))
    if not theory.strip().endswith("."):
        theory = theory.strip() + "."
    if not query.strip().endswith("."):
        query = query.strip() + "."
    return theory, query

ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_(),.;!&|<>- =\n\r\t~'")

def sanitize(s: str) -> str:
    s = s.replace("```","").replace("\u200b","")
    # Remove stray markdown and keep only allowed characters
    s2 = "".join(ch for ch in s if ch in ALLOWED or ch in "∀∃↔→¬∧∨⊕")
    return s2

def _top_level_split(expr: str, sep: str) -> list:
    parts = []
    depth = 0
    buff = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth-1)
        # handle multi-char sep in future if needed
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
    # Expand top-level occurrences of A ⊕ B into (A & -B) | (-A & B)
    # Repeat until no ⊕ at top-level.
    while True:
        parts = _top_level_split(expr, "⊕")
        if len(parts) == 1:
            return expr
        # rebuild via pairwise expansion left-associative
        a = parts[0]
        for b in parts[1:]:
            a = f"( ({a}) & -({b}) ) | ( -({a}) & ({b}) )"
        expr = a

def replace_unicode_symbols(s: str) -> str:
    s = s.replace("∀", "all ")
    s = s.replace("∃", "exists ")
    s = s.replace("↔", "<->")
    s = s.replace("→", "->")
    s = s.replace("¬", "-")
    s = s.replace("∧", "&")
    s = s.replace("∨", "|")
    return s

def normalize_for_prover9(block: str) -> str:
    # split into sentences by '.' keep only non-empty
    sentences = [t.strip() for t in block.split(".") if t.strip()]
    out = []
    for sent in sentences:
        sent = expand_xor(sent)
        sent = replace_unicode_symbols(sent)
        # collapse multiple spaces
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent.endswith("."):
            sent += "."
        out.append(sent)
    return "\n".join(out)
