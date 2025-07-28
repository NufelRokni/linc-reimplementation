from typing import List
from textwrap import dedent
from .datasets import Sample

BASE = dedent("""\
You are a careful logician. Follow the format exactly.
For all tasks, decide whether the CONCLUSION logically follows from the PREMISES.
Answer with one of: True, False, Uncertain.
""").strip()

SCRATCHPAD_GUIDE = dedent("""\
You may think step by step in <REASONING>...</REASONING>, translating to First-Order Logic (FOL).
After reasoning, produce exactly two blocks and then an answer:

<FOL_THEORY>
<one or more FOL sentences, each ending with a period>
</FOL_THEORY>

<FOL_QUERY>
<a single FOL sentence ending with a period>
</FOL_QUERY>

<ANSWER>True|False|Uncertain</ANSWER>

Do NOT put natural language inside FOL blocks. Use symbols: all, exists, ->, &, |, -, <->. Avoid unicode if possible.
""").strip()

COT_GUIDE = dedent("""\
Think step by step. Conclude with <ANSWER>True|False|Uncertain</ANSWER>.
""").strip()

NAIVE_GUIDE = dedent("""\
Output only <ANSWER>True|False|Uncertain</ANSWER>.
""").strip()

def few_shot_examples(k: int, dataset: str) -> str:
    # Minimal, diverse examples; can be replaced with your curated set
    shots = [
        dedent("""\
        <EXAMPLE>
        <PREMISES>
        - All A are B.
        - All B are C.
        - A(a).
        </PREMISES>
        <CONCLUSION> C(a). </CONCLUSION>
        <EVALUATE>
        <REASONING> If all A are B and all B are C, then all A are C; since A(a), C(a). </REASONING>
        <FOL_THEORY>
        all x (A(x) -> B(x)).
        all x (B(x) -> C(x)).
        A(a).
        </FOL_THEORY>
        <FOL_QUERY>
        C(a).
        </FOL_QUERY>
        <ANSWER>True</ANSWER>
        </EVALUATE>
        </EXAMPLE>
        """).strip(),
        dedent("""\
        <EXAMPLE>
        <PREMISES>
        - No cats are dogs.
        - Cat(milo).
        </PREMISES>
        <CONCLUSION> Dog(milo). </CONCLUSION>
        <EVALUATE>
        <REASONING> From "No cats are dogs" and Cat(milo), Dog(milo) is contradicted. </REASONING>
        <FOL_THEORY>
        all x (Cat(x) -> -Dog(x)).
        Cat(milo).
        </FOL_THEORY>
        <FOL_QUERY>
        Dog(milo).
        </FOL_QUERY>
        <ANSWER>False</ANSWER>
        </EVALUATE>
        </EXAMPLE>
        """).strip(),
        dedent("""\
        <EXAMPLE>
        <PREMISES>
        - Some person is a runner.
        - All runners are athletes.
        </PREMISES>
        <CONCLUSION> There exists an athlete. </CONCLUSION>
        <EVALUATE>
        <REASONING> Existential premise + universal rule implies an athlete exists, but the individual might not be named. </REASONING>
        <FOL_THEORY>
        exists x (Runner(x)).
        all x (Runner(x) -> Athlete(x)).
        </FOL_THEORY>
        <FOL_QUERY>
        exists x (Athlete(x)).
        </FOL_QUERY>
        <ANSWER>True</ANSWER>
        </EVALUATE>
        </EXAMPLE>
        """).strip(),
        dedent("""\
        <EXAMPLE>
        <PREMISES>
        - Everything is either P or Q.
        - Not everything is P.
        </PREMISES>
        <CONCLUSION> Something is Q. </CONCLUSION>
        <EVALUATE>
        <REASONING> From the partition and "not all P", it follows that some Q. </REASONING>
        <FOL_THEORY>
        all x (P(x) | Q(x)).
        -all x P(x).
        </FOL_THEORY>
        <FOL_QUERY>
        exists x Q(x).
        </FOL_QUERY>
        <ANSWER>True</ANSWER>
        </EVALUATE>
        </EXAMPLE>
        """).strip(),
    ]
    return "\n\n".join(shots[:max(0, min(k, len(shots)))])

def format_premises(premises: List[str]) -> str:
    return "\n".join(f"- {p.strip()}" for p in premises)

def build_prompt(sample: Sample, method: str, shots: int, dataset: str) -> str:
    header = BASE
    if method in ("scratchpad","linc"):
        guide = SCRATCHPAD_GUIDE
    elif method == "cot":
        guide = COT_GUIDE
    else:
        guide = NAIVE_GUIDE

    fs = few_shot_examples(shots, dataset)

    body = []
    body.append("<TASK>")
    body.append("<PREMISES>")
    body.append(format_premises(sample.premises))
    body.append("</PREMISES>")
    body.append("<CONCLUSION> " + sample.conclusion.strip() + " </CONCLUSION>")
    body.append("</TASK>")

    prompt = "\n\n".join([header, guide, fs, "\n".join(body)]).strip() + "\n"
    return prompt
