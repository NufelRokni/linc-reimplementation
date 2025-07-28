from dataclasses import dataclass
from typing import List, Iterable
from datasets import load_dataset

@dataclass
class Sample:
    id: str
    premises: List[str]
    conclusion: str
    label: str  # "True" | "False" | "Uncertain"

def _split_sentences(text: str) -> List[str]:
    # Simple splitter; FOLIO premises are already sentence-like; keep conservative
    parts = [s.strip() for s in text.replace("\n"," ").split(".") if s.strip()]
    return [p + "." for p in parts]

def load_folio(split: str = "validation", source: str = "tasksource") -> List[Sample]:
    if split in ("val","dev"):
        split = "validation"

    if source == "tasksource":
        ds = load_dataset("tasksource/folio", split=split)
        # fields: premises (str), conclusion (str), label (str), story_id, example_id
        out = []
        for row in ds:
            prem = row["premises"]
            concl = row["conclusion"]
            label = row["label"].capitalize()
            out.append(Sample(
                id=str(row.get("example_id", row.get("story_id", ""))),
                premises=_split_sentences(prem),
                conclusion=concl.strip(),
                label=label
            ))
        return out
    elif source == "yale":
        ds = load_dataset("yale-nlp/FOLIO", split=split)  # requires access grant
        out = []
        for row in ds:
            prem = row.get("premises", row.get("context",""))
            concl = row.get("conclusion", row.get("hypothesis","")).strip()
            label = str(row.get("label","Uncertain")).capitalize()
            out.append(Sample(
                id=str(row.get("example_id", "")),
                premises=_split_sentences(prem),
                conclusion=concl,
                label=label
            ))
        return out
    else:
        raise ValueError(f"Unknown source: {source}")
