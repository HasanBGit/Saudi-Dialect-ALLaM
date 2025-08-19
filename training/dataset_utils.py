from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any, Tuple
from datasets import Dataset

# Match a LEADING dialect control tag
TAG_RE = re.compile(r'^\s*<\s*DIALECT\s*=\s*(HIJAZI|NAJDI)\s*>\s*', re.IGNORECASE)
EOS = "</s>"

def load_jsonl(path: Path) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instr = (obj.get("instruction") or "").strip()
            resp  = (obj.get("response") or "").strip()
            if instr and resp:
                rows.append({"instruction": instr, "response": resp})
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return Dataset.from_list(rows)

def fmt_token(example: Dict[str, Any]) -> Dict[str, str]:
    """Keep tags as-is."""
    instr = example["instruction"]
    resp  = example["response"]
    return {"text": f"### Instruction:\n{instr}\n\n### Response:\n{resp}{EOS}"}

def fmt_no_token_counter():
    """
    Returns (formatter_fn, counters_dict). Formatter strips a LEADING tag if present.
    counters_dict keeps {'count': removed, 'total': seen}.
    """
    counters = {"count": 0, "total": 0}

    def _fmt(example: Dict[str, Any]) -> Dict[str, str]:
        instr = example["instruction"]
        counters["total"] += 1
        stripped = TAG_RE.sub("", instr).lstrip()
        if stripped != instr:
            counters["count"] += 1
        resp = example["response"]
        return {"text": f"### Instruction:\n{stripped}\n\n### Response:\n{resp}{EOS}"}

    return _fmt, counters

def keep_only_text(ds: Dataset) -> Dataset:
    return ds.remove_columns([c for c in ds.column_names if c != "text"])
