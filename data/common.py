import json, re, unicodedata
from pathlib import Path

# ---------- Default IO ----------
DEFAULT_IN  = Path("./saudi_dataset_all.jsonl")
OUT_CLEAN   = Path("saudi_dataset_all.clean.jsonl")
OUT_BAL     = Path("saudi_dataset_all.clean.balanced.jsonl")
OUT_CSV     = Path("saudi_dataset_balanced_no_tags.csv")  # final output by default

# ---------- Repro / balancing ----------
SEED = 42
BALANCE_MODE = "downsample"   # or "oversample"

# ---------- Filtering ----------
BANNED_TOPICS = {
    "politics_governance",
    "religion_faith",
}

# ---------- Regex / helpers ----------
# Accept both documented short tags (<HIJAZI>/<NAJDI>) and the canonical
# control form emitted by step_add_tags (<DIALECT=HIJAZI>/<DIALECT=NAJDI>).
DIALECT_TAG_BODY = r'<\s*(?:DIALECT\s*=\s*)?(HIJAZI|NAJDI)\s*>'
DIALECT_TAG_RE = re.compile(DIALECT_TAG_BODY, re.IGNORECASE)
TAG_RE = re.compile(r'^\s*' + DIALECT_TAG_BODY + r'\s*', re.IGNORECASE)

def strip_cf(s: str) -> str:
    """Strip zero-width/formatting controls (avoid hidden characters)."""
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cf")

ALIASES = {
    "hijazi":"Hijazi","hejazi":"Hijazi","hejaz":"Hijazi","hijaz":"Hijazi","حجازي":"Hijazi",
    "najdi":"Najdi","najdy":"Najdi","najd":"Najdi","نجدي":"Najdi","النجدي":"Najdi",
    "HIJAZI":"Hijazi","NAJDI":"Najdi",
}

def canon_dialect(d):
    """Return 'Hijazi'/'Najdi' or None."""
    if d is None:
        return None
    s = strip_cf(str(d)).strip()
    if s in ("Hijazi","Najdi"):
        return s
    low = s.lower()
    if low in ALIASES:
        return ALIASES[low]
    if s in ALIASES:
        return ALIASES[s]
    return None

def tag_for_dialect(d):
    """Return 'HIJAZI' or 'NAJDI' for tag insertion (optional step)."""
    if not d:
        return None
    dl = strip_cf(str(d)).strip().lower()
    if dl.startswith("hij"):
        return "HIJAZI"
    if dl.startswith("naj"):
        return "NAJDI"
    return None

def get_topic(obj) -> str:
    meta = obj.get("meta") or {}
    return str(meta.get("topic", "")).strip()

def read_jsonl(path: Path):
    """Yield non-empty JSONL records and fail closed on malformed JSON."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
