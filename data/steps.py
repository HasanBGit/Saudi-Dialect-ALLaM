import csv, random
from pathlib import Path
from collections import defaultdict, Counter

from common import (
    TAG_RE, canon_dialect, tag_for_dialect, get_topic,
    read_jsonl, write_jsonl, BANNED_TOPICS
)

# ---------- CLEAN + FILTER ----------
def step_clean(in_path: Path, out_path: Path):
    """
    Keep only instruction/response/dialect; strip a leading <DIALECT=...> from instruction.
    Drop rows whose meta.topic is in BANNED_TOPICS.
    """
    total = written = 0
    dropped_empty = dropped_banned = 0
    input_dialects = Counter()
    output_dialects = Counter()
    topic_counts = Counter()

    out_rows = []
    for obj in read_jsonl(in_path):
        total += 1

        # topic filter
        topic = get_topic(obj)
        if topic:
            topic_counts[topic] += 1
        if topic in BANNED_TOPICS:
            dropped_banned += 1
            continue

        instr = obj.get("instruction", "")
        resp  = obj.get("response", "")
        dia   = obj.get("dialect", None)  # keep verbatim

        if not (instr and resp):
            dropped_empty += 1
            continue

        instr_clean = TAG_RE.sub("", instr or "").strip()
        out = {"instruction": instr_clean, "response": resp, "dialect": dia}
        out_rows.append(out)

        input_dialects[dia]  += 1
        output_dialects[dia] += 1
        written += 1

    write_jsonl(out_path, out_rows)

    print(f"[CLEAN]  Input rows: {total:,}  |  Written: {written:,}  -> {out_path}")
    print(f"[CLEAN]  Dropped (empty instr/resp): {dropped_empty:,}  |  Dropped (banned topics): {dropped_banned:,}")
    if topic_counts:
        top10 = topic_counts.most_common(10)
        print("[CLEAN]  Top topics (pre-filter, up to 10):")
        for k,v in top10:
            print(f"         - {k:25s}: {v:,}")
    preserved = (input_dialects == output_dialects)
    print(f"[CLEAN]  Dialect column preserved verbatim: {'YES' if preserved else 'CHECK'}")

# ---------- BALANCE ----------
def step_balance(in_path: Path, out_path: Path, mode: str = "downsample", seed: int = 42):
    """
    Balance to 50/50 Hijazi vs Najdi. Drop Unknown.
    mode: 'downsample' or 'oversample'
    """
    rows = list(read_jsonl(in_path))

    # group by canonical dialect; drop unknown
    groups = defaultdict(list)
    raw_counts = Counter()
    for r in rows:
        d = r.get("dialect", None)
        raw_counts[d] += 1
        c = canon_dialect(d)
        if c in ("Hijazi","Najdi"):
            groups[c].append(r)

    kept_before = len(groups["Hijazi"]) + len(groups["Najdi"])
    dropped_unknown = len(rows) - kept_before

    random.seed(seed)
    for k in ("Hijazi","Najdi"):
        random.shuffle(groups[k])

    if mode.lower() == "oversample":
        m = max(len(groups["Hijazi"]), len(groups["Najdi"]))
        def oversample(lst, target):
            if not lst:
                return []
            q, r = divmod(target, len(lst))
            out = lst * q + random.sample(lst, r)
            random.shuffle(out)
            return out
        balanced = oversample(groups["Hijazi"], m) + oversample(groups["Najdi"], m)
    else:
        n = min(len(groups["Hijazi"]), len(groups["Najdi"]))
        balanced = groups["Hijazi"][:n] + groups["Najdi"][:n]

    random.shuffle(balanced)

    # write only the three fields, do NOT touch dialect string
    out_min = [ {
        "instruction": r.get("instruction",""),
        "response":    r.get("response",""),
        "dialect":     r.get("dialect", None)
    } for r in balanced ]
    write_jsonl(out_path, out_min)

    after = Counter(canon_dialect(r.get("dialect")) for r in balanced)
    total_after = sum(after.values()) or 1

    print(f"[BAL]    Kept before balance (Hijazi+Najdi): {kept_before:,} | Dropped Unknown: {dropped_unknown:,}")
    print(f"[BAL]    Output -> {out_path}")
    print(f"[BAL]    Hijazi: {after['Hijazi']:>5}  ({after['Hijazi']/total_after:6.2%})")
    print(f"[BAL]    Najdi : {after['Najdi']:>5}  ({after['Najdi']/total_after:6.2%})")

# ---------- OPTIONAL: ADD TAGS (not used for final CSV; kept for completeness) ----------
def step_add_tags(in_path: Path, out_path: Path):
    total = updated = already = unknown = 0
    out_rows = []
    for obj in read_jsonl(in_path):
        total += 1
        instr = obj.get("instruction", "")
        resp  = obj.get("response", "")
        dia   = obj.get("dialect", "")
        if TAG_RE.match(instr or ""):
            out_rows.append({"instruction": instr, "response": resp, "dialect": dia})
            already += 1
            continue
        tag = tag_for_dialect(dia)
        if not tag:
            out_rows.append({"instruction": instr, "response": resp, "dialect": dia})
            unknown += 1
            continue
        new_instr = f"<DIALECT={tag}>" + (instr.lstrip() if isinstance(instr, str) else "")
        out_rows.append({"instruction": new_instr, "response": resp, "dialect": dia})
        updated += 1
    write_jsonl(out_path, out_rows)
    print(f"[TAGS]   Total: {total:,} | Updated: {updated:,} | Already-tagged: {already:,} | Unknown dialect: {unknown:,}")
    print(f"[TAGS]   Output -> {out_path}")

# ---------- EXPORT CSV (final step) ----------
def step_export_csv(in_path: Path, csv_path: Path):
    """
    Export rows (instruction/response/dialect) to CSV.
    Intended after step_balance if your final set should be no-tag + balanced.
    """
    count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instruction", "response", "dialect"])
        for r in read_jsonl(in_path):
            w.writerow([r.get("instruction",""), r.get("response",""), r.get("dialect","")])
            count += 1
    print(f"[CSV]    Wrote {count:,} rows -> {csv_path}")
