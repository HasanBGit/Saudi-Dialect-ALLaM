from __future__ import annotations
import re, json, csv
from pathlib import Path
from typing import List
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm
import sacrebleu
from bert_score import score as bertscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# MARBERTv2 5-way written dialect classifier
DID_ID = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"

def read_jsonl(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s: rows.append(json.loads(s))
    return rows

def load_dialect_clf(device: str):
    tok  = AutoTokenizer.from_pretrained(DID_ID, use_fast=True)
    mdl  = AutoModelForSequenceClassification.from_pretrained(DID_ID).to(device).eval()
    id2label = mdl.config.id2label
    label2id = {(v if isinstance(v,str) else v.get("name","")).upper(): int(k) for k,v in id2label.items()}
    msa_idx = None
    for raw, idx in label2id.items():
        if "MSA" in raw:
            msa_idx = idx; break
    return tok, mdl, id2label, msa_idx

@torch.no_grad()
def camel_probs_batched(texts: List[str], tok, mdl, device: str, bs: int):
    all_probs=[]
    for i in tqdm(range(0, len(texts), bs), desc="Dialect scoring (MARBERTv2)", leave=False):
        chunk = texts[i:i+bs]
        batch = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        logits = mdl(**batch).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs) if all_probs else np.zeros((0, mdl.config.num_labels))

def labels_from_probs(probs, id2label):
    ids = probs.argmax(axis=1)
    out=[]
    for i in ids:
        key = str(i)
        lab = id2label[key] if key in id2label else id2label[i]
        lab = lab if isinstance(lab, str) else lab.get("name","")
        out.append(lab.upper().strip())
    return out

def is_saudi(lbl: str) -> bool:
    return "GLF" in lbl

def tag_echo_rate(texts: List[str]) -> float:
    patt = re.compile(r'<\s*DIALECT\s*=\s*[^>]+>', re.IGNORECASE)
    return 100.0 * float(np.mean([bool(patt.search(t)) for t in texts]))

def diversity_metrics(preds: List[str]):
    def ngrams(tokens, n): return list(zip(*[tokens[i:] for i in range(n)]))
    total_bi=total_tri=0; uniq_bi=set(); uniq_tri=set()
    for p in preds:
        t=p.split()
        b=ngrams(t,2); g=ngrams(t,3)
        total_bi += max(1,len(b)); total_tri += max(1,len(g))
        uniq_bi.update(b); uniq_tri.update(g)
    d2 = len(uniq_bi)/total_bi if total_bi else 0.0
    d3 = len(uniq_tri)/total_tri if total_tri else 0.0
    if len(preds) < 2:
        sbleu = 0.0
    else:
        scores=[]
        for i in range(len(preds)):
            hyp=[preds[i]]
            refs=[[p for j,p in enumerate(preds) if j!=i]]
            scores.append(sacrebleu.corpus_bleu(hyp, refs).score)
        sbleu = float(np.mean(scores))
    return d2, d3, sbleu

_ar_punct = r"[^\w\s\u0600-\u06FF]"
_ar_tatweel = "\u0640"
_ar_diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")

def normalize_ar(text: str) -> str:
    t = text.replace(_ar_tatweel, "")
    t = _ar_diacritics.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(_ar_punct, " ", t)
    return t.strip()

def near_duplicate_rate(preds: List[str], thr=0.90) -> float:
    if len(preds) < 2:
        return 0.0
    norm = [normalize_ar(p) for p in preds]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=1)
    X = vec.fit_transform(norm)
    sims = cosine_similarity(X)
    n = X.shape[0]; cnt = 0; denom = n*(n-1)/2
    for i in range(n):
        for j in range(i+1, n):
            if sims[i, j] >= thr:
                cnt += 1
    return 100.0 * cnt / max(1, denom)

def evaluate_one_pred_file(pred_path: Path, device: str, bs_clf: int, out_dir: Path) -> list:
    tok, clf, id2label, msa_idx = load_dialect_clf(device)
    name  = pred_path.stem
    rows  = read_jsonl(pred_path)
    preds = [r["pred"] for r in rows]
    golds = [r["gold"] for r in rows]

    probs = camel_probs_batched(preds, tok, clf, device=device, bs=bs_clf)
    labs  = labels_from_probs(probs, id2label)
    conf  = probs.max(axis=1)

    print(f"{name} label counts:", Counter(labs))

    msa_leak   = float(np.mean(probs[:, msa_idx]))*100.0 if msa_idx is not None else 0.0
    saudi_rate = 100.0*float(np.mean([is_saudi(x) for x in labs]))
    low_conf   = 100.0*float(np.mean(conf < 0.55))
    echo       = tag_echo_rate(preds)

    chrf = sacrebleu.corpus_chrf(preds, [golds]).score
    P,R,F = bertscore(preds, golds, lang="ar", rescale_with_baseline=True)
    bert_f1 = float(F.mean().item())

    d2, d3, sbleu = diversity_metrics(preds)
    near_dup = near_duplicate_rate(preds, thr=0.90)

    with (out_dir / f"{name}_report.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model": name, "n": len(rows),
            "saudi_rate_pct": saudi_rate,
            "msa_leak_pct": msa_leak,
            "low_conf_pct": low_conf,
            "tag_echo_pct": echo,
            "chrF++": chrf,
            "BERTScore_F1": bert_f1,
            "distinct2": d2, "distinct3": d3, "selfBLEU": sbleu,
            "near_duplicate_pct": near_dup
        }, f, ensure_ascii=False, indent=2)

    return [name, round(saudi_rate,2), round(msa_leak,2), round(low_conf,2),
            round(echo,2), round(chrf,2), round(bert_f1,4),
            round(d2,4), round(d3,4), round(sbleu,2), round(near_dup,2)]

def save_summary_csv(summary: list[list], out_dir: Path, filename: str = "summary_saudi_only.csv") -> Path:
    csv_path = out_dir / filename
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model","Saudi% (GLF) ↑","MSA leak% ↓","Low-conf% ↓",
                    "Tag-echo% ↓","chrF++ ↑","BERTScore F1 ↑",
                    "distinct-2 ↑","distinct-3 ↑","Self-BLEU ↓","Near-dup% ↓"])
        for row in summary:
            w.writerow(row)
    print("\n== Summary ==")
    for row in summary:
        print(row)
    print(f"\nSaved: {csv_path}")
    return csv_path
