"""Score every sampled comment with the three pre-registered emotion instruments.

M1: j-hartmann/emotion-english-distilroberta-base (softmax over 7 labels)
M1m: M1 applied after masking AI keywords (topic-word masking check)
M2: SamLowe/roberta-base-go_emotions (sigmoid over 28 labels), aggregated to Ekman
    categories with the official GoEmotions ekman_mapping.json (max over members)
M3: NRC Emotion Lexicon (share of tokens associated with each emotion)

Usage: python3 03_classify.py --seed 2026
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from keywords import mask_ai_terms

ROOT = Path(__file__).resolve().parents[1]
EMOS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# Official mapping: github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json
EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@torch.no_grad()
def run_model(name: str, texts: list[str], sigmoid: bool, bs: int = 64) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    # sort by length so that batches are padded efficiently; restore order afterwards
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=512,
                  padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits.float()
        p = torch.sigmoid(logits) if sigmoid else torch.softmax(logits, dim=-1)
        out[idx] = p.cpu().numpy()
        if (s // bs) % 200 == 0:
            print(f"  {name}: {s}/{len(texts)}", flush=True)
    return pd.DataFrame(out, columns=labels)


def load_nrc() -> dict[str, set[str]]:
    path = ROOT / "data" / "nrc" / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    lex: dict[str, set[str]] = {e: set() for e in EMOS}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3 and parts[1] in lex and parts[2] == "1":
            lex[parts[1]].add(parts[0])
    return lex


def nrc_scores(texts: list[str], lex: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for t in texts:
        toks = TOKEN_RE.findall(t.lower())
        n = max(len(toks), 1)
        rows.append({e: sum(w in lex[e] for w in toks) / n for e in EMOS})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    df = pd.read_parquet(ROOT / "data" / f"comments_seed{args.seed}.parquet").reset_index(drop=True)
    texts = df["text"].tolist()
    print(f"{len(texts)} comments on {DEVICE}")

    m1 = run_model("j-hartmann/emotion-english-distilroberta-base", texts, sigmoid=False)
    m1m = run_model("j-hartmann/emotion-english-distilroberta-base",
                    [mask_ai_terms(t) for t in texts], sigmoid=False)
    go = run_model("SamLowe/roberta-base-go_emotions", texts, sigmoid=True)
    m2 = pd.DataFrame({e: go[members].max(axis=1) for e, members in EKMAN.items()})
    m3 = nrc_scores(texts, load_nrc())

    res = df.drop(columns=["text"]).copy()
    for tag, m in [("m1", m1), ("m1m", m1m), ("m2", m2), ("m3", m3)]:
        for c in m.columns:
            res[f"{tag}_{c}"] = m[c].values
    res["n_words"] = [len(t.split()) for t in texts]
    out = ROOT / "results" / f"scores_seed{args.seed}.parquet"
    res.to_parquet(out)
    print(f"saved {out} {res.shape}")


if __name__ == "__main__":
    main()
