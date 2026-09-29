"""R-F(ii) topic-matched control (additional analysis plan).

For every AI story in the primary sample (seed 2026), one non-AI story from the same month is
matched without replacement: candidates whose points and comment counts are within a factor of
two (relaxed to the whole month if none) and, among them, the highest cosine similarity of the
title embeddings (all-MiniLM-L6-v2). Comments of matched stories are sampled with the same rule
(<= 40 comments, >= 5 words) and scored with M1, M2, and M3; the comparative ITS and placebo
calibration are then run with the matched stories as the control group.
Outputs results/scores_matched_seed2026.parquet, results/matched_control.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name):
    s = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


fetch = _load("02_sample_and_fetch")
clf = _load("03_classify")
its = _load("04_its")


def match() -> pd.DataFrame:
    topics = pd.read_parquet(ROOT / "data" / "story_topics.parquet")
    emb = np.load(ROOT / "data" / "title_embeddings.npy")
    topics["row"] = np.arange(len(topics))
    main = pd.read_parquet(ROOT / "results" / "scores_seed2026.parquet")
    ai_ids = main.loc[main.group == "ai", "story_id"].unique()
    ai = topics[topics.objectID.isin(ai_ids)]
    used, pairs = set(), []
    for _, a in ai.iterrows():
        cand = topics[(topics.month == a.month) & (~topics.ai_broad) & (~topics.objectID.isin(used))]
        close = cand[(cand.points.between(a.points / 2, a.points * 2)) &
                     (cand.num_comments.between(a.num_comments / 2, a.num_comments * 2))]
        pool = close if len(close) else cand
        sims = emb[pool.row.values] @ emb[a.row]
        b = pool.iloc[int(np.argmax(sims))]
        used.add(b.objectID)
        pairs.append({"ai_story": a.objectID, "matched_story": b.objectID, "month": a.month,
                      "similarity": float(sims.max()), "relaxed": len(close) == 0})
    return pd.DataFrame(pairs)


def main() -> None:
    pairs = match()
    pairs.to_csv(ROOT / "results" / "matched_pairs_seed2026.csv", index=False)
    print(f"{len(pairs)} pairs; mean similarity {pairs.similarity.mean():.3f}; relaxed {pairs.relaxed.mean():.3f}")
    rng = np.random.default_rng(2026)
    rows = []
    for _, p in pairs.iterrows():
        cs = fetch.fetch_comments(p.matched_story)
        k = min(fetch.N_COMMENTS, len(cs))
        for j in (np.sort(rng.choice(len(cs), size=k, replace=False)) if k else []):
            rows.append({"story_id": p.matched_story, "month": p.month, "group": "control", **cs[j]})
    mc = pd.DataFrame(rows).reset_index(drop=True)
    texts = mc["text"].tolist()
    m1 = clf.run_model("j-hartmann/emotion-english-distilroberta-base", texts, sigmoid=False)
    go = clf.run_model("SamLowe/roberta-base-go_emotions", texts, sigmoid=True)
    m2 = pd.DataFrame({e: go[mem].max(axis=1) for e, mem in clf.EKMAN.items()})
    m3 = clf.nrc_scores(texts, clf.load_nrc())
    for tag, m in [("m1", m1), ("m2", m2), ("m3", m3)]:
        for c in m.columns:
            mc[f"{tag}_{c}"] = m[c].values
    mc = mc.drop(columns=["text"])
    mc.to_parquet(ROOT / "results" / "scores_matched_seed2026.parquet")

    main = pd.read_parquet(ROOT / "results" / "scores_seed2026.parquet")
    df = pd.concat([main[main.group == "ai"], mc], ignore_index=True)
    df["t"] = its.month_index(df["month"]); df = df[(df.t >= 0) & (df.t <= 137)]
    out = []
    for tag in ["m1", "m2", "m3"]:
        mm = its.monthly_means(df, [f"{tag}_fear"])
        t = mm.index.values.astype(float)
        d = mm[(f"{tag}_fear", "ai")].values - mm[(f"{tag}_fear", "control")].values
        r = its.its_fit(d, t, its.T0)
        pre = t < its.T0
        bs = [its.its_fit(d[pre], t[pre], p0)["b2_level"] for p0 in its.PLACEBO_T0]
        out.append({"instrument": tag, "b2": r["b2_level"], "p_b2": r["p_b2"],
                    "std_b2": r["b2_level"] / d[pre].std(ddof=1),
                    "placebo_exceed": float(np.mean(np.abs(bs) >= abs(r["b2_level"]))),
                    "n_control_comments": len(mc)})
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "results" / "matched_control.csv", index=False)
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
