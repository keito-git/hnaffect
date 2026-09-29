"""R-G pseudo-treated controls and R-F(iii) synthetic control (fixed in the additional plan).

Uses every available scores_seed*.parquet, de-duplicated by comment id, joined with the
topic of each story from data/story_topics.parquet (k=20 k-means over non-AI titles).

R-G: each non-AI topic k is treated as if it were the AI group; control = all other non-AI
     topics; comparative ITS at T0 (HAC lag 4) for the 7 M1 emotions -> share with p<.05.
R-F(iii): synthetic control for the M1 fear series of the AI group from the 20 topic series
     (weights >= 0, sum 1, fitted on the pre-period); in-space placebo with each topic treated.
Outputs results/pseudo_treated.csv, results/synthetic_control.csv
"""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("its", HERE / "04_its.py")
its = importlib.util.module_from_spec(spec)
spec.loader.exec_module(its)
T0 = its.T0


def load() -> pd.DataFrame:
    files = sorted((ROOT / "results").glob("scores_seed*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("cid")
    topics = pd.read_parquet(ROOT / "data" / "story_topics.parquet")[["objectID", "topic"]]
    df = df.merge(topics, left_on="story_id", right_on="objectID", how="left")
    df["t"] = its.month_index(df["month"])
    df = df[(df["t"] >= 0) & (df["t"] <= 137)]
    print(f"{len(files)} samplings, {len(df)} unique comments")
    return df


def pseudo_treated(df: pd.DataFrame) -> pd.DataFrame:
    ctl = df[df["group"] == "control"]
    rows = []
    for k in sorted(ctl["topic"].dropna().unique()):
        tr = ctl[ctl["topic"] == k].groupby("t")[[f"m1_{e}" for e in its.EMO7]].mean()
        co = ctl[ctl["topic"] != k].groupby("t")[[f"m1_{e}" for e in its.EMO7]].mean()
        j = tr.index.intersection(co.index)
        t = j.values.astype(float)
        for e in its.EMO7:
            y = (tr.loc[j, f"m1_{e}"] - co.loc[j, f"m1_{e}"]).values
            r = its.its_fit(y, t, T0)
            rows.append({"topic": int(k), "emotion": e, "n_months": len(j), "b2": r["b2_level"],
                         "p_b2": r["p_b2"], "b3": r["b3_slope"], "p_b3": r["p_b3"]})
    return pd.DataFrame(rows)


def synth(y_pre: np.ndarray, X_pre: np.ndarray) -> np.ndarray:
    n = X_pre.shape[1]
    res = minimize(lambda w: np.sum((y_pre - X_pre @ w) ** 2), np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


def synthetic_control(df: pd.DataFrame) -> pd.DataFrame:
    col = "m1_fear"
    series = {"AI": df[df["group"] == "ai"].groupby("t")[col].mean()}
    ctl = df[df["group"] == "control"]
    for k in sorted(ctl["topic"].dropna().unique()):
        series[f"topic{int(k)}"] = ctl[ctl["topic"] == k].groupby("t")[col].mean()
    S = pd.DataFrame(series).dropna()
    t = S.index.values
    pre, post = t < T0, t >= T0
    rows = []
    for unit in S.columns:
        donors = [c for c in S.columns if c != unit and c != "AI"]
        w = synth(S.loc[pre, unit].values, S.loc[pre, donors].values)
        gap = S[unit].values - S[donors].values @ w
        rmspe_pre = np.sqrt(np.mean(gap[pre] ** 2))
        rmspe_post = np.sqrt(np.mean(gap[post] ** 2))
        rows.append({"unit": unit, "post_mean_gap": gap[post].mean(), "pre_mean_gap": gap[pre].mean(),
                     "rmspe_pre": rmspe_pre, "ratio": rmspe_post / rmspe_pre,
                     "top_donors": ";".join(f"{d}:{x:.2f}" for d, x in sorted(zip(donors, w), key=lambda z: -z[1])[:3])})
    out = pd.DataFrame(rows)
    out["rank_ratio"] = out["ratio"].rank(ascending=False).astype(int)
    out["n_months"] = len(S)
    return out


def main() -> None:
    df = load()
    pt = pseudo_treated(df)
    pt.to_csv(ROOT / "results" / "pseudo_treated.csv", index=False)
    print("pseudo-treated: share p_b2<.05 =", round((pt.p_b2 < 0.05).mean(), 3),
          "| share p_b3<.05 =", round((pt.p_b3 < 0.05).mean(), 3),
          "| fear only b2 =", round((pt[pt.emotion == 'fear'].p_b2 < 0.05).mean(), 3),
          "| months per topic min", pt.n_months.min())
    sc = synthetic_control(df)
    sc.to_csv(ROOT / "results" / "synthetic_control.csv", index=False)
    print(sc.sort_values("rank_ratio").head(6).round(4).to_string())
    print(sc[sc.unit == "AI"].round(4).to_string())


if __name__ == "__main__":
    main()
