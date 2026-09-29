"""R-B two-stage story-cluster bootstrap (additional analysis plan).

Pool = all available samplings (scores_seed*.parquet), de-duplicated by comment id.
Each replicate: for every month x group, draw 30 stories with replacement from the stories in
the pool, then draw min(40, n_i) comments with replacement within each drawn story; monthly
means are comment-weighted as in the main analysis. B = 1000, seed 0.
Per replicate: comparative ITS for the 7 M1 emotions (beta_2, beta_3, p), Holm over the 14
tests, and beta_2 for fear under M2 and M3.
Outputs results/bootstrap.csv, results/bootstrap_summary.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_s = importlib.util.spec_from_file_location("its", HERE / "04_its.py"); its = importlib.util.module_from_spec(_s); _s.loader.exec_module(its)
B, N_ST, N_CM = 1000, 30, 40
COLS = [f"m1_{e}" for e in its.EMO7] + ["m2_fear", "m3_fear"]


def main() -> None:
    files = sorted((ROOT / "results").glob("scores_seed*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("cid")
    df["t"] = its.month_index(df["month"]); df = df[(df.t >= 0) & (df.t <= 137)]
    print(f"pool: {len(files)} samplings, {len(df)} comments, {df.story_id.nunique()} stories")
    cells = {}
    for (t, g), sub in df.groupby(["t", "group"]):
        cells[(t, g)] = [s[COLS].to_numpy() for _, s in sub.groupby("story_id")]
    months = sorted({t for t, _ in cells})
    tt = np.array(months, dtype=float)
    rng = np.random.default_rng(0)
    rows = []
    for b in range(B):
        means = {}
        for key, stories in cells.items():
            pick = rng.integers(0, len(stories), N_ST)
            tot, n = np.zeros(len(COLS)), 0
            for i in pick:
                arr = stories[i]
                k = min(N_CM, len(arr))
                tot += arr[rng.integers(0, len(arr), k)].sum(axis=0); n += k
            means[key] = tot / n
        gap = np.array([means[(t, "ai")] - means[(t, "control")] for t in months])
        rec = {"b": b}
        pv = []
        for j, e in enumerate(its.EMO7):
            r = its.its_fit(gap[:, j], tt, its.T0)
            rec[f"b2_{e}"], rec[f"p2_{e}"], rec[f"b3_{e}"], rec[f"p3_{e}"] = r["b2_level"], r["p_b2"], r["b3_slope"], r["p_b3"]
            pv.append(r["p_b2"])
        pv += [rec[f"p3_{e}"] for e in its.EMO7]
        adj = multipletests(pv, method="holm")[1]
        for j, e in enumerate(its.EMO7):
            rec[f"h2_{e}"], rec[f"h3_{e}"] = adj[j], adj[7 + j]
        rec["b2_fear_m2"] = its.its_fit(gap[:, 7], tt, its.T0)["b2_level"]
        rec["b2_fear_m3"] = its.its_fit(gap[:, 8], tt, its.T0)["b2_level"]
        rows.append(rec)
        if b % 100 == 0:
            print("replicate", b, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "bootstrap.csv", index=False)
    f = out
    summ = {
        "B": B, "n_samplings_in_pool": len(files), "n_comments_pool": int(len(df)),
        "fear_b2_mean_pp": 100 * f.b2_fear.mean(),
        "fear_b2_q025_pp": 100 * f.b2_fear.quantile(0.025), "fear_b2_q975_pp": 100 * f.b2_fear.quantile(0.975),
        "share_fear_b2_pos": float((f.b2_fear > 0).mean()),
        "share_fear_nominal_p05": float((f.p2_fear < 0.05).mean()),
        "share_fear_holm_p05": float((f.h2_fear < 0.05).mean()),
        "share_sign_agree_3": float(((f.b2_fear > 0) & (f.b2_fear_m2 > 0) & (f.b2_fear_m3 > 0)).mean()),
        "share_surprise_b3_neg": float((f.b3_surprise < 0).mean()),
        "share_anger_b3_pos": float((f.b3_anger > 0).mean()),
        "share_any_holm_p05": float(((f[[c for c in f if c.startswith("h")]] < 0.05).any(axis=1)).mean()),
    }
    json.dump(summ, open(ROOT / "results" / "bootstrap_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
