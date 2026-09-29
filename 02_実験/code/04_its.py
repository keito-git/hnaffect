"""Comparative interrupted time series (pre-registered analysis).

Usage: python3 04_its.py --seed 2026
Outputs (results/its_seed{seed}/):
  monthly.csv          monthly group means per instrument x emotion
  its_main.csv         comparative ITS (AI - control) and single-series ITS (AI only)
  its_robust.csv       stable-keyword subset and topic-masked M1
  comment_level.csv    comment-level model with month FE and story-clustered SE
  placebo.csv          placebo-intervention estimates (pre-period only)
  local_contrast.csv   post-hoc: mean gap 12 months after vs before T0 (trend-free)
  prestart_curve.csv   post-hoc: comparative ITS with the pre-period starting each January 2015-2021
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
T0 = 95  # 2022-12 with 2015-01 = 0
HAC_LAG = 4
EMO7 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
INSTR = {"m1": EMO7, "m1m": EMO7, "m2": EMO7,
         "m3": ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}
PLACEBO_T0 = range(24, 78)  # 2017-01 .. 2021-06
PRE_END = T0 - 1            # placebo fits use t <= 94 only (pitfall_081)
PLACEBO_SPARSE = list(range(24, 78, 6))  # post-hoc: 2017-01, 2017-07, ..., 2021-01


def month_index(m: pd.Series) -> pd.Series:
    y, mo = m.str.slice(0, 4).astype(int), m.str.slice(5, 7).astype(int)
    return (y - 2015) * 12 + (mo - 1)


def its_fit(y: np.ndarray, t: np.ndarray, t0: int) -> dict:
    post = (t >= t0).astype(float)
    X = sm.add_constant(np.column_stack([t, post, (t - t0) * post]))
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
    return {"b1_trend": r.params[1], "b2_level": r.params[2], "b3_slope": r.params[3],
            "se_b2": r.bse[2], "se_b3": r.bse[3], "p_b2": r.pvalues[2], "p_b3": r.pvalues[3],
            "ci_b2_lo": r.conf_int()[2, 0], "ci_b2_hi": r.conf_int()[2, 1],
            "ci_b3_lo": r.conf_int()[3, 0], "ci_b3_hi": r.conf_int()[3, 1]}


def monthly_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["t", "group"])[cols].mean().unstack("group")
    return g


def series_table(df: pd.DataFrame, label: str, instr: dict) -> list[dict]:
    rows = []
    for tag, emos in instr.items():
        cols = [f"{tag}_{e}" for e in emos]
        mm = monthly_means(df, cols)
        t = mm.index.values.astype(float)
        for e, c in zip(emos, cols):
            ai, ctl = mm[(c, "ai")].values, mm[(c, "control")].values
            pre_mean = ai[t < T0].mean()
            sd_pre = (ai - ctl)[t < T0].std(ddof=1)
            for kind, y in [("comparative", ai - ctl), ("single_ai", ai), ("single_control", ctl)]:
                r = its_fit(y, t, T0)
                rows.append({"sample": label, "instrument": tag, "emotion": e, "model": kind,
                             "pre_mean_ai": pre_mean, "rel_b2": r["b2_level"] / pre_mean,
                             "std_b2": r["b2_level"] / sd_pre, **r})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = ROOT / "results" / f"its_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "results" / f"scores_seed{args.seed}.parquet")
    df["t"] = month_index(df["month"])
    df = df[(df["t"] >= 0) & (df["t"] <= 137)]
    counts = df.groupby(["t", "group"]).agg(n_comments=("cid", "size"), n_stories=("story_id", "nunique"))
    counts.to_csv(out / "counts.csv")

    all_cols = [f"{k}_{e}" for k, v in INSTR.items() for e in v]
    monthly_means(df, all_cols).to_csv(out / "monthly.csv")

    # main analysis
    main_rows = pd.DataFrame(series_table(df, "all", {k: INSTR[k] for k in ["m1", "m2", "m3"]}))
    # Holm over the 14 M1 comparative tests (7 emotions x {b2, b3}), fixed before data collection
    fam = main_rows[(main_rows.instrument == "m1") & (main_rows.model == "comparative")]
    pvals = np.concatenate([fam["p_b2"].values, fam["p_b3"].values])
    adj = multipletests(pvals, method="holm")[1]
    main_rows["p_b2_holm"], main_rows["p_b3_holm"] = np.nan, np.nan
    main_rows.loc[fam.index, "p_b2_holm"] = adj[:len(fam)]
    main_rows.loc[fam.index, "p_b3_holm"] = adj[len(fam):]
    main_rows.to_csv(out / "its_main.csv", index=False)

    # robustness: stable keyword subset (AI group restricted to stable-set titles) and masked M1
    stable = df[(df["group"] == "control") | (df["ai_stable"])]
    rob = series_table(stable, "stable_keywords", {"m1": EMO7}) + \
        series_table(df, "topic_masked", {"m1m": EMO7})
    pd.DataFrame(rob).to_csv(out / "its_robust.csv", index=False)

    # post-hoc (NOT pre-registered) sensitivity to the pre-trend specification:
    # (a) trend-free local contrast: mean gap in the 12 months after vs the 12 months before T0
    loc_rows = []
    for tag in ["m1", "m2", "m3"]:
        cols = [f"{tag}_{e}" for e in INSTR[tag]]
        mm = monthly_means(df, cols)
        t = mm.index.values
        for e, c in zip(INSTR[tag], cols):
            gap = mm[(c, "ai")].values - mm[(c, "control")].values
            pre, post = gap[(t >= T0 - 12) & (t < T0)], gap[(t >= T0) & (t < T0 + 12)]
            from scipy.stats import ttest_ind
            tt = ttest_ind(post, pre, equal_var=False)
            loc_rows.append({"instrument": tag, "emotion": e, "diff": post.mean() - pre.mean(),
                             "p_welch": tt.pvalue})
    pd.DataFrame(loc_rows).to_csv(out / "local_contrast.csv", index=False)

    # (b) specification curve over the start of the pre-period (every January 2015-2021)
    sc_rows = []
    for start_year in range(2015, 2022):
        t_start = (start_year - 2015) * 12
        for r in series_table(df[df["t"] >= t_start], f"start{start_year}",
                              {"m1": EMO7, "m2": EMO7, "m3": INSTR["m3"]}):
            if r["model"] == "comparative":
                sc_rows.append({"start_year": start_year, **r})
    pd.DataFrame(sc_rows).to_csv(out / "prestart_curve.csv", index=False)

    # comment-level model with month fixed effects and story-clustered SE (M1)
    df["ai"] = (df["group"] == "ai").astype(float)
    df["post"] = (df["t"] >= T0).astype(float)
    df["ai_t"] = df["ai"] * df["t"]
    df["ai_post"] = df["ai"] * df["post"]
    df["ai_slope"] = df["ai"] * (df["t"] - T0) * df["post"]
    cl_rows = []
    for e in EMO7:
        r = smf.ols(f"m1_{e} ~ C(t) + ai + ai_t + ai_post + ai_slope", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(df["story_id"])[0]})
        cl_rows.append({"emotion": e, "b2_level": r.params["ai_post"], "se_b2": r.bse["ai_post"],
                        "p_b2": r.pvalues["ai_post"], "b3_slope": r.params["ai_slope"],
                        "se_b3": r.bse["ai_slope"], "p_b3": r.pvalues["ai_slope"]})
    pd.DataFrame(cl_rows).to_csv(out / "comment_level.csv", index=False)

    # placebo interventions using pre-period only
    pl_rows = []
    for tag in ["m1", "m2", "m3"]:
        cols = [f"{tag}_{e}" for e in INSTR[tag]]
        mm = monthly_means(df[df["t"] <= PRE_END], cols)
        t = mm.index.values.astype(float)
        for e, c in zip(INSTR[tag], cols):
            d = mm[(c, "ai")].values - mm[(c, "control")].values
            for p0 in PLACEBO_T0:
                r = its_fit(d, t, p0)
                pl_rows.append({"instrument": tag, "emotion": e, "placebo_t0": p0,
                                "window_end": int(t.max()), "b2_level": r["b2_level"], "p_b2": r["p_b2"]})
    pl = pd.DataFrame(pl_rows)
    assert (pl["window_end"] < T0).all(), "placebo window overlaps the true intervention"
    pl.to_csv(out / "placebo.csv", index=False)

    comp = main_rows[main_rows.model == "comparative"].set_index(["instrument", "emotion"])
    summ = []
    for (tag, e), g in pl.groupby(["instrument", "emotion"]):
        true_b2 = comp.loc[(tag, e), "b2_level"]
        exceed = g[g["b2_level"].abs() >= abs(true_b2)]
        summ.append({"instrument": tag, "emotion": e, "true_b2": true_b2, "n_placebo": len(g),
                     "emp_p": float(len(exceed) / len(g)),
                     # post-hoc description (not pre-registered): same-sign exceedances only
                     "emp_p_same_sign": float((np.sign(g["b2_level"]) * g["b2_level"].abs()
                                               * np.sign(true_b2) >= abs(true_b2)).mean()),
                     "n_exceed": int(len(exceed)),
                     "exceed_t0_min": int(exceed["placebo_t0"].min()) if len(exceed) else -1,
                     "exceed_t0_max": int(exceed["placebo_t0"].max()) if len(exceed) else -1,
                     "exceed_opposite_sign": int((np.sign(exceed["b2_level"]) != np.sign(true_b2)).sum()),
                     # post-hoc: non-overlapping subset (every 6 months) to reduce dependence
                     "n_placebo_sparse": int(g["placebo_t0"].isin(PLACEBO_SPARSE).sum()),
                     "emp_p_sparse": float((g[g["placebo_t0"].isin(PLACEBO_SPARSE)]["b2_level"].abs()
                                            >= abs(true_b2)).mean()),
                     "share_placebo_p05": float((g["p_b2"] < 0.05).mean())})
    pd.DataFrame(summ).to_csv(out / "placebo_summary.csv", index=False)

    # post-hoc (NOT pre-registered): sensitivity of the HAC inference to the lag length,
    # for the true intervention and for the placebo false-positive rate (M1 only)
    global HAC_LAG
    hac_rows = []
    cols = [f"m1_{e}" for e in EMO7]
    mm_all, mm_pre = monthly_means(df, cols), monthly_means(df[df["t"] <= PRE_END], cols)
    t_all, t_pre = mm_all.index.values.astype(float), mm_pre.index.values.astype(float)
    main_lag = HAC_LAG
    for lag in [2, 4, 8, 12]:
        HAC_LAG = lag
        for e, c in zip(EMO7, cols):
            r = its_fit(mm_all[(c, "ai")].values - mm_all[(c, "control")].values, t_all, T0)
            d_pre = mm_pre[(c, "ai")].values - mm_pre[(c, "control")].values
            fp = np.mean([its_fit(d_pre, t_pre, p0)["p_b2"] < 0.05 for p0 in PLACEBO_T0])
            hac_rows.append({"lag": lag, "emotion": e, "p_b2": r["p_b2"], "se_b2": r["se_b2"],
                             "placebo_share_p05": fp})
    HAC_LAG = main_lag
    pd.DataFrame(hac_rows).to_csv(out / "hac_sensitivity.csv", index=False)
    json.dump({"n_comments": int(len(df)), "n_stories": int(df["story_id"].nunique()),
               "by_group": df.groupby("group").size().to_dict()},
              open(out / "meta.json", "w"), indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
