"""Post-hoc (not in the analysis plan): placebo calibration of the slope change beta_3.

For each sampling (seeds 2026-2028) and the pooled sample, the comparative ITS is fitted at
every placebo date using pre-release data only, and the empirical p-value of the true beta_3
is the share of placebo |beta_3| >= |true beta_3|. The share of placebo dates with nominal
p < 0.05 for beta_3 is also reported.
Outputs results/slope_placebo.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("its", HERE / "04_its.py")
its = importlib.util.module_from_spec(spec)
spec.loader.exec_module(its)
SEEDS = [2026, 2027, 2028]


def load(sample: str) -> pd.DataFrame:
    if sample == "pooled":
        df = pd.concat([pd.read_parquet(ROOT / "results" / f"scores_seed{s}.parquet") for s in SEEDS])
        df = df.drop_duplicates("cid")
    else:
        df = pd.read_parquet(ROOT / "results" / f"scores_seed{sample}.parquet")
    df = df.reset_index(drop=True)
    df["t"] = its.month_index(df["month"])
    return df[(df["t"] >= 0) & (df["t"] <= 137)]


def main() -> None:
    rows = []
    for sample in [str(s) for s in SEEDS] + ["pooled"]:
        df = load(sample)
        for tag in ["m1", "m2"]:
            cols = [f"{tag}_{e}" for e in its.EMO7]
            mm_all = its.monthly_means(df, cols)
            mm_pre = its.monthly_means(df[df["t"] <= its.PRE_END], cols)
            t_all, t_pre = mm_all.index.values.astype(float), mm_pre.index.values.astype(float)
            for e, c in zip(its.EMO7, cols):
                true = its.its_fit(mm_all[(c, "ai")].values - mm_all[(c, "control")].values, t_all, its.T0)
                d_pre = mm_pre[(c, "ai")].values - mm_pre[(c, "control")].values
                fits = [its.its_fit(d_pre, t_pre, p0) for p0 in its.PLACEBO_T0]
                b3 = np.array([f["b3_slope"] for f in fits])
                rows.append({"sample": sample, "instrument": tag, "emotion": e,
                             "true_b3": true["b3_slope"], "p_b3": true["p_b3"],
                             "emp_p_b3": float((np.abs(b3) >= abs(true["b3_slope"])).mean()),
                             "placebo_share_p05_b3": float(np.mean([f["p_b3"] < 0.05 for f in fits]))})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "slope_placebo.csv", index=False)
    show = out[(out.instrument == "m1") & out.emotion.isin(["anger", "surprise", "fear"])]
    print(show.assign(b3pp=show.true_b3 * 100).round(4).to_string())
    print("mean placebo share p<.05 for b3 (M1):",
          out[out.instrument == "m1"].groupby("sample").placebo_share_p05_b3.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
