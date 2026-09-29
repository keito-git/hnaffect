"""Additional analyses, each fixed in writing before being run and reported as unplanned with respect to the
original plan): R-C story weighting, R-D seasonality, R-E functional form.

Usage: python3 12_spec_sensitivity.py --samples 2026 pooled
"pooled" merges every available scores_seed*.parquet (de-duplicated by comment id).
Output: results/spec_sensitivity.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("its", HERE / "04_its.py")
its = importlib.util.module_from_spec(spec)
spec.loader.exec_module(its)
T0, LAG = its.T0, its.HAC_LAG


def load(sample: str) -> pd.DataFrame:
    if sample == "pooled":
        files = sorted((ROOT / "results").glob("scores_seed*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("cid")
    else:
        df = pd.read_parquet(ROOT / "results" / f"scores_seed{sample}.parquet")
    df = df.reset_index(drop=True)
    df["t"] = its.month_index(df["month"])
    return df[(df["t"] >= 0) & (df["t"] <= 137)]


def gap_series(df: pd.DataFrame, col: str, weighting: str) -> tuple[np.ndarray, np.ndarray]:
    if weighting == "comment":
        g = df.groupby(["t", "group"])[col].mean().unstack("group")
    else:  # story-weighted: mean within story, then equal weight per story
        s = df.groupby(["t", "group", "story_id"])[col].mean()
        g = s.groupby(["t", "group"]).mean().unstack("group")
    return g.index.values.astype(float), (g["ai"] - g["control"]).values


def rcs_basis(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline (Harrell); linear beyond the outer knots."""
    k = knots
    def p3(x):
        return np.clip(x, 0, None) ** 3
    cols = []
    for j in range(len(k) - 2):
        cols.append(p3(t - k[j]) - p3(t - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                    + p3(t - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
    B = np.column_stack(cols) / (k[-1] - k[0]) ** 2
    return B


def fit(y: np.ndarray, t: np.ndarray, extra: np.ndarray | None = None, arima: bool = False) -> dict:
    post = (t >= T0).astype(float)
    X = np.column_stack([t, post, (t - T0) * post])
    if extra is not None:
        X = np.column_stack([X, extra])
    if arima:
        m = sm.tsa.SARIMAX(y, exog=X, order=(1, 0, 0), trend="c").fit(disp=False)
        names = m.model.exog_names
        i = 1  # exog order: x1=t, x2=post, x3=slope
        b2, se = m.params[names.index("x2") + 1], m.bse[names.index("x2") + 1]
        from scipy.stats import norm
        return {"b2": b2, "p": 2 * (1 - norm.cdf(abs(b2 / se)))}
    r = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": LAG})
    return {"b2": r.params[2], "p": r.pvalues[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["2026", "pooled"])
    args = ap.parse_args()
    rows = []
    for sample in args.samples:
        df = load(sample)
        for tag in ["m1", "m2", "m3"]:
            col = f"{tag}_fear"
            for weighting in ["comment", "story"]:
                t, y = gap_series(df, col, weighting)
                specs = {"linear": fit(y, t)}
                if weighting == "comment":
                    moy = (t.astype(int) % 12)
                    specs["month_FE"] = fit(y, t, extra=np.column_stack([(moy == m).astype(float) for m in range(1, 12)]))
                    four = np.column_stack([f(2 * np.pi * k * moy / 12) for k in (1, 2) for f in (np.sin, np.cos)])
                    specs["fourier_K2"] = fit(y, t, extra=four)
                    specs["quadratic"] = fit(y, t, extra=(t ** 2)[:, None])
                    pre_t = t[t < T0]
                    knots = np.quantile(pre_t, [0.05, 0.35, 0.65, 0.95])
                    specs["rcs4_pre"] = fit(y, t, extra=rcs_basis(t, knots))
                    specs["arima100"] = fit(y, t, arima=True)
                    for w in (12, 24, 36):
                        m = (t >= T0 - w) & (t < T0 + w)
                        specs[f"local_pm{w}"] = fit(y[m], t[m])
                for name, r in specs.items():
                    rows.append({"sample": sample, "instrument": tag, "weighting": weighting,
                                 "spec": name, "b2": r["b2"], "p": r["p"]})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "spec_sensitivity.csv", index=False)
    show = out[out.instrument == "m1"].assign(b2pp=lambda d: d.b2 * 100)
    print(show[["sample", "weighting", "spec", "b2pp", "p"]].round(4).to_string())


if __name__ == "__main__":
    main()
