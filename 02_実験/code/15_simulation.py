"""R-H simulation study (additional analysis plan).

Data-generating process calibrated on Hacker News (pooled samplings, M1 fear, pre-release months):
  control_t = C_t,              C_t  ~ local level + AR(1) noise fitted to the control series
  ai_t      = C_t + G_t + d*sd*P_t, G_t ~ local level + AR(1) noise fitted to the gap series
Structural change is represented by the stochastic local level (a deviation from the plan's
"break size and frequency", recorded in the plan addendum). d in {0, 0.5, 1.0, 1.5} x SD of
the simulated pre-release gap. 2000 replications per d.
Methods (reject at 0.05, two-sided):
  single     : linear ITS on ai_t, HAC lag 4
  cits       : linear comparative ITS on the gap, HAC lag 4
  cits_plac  : cits and placebo exceedance rate < 0.05 (54 placebo dates on pre data)
  local24    : linear comparative ITS within T0 +- 24 months, HAC lag 4
  rcs        : comparative ITS with a restricted cubic spline pre-trend, HAC lag 4
Output: results/simulation.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_s = importlib.util.spec_from_file_location("its", HERE / "04_its.py"); its = importlib.util.module_from_spec(_s); _s.loader.exec_module(its)
_s = importlib.util.spec_from_file_location("spec", HERE / "12_spec_sensitivity.py"); spc = importlib.util.module_from_spec(_s); _s.loader.exec_module(spc)
T, T0, R = 138, its.T0, 2000
rng = np.random.default_rng(2026)
t = np.arange(T, dtype=float)
POST = (t >= T0).astype(float)
X_LIN = sm.add_constant(np.column_stack([t, POST, (t - T0) * POST]))
KN = np.quantile(t[t < T0], [0.05, 0.35, 0.65, 0.95])
X_RCS = np.column_stack([X_LIN, spc.rcs_basis(t, KN)])
W = (t >= T0 - 24) & (t < T0 + 24)


def hac_p(y, X, col=2):
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return r.params[col], r.pvalues[col]


def fit_uc(y):
    m = sm.tsa.UnobservedComponents(y, level="llevel", autoregressive=1).fit(disp=False)
    p = dict(zip(m.model.param_names, m.params))
    return {"s_irr": np.sqrt(p["sigma2.irregular"]), "s_lvl": np.sqrt(p["sigma2.level"]),
            "s_ar": np.sqrt(p["sigma2.ar"]), "phi": p["ar.L1"], "mean": float(np.mean(y))}


def simulate(par, n):
    lvl = par["mean"] + np.cumsum(rng.normal(0, par["s_lvl"], (n, T)), axis=1)
    ar = np.zeros((n, T)); e = rng.normal(0, par["s_ar"], (n, T))
    for i in range(1, T):
        ar[:, i] = par["phi"] * ar[:, i - 1] + e[:, i]
    return lvl + ar + rng.normal(0, par["s_irr"], (n, T))


def placebo_rate(gap, b2):
    pre = t < T0
    tp, yp = t[pre], gap[pre]
    bs = []
    for p0 in its.PLACEBO_T0:
        post = (tp >= p0).astype(float)
        X = sm.add_constant(np.column_stack([tp, post, (tp - p0) * post]))
        bs.append(np.linalg.lstsq(X, yp, rcond=None)[0][2])
    return np.mean(np.abs(bs) >= abs(b2))


def main() -> None:
    files = sorted((ROOT / "results").glob("scores_seed*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("cid")
    df["t"] = its.month_index(df["month"]); df = df[(df.t >= 0) & (df.t <= 137)]
    g = df.groupby(["t", "group"])["m1_fear"].mean().unstack("group")
    pre = g.index.values < T0
    par_c, par_g = fit_uc(g["control"].values[pre]), fit_uc((g["ai"] - g["control"]).values[pre])
    print("control params", {k: round(v, 5) for k, v in par_c.items()})
    print("gap params", {k: round(v, 5) for k, v in par_g.items()})
    rows = []
    for scen, mult in [("no_drift", 0.0), ("half_drift", 0.5), ("fitted", 1.0)]:
      pc, pg = dict(par_c), dict(par_g)
      pc["s_lvl"] *= mult; pg["s_lvl"] *= mult
      C, G = simulate(pc, R), simulate(pg, R)
      sd_gap = G[:, t < T0].std(axis=1).mean()
      for d in [0.0, 0.5, 1.0, 1.5]:
        rej = {m: 0 for m in ["single", "cits", "cits_plac", "local24", "rcs"]}
        for i in range(R):
            gap = G[i] + d * sd_gap * POST
            ai = C[i] + gap
            rej["single"] += hac_p(ai, X_LIN)[1] < 0.05
            b2, p = hac_p(gap, X_LIN)
            rej["cits"] += p < 0.05
            rej["cits_plac"] += (p < 0.05) and (placebo_rate(gap, b2) < 0.05)
            rej["local24"] += hac_p(gap[W], sm.add_constant(np.column_stack([t[W], POST[W], (t[W] - T0) * POST[W]])))[1] < 0.05
            rej["rcs"] += hac_p(gap, X_RCS)[1] < 0.05
        for m, v in rej.items():
            rows.append({"scenario": scen, "effect_sd": d, "method": m, "reject_rate": v / R})
        print(scen, d, {m: round(v / R, 3) for m, v in rej.items()}, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "simulation.csv", index=False)
    pd.DataFrame([{"which": "control", **par_c}, {"which": "gap", **par_g}]).to_csv(ROOT / "results" / "simulation_params.csv", index=False)


if __name__ == "__main__":
    main()
