"""Plan addendum 5: robustness of the AI-mention subset.

For every sampling: beta_2 of fear with M1, topic-masked M1 (m1m), M2, and M3, restricting the
AI group to comments whose text mentions an AI keyword (control = all non-AI comments).
Pooled over the ten samplings (de-duplicated by comment id): the same, with placebo exceedance
rates, and M1 with a restricted cubic spline pre-trend.
Outputs results/mention_robustness.csv, results/mention_pooled.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_s = importlib.util.spec_from_file_location("its", HERE / "04_its.py"); its = importlib.util.module_from_spec(_s); _s.loader.exec_module(its)
_s = importlib.util.spec_from_file_location("spc", HERE / "12_spec_sensitivity.py"); spc = importlib.util.module_from_spec(_s); _s.loader.exec_module(spc)
from keywords import BROAD_RE  # noqa: E402
COLS = ["m1_fear", "m1m_fear", "m2_fear", "m3_fear"]


def load(seed: int) -> pd.DataFrame:
    com = pd.read_parquet(ROOT / "data" / f"comments_seed{seed}.parquet", columns=["cid", "group", "text"])
    sc = pd.read_parquet(ROOT / "results" / f"scores_seed{seed}.parquet")
    ai = com[com.group == "ai"].drop_duplicates("cid")  # one duplicated comment in seed 2026 (API tree)
    ment = ai.set_index("cid").text.apply(lambda s: any(r.search(s) for r in BROAD_RE))
    sc["mentions_ai"] = sc.cid.map(ment).fillna(False)
    sc["t"] = its.month_index(sc["month"])
    return sc[(sc.t >= 0) & (sc.t <= 137)]


def estimate(df: pd.DataFrame, placebo: bool = False) -> list[dict]:
    d = df[(df.group == "control") | df.mentions_ai]
    mm = its.monthly_means(d, COLS)
    t = mm.index.values.astype(float)
    rows = []
    for c in COLS:
        gap = mm[(c, "ai")].values - mm[(c, "control")].values
        ok = ~np.isnan(gap)
        r = its.its_fit(gap[ok], t[ok], its.T0)
        pre = ok & (t < its.T0)
        rec = {"measure": c, "b2": r["b2_level"], "p_b2": r["p_b2"], "std_b2": r["b2_level"] / gap[pre].std(ddof=1)}
        if placebo:
            bs = [its.its_fit(gap[pre], t[pre], p0)["b2_level"] for p0 in its.PLACEBO_T0]
            rec["placebo_exceed"] = float(np.mean(np.abs(bs) >= abs(r["b2_level"])))
            if c == "m1_fear":
                kn = np.quantile(t[pre], [0.05, 0.35, 0.65, 0.95])
                rs = spc.fit(gap[ok], t[ok], extra=spc.rcs_basis(t[ok], kn))
                rec["b2_rcs"], rec["p_rcs"] = rs["b2"], rs["p"]
        rows.append(rec)
    return rows


def main() -> None:
    seeds = sorted(int(p.stem.replace("scores_seed", "")) for p in (ROOT / "results").glob("scores_seed*.parquet"))
    rows, frames = [], []
    for sd in seeds:
        df = load(sd)
        frames.append(df)
        for r in estimate(df):
            rows.append({"seed": sd, **r})
    per = pd.DataFrame(rows)
    per.to_csv(ROOT / "results" / "mention_robustness.csv", index=False)
    pooled = pd.concat(frames).drop_duplicates("cid")
    pool = pd.DataFrame(estimate(pooled, placebo=True))
    pool["n_ai_mention_comments"] = int(((pooled.group == "ai") & pooled.mentions_ai).sum())
    pool.to_csv(ROOT / "results" / "mention_pooled.csv", index=False)
    piv = per.assign(b2pp=per.b2 * 100).pivot(index="seed", columns="measure", values="b2pp").round(2)
    print(piv.to_string())
    print(per.groupby("measure").apply(lambda g: pd.Series({"n_pos": int((g.b2 > 0).sum()), "n_p05": int((g.p_b2 < 0.05).sum())})))
    print(pool.assign(b2pp=pool.b2 * 100).round(4).to_string())


if __name__ == "__main__":
    main()
