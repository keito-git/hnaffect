"""Post-hoc (not in the analysis plan): pooled estimate over the three samplings.

Comments drawn under seeds 2026, 2027, and 2028 are merged and de-duplicated by comment id,
and the comparative ITS and placebo calibration are re-run on the pooled sample.
Outputs results/its_pooled/{its_main.csv, placebo_summary.csv, meta.json}
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("its", HERE / "04_its.py")
its = importlib.util.module_from_spec(spec)
spec.loader.exec_module(its)
SEEDS = [2026, 2027, 2028]


def main() -> None:
    out = ROOT / "results" / "its_pooled"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(ROOT / "results" / f"scores_seed{s}.parquet") for s in SEEDS])
    n_raw = len(df)
    df = df.drop_duplicates("cid").reset_index(drop=True)
    df["t"] = its.month_index(df["month"])
    df = df[(df["t"] >= 0) & (df["t"] <= 137)]

    rows = pd.DataFrame(its.series_table(df, "pooled", {k: its.INSTR[k] for k in ["m1", "m2", "m3"]}))
    fam = rows[(rows.instrument == "m1") & (rows.model == "comparative")]
    from statsmodels.stats.multitest import multipletests
    adj = multipletests(np.concatenate([fam.p_b2.values, fam.p_b3.values]), method="holm")[1]
    rows["p_b2_holm"], rows["p_b3_holm"] = np.nan, np.nan
    rows.loc[fam.index, "p_b2_holm"] = adj[:len(fam)]
    rows.loc[fam.index, "p_b3_holm"] = adj[len(fam):]
    rows.to_csv(out / "its_main.csv", index=False)

    comp = rows[rows.model == "comparative"].set_index(["instrument", "emotion"])
    summ = []
    for tag in ["m1", "m2", "m3"]:
        cols = [f"{tag}_{e}" for e in its.INSTR[tag]]
        mm = its.monthly_means(df[df["t"] <= its.PRE_END], cols)
        t = mm.index.values.astype(float)
        for e, c in zip(its.INSTR[tag], cols):
            d = mm[(c, "ai")].values - mm[(c, "control")].values
            b = np.array([its.its_fit(d, t, p0)["b2_level"] for p0 in its.PLACEBO_T0])
            true_b2 = comp.loc[(tag, e), "b2_level"]
            summ.append({"instrument": tag, "emotion": e, "true_b2": true_b2,
                         "emp_p": float((np.abs(b) >= abs(true_b2)).mean())})
    pd.DataFrame(summ).to_csv(out / "placebo_summary.csv", index=False)
    json.dump({"n_comments_raw": n_raw, "n_comments_unique": int(len(df)),
               "n_stories": int(df["story_id"].nunique())}, open(out / "meta.json", "w"), indent=1)
    f = comp.loc[("m1", "fear")]
    print(f"pooled unique comments {len(df)} (raw {n_raw}); M1 fear b2={100*f.b2_level:.3f}pp "
          f"p={f.p_b2:.4f} holm={f.p_b2_holm:.4f} std={f.std_b2:.2f}")
    print(pd.DataFrame(summ).query("instrument=='m1'").round(3).to_string())


if __name__ == "__main__":
    main()
