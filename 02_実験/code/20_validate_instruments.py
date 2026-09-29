"""R-K instrument validity against the LLM reference, and R-M title check (additional plan).

The reference is built from two local LLM annotators (19_llm_annotate.py); it is NOT ground truth.
Outputs results/llm_annot/validation.json, validation_emotions.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "02_実験" / "results"
KIT = ROOT / "04_人手注釈キット"
A = RES / "llm_annot"
EMO6 = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, bins - 1)
    return float(sum(abs(prob[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main() -> None:
    q, p = pd.read_csv(A / "qwen14_comments.csv"), pd.read_csv(A / "llama_comments.csv")
    key = pd.read_csv(KIT / "comments_key_DO_NOT_SHOW_ANNOTATORS.csv")
    sc = pd.read_parquet(RES / "scores_seed2026.parquet")
    d = key.merge(sc, on="cid", how="left", suffixes=("", "_s"))
    res = {"reference": "mean of Qwen2.5-14B-Instruct (4-bit) and Llama-3.1-8B-Instruct (Q4); not ground truth", "agreement": {}}
    ref = pd.DataFrame({"item_id": q.item_id})
    for e in EMO6 + ["neutral"]:
        a, b = pd.to_numeric(q[e], errors="coerce"), pd.to_numeric(p[e], errors="coerce")
        ok = a.notna() & b.notna()
        res["agreement"][e] = float(cohen_kappa_score(a[ok].astype(int), b[ok].astype(int), weights="quadratic"))
        ref[e] = (a + b) / 2
    tq, tp = q["target"].astype(str), p["target"].astype(str)
    res["agreement"]["target"] = float(cohen_kappa_score(tq, tp))
    ref["target"] = np.where(tq == tp, tq, np.nan)
    d = d.merge(ref, on="item_id")
    rows = []
    for e in EMO6 + ["neutral"]:
        y = d[e].values; yb = (y >= 1).astype(int); ok = ~np.isnan(y)
        for tag in ["m1", "m2", "m3"]:
            col = f"{tag}_{e}"
            if col not in d:
                continue
            s = d[col].values
            r = {"emotion": e, "instrument": tag, "spearman": spearmanr(s[ok], y[ok]).statistic,
                 "prevalence": float(yb[ok].mean())}
            if yb[ok].min() != yb[ok].max():
                r["auc"] = roc_auc_score(yb[ok], s[ok])
            if tag in ("m1", "m2"):
                r["brier"] = float(np.mean((s[ok] - yb[ok]) ** 2))
            if tag == "m1":
                r["ece"] = ece(s[ok], yb[ok])
            # aggregate: difference-in-differences sign over the 4 group x period cells
            def did(v):
                m = d.assign(v=v).groupby(["group", "period"]).v.mean()
                return (m["ai", "post"] - m["ai", "pre"]) - (m["control", "post"] - m["control", "pre"])
            r["did_ref"], r["did_instr"] = did(y), did(s)
            r["did_sign_agree"] = bool(np.sign(r["did_ref"]) == np.sign(r["did_instr"]))
            rows.append(r)
    val = pd.DataFrame(rows)
    val.to_csv(A / "validation_emotions.csv", index=False)
    # agreement between instruments on the whole primary sample
    inter = {}
    for e in EMO6:
        inter[e] = {"m1_m2": spearmanr(sc[f"m1_{e}"], sc[f"m2_{e}"]).statistic,
                    "m1_m3": spearmanr(sc[f"m1_{e}"], sc[f"m3_{e}"]).statistic,
                    "m2_m3": spearmanr(sc[f"m2_{e}"], sc[f"m3_{e}"]).statistic}
    res["instrument_agreement"] = inter
    # target shares among AI-group comments by period
    ai = d[d.group == "ai"]
    for per in ["pre", "post"]:
        x = ai[(ai.period == per) & ai.target.notna()]
        res[f"ai_target_share_{per}"] = float(x.target.isin(["ai_tech", "ai_org", "ai_product"]).mean())
        res[f"ai_target_n_{per}"] = int(len(x))
    # titles (R-M)
    tq, tp = pd.read_csv(A / "qwen14_titles.csv"), pd.read_csv(A / "llama_titles.csv")
    tk = pd.read_csv(KIT / "titles_key_DO_NOT_SHOW_ANNOTATORS.csv")
    a, b = pd.to_numeric(tq.is_ai_topic, errors="coerce"), pd.to_numeric(tp.is_ai_topic, errors="coerce")
    res["agreement"]["title"] = float(cohen_kappa_score(a.fillna(-1).astype(int), b.fillna(-1).astype(int)))
    t = tk.assign(ref=np.where(a == b, a, np.nan))
    el = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet"); el = el[el.num_comments >= 10]
    el["stratum"] = pd.cut(el.month.str[:4].astype(int), [2014, 2018, 2022, 2026],
                           labels=["2015-2018", "2019-2022", "2023-2026"]).astype(str)
    pop = el.groupby(["stratum", "ai_broad"]).size()
    tpw = fnw = 0.0; res["titles"] = {}
    for s in ["2015-2018", "2019-2022", "2023-2026"]:
        ai_s, ct_s = t[(t.stratum == s) & t.ai_broad & t.ref.notna()], t[(t.stratum == s) & ~t.ai_broad & t.ref.notna()]
        prec, cont = ai_s.ref.mean(), ct_s.ref.mean()
        tpw += prec * pop[(s, True)]; fnw += cont * pop[(s, False)]
        res["titles"][s] = {"precision": prec, "contamination": cont,
                            "recall": prec * pop[(s, True)] / (prec * pop[(s, True)] + cont * pop[(s, False)])}
    res["titles"]["recall_overall"] = tpw / (tpw + fnw)
    res["titles"]["n_agreed"] = int(t.ref.notna().sum())
    json.dump(res, open(A / "validation.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])
    print(val[val.instrument == "m1"].round(3).to_string())


if __name__ == "__main__":
    main()
