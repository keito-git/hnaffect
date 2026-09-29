"""R-L emotion target (substitute for human target annotation).

Zero-shot NLI (facebook/bart-large-mnli) scores whether the emotion of each
AI-group comment in the primary sample is directed at AI, an AI company, or an AI product
(entailment probability >= 0.5). The NLI decision is validated against the LLM reference
(19/20), and the comparative ITS for M1 fear is re-estimated with the AI series restricted to
(i) NLI AI-targeted comments and (ii) comments whose text mentions an AI keyword.
Outputs results/target_subset.csv, results/target_subset.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_s = importlib.util.spec_from_file_location("its", HERE / "04_its.py"); its = importlib.util.module_from_spec(_s); _s.loader.exec_module(its)
from keywords import BROAD_RE  # noqa: E402

HYP = "The emotion in this comment is directed at artificial intelligence, an AI company, or an AI product."
MODEL = "facebook/bart-large-mnli"  # plan addendum 3 (bandwidth)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mention-only", action="store_true")
    args = ap.parse_args()
    seed = args.seed
    com = pd.read_parquet(ROOT / "data" / f"comments_seed{seed}.parquet")
    sc = pd.read_parquet(ROOT / "results" / f"scores_seed{seed}.parquet")
    ai = com[com.group == "ai"].reset_index(drop=True)
    cache = ROOT / "results" / f"nli_target_seed{seed}.parquet"
    if args.mention_only:
        prob = pd.DataFrame({"cid": ai.cid.values, "p_ai_target": np.nan})
    elif cache.exists():
        prob = pd.read_parquet(cache)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float16).to("mps").eval()
        ent, con = mdl.config.label2id["entailment"], mdl.config.label2id["contradiction"]
        # literal special tokens such as "</s>" occur in code snippets on HN; neutralize them
        texts = (ai.text.str.slice(0, 1500).str.replace("</s>", "< /s>", regex=False)
                 .str.replace("<s>", "< s>", regex=False)).tolist()
        order = np.argsort([len(x) for x in texts])
        p = np.zeros(len(ai))
        with torch.no_grad():
            for i in range(0, len(ai), 64):
                idx = order[i:i + 64]
                enc = tok([texts[j] for j in idx], [HYP] * len(idx), truncation="only_first", max_length=512,
                          padding=True, return_tensors="pt").to("mps")
                lg = mdl(**enc).logits.float()
                p[idx] = torch.softmax(lg[:, [con, ent]], dim=-1)[:, 1].cpu().numpy()
                if (i // 64) % 200 == 0:
                    print(f"  nli {i}/{len(ai)}", flush=True)
        prob = pd.DataFrame({"cid": ai.cid.values, "p_ai_target": p})
        prob.to_parquet(cache)
    ai = ai.merge(prob, on="cid")
    ai["nli_ai"] = ai.p_ai_target >= 0.5  # all False in mention-only mode
    ai["mentions_ai"] = ai.text.apply(lambda s: any(r.search(s) for r in BROAD_RE))
    res = {"model": MODEL, "hypothesis": HYP}
    for per, m in [("pre", ai.month < "2022-12"), ("post", ai.month >= "2022-12")]:
        res[f"share_nli_{per}"] = float(ai[m].nli_ai.mean())
        res[f"share_mention_{per}"] = float(ai[m].mentions_ai.mean())
    # validation against the LLM reference on the annotation sample
    ref_path = ROOT / "results" / "llm_annot"
    if seed == 2026 and (ref_path / "qwen14_comments.csv").exists():
        q, p2 = pd.read_csv(ref_path / "qwen14_comments.csv"), pd.read_csv(ref_path / "llama_comments.csv")
        key = pd.read_csv(ROOT.parent / "04_人手注釈キット" / "comments_key_DO_NOT_SHOW_ANNOTATORS.csv")
        r = key.assign(tq=q.target.astype(str), tp=p2.target.astype(str))
        r = r[(r.group == "ai") & (r.tq == r.tp)]
        r["ref_ai"] = r.tq.isin(["ai_tech", "ai_org", "ai_product"])
        r = r.merge(ai[["cid", "nli_ai"]], on="cid")
        tp_ = int((r.nli_ai & r.ref_ai).sum())
        res["validation_n"] = int(len(r))
        res["nli_precision"] = tp_ / max(int(r.nli_ai.sum()), 1)
        res["nli_recall"] = tp_ / max(int(r.ref_ai.sum()), 1)
    # re-analysis
    sc["t"] = its.month_index(sc["month"]); sc = sc[(sc.t >= 0) & (sc.t <= 137)]
    ctl = sc[sc.group == "control"]
    rows = []
    subsets = [("all", ai.cid), ("mentions_ai", ai.cid[ai.mentions_ai])]
    if not args.mention_only:
        subsets.insert(1, ("nli_ai_target", ai.cid[ai.nli_ai]))
    for name, keep in subsets:
        d = pd.concat([sc[(sc.group == "ai") & sc.cid.isin(set(keep))], ctl])
        mm = its.monthly_means(d, ["m1_fear"])
        t = mm.index.values.astype(float)
        gap = mm[("m1_fear", "ai")].values - mm[("m1_fear", "control")].values
        ok = ~np.isnan(gap)
        r2 = its.its_fit(gap[ok], t[ok], its.T0)
        pre = ok & (t < its.T0)
        bs = [its.its_fit(gap[pre], t[pre], p0)["b2_level"] for p0 in its.PLACEBO_T0]
        rows.append({"subset": name, "n_ai_comments": int(len(set(keep))), "b2": r2["b2_level"], "p_b2": r2["p_b2"],
                     "placebo_exceed": float(np.mean(np.abs(bs) >= abs(r2["b2_level"])))})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / f"target_subset_seed{seed}.csv", index=False)
    if seed == 2026 and not args.mention_only:
        out.to_csv(ROOT / "results" / "target_subset.csv", index=False)
        json.dump(res, open(ROOT / "results" / "target_subset.json", "w"), indent=1)
    else:
        json.dump(res, open(ROOT / "results" / f"target_subset_seed{seed}.json", "w"), indent=1)
    print(json.dumps(res, indent=1)); print(out.round(4).to_string())


if __name__ == "__main__":
    main()
