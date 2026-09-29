"""Precision/recall of the title keyword rule against blind coding of 300 titles.

The coder is an LLM (Claude), not a human; it coded blind_titles.csv without seeing the
predicted labels. Criterion: the main topic of the title is AI/ML (systems, research,
companies, policy, or AI hardware).
"""
from pathlib import Path
import json
import pandas as pd

D = Path(__file__).resolve().parents[1] / "results" / "title_check"
# row indices coded as AI-related (blind)
AI_ROWS = {1,3,4,8,10,13,16,17,20,21,24,26,27,28,29,30,31,32,36,37,38,43,44,46,48,50,52,60,64,67,
           68,69,72,74,76,77,78,80,81,82,83,86,89,90,91,92,93,99,101,103,105,107,109,110,112,116,
           118,119,120,121,122,123,124,125,130,131,133,134,136,137,138,143,144,145,146,148,149,150,
           151,155,157,160,161,162,165,166,172,174,176,178,179,181,183,185,187,188,192,193,196,199,
           200,201,202,204,205,209,211,213,214,215,216,217,218,219,220,221,223,225,226,227,228,229,
           233,234,235,237,238,240,243,246,248,250,252,255,257,263,264,266,271,273,277,281,283,285,
           286,287,290,291,292,294,296,297}
key = pd.read_csv(D / "key_do_not_open_before_coding.csv", index_col=0)
key["coded_ai"] = key.index.isin(AI_ROWS)
key["pred_ai"] = key["pred"] == "ai"
tp = int((key.pred_ai & key.coded_ai).sum()); fp = int((key.pred_ai & ~key.coded_ai).sum())
fn = int((~key.pred_ai & key.coded_ai).sum()); tn = int((~key.pred_ai & ~key.coded_ai).sum())
res = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
       "precision": tp / (tp + fp), "control_contamination": fn / (fn + tn)}
ctl = key[~key.pred_ai]
for per, m in [("pre", ctl.month < "2022-12"), ("post", ctl.month >= "2022-12")]:
    res[f"ctl_{per}_n"] = int(m.sum())
    res[f"ctl_{per}_fn"] = int((ctl[m].coded_ai).sum())
key.to_csv(D / "coded.csv")
json.dump(res, open(D / "title_check.json", "w"), indent=1)
print(res)
print("FP:", key[key.pred_ai & ~key.coded_ai].title.tolist())
print("FN:", key[~key.pred_ai & key.coded_ai].title.tolist())
