"""Stratified title check (external review item 9), coded blind by an LLM (Claude Opus 5),
not by humans; the annotation kit of the project uses the same 501 titles.

Criterion: the main topic of the title is AI/ML (systems, research, companies, policy, or AI
hardware); autonomous driving and robotics without an AI focus are coded 0, as in 07.
Reports, per period stratum, the keyword rule's precision and the contamination of the control
group, and a population-weighted recall.
Output: results/title_check/title_check_stratified.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "04_人手注釈キット"
AI_ROWS = {
    1, 4, 7, 12, 14, 15, 16, 17, 21, 24, 25, 26, 27, 37, 41, 46, 47, 48, 50, 51, 52, 53, 56, 60, 62,
    68, 71, 72, 73, 77, 80, 81, 82, 84, 87, 88, 94, 97, 98, 102, 105, 108, 110, 113, 116, 118, 122,
    125, 127, 129, 132, 133, 136, 140, 141, 143, 145, 149, 150, 153, 156, 162, 163, 167, 168, 169,
    173, 174, 175, 177, 178, 184, 191, 205, 206, 208, 211, 212, 219, 220, 222, 224, 226, 227, 231,
    236, 237, 242, 243, 244, 246, 249, 252, 253, 255,
    260, 262, 263, 265, 266, 267, 269, 273, 275, 276, 277, 278, 281, 283, 294, 295, 298, 305, 307,
    309, 310, 314, 317, 318, 321, 322, 328, 329, 330, 332, 333, 334, 340, 343, 344, 346, 347, 351,
    354, 357, 359, 362, 363, 364, 365, 367, 374, 377, 381, 382, 384, 385, 393, 398, 399, 400, 405,
    408, 409, 414, 420, 421, 426, 427, 428, 435, 437, 438, 441, 449, 450, 456, 463, 465, 466, 469,
    470, 472, 479, 480, 483, 484, 485, 487, 488, 491, 492, 493, 498, 500,
}


def main() -> None:
    sheet = pd.read_csv(KIT / "titles_annotation_sheet.csv")
    key = pd.read_csv(KIT / "titles_key_DO_NOT_SHOW_ANNOTATORS.csv")
    d = sheet.merge(key, on="item_id")
    d["coded_ai"] = d.index.isin(AI_ROWS)
    el = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet")
    el = el[el.num_comments >= 10]
    el["stratum"] = pd.cut(el.month.str[:4].astype(int), [2014, 2018, 2022, 2026],
                           labels=["2015-2018", "2019-2022", "2023-2026"]).astype(str)
    pop = el.groupby(["stratum", "ai_broad"]).size()
    res = {"coder": "LLM (Claude Opus 5), blind to the keyword label", "strata": {}}
    tp_w = fn_w = 0.0
    for s in ["2015-2018", "2019-2022", "2023-2026"]:
        a = d[(d.stratum == s) & d.ai_broad]; c = d[(d.stratum == s) & ~d.ai_broad]
        prec, cont = a.coded_ai.mean(), c.coded_ai.mean()
        n_ai, n_ctl = pop[(s, True)], pop[(s, False)]
        tp_w += prec * n_ai; fn_w += cont * n_ctl
        res["strata"][s] = {"n_ai": len(a), "precision": prec, "n_ctl": len(c), "fp_in_ai": int((~a.coded_ai).sum()),
                            "contamination": cont, "fn_in_ctl": int(c.coded_ai.sum()),
                            "recall": prec * n_ai / (prec * n_ai + cont * n_ctl)}
    res["recall_overall"] = tp_w / (tp_w + fn_w)
    res["precision_overall"] = float(d[d.ai_broad].coded_ai.mean())
    res["contamination_overall"] = float(d[~d.ai_broad].coded_ai.mean())
    res["false_positive_titles"] = d[d.ai_broad & ~d.coded_ai].title.tolist()
    res["false_negative_titles"] = d[~d.ai_broad & d.coded_ai].title.tolist()
    (ROOT / "02_実験" / "results" / "title_check").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "02_実験" / "results" / "title_check" / "title_check_stratified.json", "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
