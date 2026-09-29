"""Post-hoc (not in the analysis plan): sensitivity of M2 to the Ekman aggregation rule.

Re-scores the seed-2026 comments with the GoEmotions model, stores all 28 label
probabilities, and compares three aggregations of member labels into Ekman categories:
max (main analysis), mean, and noisy-OR 1 - prod(1 - p).
Outputs results/its_seed2026/m2_aggregation.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


clf = _load("03_classify")
its = _load("04_its")


def main(seed: int = 2026) -> None:
    df = pd.read_parquet(ROOT / "data" / f"comments_seed{seed}.parquet").reset_index(drop=True)
    raw_path = ROOT / "results" / f"goemotions28_seed{seed}.parquet"
    if raw_path.exists():
        go = pd.read_parquet(raw_path)
    else:
        go = clf.run_model("SamLowe/roberta-base-go_emotions", df["text"].tolist(), sigmoid=True)
        go.to_parquet(raw_path)
    scores = pd.read_parquet(ROOT / "results" / f"scores_seed{seed}.parquet").reset_index(drop=True)
    # sanity check: the max aggregation must reproduce the stored M2 scores
    for e, members in clf.EKMAN.items():
        assert np.allclose(go[members].max(axis=1).values, scores[f"m2_{e}"].values, atol=1e-5), e

    base = scores[["story_id", "month", "group", "ai_stable", "cid"]].copy()
    base["t"] = its.month_index(base["month"])
    rows = []
    for agg in ["max", "mean", "noisyor"]:
        d = base.copy()
        for e, members in clf.EKMAN.items():
            p = go[members]
            if agg == "max":
                v = p.max(axis=1)
            elif agg == "mean":
                v = p.mean(axis=1)
            else:
                v = 1 - (1 - p).prod(axis=1)
            d[f"m2_{e}"] = v.values
        for r in its.series_table(d, agg, {"m2": its.EMO7}):
            if r["model"] == "comparative":
                rows.append({"aggregation": agg, **r})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / f"its_seed{seed}" / "m2_aggregation.csv", index=False)
    print(out[["aggregation", "emotion", "b2_level", "p_b2", "std_b2"]].round(4).to_string())


if __name__ == "__main__":
    main()
