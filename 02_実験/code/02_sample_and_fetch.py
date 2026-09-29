"""Sample AI / control stories per month and fetch their comments.

Usage: python3 02_sample_and_fetch.py --seed 2026
Sampling protocol (pre-registered): stories with points >= 20 and num_comments >= 10;
per month and group, up to 30 stories drawn uniformly at random; per story, up to 40
comments (any depth) with >= 5 words after HTML stripping, drawn uniformly at random.
Comment trees are cached per story, so re-sampling with other seeds reuses downloads.
"""
import argparse
import gzip
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from keywords import is_ai

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "stories_daily"
CACHE = ROOT / "data" / "items"
CACHE.mkdir(parents=True, exist_ok=True)
ITEM_API = "https://hn.algolia.com/api/v1/items/{}"
N_STORIES, N_COMMENTS, MIN_WORDS = 30, 40, 5
TAG_RE = re.compile(r"<[^>]+>")


def load_stories() -> pd.DataFrame:
    rows = []
    for f in sorted(DAILY.glob("*.json")):
        rows.extend(json.loads(f.read_text()))
    df = pd.DataFrame(rows).drop_duplicates("objectID")
    df["title"] = df["title"].fillna("")
    df["month"] = pd.to_datetime(df["created_at_i"], unit="s", utc=True).dt.strftime("%Y-%m")
    df["ai_broad"] = df["title"].map(is_ai)
    df["ai_stable"] = df["title"].map(lambda t: is_ai(t, stable=True))
    return df


def clean(text: str) -> str:
    text = html.unescape(TAG_RE.sub(" ", text.replace("<p>", "\n")))
    return re.sub(r"\s+", " ", text).strip()


def flatten(node: dict, depth: int, out: list) -> None:
    for ch in node.get("children") or []:
        if ch.get("type") == "comment" and ch.get("text"):
            t = clean(ch["text"])
            if len(t.split()) >= MIN_WORDS:
                out.append({"cid": ch["id"], "depth": depth, "t": ch.get("created_at_i"), "text": t})
        flatten(ch, depth + 1, out)


def fetch_comments(story_id: str) -> list[dict]:
    f = CACHE / f"{story_id}.json.gz"
    if f.exists():
        return json.loads(gzip.decompress(f.read_bytes()))
    for attempt in range(6):
        try:
            r = requests.get(ITEM_API.format(story_id), timeout=60)
            if r.status_code == 200:
                out: list[dict] = []
                flatten(r.json(), 1, out)
                tmp = f.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(json.dumps(out).encode()))
                tmp.rename(f)
                time.sleep(0.3)
                return out
            if r.status_code == 404:
                return []
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed story {story_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    st = load_stories()
    st.to_parquet(ROOT / "data" / "stories_all.parquet")
    elig = st[st["num_comments"] >= 10]
    picks = []
    for (month, grp), g in elig.groupby(["month", "ai_broad"]):
        k = min(N_STORIES, len(g))
        idx = rng.choice(len(g), size=k, replace=False)
        s = g.iloc[np.sort(idx)].copy()
        s["group"] = "ai" if grp else "control"
        picks.append(s)
    picks = pd.concat(picks)
    print(f"sampled stories: {len(picks)}", picks.groupby("group").size().to_dict())

    ids = picks["objectID"].tolist()
    with ThreadPoolExecutor(max_workers=4) as ex:
        trees = dict(zip(ids, ex.map(fetch_comments, ids)))

    rows = []
    for _, s in picks.iterrows():
        cs = trees[s["objectID"]]
        k = min(N_COMMENTS, len(cs))
        for j in np.sort(rng.choice(len(cs), size=k, replace=False)) if k else []:
            c = cs[j]
            rows.append({
                "story_id": s["objectID"], "month": s["month"], "group": s["group"],
                "ai_stable": s["ai_stable"], "points": s["points"], **c,
            })
    df = pd.DataFrame(rows)
    out = ROOT / "data" / f"comments_seed{args.seed}.parquet"
    df.to_parquet(out)
    print(f"comments: {len(df)} -> {out}")
    print(df.groupby("group").size().to_dict())


if __name__ == "__main__":
    main()
