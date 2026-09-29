"""Collect metadata of all HN stories with points >= 20 (2015-01-01 to 2026-06-30), one API call per day.

Resumable: each day is written to its own JSON file.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "stories_daily"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://hn.algolia.com/api/v1/search_by_date"
START = datetime(2015, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 7, 1, tzinfo=timezone.utc)
KEEP = ["objectID", "title", "url", "points", "num_comments", "created_at_i"]


def fetch_day(t0: int) -> list[dict]:
    params = {
        "tags": "story",
        "hitsPerPage": 1000,
        "numericFilters": f"created_at_i>={t0},created_at_i<{t0 + 86400},points>=20",
    }
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                assert d["nbHits"] <= 1000, f"day {t0} exceeds 1000 hits"
                return [{k: h.get(k) for k in KEEP} for h in d["hits"]]
            time.sleep(5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed day {t0}")


def _one(day: datetime) -> None:
    f = OUT / f"{day:%Y-%m-%d}.json"
    if f.exists():
        return
    hits = fetch_day(int(day.timestamp()))
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(hits))
    tmp.rename(f)  # atomic: a partially written day is never treated as done
    time.sleep(0.4)


def main() -> None:
    days = []
    day = START
    while day < END:
        days.append(day)
        day += timedelta(days=1)
    # 4 workers keep us well below the Algolia limit of 10,000 requests/hour
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_one, days))
    n = len(list(OUT.glob("*.json")))
    print(f"done: {n}/{len(days)} days")


if __name__ == "__main__":
    main()
