"""Embed all eligible story titles and cluster non-AI stories into topics (R-F, R-G).

Model: sentence-transformers/all-MiniLM-L6-v2. Non-AI stories are clustered with k-means (k=20,
random_state=0). Outputs data/title_embeddings.npy, data/story_topics.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
K = 20


def main() -> None:
    st = pd.read_parquet(ROOT / "data" / "stories_all.parquet")
    st = st[st.num_comments >= 10].reset_index(drop=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    emb = model.encode(st["title"].tolist(), batch_size=512, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    np.save(ROOT / "data" / "title_embeddings.npy", emb)
    st["topic"] = -1
    non_ai = ~st["ai_broad"]
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(emb[non_ai.values])
    st.loc[non_ai, "topic"] = km.labels_
    st[["objectID", "month", "title", "points", "num_comments", "ai_broad", "ai_stable", "topic"]] \
        .to_parquet(ROOT / "data" / "story_topics.parquet")
    # describe clusters by their most central titles
    for k in range(K):
        idx = np.where(st["topic"].values == k)[0]
        c = km.cluster_centers_[k]
        top = idx[np.argsort(-(emb[idx] @ c))[:4]]
        print(k, len(idx), " | ".join(st.loc[top, "title"].str.slice(0, 40)))


if __name__ == "__main__":
    main()
