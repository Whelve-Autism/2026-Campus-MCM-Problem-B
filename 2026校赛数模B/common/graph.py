from __future__ import annotations

import pandas as pd


def undirected_edge_max_similarity(edges: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Merge directed KNN edges to undirected; keep max Tanimoto for each pair."""
    best: dict[tuple[int, int], float] = {}
    for _, r in edges.iterrows():
        i, j = int(r["Source"]), int(r["Target"])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        s = float(r["Tanimoto_Similarity"])
        best[(a, b)] = max(best.get((a, b), 0.0), s)
    return best


def adjacency_from_pairs(pairs: dict[tuple[int, int], float]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    for (a, b) in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj
