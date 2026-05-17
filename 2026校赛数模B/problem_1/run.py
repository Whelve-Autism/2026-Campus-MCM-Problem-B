"""
Problem 1: 2D manifold clustering (DBSCAN), hotspot labeling, region stats, representatives.

Run from repo root:  python -m problem_1.run
Writes CSVs under problem_1/outputs/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.graph import undirected_edge_max_similarity
from common.loaders import load_edges, load_manifest, manifest_phys_cols
from common.paths import ROOT as PROJECT_ROOT

from problem_1 import config as cfg

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + cfg.EPS)


def region_network_stats(
    region_ids: np.ndarray,
    id_list: np.ndarray,
    undirected_pairs: dict[tuple[int, int], float],
) -> dict[int, tuple[int, int, float, float]]:
    """
    region_id -> (internal_edges, touching_edges, mean_internal_tanimoto, cross_region_edge_ratio).
    cross_region_edge_ratio = (touch - in) / touch for edges incident to the region.
    """
    id_to_region = {int(i): int(r) for i, r in zip(id_list, region_ids)}
    acc = {
        int(r): {"in": 0, "touch": 0, "sim_sum": 0.0}
        for r in np.unique(region_ids)
    }

    for (a, b), sim in undirected_pairs.items():
        ra = id_to_region.get(a)
        rb = id_to_region.get(b)
        if ra is None or rb is None:
            continue
        if ra == rb:
            acc[ra]["in"] += 1
            acc[ra]["touch"] += 1
            acc[ra]["sim_sum"] += sim
        else:
            acc[ra]["touch"] += 1
            acc[rb]["touch"] += 1

    out: dict[int, tuple[int, int, float, float]] = {}
    for r, s in acc.items():
        ein, etouch = s["in"], s["touch"]
        mean_sim = s["sim_sum"] / (ein + cfg.EPS) if ein > 0 else 0.0
        cross_ratio = (etouch - ein) / (etouch + cfg.EPS)
        out[r] = (ein, etouch, mean_sim, cross_ratio)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    edges = load_edges()
    phys_cols = manifest_phys_cols(manifest)

    df = manifest.dropna(subset=["Manifold_X", "Manifold_Y", "Bioactivity_Score"]).copy()
    df.to_csv(OUT_DIR / "01_manifest_clean.csv", index=False, encoding="utf-8-sig")

    xy = df[["Manifold_X", "Manifold_Y"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    xy_std = scaler.fit_transform(xy)

    clustering = DBSCAN(eps=cfg.DBSCAN_EPS, min_samples=cfg.DBSCAN_MIN_SAMPLES)
    labels = clustering.fit_predict(xy_std)
    df["cluster_dbscan"] = labels

    activity = df["Bioactivity_Score"].to_numpy(dtype=float)
    q75_act = float(np.quantile(activity, cfg.HIGH_ACTIVITY_QUANTILE))
    high_mask = activity >= q75_act

    ids = df["ID"].to_numpy()

    undirected = undirected_edge_max_similarity(edges)
    adj: dict[int, set[int]] = {}
    for (a, b) in undirected:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    deg = np.array([len(adj.get(int(i), set())) for i in ids], dtype=float)
    deg_norm = _safe_minmax(deg) if deg.max() > 0 else np.zeros_like(deg)

    act_norm = _safe_minmax(activity)

    uniq_regions = sorted(set(labels.tolist()))
    rows_summary = []
    region_hotspot: dict[int, str] = {}

    net_stat = region_network_stats(labels, ids, undirected)

    for rid in uniq_regions:
        m = labels == rid
        n = int(m.sum())
        if n == 0:
            continue
        sub_act = activity[m]
        mean_a = float(np.mean(sub_act))
        med_a = float(np.median(sub_act))
        p_high = float(np.mean(high_mask[m]))
        ein, etouch, mean_sim, cross_ratio = net_stat.get(int(rid), (0, 0, 0.0, 0.0))
        q_ratio = ein / (etouch + cfg.EPS)

        zmeans = {f"mean_{c}": float(df.loc[m, c].mean()) for c in phys_cols if c in df.columns}

        rows_summary.append(
            {
                "region_id": int(rid),
                "n_molecules": n,
                "mean_bioactivity": mean_a,
                "median_bioactivity": med_a,
                "high_activity_ratio": p_high,
                "internal_edge_count": ein,
                "touching_edge_count": etouch,
                "internal_edge_ratio_q": q_ratio,
                "mean_internal_tanimoto": mean_sim,
                "cross_region_edge_ratio": cross_ratio,
                **zmeans,
            }
        )

    reg_df = pd.DataFrame(rows_summary)
    mean_col = reg_df["mean_bioactivity"].to_numpy(dtype=float)
    norm_mean_a = _safe_minmax(mean_col)
    reg_df["norm_mean_bioactivity"] = norm_mean_a
    reg_df["H_score"] = cfg.LAMBDA1 * norm_mean_a + cfg.LAMBDA2 * reg_df[
        "high_activity_ratio"
    ].to_numpy(dtype=float)

    h_vals = reg_df["H_score"].to_numpy(dtype=float)
    thr_h = float(np.quantile(h_vals, cfg.HOTSPOT_QUANTILE))
    reg_df["region_type"] = np.where(reg_df["H_score"] >= thr_h, "hotspot", "normal")
    for _, r in reg_df.iterrows():
        region_hotspot[int(r["region_id"])] = str(r["region_type"])

    reg_df.to_csv(OUT_DIR / "02_region_summary.csv", index=False, encoding="utf-8-sig")

    df["region_id"] = df["cluster_dbscan"]
    df["region_type"] = df["region_id"].map(region_hotspot).fillna("unknown")

    centroids = df.groupby("region_id", as_index=False)[["Manifold_X", "Manifold_Y"]].mean()
    centroids = centroids.rename(columns={"Manifold_X": "cx", "Manifold_Y": "cy"})
    df = df.merge(centroids, on="region_id", how="left")
    dist_center = np.sqrt(
        (df["Manifold_X"] - df["cx"]) ** 2 + (df["Manifold_Y"] - df["cy"]) ** 2
    ).to_numpy()
    dc_by_region: dict[int, np.ndarray] = {}
    for rid in uniq_regions:
        m = labels == rid
        dc_by_region[int(rid)] = dist_center[m]

    cent_scores = np.zeros(len(df))
    for i, rid in enumerate(labels):
        dloc = dist_center[i]
        subset = dc_by_region[int(rid)]
        lo, hi = np.min(subset), np.max(subset)
        cent_scores[i] = 1.0 - (dloc - lo) / (hi - lo + cfg.EPS)

    rep_score = cfg.OMEGA1 * act_norm + cfg.OMEGA2 * cent_scores + cfg.OMEGA3 * deg_norm
    df["degree"] = deg
    df["deg_norm"] = deg_norm
    df["cent_score"] = cent_scores
    df["rep_score"] = rep_score
    df["is_high_activity_global_q75"] = high_mask

    df.to_csv(OUT_DIR / "03_molecules_with_regions.csv", index=False, encoding="utf-8-sig")

    reps = []
    for rid in uniq_regions:
        sub = df.loc[df["region_id"] == rid].copy()
        if sub.empty:
            continue
        top = sub.nlargest(3, "rep_score")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            reps.append(
                {
                    "region_id": int(rid),
                    "rank_in_region": rank,
                    "ID": int(row["ID"]),
                    "Bioactivity_Score": float(row["Bioactivity_Score"]),
                    "cent_score": float(row["cent_score"]),
                    "deg_norm": float(row["deg_norm"]),
                    "degree": int(row["degree"]),
                    "rep_score": float(row["rep_score"]),
                    "region_type": str(row["region_type"]),
                    "Manifold_X": float(row["Manifold_X"]),
                    "Manifold_Y": float(row["Manifold_Y"]),
                }
            )
    pd.DataFrame(reps).to_csv(
        OUT_DIR / "04_representative_molecules.csv", index=False, encoding="utf-8-sig"
    )

    meta = pd.DataFrame(
        [
            {"key": "global_activity_Q75", "value": q75_act},
            {"key": "hotspot_H_threshold_Q75", "value": thr_h},
            {"key": "DBSCAN_EPS_std_xy", "value": cfg.DBSCAN_EPS},
            {"key": "DBSCAN_MIN_SAMPLES", "value": cfg.DBSCAN_MIN_SAMPLES},
            {"key": "n_molecules_used", "value": len(df)},
            {"key": "n_regions", "value": len(uniq_regions)},
        ]
    )
    meta.to_csv(OUT_DIR / "05_run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[problem_1] done -> {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
