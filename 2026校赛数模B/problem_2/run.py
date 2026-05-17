"""
Problem 2: KNN 边活性悬崖、分子局部敏感度、区域稳定性、活性–敏感关联。

依赖 problem_1 分区结果。运行：python -m problem_2.run
中间结果：problem_2/outputs/
论文表图：python scripts/export_problem2_results.py -> out/problem_2/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.graph import undirected_edge_max_similarity
from common.loaders import load_edges, load_manifest

from problem_2 import config as cfg

OUT_DIR = Path(__file__).resolve().parent / "outputs"
P1_OUT = ROOT / "problem_1" / "outputs"


def _safe_minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    return (arr - lo) / (hi - lo + cfg.EPS)


def _paper_region_labels(region_ids: list[int]) -> dict[int, str]:
    subs = "₀₁₂₃₄₅₆₇₈₉"
    labels: dict[int, str] = {}
    if -1 in region_ids:
        labels[-1] = "R₀"
    pos = sorted(r for r in set(region_ids) if r >= 0)
    for i, rid in enumerate(pos, start=1):
        s = str(i)
        labels[rid] = "R" + "".join(subs[int(c)] for c in s)
    return labels


def _pair_region_label(a: int, b: int, id_to_region: dict[int, int], rmap: dict[int, str]) -> str:
    ra, rb = id_to_region.get(a), id_to_region.get(b)
    if ra is None or rb is None:
        return ""
    la, lb = rmap.get(ra, f"R{ra}"), rmap.get(rb, f"R{rb}")
    return la if ra == rb else f"{la}–{lb}"


def _stability_judgment(
    mean_ls: float,
    u_r: float,
    v_r: float,
    ref_ls: float,
    ref_u: float,
    ref_v: float,
) -> str:
    flags = sum([mean_ls > ref_ls, u_r > ref_u, v_r > ref_v])
    if flags == 0:
        return "较稳定"
    if flags == 1:
        return "中等风险"
    return "高风险"


def _quadrant_label(high_act: bool, high_ls: bool) -> str:
    if high_act and not high_ls:
        return "高活性-低敏感"
    if high_act and high_ls:
        return "高活性-高敏感"
    if not high_act and not high_ls:
        return "普通活性-低敏感"
    return "普通活性-高敏感"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mol_path = P1_OUT / "03_molecules_with_regions.csv"
    if not mol_path.exists():
        raise FileNotFoundError(f"Run problem_1 first; missing {mol_path}")

    df_regions = pd.read_csv(mol_path)
    manifest = load_manifest()
    edges = load_edges()

    ids = df_regions["ID"].astype(int).tolist()
    id_to_region = dict(zip(df_regions["ID"].astype(int), df_regions["region_id"].astype(int)))
    rmap = _paper_region_labels(df_regions["region_id"].astype(int).tolist())

    act_raw = manifest.set_index("ID")["Bioactivity_Score"]
    act_norm_map = dict(zip(df_regions["ID"].astype(int), _safe_minmax(df_regions["Bioactivity_Score"].to_numpy())))

    undirected = undirected_edge_max_similarity(edges)

    cliff_rows = []
    for (a, b), sim in undirected.items():
        if a not in act_norm_map or b not in act_norm_map:
            continue
        ai, aj = act_norm_map[a], act_norm_map[b]
        ar, br = float(act_raw.get(a, np.nan)), float(act_raw.get(b, np.nan))
        delta_a = abs(ai - aj)
        c_raw = float(sim) * delta_a
        cliff_rows.append(
            {
                "id_min": min(a, b),
                "id_max": max(a, b),
                "Source": a,
                "Target": b,
                "Tanimoto_Similarity": float(sim),
                "activity_i_raw": ar,
                "activity_j_raw": br,
                "activity_i_norm": ai,
                "activity_j_norm": aj,
                "delta_activity_norm": delta_a,
                "cliff_strength_raw": c_raw,
                "region_pair": _pair_region_label(a, b, id_to_region, rmap),
            }
        )

    cliff_df = pd.DataFrame(cliff_rows)
    if cliff_df.empty:
        raise RuntimeError("No undirected edges produced cliff records.")

    cliff_df["cliff_strength_norm"] = _safe_minmax(cliff_df["cliff_strength_raw"].to_numpy())
    q75_c = float(np.quantile(cliff_df["cliff_strength_norm"], cfg.CLIFF_QUANTILE_MODERATE))
    q90_c = float(np.quantile(cliff_df["cliff_strength_norm"], cfg.CLIFF_QUANTILE_STRONG))
    cliff_df["is_cliff_moderate"] = cliff_df["cliff_strength_norm"] >= q75_c
    cliff_df["is_cliff_strong"] = cliff_df["cliff_strength_norm"] >= q90_c

    cliffs_by_node: dict[int, list[float]] = {i: [] for i in ids}
    for _, r in cliff_df.iterrows():
        cn = float(r["cliff_strength_norm"])
        for mid in (int(r["Source"]), int(r["Target"])):
            if mid in cliffs_by_node:
                cliffs_by_node[mid].append(cn)

    ls_rows = []
    for mid in ids:
        vals = cliffs_by_node.get(mid, [])
        if vals:
            ls_mean = float(np.mean(vals))
            ls_max = float(np.max(vals))
            ls = cfg.ALPHA * ls_mean + (1.0 - cfg.ALPHA) * ls_max
        else:
            ls_mean = ls_max = ls = 0.0
        ls_rows.append(
            {
                "ID": mid,
                "local_sensitivity": ls,
                "neighbor_cliff_mean": ls_mean,
                "neighbor_cliff_max": ls_max,
                "neighbor_count": len(vals),
            }
        )

    merged = df_regions.merge(pd.DataFrame(ls_rows), on="ID", how="left")
    merged["activity_minmax"] = _safe_minmax(merged["Bioactivity_Score"].to_numpy(dtype=float))
    merged["paper_region"] = merged["region_id"].map(rmap)

    q_act = float(np.quantile(merged["activity_minmax"], cfg.HIGH_ACTIVITY_QUANTILE))
    q_ls = float(np.quantile(merged["local_sensitivity"], cfg.HIGH_LS_QUANTILE))
    merged["is_high_activity"] = merged["activity_minmax"] >= q_act
    merged["is_high_local_sensitivity"] = merged["local_sensitivity"] >= q_ls
    merged["quadrant_type"] = [
        _quadrant_label(bool(a), bool(s))
        for a, s in zip(merged["is_high_activity"], merged["is_high_local_sensitivity"])
    ]

    merged.to_csv(OUT_DIR / "02_molecule_local_sensitivity.csv", index=False, encoding="utf-8-sig")

    internal_total: dict[int, int] = {}
    internal_cliff: dict[int, int] = {}
    for _, row in cliff_df.iterrows():
        a, b = int(row["Source"]), int(row["Target"])
        ra, rb = id_to_region.get(a), id_to_region.get(b)
        if ra is None or rb is None or ra != rb:
            continue
        internal_total[ra] = internal_total.get(ra, 0) + 1
        if bool(row["is_cliff_moderate"]):
            internal_cliff[ra] = internal_cliff.get(ra, 0) + 1

    stab_rows = []
    for rid, sub in merged.groupby("region_id"):
        rid = int(rid)
        mean_ls = float(sub["local_sensitivity"].mean())
        u_r = float(sub["is_high_local_sensitivity"].mean())
        ein = internal_total.get(rid, 0)
        v_r = internal_cliff.get(rid, 0) / (ein + cfg.EPS)
        stab_rows.append(
            {
                "region_id": rid,
                "paper_region": rmap.get(rid, ""),
                "region_type": sub["region_type"].iloc[0],
                "n_molecules": len(sub),
                "mean_bioactivity": float(sub["Bioactivity_Score"].mean()),
                "mean_local_sensitivity": mean_ls,
                "high_ls_ratio": u_r,
                "internal_edge_count": ein,
                "internal_cliff_edge_count": internal_cliff.get(rid, 0),
                "internal_cliff_edge_ratio": v_r,
            }
        )

    stab_df = pd.DataFrame(stab_rows).sort_values("region_id")
    ref_ls = float(stab_df["mean_local_sensitivity"].median())
    ref_u = float(stab_df["high_ls_ratio"].median())
    ref_v = float(stab_df["internal_cliff_edge_ratio"].median())
    stab_df["stability_judgment"] = [
        _stability_judgment(r.mean_local_sensitivity, r.high_ls_ratio, r.internal_cliff_edge_ratio, ref_ls, ref_u, ref_v)
        for r in stab_df.itertuples()
    ]
    stab_df.to_csv(OUT_DIR / "03_region_stability.csv", index=False, encoding="utf-8-sig")

    a_norm = merged["activity_minmax"].to_numpy(dtype=float)
    ls_arr = merged["local_sensitivity"].to_numpy(dtype=float)
    pr, pp = pearsonr(a_norm, ls_arr)
    sr, sp = spearmanr(a_norm, ls_arr)

    pd.DataFrame(
        [
            {"metric": "pearson_r", "value": float(pr), "p_value": float(pp)},
            {"metric": "spearman_rho", "value": float(sr), "p_value": float(sp)},
            {"metric": "Q75_activity_norm", "value": q_act, "p_value": np.nan},
            {"metric": "Q75_local_sensitivity", "value": q_ls, "p_value": np.nan},
            {"metric": "cliff_norm_Q75", "value": q75_c, "p_value": np.nan},
            {"metric": "cliff_norm_Q90", "value": q90_c, "p_value": np.nan},
        ]
    ).to_csv(OUT_DIR / "04_activity_ls_correlation.csv", index=False, encoding="utf-8-sig")

    cliff_df.to_csv(OUT_DIR / "01_edges_cliff.csv", index=False, encoding="utf-8-sig")

    top_pairs = (
        cliff_df.nlargest(cfg.TOP_CLIFF_PAIRS_PUBLISH, "cliff_strength_norm")
        .reset_index(drop=True)
        .reset_index(names="rank")
    )
    top_pairs["rank"] = top_pairs["rank"] + 1
    top_pairs.to_csv(OUT_DIR / "05_top_cliff_pairs_publish.csv", index=False, encoding="utf-8-sig")

    merged["quadrant_type"].value_counts().reset_index().to_csv(
        OUT_DIR / "06_quadrant_counts.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        [
            {"key": "n_edges", "value": len(cliff_df)},
            {"key": "n_molecules", "value": len(merged)},
            {"key": "pearson_r_norm_activity_vs_LS", "value": float(pr)},
            {"key": "spearman_rho", "value": float(sr)},
        ]
    ).to_csv(OUT_DIR / "07_run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[problem_2] done -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
