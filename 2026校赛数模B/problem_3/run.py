"""
Problem 3: greedy selection of 8 molecules (activity, low LS, diversity, region coverage).

Requires problem_1 and problem_2 CSV outputs. Run: python -m problem_3.run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem_3 import config as cfg

OUT_DIR = Path(__file__).resolve().parent / "outputs"
P2_MOL = ROOT / "problem_2" / "outputs" / "02_molecule_local_sensitivity.csv"


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / (hi - lo + cfg.EPS)


def _region_type_cn(t: str) -> str:
    return "高活性热点区域" if t == "hotspot" else "普通区域"


def _pairwise_mean_dist(xy: np.ndarray) -> float:
    n = len(xy)
    if n < 2:
        return 0.0
    dsum = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dsum += float(np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
            cnt += 1
    return dsum / cnt


def _set_metrics(df: pd.DataFrame, ids: list[int]) -> dict[str, float]:
    sub = df[df["ID"].isin(ids)]
    ls = sub["local_sensitivity"].to_numpy(dtype=float)
    q75 = float(df["local_sensitivity"].quantile(0.75))
    xy = sub[["Manifold_X", "Manifold_Y"]].to_numpy(dtype=float)
    return {
        "mean_bioactivity": float(sub["Bioactivity_Score"].mean()),
        "mean_activity_norm": float(sub["activity_minmax"].mean()),
        "mean_local_sensitivity": float(ls.mean()),
        "high_ls_count": int((ls > q75).sum()),
        "n_regions_covered": int(sub["region_id"].nunique()),
        "mean_pairwise_dist": _pairwise_mean_dist(xy),
    }


def _pick_reason(
    *,
    step: int,
    is_new_region: bool,
    d_norm: float,
    activity_norm: float,
    l_norm: float,
    u_norm: float,
) -> str:
    if step == 1:
        return "净收益最高"
    if is_new_region:
        return "覆盖新区域"
    if d_norm >= 0.12:
        return "空间补充性强"
    if activity_norm >= 0.55 and l_norm <= 0.40:
        return "高活性低敏感"
    if u_norm >= 0.65:
        return "净收益较高"
    return "综合边际收益较高"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not P2_MOL.exists():
        raise FileNotFoundError(f"Run problem_2 first; missing {P2_MOL}")

    df = pd.read_csv(P2_MOL)
    need = {
        "ID",
        "Bioactivity_Score",
        "activity_minmax",
        "local_sensitivity",
        "region_id",
        "Manifold_X",
        "Manifold_Y",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    A = df["activity_minmax"].to_numpy(dtype=float)
    ls = df["local_sensitivity"].to_numpy(dtype=float)
    L = _safe_minmax(ls)
    U = cfg.ALPHA * A - cfg.BETA * L
    U_rank = _safe_minmax(U)

    xy = df[["Manifold_X", "Manifold_Y"]].to_numpy(dtype=float)
    xmax, xmin = xy[:, 0].max(), xy[:, 0].min()
    ymax, ymin = xy[:, 1].max(), xy[:, 1].min()
    bbox_diag = float(np.hypot(xmax - xmin, ymax - ymin)) + cfg.EPS

    ids = df["ID"].astype(int).tolist()
    rid_arr = df["region_id"].to_numpy()
    id_to_idx = {i: k for k, i in enumerate(ids)}

    scores_tbl = df.assign(L_norm=L, U_i=U, U_rank=U_rank)[
        ["ID", "Bioactivity_Score", "activity_minmax", "local_sensitivity", "L_norm", "U_i", "U_rank", "region_id"]
    ]
    scores_tbl.to_csv(OUT_DIR / "01_all_molecule_scores.csv", index=False, encoding="utf-8-sig")

    selected: list[int] = []
    step_rows: list[dict] = []

    first = int(df.iloc[int(np.argmax(U))]["ID"])
    fi = id_to_idx[first]
    selected.append(first)
    step_rows.append(
        {
            "step": 1,
            "ID": first,
            "marginal_score": float(U_rank[fi]),
            "U_rank": float(U_rank[fi]),
            "diversity_norm": 0.0,
            "region_bonus": 0.0,
            "is_new_region": False,
            "reason": _pick_reason(
                step=1,
                is_new_region=False,
                d_norm=0.0,
                activity_norm=float(A[fi]),
                l_norm=float(L[fi]),
                u_norm=float(U_rank[fi]),
            ),
        }
    )

    while len(selected) < cfg.K_SELECT:
        sel_idx = [id_to_idx[s] for s in selected]
        sel_regions = {int(rid_arr[j]) for j in sel_idx}

        best_id: int | None = None
        best_score = -np.inf
        best_meta: dict = {}

        for cand_id in ids:
            if cand_id in selected:
                continue
            ci = id_to_idx[cand_id]
            dists = np.hypot(xy[ci, 0] - xy[sel_idx, 0], xy[ci, 1] - xy[sel_idx, 1])
            dmin = float(np.min(dists))
            d_norm = dmin / bbox_diag
            is_new = int(rid_arr[ci]) not in sel_regions
            b = 1.0 if is_new else 0.0
            marginal = cfg.ETA_U * U_rank[ci] + cfg.ETA_DIV * d_norm + cfg.ETA_REGION * b
            if marginal > best_score:
                best_score = marginal
                best_id = cand_id
                best_meta = {
                    "marginal_score": float(marginal),
                    "U_rank": float(U_rank[ci]),
                    "diversity_norm": d_norm,
                    "region_bonus": b,
                    "is_new_region": is_new,
                }

        if best_id is None:
            break
        bi = id_to_idx[best_id]
        reason = _pick_reason(
            step=len(selected) + 1,
            is_new_region=bool(best_meta["is_new_region"]),
            d_norm=float(best_meta["diversity_norm"]),
            activity_norm=float(A[bi]),
            l_norm=float(L[bi]),
            u_norm=float(U_rank[bi]),
        )
        selected.append(best_id)
        step_rows.append({"step": len(selected), "ID": best_id, "reason": reason, **best_meta})

    sel_rows = []
    for rank, mid in enumerate(selected, start=1):
        row = df.loc[df["ID"] == mid].iloc[0]
        st = step_rows[rank - 1]
        pr = str(row["paper_region"]) if "paper_region" in df.columns else f"R_{int(row['region_id'])+1}"
        rt = row["region_type"] if "region_type" in df.columns else ""
        sel_rows.append(
            {
                "rank": rank,
                "ID": int(mid),
                "paper_region": pr,
                "region_id": int(row["region_id"]),
                "region_type": rt,
                "region_type_cn": _region_type_cn(rt) if rt else "",
                "Bioactivity_Score": float(row["Bioactivity_Score"]),
                "activity_minmax": float(row["activity_minmax"]),
                "local_sensitivity": float(row["local_sensitivity"]),
                "L_norm": float(L[id_to_idx[mid]]),
                "U_i": float(U[id_to_idx[mid]]),
                "U_rank": float(U_rank[id_to_idx[mid]]),
                "Manifold_X": float(row["Manifold_X"]),
                "Manifold_Y": float(row["Manifold_Y"]),
                "marginal_score": st.get("marginal_score", np.nan),
                "diversity_norm": st.get("diversity_norm", np.nan),
                "region_bonus": st.get("region_bonus", np.nan),
                "reason": st["reason"],
            }
        )

    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(OUT_DIR / "02_selected_8_molecules.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(step_rows).to_csv(OUT_DIR / "03_selection_steps.csv", index=False, encoding="utf-8-sig")

    model_ids = selected
    baseline_ids = df.nlargest(cfg.K_SELECT, "Bioactivity_Score")["ID"].astype(int).tolist()

    m_model = _set_metrics(df, model_ids)
    m_base = _set_metrics(df, baseline_ids)

    cmp_rows = [
        {"metric": "mean_bioactivity", "label": "平均活性得分", "model": m_model["mean_bioactivity"], "baseline": m_base["mean_bioactivity"]},
        {"metric": "mean_activity_norm", "label": "平均归一化活性", "model": m_model["mean_activity_norm"], "baseline": m_base["mean_activity_norm"]},
        {"metric": "mean_local_sensitivity", "label": "平均局部敏感度", "model": m_model["mean_local_sensitivity"], "baseline": m_base["mean_local_sensitivity"]},
        {"metric": "high_ls_count", "label": "高敏感分子数量", "model": m_model["high_ls_count"], "baseline": m_base["high_ls_count"]},
        {"metric": "n_regions_covered", "label": "覆盖区域数", "model": m_model["n_regions_covered"], "baseline": m_base["n_regions_covered"]},
        {"metric": "mean_pairwise_dist", "label": "平均两两空间距离", "model": m_model["mean_pairwise_dist"], "baseline": m_base["mean_pairwise_dist"]},
    ]
    pd.DataFrame(cmp_rows).to_csv(OUT_DIR / "04_model_vs_activity_top8.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {"key": "ALPHA", "value": cfg.ALPHA},
            {"key": "BETA", "value": cfg.BETA},
            {"key": "ETA_U", "value": cfg.ETA_U},
            {"key": "ETA_DIV", "value": cfg.ETA_DIV},
            {"key": "ETA_REGION", "value": cfg.ETA_REGION},
            {"key": "K_SELECT", "value": cfg.K_SELECT},
            {"key": "baseline_rule", "value": "top8_Bioactivity_Score"},
        ]
    ).to_csv(OUT_DIR / "05_run_meta.csv", index=False, encoding="utf-8-sig")

    print("[problem_3] selected IDs:", selected)
    print(f"[problem_3] regions covered: {m_model['n_regions_covered']}, baseline: {m_base['n_regions_covered']}")
    print(f"[problem_3] done -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
