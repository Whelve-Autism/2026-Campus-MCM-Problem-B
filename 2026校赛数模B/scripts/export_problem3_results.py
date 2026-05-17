"""
问题三求解结果：表 1–2 + 图 1–2（各 5:4 单幅）-> out/problem_3/

运行：python scripts/export_problem3_results.py
依赖：python -m problem_1.run && python -m problem_2.run && python -m problem_3.run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out" / "problem_3"
P2_OUT = ROOT / "problem_2" / "outputs"
P3_OUT = ROOT / "problem_3" / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- 5:4 画布（与问题一、二一致） ----------
FIG_W = 5.0
FIG_H = 4.0
ML_IN = 0.54
MR_IN = 0.24
MB_IN = 0.56
MT_IN = 0.20
XLABEL_PAD = 6.0
PANEL_W_FULL = FIG_W - ML_IN - MR_IN
PANEL_H = FIG_H - MB_IN - MT_IN

SPINE_LW = 1.05
TICK_LEN = 4.0
AXIS_LABEL_FS = 10.0
TICK_FS = 9.0
LABELPAD = 3.0
SCATTER_BG = 12
SCATTER_SEL = 72
BAR_EDGE_LW = 0.95

_FONT_MATCH_ORDER = [
    "SimSun",
    "NSimSun",
    "STSong",
    "Songti SC",
    "Source Han Serif SC",
    "Noto Serif CJK SC",
]

XLABEL = "流形空间横坐标"
YLABEL = "流形空间纵坐标"
BAR_EDGE_COLORS = ("#d62828", "#06a77d", "#2563eb")  # 红、绿、蓝
HATCH_CYCLE = ("xxxx", "----", "////")
BAR_LABELS = ("归一化活性", "归一化敏感度", "净收益")


def _panel_rect() -> list[float]:
    return [ML_IN / FIG_W, MB_IN / FIG_H, PANEL_W_FULL / FIG_W, PANEL_H / FIG_H]


def _fig_panel_54() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(_panel_rect())
    return fig, ax


def _fp_size(fp: FontProperties, size_pt: float) -> FontProperties:
    try:
        return FontProperties(fname=fp.get_file(), size=size_pt)
    except Exception:
        return fp


def _fc_file(pattern: str) -> str | None:
    try:
        proc = subprocess.run(
            ["fc-match", "-f", "%{file}", pattern],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        path = proc.stdout.strip()
        if path and Path(path).is_file():
            return path
    except Exception:
        pass
    return None


def _font_prop() -> FontProperties:
    plt.rcParams["axes.unicode_minus"] = False
    for pattern in _FONT_MATCH_ORDER:
        path = _fc_file(pattern)
        if not path or "notosanscjk" in path.lower():
            continue
        return FontProperties(fname=path)
    raise RuntimeError("无法解析宋体/思源宋体系字体。")


def _style_hollow_bar(patch, edge_c: str, hatch: str) -> None:
    patch.set_facecolor("white")
    patch.set_edgecolor(edge_c)
    patch.set_hatch(hatch)
    patch.set_linewidth(BAR_EDGE_LW)
    if hasattr(patch, "set_hatch_color"):
        patch.set_hatch_color(edge_c)


def _style_manifold_ax(ax: plt.Axes, fp_tick: FontProperties, fp_label: FontProperties) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", which="major", length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    ax.set_xlabel(XLABEL, fontproperties=fp_label, labelpad=XLABEL_PAD)
    ax.set_ylabel(YLABEL, fontproperties=fp_label, labelpad=LABELPAD)
    ax.xaxis.label.set_clip_on(False)
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)


def _style_bar_ax(ax: plt.Axes, fp_tick: FontProperties, fp_label: FontProperties) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", which="major", length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    ax.set_xlabel("分子ID", fontproperties=fp_label, labelpad=LABELPAD)
    ax.set_ylabel("归一化指标值", fontproperties=fp_label, labelpad=LABELPAD)
    ax.set_ylim(0.0, 1.05)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
        t.set_rotation(35)
        t.set_ha("right")
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)


def _save_fig(fig: plt.Figure, stem: Path, *, pad_inches: float = 0.02) -> None:
    kw = dict(facecolor="white", edgecolor="none", pad_inches=pad_inches)
    fig.savefig(stem.with_suffix(".pdf"), **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)


def _ensure_outputs() -> None:
    need = P3_OUT / "02_selected_8_molecules.csv"
    if need.exists() and "U_i" in need.read_text(encoding="utf-8", errors="ignore"):
        return
    subprocess.run([sys.executable, "-m", "problem_3.run"], cwd=str(ROOT), check=True)


def _write_table1(path: Path, sel: pd.DataFrame) -> None:
    lines = [
        "表1 最终推荐的8个候选分子",
        "说明：U_i=α·A_i-β·L_i（α=0.6, β=0.4）；入选顺序为贪心边际收益求解结果。",
        "",
        "序号\t分子ID\t所属区域\t区域类型\t活性得分\t局部敏感度\t净收益U_i\t推荐理由",
    ]
    for _, r in sel.sort_values("rank").iterrows():
        reg = r.get("paper_region", f"R_{int(r['region_id'])+1}")
        rtcn = r.get("region_type_cn") or ("高活性热点区域" if r.get("region_type") == "hotspot" else "普通区域")
        lines.append(
            f"{int(r['rank'])}\t{int(r['ID'])}\t{reg}\t{rtcn}\t"
            f"{r['Bioactivity_Score']:.4f}\t{r['local_sensitivity']:.4f}\t{r['U_i']:.4f}\t{r['reason']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table2(path: Path, cmp_df: pd.DataFrame) -> None:
    lines = [
        "表2 本文优选模型与单纯活性排序的对比",
        "说明：对比集合分别为本文贪心优选的8个分子与按 Bioactivity_Score 降序的前8个分子。",
        "",
        "指标\t本文优选模型\t仅按活性排序\t说明",
    ]
    notes = {
        "mean_bioactivity": "保持较高活性",
        "mean_local_sensitivity": "风险更低",
        "high_ls_count": "控制不稳定样本",
        "n_regions_covered": "多样性更好",
        "mean_pairwise_dist": "空间更分散",
    }
    for _, r in cmp_df.iterrows():
        key = r["metric"]
        if key == "mean_activity_norm":
            continue
        label = r["label"]
        if key in ("high_ls_count", "n_regions_covered"):
            v_m, v_b = f"{int(r['model'])}", f"{int(r['baseline'])}"
        elif key == "mean_pairwise_dist":
            v_m, v_b = f"{r['model']:.2f}", f"{r['baseline']:.2f}"
        else:
            v_m, v_b = f"{r['model']:.4f}", f"{r['baseline']:.4f}"
        lines.append(f"{label}\t{v_m}\t{v_b}\t{notes.get(key, '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_fig1_space(
    mol: pd.DataFrame,
    sel: pd.DataFrame,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    stem: Path,
) -> None:
    fig, ax = _fig_panel_54()
    sel_ids = set(sel["ID"].astype(int))
    bg = mol[~mol["ID"].isin(sel_ids)]
    ax.scatter(
        bg["Manifold_X"],
        bg["Manifold_Y"],
        s=SCATTER_BG,
        c="#c8c8c8",
        alpha=0.45,
        linewidths=0,
        zorder=1,
    )
    ax.scatter(
        sel["Manifold_X"],
        sel["Manifold_Y"],
        s=SCATTER_SEL,
        c="#d62828",
        alpha=0.92,
        edgecolors="#1a1a1a",
        linewidths=0.6,
        zorder=3,
    )
    for _, r in sel.iterrows():
        mid = int(r["ID"])
        # ID=1 标注在红点左下角，其余仍在右上角
        xytext = (-4, -6) if mid == 1 else (3, 3)
        ha = "right" if mid == 1 else "left"
        va = "top" if mid == 1 else "bottom"
        ax.annotate(
            str(mid),
            (r["Manifold_X"], r["Manifold_Y"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=7.5,
            fontproperties=fp_tick,
            color="#1a1a1a",
            ha=ha,
            va=va,
            zorder=4,
        )
    _style_manifold_ax(ax, fp_tick, fp_label)
    _save_fig(fig, stem)


def _plot_fig2_bars(
    sel: pd.DataFrame,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    stem: Path,
) -> None:
    plt.rcParams["hatch.linewidth"] = 0.85
    fig, ax = _fig_panel_54()

    order = sel.sort_values("rank")
    x = np.arange(len(order))
    bar_w = 0.18
    bar_gap = 0.05
    half = bar_w + bar_gap
    offsets = (-half, 0.0, half)
    u_plot = (
        order["U_rank"].to_numpy(dtype=float)
        if "U_rank" in order.columns
        else _safe_minmax(order["U_i"].to_numpy(dtype=float))
    )
    series = [
        order["activity_minmax"].to_numpy(dtype=float),
        order["L_norm"].to_numpy(dtype=float)
        if "L_norm" in order.columns
        else _safe_minmax(order["local_sensitivity"].to_numpy(dtype=float)),
        u_plot,
    ]

    for k, (vals, edge_c, hatch, lab) in enumerate(zip(series, BAR_EDGE_COLORS, HATCH_CYCLE, BAR_LABELS)):
        bars = ax.bar(x + offsets[k], vals, width=bar_w, zorder=2)
        for patch in bars:
            _style_hollow_bar(patch, edge_c, hatch)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(i)) for i in order["ID"]])
    _style_bar_ax(ax, fp_tick, fp_label)

    handles = [
        Patch(facecolor="white", edgecolor=c, hatch=h, linewidth=BAR_EDGE_LW, label=lab)
        for c, h, lab in zip(BAR_EDGE_COLORS, HATCH_CYCLE, BAR_LABELS)
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=True,
        fontsize=8,
        prop=fp_tick,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    leg.get_frame().set_linewidth(SPINE_LW)
    _save_fig(fig, stem)


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo + 1e-12)


def _remove_legacy_outputs() -> None:
    legacy = OUT_DIR / "图1_候选分子空间分布与指标特征二联图"
    for ext in (".pdf", ".png"):
        p = legacy.with_suffix(ext)
        if p.exists():
            p.unlink()


def main() -> None:
    _ensure_outputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _remove_legacy_outputs()

    fp0 = _font_prop()
    fp_tick = _fp_size(fp0, TICK_FS)
    fp_label = _fp_size(fp0, AXIS_LABEL_FS)

    mol = pd.read_csv(P2_OUT / "02_molecule_local_sensitivity.csv")
    sel = pd.read_csv(P3_OUT / "02_selected_8_molecules.csv")
    cmp_df = pd.read_csv(P3_OUT / "04_model_vs_activity_top8.csv")

    _write_table1(OUT_DIR / "表1_最终推荐的8个候选分子.txt", sel)
    _write_table2(OUT_DIR / "表2_本文模型与单纯活性排序的对比.txt", cmp_df)
    _plot_fig1_space(mol, sel, fp_tick, fp_label, OUT_DIR / "图1_候选分子二维空间分布")
    _plot_fig2_bars(sel, fp_tick, fp_label, OUT_DIR / "图2_候选分子主要指标柱状图")

    m_reg = int(sel["region_id"].nunique())
    manifest = [
        "问题三结果导出清单（2 表 + 2 图）",
        f"入选分子ID: {', '.join(str(int(i)) for i in sel.sort_values('rank')['ID'])}",
        f"覆盖区域数: {m_reg}",
        "",
        "表1_最终推荐的8个候选分子.txt",
        "表2_本文模型与单纯活性排序的对比.txt",
        "图1_候选分子二维空间分布.pdf",
        "图2_候选分子主要指标柱状图.pdf",
    ]
    (OUT_DIR / "导出清单.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"[export_problem3] -> {OUT_DIR}")
    for line in manifest[4:]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
