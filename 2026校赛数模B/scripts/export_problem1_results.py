"""
问题一求解结果：按论文定稿方案导出图 1–4、表 1–3 至 out/problem_1/。
（不含可选图 5）

运行：python scripts/export_problem1_results.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out" / "problem_1"
P1_OUT = ROOT / "problem_1" / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.loaders import manifest_phys_cols

# ---------- 5:4 主面板（图 1/2/3 单幅对齐） ----------
FIG_W_54 = 5.0
FIG_H_54 = 4.0
ML_IN = 0.54
MR_IN = 0.24
MB_IN = 0.56
MB_IN_FIG3 = 0.72
MT_IN = 0.20
XLABEL_PAD = 6.0
FIG3_XLABEL_PAD = 4.0
PANEL_W_IN = FIG_W_54 - ML_IN - MR_IN
PANEL_H_IN = FIG_H_54 - MB_IN - MT_IN

SPINE_LW = 1.05
TICK_LEN = 4.0
AXIS_LABEL_FS = 10.0
TICK_FS = 9.0
LABELPAD = 3.0
# 图4 热力图：初始画布约 3:1，导出时用 tight 裁去左右空白（保留文字）
FIG_W_31 = 10.8
FIG_H_31 = FIG_W_31 / 3.0
HM_PAD_INCHES = 0.08
SCATTER_SIZE = 22
CENTROID_SIZE = 140
HIST_EDGE_LW = 0.95

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

_SUBS = "₀₁₂₃₄₅₆₇₈₉"
_REGION_CMAP = plt.colormaps["tab20"].resampled(20)

# 柱色：左红 → 彩虹 → 右紫（镂空描边同色）
BAR_CMAP = LinearSegmentedColormap.from_list(
    "red_rainbow_purple",
    ["#d62828", "#fb8500", "#ffd166", "#06d6a0", "#3a86ff", "#5a4fcf", "#7b2cbf"],
    N=256,
)
HATCH_CYCLE = ["xxxx", "----", "////", "\\\\\\\\", "++++", "|||", "oooo"]


def _panel_axes_rect(*, mb_in: float = MB_IN) -> list[float]:
    h_in = FIG_H_54 - mb_in - MT_IN
    return [ML_IN / FIG_W_54, mb_in / FIG_H_54, PANEL_W_IN / FIG_W_54, h_in / FIG_H_54]


def _fig_with_panel(*, mb_in: float = MB_IN) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W_54, FIG_H_54))
    ax = fig.add_axes(_panel_axes_rect(mb_in=mb_in))
    return fig, ax


def _int_to_subscript(n: int) -> str:
    s = str(n)
    return "".join(_SUBS[int(c)] for c in s)


def _paper_region_labels(region_ids: list[int]) -> dict[int, tuple[str, str]]:
    """
    DBSCAN 簇 -> (正文 Unicode 下标, matplotlib 数学下标)。
    例：簇 0 对应 R₁，图中为 $R_{1}$。
    """
    uniq = sorted(set(region_ids))
    labels: dict[int, tuple[str, str]] = {}
    if -1 in uniq:
        labels[-1] = ("R₀", r"$R_{0}$")
    pos = sorted(r for r in uniq if r >= 0)
    for i, rid in enumerate(pos, start=1):
        labels[rid] = (f"R{_int_to_subscript(i)}", f"$R_{{{i}}}$")
    return labels


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


def _font_prop_from_fc_match() -> FontProperties:
    plt.rcParams["axes.unicode_minus"] = False
    for pattern in _FONT_MATCH_ORDER:
        path = _fc_file(pattern)
        if not path:
            continue
        if "notosanscjk" in path.lower():
            continue
        return FontProperties(fname=path)
    raise RuntimeError("无法解析宋体/思源宋体系字体。")


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


def _style_stat_ax(
    ax: plt.Axes,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", which="major", length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    ax.tick_params(axis="x", pad=4)
    ax.set_xlabel(xlabel, fontproperties=fp_label, labelpad=FIG3_XLABEL_PAD)
    ax.set_ylabel(ylabel, fontproperties=fp_label, labelpad=LABELPAD)
    ax.xaxis.label.set_clip_on(False)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
        t.set_rotation(0)
        t.set_ha("center")
        t.set_clip_on(False)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)


def _save_fig_fig3(fig: plt.Figure, stem: Path) -> None:
    kw = dict(facecolor="white", edgecolor="none", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".pdf"), **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)


def _style_hollow_bar(patch, edge_c: str, hatch: str) -> None:
    patch.set_facecolor("white")
    patch.set_edgecolor(edge_c)
    patch.set_hatch(hatch)
    patch.set_linewidth(HIST_EDGE_LW)
    if hasattr(patch, "set_hatch_color"):
        patch.set_hatch_color(edge_c)


def _ensure_problem1_outputs() -> None:
    reg_path = P1_OUT / "02_region_summary.csv"
    mol_path = P1_OUT / "03_molecules_with_regions.csv"
    ok = (
        mol_path.exists()
        and reg_path.exists()
        and "mean_internal_tanimoto" in reg_path.read_text(encoding="utf-8", errors="ignore")
    )
    if ok:
        return
    subprocess.run([sys.executable, "-m", "problem_1.run"], cwd=str(ROOT), check=True)


def _attach_region_labels(reg: pd.DataFrame, label_map: dict[int, tuple[str, str]]) -> pd.DataFrame:
    reg = reg.copy()
    reg["paper_region"] = reg["region_id"].map(lambda r: label_map[int(r)][0])
    reg["paper_region_tex"] = reg["region_id"].map(lambda r: label_map[int(r)][1])
    return reg


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mol = pd.read_csv(P1_OUT / "03_molecules_with_regions.csv")
    reg = pd.read_csv(P1_OUT / "02_region_summary.csv")
    reps = pd.read_csv(P1_OUT / "04_representative_molecules.csv")
    if "mean_internal_tanimoto" not in reg.columns:
        raise RuntimeError("02_region_summary 缺少网络字段，请重新运行 python -m problem_1.run")
    label_map = _paper_region_labels(reg["region_id"].astype(int).tolist())
    reg = _attach_region_labels(reg, label_map)
    reg = reg.sort_values("region_id").reset_index(drop=True)
    n_total = int(mol.shape[0])
    reg["pct_molecules"] = reg["n_molecules"] / n_total * 100.0
    mol["paper_region_tex"] = mol["region_id"].map(lambda r: label_map[int(r)][1])
    return mol, reg, reps


def _write_table1(reg: pd.DataFrame, path: Path) -> None:
    lines = [
        "表1 区域综合统计与高活性热点判定结果",
        "字段说明：区域编号 Rₖ 表示 R 的下标 k（正文与图一致）；代码簇标签为 DBSCAN 簇 ID。",
        "",
        "区域编号\t代码簇标签\t分子数量\t占比(%)\t平均活性\t高活性比例\t活性评分H_r\t区域类型",
    ]
    for _, r in reg.iterrows():
        lines.append(
            f"{r['paper_region']}\t{int(r['region_id'])}\t{int(r['n_molecules'])}\t"
            f"{r['pct_molecules']:.2f}\t{r['mean_bioactivity']:.6f}\t"
            f"{r['high_activity_ratio']:.4f}\t{r['H_score']:.6f}\t{r['region_type']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table2(reg: pd.DataFrame, path: Path) -> None:
    lines = [
        "表2 不同区域的近邻相似网络统计",
        "",
        "区域编号\t代码簇标签\t内部边数\t内部边比例q_r\t平均内部相似度\t跨区域边比例",
    ]
    for _, r in reg.iterrows():
        lines.append(
            f"{r['paper_region']}\t{int(r['region_id'])}\t{int(r['internal_edge_count'])}\t"
            f"{r['internal_edge_ratio_q']:.4f}\t{r['mean_internal_tanimoto']:.6f}\t"
            f"{r['cross_region_edge_ratio']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table3(reps: pd.DataFrame, reg: pd.DataFrame, path: Path) -> None:
    top1 = reps.loc[reps["rank_in_region"] == 1].copy()
    top1 = top1.merge(reg[["region_id", "paper_region"]], on="region_id", how="left")
    top1 = top1.sort_values("region_id")
    lines = [
        "表3 各区域代表性分子选取结果（每区 Top 1）",
        "",
        "区域编号\t代码簇标签\t分子ID\t活性得分\t中心性Cent_i\t连接性Deg_i\t代表性Rep_i\t区域类型",
    ]
    for _, r in top1.iterrows():
        lines.append(
            f"{r['paper_region']}\t{int(r['region_id'])}\t{int(r['ID'])}\t"
            f"{r['Bioactivity_Score']:.6f}\t{r.get('cent_score', np.nan):.6f}\t"
            f"{r.get('deg_norm', np.nan):.6f}\t{r['rep_score']:.6f}\t{r['region_type']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _region_color_map(region_ids: list[int]) -> dict[int, tuple]:
    uniq = sorted(set(region_ids))
    colors: dict[int, tuple] = {}
    if -1 in uniq:
        colors[-1] = (0.55, 0.55, 0.55, 0.85)
    pos = [r for r in uniq if r >= 0]
    for i, rid in enumerate(pos):
        colors[rid] = _REGION_CMAP(i % 20)
    return colors


def _plot_fig1_partition(
    mol: pd.DataFrame,
    reg: pd.DataFrame,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    stem: Path,
) -> None:
    fig, ax = _fig_with_panel()
    colors = _region_color_map(mol["region_id"].astype(int).tolist())
    legend_handles: list[Patch] = []

    for rid in sorted(mol["region_id"].unique()):
        rid = int(rid)
        sub = mol.loc[mol["region_id"] == rid]
        c = colors[rid]
        ax.scatter(
            sub["Manifold_X"],
            sub["Manifold_Y"],
            s=SCATTER_SIZE,
            c=[c],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.25,
            zorder=2,
        )
        tex = reg.loc[reg["region_id"] == rid, "paper_region_tex"].iloc[0]
        legend_handles.append(
            Patch(facecolor=c, edgecolor="white", linewidth=0.4, label=tex)
        )

    for _, r in reg.iterrows():
        rid = int(r["region_id"])
        cx = float(mol.loc[mol["region_id"] == rid, "Manifold_X"].mean())
        cy = float(mol.loc[mol["region_id"] == rid, "Manifold_Y"].mean())
        ax.scatter(cx, cy, marker="*", s=CENTROID_SIZE, c="black", zorder=5, linewidths=0)
        ax.annotate(
            r["paper_region_tex"],
            (cx, cy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            zorder=6,
        )

    _style_manifold_ax(ax, fp_tick, fp_label)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=11,
            markerfacecolor="black",
            markeredgecolor="black",
            label="区域质心",
        )
    )
    ax.legend(
        handles=legend_handles,
        prop=fp_tick,
        loc="upper left",
        frameon=True,
        fontsize=7.5,
        ncol=2,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.45,
        labelspacing=0.35,
    )
    _save_fig(fig, stem)


def _plot_fig2_hotspot(
    mol: pd.DataFrame,
    reg: pd.DataFrame,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    stem: Path,
) -> None:
    fig, ax = _fig_with_panel()
    hotspot_ids = set(reg.loc[reg["region_type"] == "hotspot", "region_id"].astype(int))

    ax.scatter(
        mol["Manifold_X"],
        mol["Manifold_Y"],
        s=SCATTER_SIZE * 0.85,
        c="#d9d9d9",
        alpha=0.55,
        edgecolors="none",
        zorder=1,
    )

    warm = plt.cm.OrRd
    for i, rid in enumerate(sorted(hotspot_ids)):
        sub = mol.loc[mol["region_id"] == rid]
        if sub.empty:
            continue
        ax.scatter(
            sub["Manifold_X"],
            sub["Manifold_Y"],
            s=SCATTER_SIZE * 1.15,
            c=[warm(0.45 + 0.12 * (i % 5))],
            alpha=0.88,
            edgecolors="#8b0000",
            linewidths=0.35,
            zorder=3,
        )

    for _, r in reg.iterrows():
        if r["region_type"] != "hotspot":
            continue
        rid = int(r["region_id"])
        cx = float(mol.loc[mol["region_id"] == rid, "Manifold_X"].mean())
        cy = float(mol.loc[mol["region_id"] == rid, "Manifold_Y"].mean())
        ax.scatter(cx, cy, marker="*", s=CENTROID_SIZE, c="black", zorder=5, linewidths=0)

    _style_manifold_ax(ax, fp_tick, fp_label)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9d9d9", markersize=7, label="普通区域分子"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=warm(0.65),
            markeredgecolor="#8b0000",
            markersize=7,
            label="高活性热点区域分子",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=11,
            markerfacecolor="black",
            markeredgecolor="black",
            label="热点区域质心",
        ),
    ]
    ax.legend(handles=handles, prop=fp_tick, loc="upper left", frameon=True, fontsize=8)
    _save_fig(fig, stem)


def _save_fig(fig: plt.Figure, stem: Path, *, pad_inches: float = 0.02) -> None:
    kw = dict(facecolor="white", edgecolor="none", pad_inches=pad_inches)
    fig.savefig(stem.with_suffix(".pdf"), **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)


def _save_fig_heatmap(fig: plt.Figure, stem: Path) -> None:
    """按内容紧包围裁剪左右留白，pad 防止刻度/色条文字被裁。"""
    fig.canvas.draw()
    kw = dict(facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=HM_PAD_INCHES)
    fig.savefig(stem.with_suffix(".pdf"), **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)


def _region_bar_color(i: int, n: int) -> str:
    return BAR_CMAP(i / max(n - 1, 1))


def _ordered_regions(reg: pd.DataFrame) -> tuple[list[int], list[str]]:
    order = reg.sort_values("region_id")["region_id"].astype(int).tolist()
    labels = [reg.loc[reg["region_id"] == rid, "paper_region_tex"].iloc[0] for rid in order]
    return order, labels


def _plot_fig3_boxplot(
    mol: pd.DataFrame,
    reg: pd.DataFrame,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    stem: Path,
) -> None:
    fig, ax = _fig_with_panel(mb_in=MB_IN_FIG3)
    order, labels = _ordered_regions(reg)
    data = [mol.loc[mol["region_id"] == rid, "Bioactivity_Score"].to_numpy(dtype=float) for rid in order]
    n = len(order)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for i, patch in enumerate(bp["boxes"]):
        edge_c = _region_bar_color(i, n)
        _style_hollow_bar(patch, edge_c, HATCH_CYCLE[i % len(HATCH_CYCLE)])

        bp["medians"][i].set_color("black")
        bp["medians"][i].set_linewidth(1.15)
        for j in (2 * i, 2 * i + 1):
            bp["whiskers"][j].set_color(edge_c)
            bp["whiskers"][j].set_linewidth(1.0)
            bp["caps"][j].set_color(edge_c)
            bp["caps"][j].set_linewidth(1.0)
        flier = bp["fliers"][i]
        flier.set_markerfacecolor(edge_c)
        flier.set_markeredgecolor(edge_c)
        flier.set_alpha(0.55)

        xpos = i + 1
        med_y = float(np.median(data[i]))
        ax.plot(
            xpos,
            med_y,
            marker="*",
            markersize=9,
            color="black",
            markeredgecolor="black",
            linestyle="None",
            zorder=6,
        )

    _style_stat_ax(ax, fp_tick, fp_label, xlabel="区域编号", ylabel="活性得分")
    _save_fig_fig3(fig, stem)


def _plot_fig3_bar_ratio(
    reg: pd.DataFrame,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    stem: Path,
) -> None:
    """与图3 箱线图同一 5:4 画布、主面板位置与轴样式。"""
    plt.rcParams["hatch.linewidth"] = 0.85
    fig, ax = _fig_with_panel(mb_in=MB_IN_FIG3)
    order, labels = _ordered_regions(reg)
    ratios = reg.sort_values("region_id")["high_activity_ratio"].to_numpy(dtype=float)
    n = len(order)
    x = np.arange(1, n + 1, dtype=float)
    bar_w = 0.55

    for i, (xi, yi) in enumerate(zip(x, ratios)):
        edge_c = _region_bar_color(i, n)
        bars = ax.bar(xi, yi, width=bar_w, zorder=2)
        _style_hollow_bar(bars[0], edge_c, HATCH_CYCLE[i % len(HATCH_CYCLE)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, min(1.05, float(ratios.max()) * 1.12 + 0.04))
    _style_stat_ax(ax, fp_tick, fp_label, xlabel="区域编号", ylabel="高活性分子比例")
    _save_fig_fig3(fig, stem)


def _plot_fig4_phys_heatmap(
    mol: pd.DataFrame,
    reg: pd.DataFrame,
    fp_tick: FontProperties,
    stem: Path,
) -> None:
    phys = manifest_phys_cols(mol)
    if not phys:
        raise RuntimeError("未找到理化指标列。")

    z = mol[phys].astype(float).copy()
    for c in phys:
        mu, sig = z[c].mean(), z[c].std()
        z[c] = (z[c] - mu) / (sig + 1e-12)

    rows = []
    ylabels_tex = []
    for _, r in reg.sort_values("region_id").iterrows():
        rid = int(r["region_id"])
        sub = z.loc[mol["region_id"] == rid, phys]
        rows.append(sub.mean().to_numpy(dtype=float))
        ylabels_tex.append(r["paper_region_tex"])

    mat = np.array(rows)
    vmin, vmax = -1.5, 1.5
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(FIG_W_31, FIG_H_31), layout="constrained")

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    ax.set_xticks(np.arange(len(phys)))
    ax.set_xticklabels(phys, rotation=18, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(ylabels_tex)))
    ax.set_yticklabels(ylabels_tex, fontsize=8)
    ax.tick_params(axis="both", length=TICK_LEN, width=SPINE_LW, pad=2)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
        t.set_clip_on(False)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)
        t.set_clip_on(False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat[i, j])
            rgba = cmap(norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.52 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=tc,
                fontsize=6.5,
                fontproperties=fp_tick,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.015, aspect=32)
    cbar.ax.tick_params(length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    cbar.set_label("标准化均值", fontproperties=fp_tick, fontsize=9, labelpad=2)
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(fp_tick)

    _save_fig_heatmap(fig, stem)


def _remove_legacy_outputs() -> None:
    legacy = OUT_DIR / "图3_区域活性差异二联图.pdf"
    if legacy.exists():
        legacy.unlink()
    for ext in (".pdf", ".png"):
        p = OUT_DIR / f"图3_区域活性差异二联图{ext}"
        if p.exists():
            p.unlink()


def main() -> None:
    _ensure_problem1_outputs()
    _remove_legacy_outputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fp0 = _font_prop_from_fc_match()
    fp_tick = _fp_size(fp0, TICK_FS)
    fp_label = _fp_size(fp0, AXIS_LABEL_FS)

    mol, reg, reps = _load_tables()

    _write_table1(reg, OUT_DIR / "表1_区域综合统计与热点判定.txt")
    _write_table2(reg, OUT_DIR / "表2_区域近邻网络统计.txt")
    _write_table3(reps, reg, OUT_DIR / "表3_各区域代表性分子.txt")

    _plot_fig1_partition(mol, reg, fp_tick, fp_label, OUT_DIR / "图1_二维流形空间分区结果")
    _plot_fig2_hotspot(mol, reg, fp_tick, fp_label, OUT_DIR / "图2_高活性热点区域识别结果")
    _plot_fig3_boxplot(mol, reg, fp_tick, fp_label, OUT_DIR / "图3_区域活性箱线图")
    _plot_fig3_bar_ratio(reg, fp_tick, fp_label, OUT_DIR / "图3_区域高活性比例柱状图")
    _plot_fig4_phys_heatmap(mol, reg, fp_tick, OUT_DIR / "图4_区域理化指标热力图")

    manifest = [
        "问题一结果导出清单（图 1–4 + 表 1–3；图3 拆为两张 5:4 单图）",
        "区域编号：Rₖ 表示 R 的下标 k，与图中 $R_{k}$ 一致",
        f"n_molecules={len(mol)}",
        f"n_regions={len(reg)}",
        f"hotspot_regions={reg.loc[reg['region_type']=='hotspot','paper_region'].tolist()}",
        "",
        "图1_二维流形空间分区结果.pdf",
        "图2_高活性热点区域识别结果.pdf",
        "图3_区域活性箱线图.pdf",
        "图3_区域高活性比例柱状图.pdf（画布 5:4，与箱线图版式一致）",
        "图4_区域理化指标热力图.pdf（画布 3:1）",
        "表1_区域综合统计与热点判定.txt",
        "表2_区域近邻网络统计.txt",
        "表3_各区域代表性分子.txt",
    ]
    (OUT_DIR / "导出清单.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"[export_problem1] -> {OUT_DIR}")
    for name in manifest[5:]:
        if name:
            print(f"  {name}")


if __name__ == "__main__":
    main()
