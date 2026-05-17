"""
近邻相似网络二联图：
  (a) KNN 边 Tanimoto 相似度分布（直方图 + KDE）
  (b) KNN 边两端活性差异 |a_i - a_j| 分布（直方图 + KDE）

单幅与二联中，主坐标轴框使用同一套英寸定位（左/下/宽高一致），尽量撑满 5:4 画布。

运行：python scripts/plot_knn_network_twin.py
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
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out" / "data_analysis"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.loaders import load_edges, load_manifest

# ---------- 画布与主面板（英寸）：单幅 5:4；二联 10:4，每块主面板与单幅完全一致 ----------
FIG_W_SINGLE = 5.0
FIG_H_SINGLE = 4.0
FIG_W_TWIN = 10.0
FIG_H_TWIN = 4.0

ML_IN = 0.54
MR_IN = 0.24
MB_IN = 0.40
MT_IN = 0.20
PANEL_W_IN = FIG_W_SINGLE - ML_IN - MR_IN
PANEL_H_IN = FIG_H_SINGLE - MB_IN - MT_IN
TWIN_GAP_IN = 0.42

SPINE_LW = 1.05
TICK_LEN = 4.0
AXIS_LABEL_FS = 10.0
TICK_FS = 9.0
LABELPAD = 2.5
HIST_BINS = 32
HIST_EDGE_LW = 0.95
KDE_COLOR = "#c1121f"
KDE_LW = 1.35
KDE_LS = "-."

# 柱色：左紫 → 中彩虹 → 右红（镂空描边/纹理同色）
BAR_CMAP = LinearSegmentedColormap.from_list(
    "purple_rainbow_red",
    ["#7b2cbf", "#5a4fcf", "#3a86ff", "#06d6a0", "#ffd166", "#fb8500", "#d62828"],
    N=256,
)
# 与 Pearson 柱状图一致的镂空纹理，按柱位从左到右循环
HATCH_CYCLE = ["xxxx", "----", "////", "\\\\\\\\", "++++", "|||", "oooo"]

_FONT_MATCH_ORDER = [
    "SimSun",
    "NSimSun",
    "STSong",
    "Songti SC",
    "Source Han Serif SC",
    "Noto Serif CJK SC",
]

XLABEL_A = "Tanimoto 相似度"
XLABEL_B = "边两端活性差异绝对值"
YLABEL = "概率密度"


def _panel_axes_rect(fig_w: float, fig_h: float, left_in: float) -> list[float]:
    """主绘图区归一化矩形（两图单幅必须共用同一 left_in 与 PANEL_*）。"""
    return [
        left_in / fig_w,
        MB_IN / fig_h,
        PANEL_W_IN / fig_w,
        PANEL_H_IN / fig_h,
    ]


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


def _font_prop_from_fc_match() -> tuple[FontProperties, str]:
    plt.rcParams["axes.unicode_minus"] = False
    last_err: str | None = None
    for pattern in _FONT_MATCH_ORDER:
        path = _fc_file(pattern)
        if not path:
            continue
        if "notosanscjk" in path.lower():
            continue
        try:
            return FontProperties(fname=path), f"{pattern} -> {path}"
        except Exception as e:
            last_err = repr(e)
            continue
    raise RuntimeError(
        "无法解析宋体/思源宋体系字体（fc-match）。请安装 fonts-noto-cjk 或 SimSun。"
        + (f" 末次错误: {last_err}" if last_err else "")
    )


def _style_axes(ax: plt.Axes, fp_tick: FontProperties, fp_label: FontProperties, xlabel: str) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", which="major", length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    ax.set_xlabel(xlabel, fontproperties=fp_label, labelpad=LABELPAD)
    ax.set_ylabel(YLABEL, fontproperties=fp_label, labelpad=LABELPAD)
    ax.xaxis.label.set_clip_on(False)
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)


def _style_hollow_hist_bars(bin_edges: np.ndarray, patches) -> None:
    """白底 + 彩色描边/纹理镂空（风格对齐 Pearson 柱状图）。"""
    lo_edge, hi_edge = float(bin_edges[0]), float(bin_edges[-1])
    span = hi_edge - lo_edge + 1e-9
    for i, patch in enumerate(patches):
        center = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        t = (center - lo_edge) / span
        edge_c = BAR_CMAP(t)
        hatch = HATCH_CYCLE[i % len(HATCH_CYCLE)]
        patch.set_facecolor("white")
        patch.set_edgecolor(edge_c)
        patch.set_hatch(hatch)
        patch.set_linewidth(HIST_EDGE_LW)
        if hasattr(patch, "set_hatch_color"):
            patch.set_hatch_color(edge_c)


def _hist_kde(
    ax: plt.Axes,
    data: np.ndarray,
    xlabel: str,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    *,
    x_min: float | None = None,
) -> None:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("分布数据为空。")

    if x_min is not None:
        data = data[data >= x_min]

    hi = float(np.max(data))
    lo = float(x_min) if x_min is not None else float(np.min(data))
    pad = 0.04 * (hi - lo + 1e-9)

    if x_min is not None:
        bin_edges = np.linspace(lo, hi, HIST_BINS + 1)
    else:
        bin_edges = HIST_BINS

    plt.rcParams["hatch.linewidth"] = 0.85
    _, bins, patches = ax.hist(data, bins=bin_edges, density=True, zorder=1)
    _style_hollow_hist_bars(bins, patches)

    try:
        kde = gaussian_kde(data)
        xs = np.linspace(lo, hi + pad, 200)
        ax.plot(xs, kde(xs), color=KDE_COLOR, linewidth=KDE_LW, linestyle=KDE_LS, zorder=3)
    except Exception:
        pass

    ax.set_xlim(lo, hi + pad)
    ax.set_ylim(bottom=0.0)
    _style_axes(ax, fp_tick, fp_label, xlabel)


def _build_edge_table() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = load_manifest()
    edges = load_edges()
    act = manifest.set_index("ID")["Bioactivity_Score"]

    edges = edges.copy()
    edges["bioactivity_source"] = edges["Source"].map(act)
    edges["bioactivity_target"] = edges["Target"].map(act)
    edges = edges.dropna(subset=["bioactivity_source", "bioactivity_target"])
    edges["activity_abs_diff"] = (edges["bioactivity_source"] - edges["bioactivity_target"]).abs()

    tanimoto = edges["Tanimoto_Similarity"].to_numpy(dtype=float)
    act_diff = edges["activity_abs_diff"].to_numpy(dtype=float)
    return edges, tanimoto, act_diff


def _new_figure_single() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))
    ax = fig.add_axes(_panel_axes_rect(FIG_W_SINGLE, FIG_H_SINGLE, ML_IN))
    return fig, ax


def _new_figure_twin() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W_TWIN, FIG_H_TWIN))
    left_x = ML_IN
    right_x = ML_IN + PANEL_W_IN + TWIN_GAP_IN
    ax_a = fig.add_axes(_panel_axes_rect(FIG_W_TWIN, FIG_H_TWIN, left_x))
    ax_b = fig.add_axes(_panel_axes_rect(FIG_W_TWIN, FIG_H_TWIN, right_x))
    return fig, ax_a, ax_b


def _save_one_panel(
    data: np.ndarray,
    xlabel: str,
    stem: Path,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    *,
    x_min: float | None = None,
) -> None:
    fig, ax = _new_figure_single()
    _hist_kde(ax, data, xlabel, fp_tick, fp_label, x_min=x_min)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    fp0, font_note = _font_prop_from_fc_match()
    fp_tick = _fp_size(fp0, TICK_FS)
    fp_label = _fp_size(fp0, AXIS_LABEL_FS)

    edges_df, tanimoto, act_diff = _build_edge_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    edges_df.to_csv(
        OUT_DIR / "近邻边_Tanimoto与活性差异.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {
                "variable": "Tanimoto_Similarity",
                "n": len(tanimoto),
                "mean": float(np.mean(tanimoto)),
                "median": float(np.median(tanimoto)),
                "std": float(np.std(tanimoto)),
                "q75": float(np.quantile(tanimoto, 0.75)),
                "q90": float(np.quantile(tanimoto, 0.90)),
            },
            {
                "variable": "activity_abs_diff",
                "n": len(act_diff),
                "mean": float(np.mean(act_diff)),
                "median": float(np.median(act_diff)),
                "std": float(np.std(act_diff)),
                "q75": float(np.quantile(act_diff, 0.75)),
                "q90": float(np.quantile(act_diff, 0.90)),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "近邻边_分布统计摘要.csv", index=False, encoding="utf-8-sig")

    fig, ax_a, ax_b = _new_figure_twin()
    _hist_kde(ax_a, tanimoto, XLABEL_A, fp_tick, fp_label)
    _hist_kde(ax_b, act_diff, XLABEL_B, fp_tick, fp_label, x_min=0.0)

    stem_twin = OUT_DIR / "近邻网络_Tanimoto与活性差异_二联图"
    fig.savefig(stem_twin.with_suffix(".pdf"))
    fig.savefig(stem_twin.with_suffix(".png"), dpi=300)
    plt.close(fig)

    _save_one_panel(tanimoto, XLABEL_A, OUT_DIR / "图a_Tanimoto相似度分布", fp_tick, fp_label)
    _save_one_panel(
        act_diff,
        XLABEL_B,
        OUT_DIR / "图b_近邻边活性差异分布",
        fp_tick,
        fp_label,
        x_min=0.0,
    )

    rect = _panel_axes_rect(FIG_W_SINGLE, FIG_H_SINGLE, ML_IN)
    (OUT_DIR / "近邻网络_二联图_meta.txt").write_text(
        f"n_edges={len(edges_df)}\n"
        f"single_canvas_in={FIG_W_SINGLE}x{FIG_H_SINGLE}\n"
        f"twin_canvas_in={FIG_W_TWIN}x{FIG_H_TWIN}\n"
        f"panel_inches={PANEL_W_IN:.3f}x{PANEL_H_IN:.3f}\n"
        f"panel_norm_rect={rect}\n"
        f"bar_style=紫-彩虹-红镂空纹理\n"
        f"font={font_note}\n",
        encoding="utf-8",
    )

    print(f"Saved: {stem_twin}.pdf / .png")
    print(f"Saved: 图a / 图b 单幅 PDF")
    print(f"panel_norm_rect (single a/b): {rect}")
    print(f"n_edges={len(edges_df)}")
    print(font_note)


if __name__ == "__main__":
    main()
