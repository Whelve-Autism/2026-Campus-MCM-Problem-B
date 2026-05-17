"""
流形分析图：两张独立图；主绘图区几何尺寸一致（英寸定位），脊线粗细、刻度与轴标签字号一致。
图2 与 hexbin 共用「主图 + 色条」横向分配；整张画布与图1同为 5:4。

运行：python scripts/plot_manifold_bioactivity_twin.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out" / "data_analysis"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.loaders import load_manifest

# ---------- 画布：两图同一尺寸，严格 5:4（宽/高 = 5/4），尽量撑满 ----------
FIG_H_IN = 4.0
FIG_W_IN = FIG_H_IN * 5.0 / 4.0

LEFT_MARGIN_IN = 0.56
BOTTOM_MARGIN_IN = 0.50
TOP_MARGIN_IN = 0.26
RIGHT_MARGIN_IN = 0.26

PANEL_H_IN = FIG_H_IN - BOTTOM_MARGIN_IN - TOP_MARGIN_IN

CB_GAP_IN = 0.22
CB_W_IN = 0.20

# 图1（无色条）：主图区占满左右边距之间
PANEL_W_FULL_IN = FIG_W_IN - LEFT_MARGIN_IN - RIGHT_MARGIN_IN
# 图2（有色条）：主图右边界内缩，左侧与底侧与图1对齐
PANEL_W_CB_IN = FIG_W_IN - LEFT_MARGIN_IN - CB_GAP_IN - CB_W_IN - RIGHT_MARGIN_IN

# ---------- 视觉统一 ----------
SPINE_LW = 1.05
TICK_LEN = 4.0
AXIS_LABEL_FS = 10.0
TICK_FS = 9.0
LEGEND_FS = 9.0
LABELPAD = 3.5
SCATTER_SIZE = 17

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
CBAR_LABEL = "活性得分"


def _format_density_level(v: float) -> str:
    if not np.isfinite(v):
        return ""
    av = abs(float(v))
    if av >= 0.1:
        s = f"{v:.3f}"
    elif av >= 1e-4:
        s = f"{v:.5f}"
    else:
        s = f"{v:.6f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


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


def _main_axes_rect(panel_w_in: float, fig_w_in: float, fig_h_in: float) -> list[float]:
    return [
        LEFT_MARGIN_IN / fig_w_in,
        BOTTOM_MARGIN_IN / fig_h_in,
        panel_w_in / fig_w_in,
        PANEL_H_IN / fig_h_in,
    ]


def _colorbar_axes_rect(panel_w_in: float, fig_w_in: float, fig_h_in: float) -> list[float]:
    x0_in = LEFT_MARGIN_IN + panel_w_in + CB_GAP_IN
    return [
        x0_in / fig_w_in,
        BOTTOM_MARGIN_IN / fig_h_in,
        CB_W_IN / fig_w_in,
        PANEL_H_IN / fig_h_in,
    ]


def _add_main_axes(fig: plt.Figure, panel_w_in: float) -> plt.Axes:
    fig.set_size_inches(FIG_W_IN, FIG_H_IN)
    return fig.add_axes(_main_axes_rect(panel_w_in, FIG_W_IN, FIG_H_IN))


def _add_colorbar_axes(fig: plt.Figure, panel_w_in: float) -> plt.Axes:
    return fig.add_axes(_colorbar_axes_rect(panel_w_in, FIG_W_IN, FIG_H_IN))


def _style_spines_and_ticks(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(
        axis="both",
        which="major",
        length=TICK_LEN,
        width=SPINE_LW,
        labelsize=TICK_FS,
    )


def _style_colorbar_axes(cb, fp_tick: FontProperties, fp_label: FontProperties, label: str) -> None:
    cb.ax.tick_params(length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    for spine in cb.ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    cb.set_label(label, fontproperties=fp_label)
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(fp_tick)
    cb.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def _axis_style(
    ax: plt.Axes,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    pad_x: float,
    pad_y: float,
) -> None:
    ax.set_xlim(xlim[0] - pad_x, xlim[1] + pad_x)
    ax.set_ylim(ylim[0] - pad_y, ylim[1] + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(XLABEL, fontproperties=fp_label, labelpad=LABELPAD)
    ax.set_ylabel(YLABEL, fontproperties=fp_label, labelpad=LABELPAD)
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    _style_spines_and_ticks(ax)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)


def _scatter_xy(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ax.scatter(x, y, s=SCATTER_SIZE, c="#6a8caf", alpha=0.38, linewidths=0, zorder=1)


def _kde_contour_and_delta(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    pad_x: float,
    pad_y: float,
) -> float:
    kde = gaussian_kde(np.vstack([x, y]))
    gx = np.linspace(xmin - pad_x, xmax + pad_x, 120)
    gy = np.linspace(ymin - pad_y, ymax + pad_y, 120)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    zmax = float(np.nanmax(Z))
    if not np.isfinite(zmax) or zmax <= 0:
        raise ValueError("密度网格无效")
    lev = np.linspace(zmax * 0.12, zmax, 8)
    ax.contour(
        X,
        Y,
        Z,
        levels=lev,
        colors="#8b0000",
        linewidths=0.95,
        alpha=0.88,
        zorder=3,
    )
    nlev = len(lev)
    if nlev < 2:
        raise ValueError("等高线条数不足")
    return float((lev[-1] - lev[0]) / (nlev - 1))


def _hexbin_with_cbar(
    ax: plt.Axes,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    fig: plt.Figure,
    cax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    hb = ax.hexbin(
        x,
        y,
        gridsize=32,
        cmap="Blues",
        mincnt=1,
        alpha=0.78,
        linewidths=0,
        zorder=2,
    )
    cb_d = fig.colorbar(hb, cax=cax)
    _style_colorbar_axes(cb_d, fp_tick, fp_label, "样本计数")


def _add_density_legend(ax: plt.Axes, fp_leg: FontProperties, mode: str, kde_delta: float | None) -> None:
    if mode == "kde" and kde_delta is not None and np.isfinite(kde_delta):
        ds = _format_density_level(kde_delta)
        contour_label = f"二维核密度估计等高线\n（相邻等值线密度差 Δ = {ds}）"
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=6.5,
                markerfacecolor="#6a8caf",
                markeredgecolor="#5a7c9f",
                alpha=0.9,
                label="各分子在流形平面上的位置",
            ),
            Line2D(
                [0],
                [0],
                color="#8b0000",
                linewidth=2.0,
                label=contour_label,
            ),
        ]
    else:
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=6.5,
                markerfacecolor="#6a8caf",
                markeredgecolor="#5a7c9f",
                alpha=0.9,
                label="各分子在流形平面上的位置",
            ),
            Patch(
                facecolor="#3182bd",
                edgecolor="#08519c",
                linewidth=0.75,
                label="格内分子个数见右侧色条",
            ),
        ]

    leg = ax.legend(
        handles=handles,
        loc="upper left",
        prop=fp_leg,
        fontsize=LEGEND_FS,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#888888",
    )
    leg.get_frame().set_linewidth(SPINE_LW * 0.85)
    for t in leg.get_texts():
        t.set_fontproperties(fp_leg)


def _draw_activity(
    ax: plt.Axes,
    fp_tick: FontProperties,
    fp_label: FontProperties,
    fig: plt.Figure,
    cax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    act: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    pad_x: float,
    pad_y: float,
) -> None:
    sc = ax.scatter(
        x,
        y,
        s=SCATTER_SIZE,
        c=act,
        cmap="plasma",
        alpha=0.92,
        linewidths=0,
    )
    _axis_style(ax, fp_tick, fp_label, (xmin, xmax), (ymin, ymax), pad_x, pad_y)
    cb = fig.colorbar(sc, cax=cax)
    _style_colorbar_axes(cb, fp_tick, fp_label, CBAR_LABEL)


def main() -> None:
    fp0, font_note = _font_prop_from_fc_match()
    fp_tick = _fp_size(fp0, TICK_FS)
    fp_label = _fp_size(fp0, AXIS_LABEL_FS)
    fp_leg = _fp_size(fp0, LEGEND_FS)

    df = load_manifest().dropna(subset=["Manifold_X", "Manifold_Y", "Bioactivity_Score"])
    x = df["Manifold_X"].to_numpy(dtype=float)
    y = df["Manifold_Y"].to_numpy(dtype=float)
    act = df["Bioactivity_Score"].to_numpy(dtype=float)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    rx = max(xmax - xmin, 1e-9)
    ry = max(ymax - ymin, 1e-9)
    pad_x = 0.03 * rx
    pad_y = 0.03 * ry

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kde_delta: float | None = None
    mode: str

    fig1 = plt.figure()
    ax1 = _add_main_axes(fig1, PANEL_W_FULL_IN)
    _scatter_xy(ax1, x, y)
    try:
        kde_delta = _kde_contour_and_delta(ax1, x, y, xmin, xmax, ymin, ymax, pad_x, pad_y)
        mode = "kde"
        _axis_style(ax1, fp_tick, fp_label, (xmin, xmax), (ymin, ymax), pad_x, pad_y)
    except Exception:
        plt.close(fig1)
        fig1 = plt.figure()
        ax1 = _add_main_axes(fig1, PANEL_W_CB_IN)
        cax1 = _add_colorbar_axes(fig1, PANEL_W_CB_IN)
        _scatter_xy(ax1, x, y)
        _hexbin_with_cbar(ax1, fp_tick, fp_label, fig1, cax1, x, y)
        mode = "hexbin"
        kde_delta = None
        _axis_style(ax1, fp_tick, fp_label, (xmin, xmax), (ymin, ymax), pad_x, pad_y)

    _add_density_legend(ax1, fp_leg, mode, kde_delta)
    stem1 = OUT_DIR / "图1_流形空间密度分布"
    fig1.savefig(stem1.with_suffix(".pdf"))
    fig1.savefig(stem1.with_suffix(".png"), dpi=300)
    plt.close(fig1)

    fig2 = plt.figure()
    ax2 = _add_main_axes(fig2, PANEL_W_CB_IN)
    cax2 = _add_colorbar_axes(fig2, PANEL_W_CB_IN)
    _draw_activity(
        ax2,
        fp_tick,
        fp_label,
        fig2,
        cax2,
        x,
        y,
        act,
        xmin,
        xmax,
        ymin,
        ymax,
        pad_x,
        pad_y,
    )
    stem2 = OUT_DIR / "图2_活性得分空间分布"
    fig2.savefig(stem2.with_suffix(".pdf"))
    fig2.savefig(stem2.with_suffix(".png"), dpi=300)
    plt.close(fig2)

    cap_path = OUT_DIR / "图_caption建议.txt"
    cap_path.write_text(
        "【图1 说明】二维流形空间中的分子聚集特征（分子位置与核密度等高线）。\n"
        "【图2 说明】活性得分在二维流形空间中的分布（颜色表示活性高低）。\n\n"
        "【LaTeX 示例】\n"
        "\\caption{二维流形空间中的分子聚集特征。}\n"
        "\\caption{活性得分在二维流形空间中的分布。}\n",
        encoding="utf-8",
    )

    meta = OUT_DIR / "图_meta.txt"
    meta.write_text(
        f"n_molecules={len(df)}\n"
        f"图1密度模式={'KDE等高线' if mode == 'kde' else '六边形分箱'}\n"
        f"画布英寸={FIG_W_IN}x{FIG_H_IN}(宽x高), 比例5:4\n"
        f"图1主面板宽={PANEL_W_FULL_IN:.3f}（无色条时横向撑满）\n"
        f"图2主面板宽={PANEL_W_CB_IN:.3f}（色条占去右侧 {CB_GAP_IN}+{CB_W_IN} in）\n"
        f"主面板高={PANEL_H_IN:.3f}, 左/下边距相同\n"
        f"脊线宽度={SPINE_LW}, 轴标签字号={AXIS_LABEL_FS}, 刻度字号={TICK_FS}, 图例字号={LEGEND_FS}\n"
        f"font={font_note}\n",
        encoding="utf-8",
    )

    print(f"Saved: {stem1}.pdf / .png")
    print(f"Saved: {stem2}.pdf / .png")
    print(font_note)


if __name__ == "__main__":
    main()
