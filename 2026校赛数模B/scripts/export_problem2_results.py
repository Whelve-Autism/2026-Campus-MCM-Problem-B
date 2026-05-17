"""
问题二求解结果：表 1–2 + 图 1–2 -> out/problem_2/

运行：python scripts/export_problem2_results.py
依赖：python -m problem_1.run && python -m problem_2.run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out" / "problem_2"
P2_OUT = ROOT / "problem_2" / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- 5:4 画布（与问题一一致）；图1 主面板缩进留色条，图2 主面板占满右侧边距 ----------
FIG_W = 5.0
FIG_H = 4.0
ML_IN = 0.54
MR_IN = 0.24
MB_IN = 0.56
MT_IN = 0.20
XLABEL_PAD = 6.0
CB_GAP_IN = 0.04
CB_W_IN = 0.14
CB_LABEL_IN = 0.20  # 色条轴右侧留给竖排标签，避免 PDF 裁切
PANEL_W_FULL = FIG_W - ML_IN - MR_IN
PANEL_W_MAIN = FIG_W - ML_IN - MR_IN - CB_GAP_IN - CB_W_IN - CB_LABEL_IN
PANEL_H = FIG_H - MB_IN - MT_IN

SPINE_LW = 1.05
TICK_LEN = 4.0
AXIS_LABEL_FS = 10.0
TICK_FS = 9.0
LABELPAD = 3.0
SCATTER_SIZE = 20

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
CBAR_LS = "局部敏感度"
LS_CMAP = "YlOrRd"  # 黄–橙–红，表示敏感度高=风险高（与问题一活性 plasma 区分）

def _panel_rect(panel_w: float) -> list[float]:
    return [ML_IN / FIG_W, MB_IN / FIG_H, panel_w / FIG_W, PANEL_H / FIG_H]


def _cbar_rect() -> list[float]:
    x0 = ML_IN + PANEL_W_MAIN + CB_GAP_IN
    return [x0 / FIG_W, MB_IN / FIG_H, CB_W_IN / FIG_W, PANEL_H / FIG_H]


def _fig_panel_54(panel_w: float) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(_panel_rect(panel_w))
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
    ax.set_xlabel(xlabel, fontproperties=fp_label, labelpad=XLABEL_PAD)
    ax.set_ylabel(ylabel, fontproperties=fp_label, labelpad=LABELPAD)
    ax.xaxis.label.set_clip_on(False)
    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)


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


def _ensure_outputs() -> None:
    need = P2_OUT / "02_molecule_local_sensitivity.csv"
    if need.exists() and "quadrant_type" in need.read_text(encoding="utf-8", errors="ignore"):
        return
    subprocess.run([sys.executable, "-m", "problem_2.run"], cwd=str(ROOT), check=True)


def _region_type_cn(t: str) -> str:
    return "高活性热点区域" if t == "hotspot" else "普通区域"


def _write_table1(path: Path) -> None:
    df = pd.read_csv(P2_OUT / "05_top_cliff_pairs_publish.csv")
    lines = [
        "表1 典型活性悬崖分子对",
        "说明：C'_{ij}=s_{ij}·|A_i-A_j|（归一化活性）；排名按 C'_{ij} 降序。",
        "",
        "排名\t分子i\t分子j\t所属区域\tTanimoto相似度\t活性差异ΔA\t悬崖强度C'",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['rank'])}\t{int(r['Source'])}\t{int(r['Target'])}\t{r['region_pair']}\t"
            f"{r['Tanimoto_Similarity']:.4f}\t{r['delta_activity_norm']:.4f}\t{r['cliff_strength_norm']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table2(path: Path) -> None:
    df = pd.read_csv(P2_OUT / "03_region_stability.csv")
    lines = [
        "表2 不同区域稳定性统计",
        "说明：u_r 为高敏感分子比例；v_r 为区域内一般悬崖边占比；稳定性判断为相对中位数规则。",
        "",
        "区域编号\t区域类型\t平均活性\t平均敏感度\t高敏感比例u_r\t悬崖边比例v_r\t稳定性判断",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['paper_region']}\t{_region_type_cn(r['region_type'])}\t"
            f"{r['mean_bioactivity']:.4f}\t{r['mean_local_sensitivity']:.4f}\t"
            f"{r['high_ls_ratio']:.4f}\t{r['internal_cliff_edge_ratio']:.4f}\t{r['stability_judgment']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_fig1_ls_space(mol: pd.DataFrame, fp_tick: FontProperties, fp_label: FontProperties, stem: Path) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(_panel_rect(PANEL_W_MAIN))
    cax = fig.add_axes(_cbar_rect())

    x = mol["Manifold_X"].to_numpy(dtype=float)
    y = mol["Manifold_Y"].to_numpy(dtype=float)
    ls = mol["local_sensitivity"].to_numpy(dtype=float)

    sc = ax.scatter(x, y, c=ls, s=SCATTER_SIZE, cmap=LS_CMAP, alpha=0.85, linewidths=0, zorder=2)
    _style_manifold_ax(ax, fp_tick, fp_label)

    cb = fig.colorbar(sc, cax=cax)
    cb.ax.tick_params(length=TICK_LEN, width=SPINE_LW, labelsize=TICK_FS)
    cb.set_label(CBAR_LS, fontproperties=fp_label, labelpad=2)
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(fp_tick)

    _save_fig(fig, stem, pad_inches=0.03)


def _plot_fig2_quadrant(mol: pd.DataFrame, corr: pd.DataFrame, fp_tick: FontProperties, fp_label: FontProperties, stem: Path) -> None:
    fig, ax = _fig_panel_54(PANEL_W_FULL)

    a = mol["activity_minmax"].to_numpy(dtype=float)
    ls = mol["local_sensitivity"].to_numpy(dtype=float)
    q_act = float(corr.loc[corr["metric"] == "Q75_activity_norm", "value"].iloc[0])
    q_ls = float(corr.loc[corr["metric"] == "Q75_local_sensitivity", "value"].iloc[0])

    ax.scatter(a, ls, s=SCATTER_SIZE, c="#5a7fa5", alpha=0.55, edgecolors="white", linewidths=0.25, zorder=2)
    ax.axvline(q_act, color="#666666", linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(q_ls, color="#666666", linestyle="--", linewidth=0.9, zorder=1)

    xmin, xmax = -0.02, 1.02
    ymin, ymax = -0.02, max(float(ls.max()) * 1.05, q_ls * 1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    pr = float(corr.loc[corr["metric"] == "pearson_r", "value"].iloc[0])
    sr = float(corr.loc[corr["metric"] == "spearman_rho", "value"].iloc[0])
    ax.text(
        0.02,
        0.98,
        f"$\\rho_P$={pr:.3f}，$\\rho_S$={sr:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        fontproperties=fp_tick,
    )

    labels_pos = [
        (0.78, 0.82, "高活性\n低敏感"),
        (0.78, 0.12, "高活性\n高敏感"),
        (0.08, 0.82, "普通活性\n低敏感"),
        (0.08, 0.12, "普通活性\n高敏感"),
    ]
    for xp, yp, txt in labels_pos:
        ax.text(
            xp,
            yp,
            txt,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            fontproperties=fp_tick,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.85),
        )

    _style_stat_ax(
        ax,
        fp_tick,
        fp_label,
        xlabel="归一化活性得分 $A_i$",
        ylabel="局部敏感度 $LS_i$",
    )

    _save_fig(fig, stem)


def _save_fig(fig: plt.Figure, stem: Path, *, pad_inches: float = 0.02) -> None:
    kw = dict(facecolor="white", edgecolor="none", pad_inches=pad_inches)
    fig.savefig(stem.with_suffix(".pdf"), **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)


def main() -> None:
    _ensure_outputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fp0 = _font_prop()
    fp_tick = _fp_size(fp0, TICK_FS)
    fp_label = _fp_size(fp0, AXIS_LABEL_FS)

    mol = pd.read_csv(P2_OUT / "02_molecule_local_sensitivity.csv")
    corr = pd.read_csv(P2_OUT / "04_activity_ls_correlation.csv")

    _write_table1(OUT_DIR / "表1_典型活性悬崖分子对.txt")
    _write_table2(OUT_DIR / "表2_不同区域稳定性统计.txt")

    _plot_fig1_ls_space(mol, fp_tick, fp_label, OUT_DIR / "图1_分子局部敏感度空间分布")
    _plot_fig2_quadrant(mol, corr, fp_tick, fp_label, OUT_DIR / "图2_活性与局部敏感度四象限图")

    pr = float(corr.loc[corr["metric"] == "pearson_r", "value"].iloc[0])
    sr = float(corr.loc[corr["metric"] == "spearman_rho", "value"].iloc[0])
    manifest = [
        "问题二结果导出清单（2 表 + 2 图）",
        f"Pearson_r={pr:.4f}, Spearman_rho={sr:.4f}",
        "",
        "表1_典型活性悬崖分子对.txt",
        "表2_不同区域稳定性统计.txt",
        "图1_分子局部敏感度空间分布.pdf",
        "图2_活性与局部敏感度四象限图.pdf",
    ]
    (OUT_DIR / "导出清单.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"[export_problem2] -> {OUT_DIR}")
    for line in manifest[3:]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
