"""
理化指标与活性得分的 Pearson 相关系数柱状图（7 项）。
画布 2:1，版式与流形图脚本一致（宋体系、脊线粗细、字号）。

运行：python scripts/plot_phys_property_pearson_bar.py
输出：out/data_analysis/理化指标与活性_Pearson相关系数柱状图.pdf / .png
      out/data_analysis/理化指标与活性_Pearson相关系数.csv
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
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out" / "data_analysis"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.loaders import load_manifest, manifest_phys_cols

# 画布 2:1（宽/高 = 2）
FIG_W_IN = 8.0
FIG_H_IN = 4.0

LEFT_MARGIN_IN = 0.62
TOP_MARGIN_IN = 0.26
RIGHT_MARGIN_IN = 0.28
# 图高比例：为横轴刻度 +「理化指标」轴标题预留的底部空白（归一化）
FIG_BOTTOM_FRAC = 0.36

SPINE_LW = 1.05
TICK_LEN = 4.0
AXIS_LABEL_FS = 10.0
TICK_FS = 9.0
LABELPAD = 4.0
XLABEL_PAD = 10.0
XTICK_PAD = 4.0

# 从左到右：红橙黄绿青蓝紫 + 镂空纹理（与样例一致）
BAR_EDGE_HATCH: list[tuple[str, str]] = [
    ("#d62728", "xxxx"),  # 红：交叉网格
    ("#ff7f0e", "----"),  # 橙：横线
    ("#e6b800", "////"),  # 黄：斜线
    ("#2ca02c", "\\\\\\\\"),  # 绿：反斜线
    ("#17becf", "++++"),  # 青：方格
    ("#1f77b4", "|||"),  # 蓝：竖线
    ("#9467bd", "oooo"),  # 紫：点阵
]

_FONT_MATCH_ORDER = [
    "SimSun",
    "NSimSun",
    "STSong",
    "Songti SC",
    "Source Han Serif SC",
    "Noto Serif CJK SC",
]

XLABEL = "理化指标"
YLABEL = "与活性得分的 Pearson 相关系数"


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


def _style_axes(ax: plt.Axes, fp_tick: FontProperties, fp_label: FontProperties) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(
        axis="both",
        which="major",
        length=TICK_LEN,
        width=SPINE_LW,
        labelsize=TICK_FS,
    )
    ax.tick_params(axis="x", pad=XTICK_PAD)
    ax.set_xlabel(XLABEL, fontproperties=fp_label, labelpad=XLABEL_PAD)
    ax.set_ylabel(YLABEL, fontproperties=fp_label, labelpad=LABELPAD)
    fmt = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(fmt)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp_tick)
        t.set_rotation(28)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    ax.xaxis.label.set_clip_on(False)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp_tick)
    ax.axhline(0.0, color="#666666", linewidth=0.85, linestyle="-", zorder=0)


def _compute_pearson(df: pd.DataFrame, phys_cols: list[str]) -> pd.DataFrame:
    rows = []
    act = df["Bioactivity_Score"]
    for col in phys_cols:
        sub = df[[col, "Bioactivity_Score"]].dropna()
        if len(sub) < 3:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(sub[col].to_numpy(), sub["Bioactivity_Score"].to_numpy())
        rows.append(
            {
                "property": col,
                "pearson_r": float(r),
                "p_value": float(p),
                "n_pairs": len(sub),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_r"] = out["pearson_r"].abs()
    return out.sort_values("abs_r", ascending=False).reset_index(drop=True)


def main() -> None:
    fp0, font_note = _font_prop_from_fc_match()
    fp_tick = _fp_size(fp0, TICK_FS)
    fp_label = _fp_size(fp0, AXIS_LABEL_FS)

    df = load_manifest().dropna(subset=["Bioactivity_Score"])
    phys_cols = manifest_phys_cols(df)
    if not phys_cols:
        raise RuntimeError("未找到理化指标列。")

    corr_df = _compute_pearson(df, phys_cols)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "理化指标与活性_Pearson相关系数.csv"
    corr_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    names = corr_df["property"].tolist()
    rs = corr_df["pearson_r"].to_numpy(dtype=float)
    nbar = len(names)
    if nbar != len(BAR_EDGE_HATCH):
        raise ValueError(f"期望 7 个理化指标，当前为 {nbar} 个。")

    plt.rcParams["hatch.linewidth"] = 0.85

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    fig.subplots_adjust(
        left=LEFT_MARGIN_IN / FIG_W_IN,
        bottom=FIG_BOTTOM_FRAC,
        right=1.0 - RIGHT_MARGIN_IN / FIG_W_IN,
        top=1.0 - TOP_MARGIN_IN / FIG_H_IN,
    )

    x = np.arange(nbar)
    width = 0.62
    for i, (xi, ri) in enumerate(zip(x, rs)):
        edge_c, hatch = BAR_EDGE_HATCH[i]
        bars = ax.bar(
            xi,
            ri,
            width=width,
            facecolor="white",
            edgecolor=edge_c,
            hatch=hatch,
            linewidth=1.05,
            zorder=2,
        )
        patch = bars[0]
        patch.set_edgecolor(edge_c)
        if hasattr(patch, "set_hatch_color"):
            patch.set_hatch_color(edge_c)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ymax = float(np.nanmax(np.abs(rs))) if np.isfinite(rs).any() else 1.0
    ymax = max(ymax * 1.12, 0.15)
    ax.set_ylim(-0.2, ymax)
    _style_axes(ax, fp_tick, fp_label)

    stem = OUT_DIR / "理化指标与活性_Pearson相关系数柱状图"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    meta = OUT_DIR / "理化指标与活性_Pearson相关系数_meta.txt"
    meta.write_text(
        f"画布英寸={FIG_W_IN}x{FIG_H_IN}, 比例2:1\n"
        f"n_indicators={len(names)}\n"
        f"bar_style=红橙黄绿青蓝紫镂空纹理\n"
        f"font={font_note}\n",
        encoding="utf-8",
    )

    print(f"Saved: {stem}.pdf / .png")
    print(f"Saved: {csv_path.name}")
    print(font_note)


if __name__ == "__main__":
    main()
