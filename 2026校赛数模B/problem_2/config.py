# Problem 2 hyperparameters (aligned with docs/plan.tex).

# LS_i = ALPHA * mean(neighbor C'_ij) + (1-ALPHA) * max(neighbor C'_ij)
ALPHA = 0.7

# Cliff edge thresholds on normalized C'_ij
CLIFF_QUANTILE_MODERATE = 0.75
CLIFF_QUANTILE_STRONG = 0.90

# High activity / high sensitivity (quadrant & u_r)
HIGH_ACTIVITY_QUANTILE = 0.75
HIGH_LS_QUANTILE = 0.75

# 表1 典型悬崖分子对行数
TOP_CLIFF_PAIRS_PUBLISH = 12

EPS = 1e-12
