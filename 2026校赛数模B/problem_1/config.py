# Problem 1 hyperparameters (tune for sensitivity analysis).

# DBSCAN on standardized Manifold_X, Manifold_Y
DBSCAN_EPS = 0.45
DBSCAN_MIN_SAMPLES = 5

# H_r = LAMBDA1 * norm(region_mean_activity) + LAMBDA2 * high_activity_ratio
LAMBDA1 = 0.5
LAMBDA2 = 0.5

# Hotspot if H_r >= quantile of H over regions
HOTSPOT_QUANTILE = 0.75

# Global high-activity cutoff on Bioactivity_Score
HIGH_ACTIVITY_QUANTILE = 0.75

# Representative score weights
OMEGA1 = 0.4
OMEGA2 = 0.3
OMEGA3 = 0.3

EPS = 1e-12
