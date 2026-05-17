# Problem 3: budget-constrained multi-objective selection (greedy).

ALPHA = 0.6  # activity weight in U_i
BETA = 0.4  # normalized LS penalty in U_i

# Greedy marginal weights (align with plan: U + diversity + region coverage)
ETA_U = 0.6
ETA_DIV = 0.25
ETA_REGION = 0.15

K_SELECT = 8
EPS = 1e-12

# Legacy aliases
W1, W2 = ALPHA, BETA
ETA1, ETA2, ETA3 = ETA_U, ETA_DIV, ETA_REGION
