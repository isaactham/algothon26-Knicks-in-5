import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T  # nInst x nt


prices_all = load_prices()
nInst, nt = prices_all.shape
logrets_all = np.diff(np.log(prices_all), axis=1)
n_obs = logrets_all.shape[1]
sig_threshold = 2 / np.sqrt(n_obs)

market_factor = logrets_all.mean(axis=0)
X = np.column_stack([np.ones(n_obs), market_factor])
residuals = np.zeros_like(logrets_all)
for i in range(nInst):
    y = logrets_all[i]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals[i] = y - X.dot(coeffs)

candidate_pairs = [(0, 21), (0, 37), (0, 50), (24, 35), (0, 35)]

print(f"Significance threshold: {sig_threshold:.4f}\n")

# ---------------------------------------------------------------------------
# 1. Cointegration on raw prices: is there a real, stable price-level
#    relationship, not just correlated wiggles?
# ---------------------------------------------------------------------------
print("=== Cointegration test on raw prices ===")
for i, j in candidate_pairs:
    _, pvalue, _ = coint(prices_all[i], prices_all[j])
    flag = "  <-- significant" if pvalue < 0.05 else ""
    print(f"({i:>2},{j:>3})   coint p-value: {pvalue:.4f}{flag}")

# ---------------------------------------------------------------------------
# 2. Lead-lag correlation on the residual series: does one instrument's
#    idiosyncratic move actually precede the other's, in either direction?
# ---------------------------------------------------------------------------
print("\n=== Lead-lag correlation on residuals ===")
lags = [1, 2, 3, 5, 10]
for i, j in candidate_pairs:
    print(f"\nPair ({i},{j}):")
    for lag in lags:
        # i leads j
        c_ij = np.corrcoef(residuals[i, :-lag], residuals[j, lag:])[0, 1]
        # j leads i
        c_ji = np.corrcoef(residuals[j, :-lag], residuals[i, lag:])[0, 1]
        flag_ij = " *" if abs(c_ij) > sig_threshold else ""
        flag_ji = " *" if abs(c_ji) > sig_threshold else ""
        print(f"  lag={lag:>2}   {i} leads {j}: {c_ij:>7.4f}{flag_ij}   "
              f"{j} leads {i}: {c_ji:>7.4f}{flag_ji}")
