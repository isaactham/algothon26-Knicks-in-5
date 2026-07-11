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

print(f"Loaded {nInst} instruments, {nt} days")
print(f"Significance threshold: {sig_threshold:.4f}\n")

# ---------------------------------------------------------------------------
# 1. Lead-lag cross-correlation: does instrument i's return today correlate
#    with instrument j's return `lag` days later? This is what you could
#    actually trade on, unlike same-day correlation.
# ---------------------------------------------------------------------------
print("=== Lead-lag cross-correlation ===")
lags = [1, 2, 3, 5]
leadlag_hits = []

for i in range(nInst):
    for j in range(nInst):
        if i == j:
            continue
        for lag in lags:
            a = logrets_all[i, :-lag]
            b = logrets_all[j, lag:]
            c = np.corrcoef(a, b)[0, 1]
            if abs(c) > sig_threshold:
                leadlag_hits.append((i, j, lag, c))

print(f"Total (leader, follower, lag) combinations tested: {nInst * (nInst - 1) * len(lags)}")
print(f"Combinations exceeding significance threshold: {len(leadlag_hits)}")

leadlag_hits.sort(key=lambda x: -abs(x[3]))
print("\nTop 10 strongest lead-lag relationships:")
print(f"{'leader':>7} {'follower':>9} {'lag':>4} {'corr':>8}")
for i, j, lag, c in leadlag_hits[:10]:
    print(f"{i:>7} {j:>9} {lag:>4} {c:>8.4f}")

# ---------------------------------------------------------------------------
# 2. Cointegration test on the most correlated pairs (from same-day corr).
#    Correlated returns don't guarantee the price levels stay tethered
#    together, cointegration is the real test for that.
# ---------------------------------------------------------------------------
print("\n=== Cointegration test on top same-day correlated pairs ===")
corr_matrix = np.corrcoef(logrets_all)
pairs = [(i, j, corr_matrix[i, j]) for i in range(nInst) for j in range(i + 1, nInst)]
pairs.sort(key=lambda x: -abs(x[2]))
top_pairs = pairs[:10]

print(f"{'pair':>10} {'same-day corr':>15} {'coint p-value':>15}")
for i, j, c in top_pairs:
    _, pvalue, _ = coint(prices_all[i], prices_all[j])
    flag = "  <-- significant" if pvalue < 0.05 else ""
    print(f"({i:>2},{j:>3}) {c:>15.4f} {pvalue:>15.4f}{flag}")
