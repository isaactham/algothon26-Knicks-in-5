import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T  # nInst x nt


prices_all = load_prices()
nInst, nt = prices_all.shape
logrets_all = np.diff(np.log(prices_all), axis=1)  # nInst x (nt-1)
n_obs = logrets_all.shape[1]

print(f"Loaded {nInst} instruments, {nt} days, {n_obs} return observations\n")

# ---------------------------------------------------------------------------
# 1. Per-instrument autocorrelation at several lags
#    Does yesterday's (or 2/3/5/10 days ago's) return predict today's, on average?
# ---------------------------------------------------------------------------
print("=== Autocorrelation at multiple lags ===")
sig_threshold = 2 / np.sqrt(n_obs)
print(f"Significance threshold (2/sqrt(n)): {sig_threshold:.4f}\n")

lags = [1, 2, 3, 5, 10]
print(f"{'lag':>4} {'mean corr':>10} {'frac |corr| > threshold':>25}")
for lag in lags:
    corrs = []
    for i in range(nInst):
        r = logrets_all[i]
        c = np.corrcoef(r[:-lag], r[lag:])[0, 1]
        corrs.append(c)
    corrs = np.array(corrs)
    frac_sig = np.mean(np.abs(corrs) > sig_threshold)
    print(f"{lag:>4} {corrs.mean():>10.4f} {frac_sig:>25.2f}")

# ---------------------------------------------------------------------------
# 2. Cross-sectional ranking stability
#    Rank all instruments by trailing N-day return. Does that ranking still
#    look similar a few days/weeks later, or is it basically reshuffled?
# ---------------------------------------------------------------------------
print("\n=== Cross-sectional ranking stability ===")
window = 10
cum_logrets = np.cumsum(logrets_all, axis=1)
trailing_ret = cum_logrets[:, window:] - cum_logrets[:, :-window]  # nInst x (n_obs - window)

for gap in [1, 5, 10, 20]:
    if trailing_ret.shape[1] <= gap:
        continue
    rank_corrs = []
    for t in range(trailing_ret.shape[1] - gap):
        rho, _ = spearmanr(trailing_ret[:, t], trailing_ret[:, t + gap])
        if not np.isnan(rho):
            rank_corrs.append(rho)
    rank_corrs = np.array(rank_corrs)
    print(f"gap={gap:>3}d   mean rank correlation: {rank_corrs.mean():.4f}   "
          f"frac positive: {(rank_corrs > 0).mean():.2f}")

# ---------------------------------------------------------------------------
# 3. Cross-instrument correlation
#    Do any instruments move together? Needed for pairs/stat-arb style ideas.
# ---------------------------------------------------------------------------
print("\n=== Cross-instrument correlation ===")
corr_matrix = np.corrcoef(logrets_all)
upper = corr_matrix[np.triu_indices(nInst, k=1)]
print(f"Total pairs: {len(upper)}")
print(f"Mean pairwise correlation: {upper.mean():.4f}")
print(f"Pairs with |correlation| > 0.5: {np.sum(np.abs(upper) > 0.5)}")
print(f"Max correlation: {upper.max():.4f}   Min correlation: {upper.min():.4f}")

# ---------------------------------------------------------------------------
# 4. Volatility clustering
#    Does yesterday's volatility predict today's? Needed for vol-based sizing
#    to be grabbing onto something real rather than noise.
# ---------------------------------------------------------------------------
print("\n=== Volatility clustering (autocorrelation of |returns|) ===")
vol_autocorrs = []
for i in range(nInst):
    absret = np.abs(logrets_all[i])
    c = np.corrcoef(absret[:-1], absret[1:])[0, 1]
    vol_autocorrs.append(c)
vol_autocorrs = np.array(vol_autocorrs)
print(f"Mean autocorrelation of |returns|: {vol_autocorrs.mean():.4f}")
print(f"Fraction with autocorrelation > {sig_threshold:.4f}: "
      f"{(vol_autocorrs > sig_threshold).mean():.2f}")
