import numpy as np
import pandas as pd


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

market_factor = logrets_all.mean(axis=0)

# ---------------------------------------------------------------------------
# 1. Compute residuals (idiosyncratic part of each instrument, factor removed)
# ---------------------------------------------------------------------------
X = np.column_stack([np.ones(n_obs), market_factor])
residuals = np.zeros_like(logrets_all)
for i in range(nInst):
    y = logrets_all[i]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals[i] = y - X.dot(coeffs)

# ---------------------------------------------------------------------------
# 2. Autocorrelation of the market factor itself, at several lags
# ---------------------------------------------------------------------------
print("=== Autocorrelation of the market factor ===")
lags = [1, 2, 3, 5, 10]
print(f"{'lag':>4} {'corr':>8}")
for lag in lags:
    c = np.corrcoef(market_factor[:-lag], market_factor[lag:])[0, 1]
    flag = "  <-- above threshold" if abs(c) > sig_threshold else ""
    print(f"{lag:>4} {c:>8.4f}{flag}")

# ---------------------------------------------------------------------------
# 3. Autocorrelation of each instrument's residual (idiosyncratic part),
#    same test as your very first check, but with the common factor removed
# ---------------------------------------------------------------------------
print("\n=== Autocorrelation of residuals (idiosyncratic part) ===")
print(f"{'lag':>4} {'mean corr':>10} {'frac |corr| > threshold':>25}")
for lag in lags:
    corrs = []
    for i in range(nInst):
        r = residuals[i]
        c = np.corrcoef(r[:-lag], r[lag:])[0, 1]
        corrs.append(c)
    corrs = np.array(corrs)
    frac_sig = np.mean(np.abs(corrs) > sig_threshold)
    print(f"{lag:>4} {corrs.mean():>10.4f} {frac_sig:>25.2f}")

# ---------------------------------------------------------------------------
# 4. Which pair still shows strong residual correlation after factor removal?
#    Worth knowing specifically, not just as a count.
# ---------------------------------------------------------------------------
print("\n=== Remaining high-residual-correlation pair(s) ===")
resid_corr = np.corrcoef(residuals)
pairs = [(i, j, resid_corr[i, j]) for i in range(nInst) for j in range(i + 1, nInst)]
pairs.sort(key=lambda x: -abs(x[2]))
for i, j, c in pairs[:5]:
    print(f"({i:>2},{j:>3})   residual corr: {c:.4f}")
