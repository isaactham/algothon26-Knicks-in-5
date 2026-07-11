import numpy as np
import pandas as pd


def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T  # nInst x nt


prices_all = load_prices()
nInst, nt = prices_all.shape
logrets_all = np.diff(np.log(prices_all), axis=1)  # nInst x n_obs
n_obs = logrets_all.shape[1]

print(f"Loaded {nInst} instruments, {nt} days\n")


def regress_on_factor(factor, label):
    print(f"=== Regressing each instrument on {label} ===")
    betas, r2s = [], []
    for i in range(nInst):
        y = logrets_all[i]
        X = np.column_stack([np.ones(n_obs), factor])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X.dot(coeffs)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        betas.append(coeffs[1])
        r2s.append(r2)

    betas = np.array(betas)
    r2s = np.array(r2s)
    print(f"Mean R^2: {r2s.mean():.4f}   Median R^2: {np.median(r2s):.4f}   Max R^2: {r2s.max():.4f}")
    print(f"Mean beta: {betas.mean():.4f}   Std of beta: {betas.std():.4f}")
    print(f"Fraction of instruments with R^2 > 0.10: {(r2s > 0.10).mean():.2f}\n")
    return betas, r2s


# ---------------------------------------------------------------------------
# 1. Use ALGO (asset 0) itself as the candidate common factor
# ---------------------------------------------------------------------------
algo_factor = logrets_all[0]
betas_algo, r2_algo = regress_on_factor(algo_factor, "ALGO (asset 0)")

# ---------------------------------------------------------------------------
# 2. Use the cross-sectional average return (excluding each instrument itself)
#    as a proxy "market" factor, in case the common driver isn't ALGO specifically
# ---------------------------------------------------------------------------
market_factor = logrets_all.mean(axis=0)
betas_mkt, r2_mkt = regress_on_factor(market_factor, "cross-sectional average (market proxy)")

# ---------------------------------------------------------------------------
# 3. Residuals after removing the market factor: how much correlation is left?
#    If the single factor explains most of the co-movement, residual
#    correlations should be much smaller than the raw correlations were.
# ---------------------------------------------------------------------------
print("=== Residual correlation after removing market factor ===")
residuals = np.zeros_like(logrets_all)
X = np.column_stack([np.ones(n_obs), market_factor])
for i in range(nInst):
    y = logrets_all[i]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals[i] = y - X.dot(coeffs)

raw_corr = np.corrcoef(logrets_all)
resid_corr = np.corrcoef(residuals)
raw_upper = raw_corr[np.triu_indices(nInst, k=1)]
resid_upper = resid_corr[np.triu_indices(nInst, k=1)]

print(f"Mean |raw pairwise correlation|: {np.abs(raw_upper).mean():.4f}")
print(f"Mean |residual pairwise correlation|: {np.abs(resid_upper).mean():.4f}")
print(f"Pairs with |raw corr| > 0.5: {np.sum(np.abs(raw_upper) > 0.5)}")
print(f"Pairs with |residual corr| > 0.5: {np.sum(np.abs(resid_upper) > 0.5)}")
