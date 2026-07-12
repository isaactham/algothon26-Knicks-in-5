import numpy as np
import pandas as pd

def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T

prices_all = load_prices()
nInst, nt = prices_all.shape
logrets_all = np.diff(np.log(prices_all), axis=1)
n_obs = logrets_all.shape[1]
sig_threshold = 2 / np.sqrt(n_obs)

algo = logrets_all[0]                    # ALGO returns
synth = logrets_all[1:]                  # the 50 synthetic instruments
synth_mean = synth.mean(axis=0)          # average synthetic return each day

print(f"Loaded {nInst} instruments, {nt} days. Significance threshold: {sig_threshold:.4f}\n")

# ---------------------------------------------------------------------------
# 1. ALGO's own character: drift, vol, and autocorrelation at many lags
# ---------------------------------------------------------------------------
print("=== 1. ALGO's own structure ===")
print(f"Mean daily return: {algo.mean():.6f}   Daily vol: {algo.std():.6f}")
print(f"Annualised drift: {algo.mean()*250:.2%}   Annualised vol: {algo.std()*np.sqrt(250):.2%}\n")

print(f"{'lag':>4} {'ALGO autocorr':>14}")
for lag in [1, 2, 3, 5, 10, 20]:
    c = np.corrcoef(algo[:-lag], algo[lag:])[0, 1]
    flag = "  <-- significant" if abs(c) > sig_threshold else ""
    print(f"{lag:>4} {c:>14.4f}{flag}")

# volatility clustering in ALGO specifically (real tickers usually have it)
absr = np.abs(algo)
volclust = np.corrcoef(absr[:-1], absr[1:])[0, 1]
print(f"\nALGO |return| autocorrelation (vol clustering): {volclust:.4f}"
      f"{'  <-- significant' if abs(volclust) > sig_threshold else ''}")

# ---------------------------------------------------------------------------
# 2. THE BIG QUESTION: does ALGO lead the synthetics?
#    If synthetics were generated from ALGO/factor + noise, ALGO today might
#    predict synthetics tomorrow. That would be structural, not fitted.
# ---------------------------------------------------------------------------
print("\n=== 2. Does ALGO lead the synthetics? ===")
print(f"{'lag':>4} {'ALGO -> synth mean':>19} {'synth mean -> ALGO':>19}")
for lag in [1, 2, 3, 5]:
    a_leads = np.corrcoef(algo[:-lag], synth_mean[lag:])[0, 1]
    s_leads = np.corrcoef(synth_mean[:-lag], algo[lag:])[0, 1]
    fa = " *" if abs(a_leads) > sig_threshold else ""
    fs = " *" if abs(s_leads) > sig_threshold else ""
    print(f"{lag:>4} {a_leads:>19.4f}{fa} {s_leads:>19.4f}{fs}")

# per-instrument version at lag 1: how many synthetics does ALGO lead individually?
lead_corrs = np.array([np.corrcoef(algo[:-1], synth[i, 1:])[0, 1] for i in range(synth.shape[0])])
print(f"\nPer-instrument ALGO->synth lag-1: mean {lead_corrs.mean():.4f}, "
      f"frac |c|>threshold: {(np.abs(lead_corrs) > sig_threshold).mean():.2f}, "
      f"frac positive: {(lead_corrs > 0).mean():.2f}")

# ---------------------------------------------------------------------------
# 3. Same-day relationship strength (for reference / hedging design)
# ---------------------------------------------------------------------------
print("\n=== 3. Same-day ALGO vs synthetics (reference) ===")
same_day = np.array([np.corrcoef(algo, synth[i])[0, 1] for i in range(synth.shape[0])])
print(f"Mean same-day corr: {same_day.mean():.4f}   Max: {same_day.max():.4f}   Min: {same_day.min():.4f}")

# ---------------------------------------------------------------------------
# 4. Is ALGO's relationship to the factor itself special?
#    Check whether ALGO IS (close to) the factor, vs just another loading on it.
# ---------------------------------------------------------------------------
print("\n=== 4. ALGO vs the common factor ===")
factor = logrets_all.mean(axis=0)
c_af = np.corrcoef(algo, factor)[0, 1]
c_sf = np.corrcoef(synth_mean, factor)[0, 1]
print(f"corr(ALGO, factor): {c_af:.4f}")
print(f"corr(synth mean, factor): {c_sf:.4f}")
print(f"corr(ALGO, synth mean): {np.corrcoef(algo, synth_mean)[0, 1]:.4f}")
