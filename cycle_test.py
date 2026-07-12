import numpy as np
import pandas as pd

def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T

prices_all = load_prices()
nInst, nt = prices_all.shape
logp = np.log(prices_all)
logrets = np.diff(logp, axis=1)
n_obs = logrets.shape[1]
factor_ret = logrets.mean(axis=0)
factor_price = logp.mean(axis=0)

print(f"Loaded {nInst} instruments, {nt} days\n")

# ---------------------------------------------------------------------------
# 1. FULL autocorrelation sweep of factor returns, lags 1..100.
#    Cycles show up as an oscillating ACF; we only ever checked 6 lags.
# ---------------------------------------------------------------------------
print("=== 1. Factor return ACF, lags 1-100 (only |acf| > threshold shown) ===")
thr = 2 / np.sqrt(n_obs)
hits = []
for lag in range(1, 101):
    c = np.corrcoef(factor_ret[:-lag], factor_ret[lag:])[0, 1]
    if abs(c) > thr:
        hits.append((lag, c))
print(f"threshold {thr:.4f}, significant lags: {len(hits)} of 100")
for lag, c in hits:
    print(f"  lag {lag:>3}: {c:>7.4f}")

# ---------------------------------------------------------------------------
# 2. Power spectrum of DETRENDED log prices (factor + each instrument).
#    A planted cycle = a sharp spike at its frequency.
# ---------------------------------------------------------------------------
def top_periods(series, k=5, min_period=4):
    x = series - np.polyval(np.polyfit(np.arange(len(series)), series, 1),
                            np.arange(len(series)))          # linear detrend
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    power[0] = 0                                             # drop DC
    order = np.argsort(power)[::-1]
    out = []
    for idx in order:
        if freqs[idx] <= 0:
            continue
        period = 1 / freqs[idx]
        if period < min_period:
            continue
        out.append((period, power[idx]))
        if len(out) == k:
            break
    total = power.sum()
    return out, total

print("\n=== 2. Dominant periods in detrended log FACTOR price ===")
tp, total = top_periods(factor_price, k=5)
for period, p in tp:
    print(f"  period {period:>7.1f} days   power share {p/total:.1%}")

# ---------------------------------------------------------------------------
# 3. THE TRANSFER TEST: do the dominant frequencies in the FIRST half
#    reappear in the SECOND half? Generator constants must persist.
#    For each instrument: find its top period in each half; compare.
# ---------------------------------------------------------------------------
print("\n=== 3. Frequency stability: first half vs second half ===")
half = nt // 2
matches, tops1, tops2 = 0, [], []
for i in range(nInst):
    t1, _ = top_periods(logp[i, :half], k=1)
    t2, _ = top_periods(logp[i, half:], k=1)
    if not t1 or not t2:
        continue
    p1, p2 = t1[0][0], t2[0][0]
    tops1.append(p1); tops2.append(p2)
    if abs(p1 - p2) / max(p1, p2) < 0.25:      # within 25% = same cycle
        matches += 1
print(f"Instruments whose dominant period matches across halves (within 25%): "
      f"{matches}/{len(tops1)}")
print(f"First-half dominant periods:  median {np.median(tops1):.1f}d, "
      f"range [{min(tops1):.1f}, {max(tops1):.1f}]")
print(f"Second-half dominant periods: median {np.median(tops2):.1f}d, "
      f"range [{min(tops2):.1f}, {max(tops2):.1f}]")

# NOTE: for a ~250-day half, a random walk's spectrum is dominated by the
# lowest frequencies, so 'dominant period near the window length' in both
# halves is NOT evidence of a cycle. Real planted cycles show as matching
# periods WELL BELOW the window length (e.g. both halves peaking at ~40d).

# ---------------------------------------------------------------------------
# 4. Calendar periodicity: mean return by day-of-week-style buckets (mod k).
#    A mod-5 or mod-7 pattern in the generator would be perfectly stable.
# ---------------------------------------------------------------------------
print("\n=== 4. Calendar periodicity in factor returns (mod-k buckets) ===")
for k in [5, 7, 10]:
    means = [factor_ret[np.arange(n_obs) % k == b].mean() for b in range(k)]
    spread = (max(means) - min(means))
    # rough significance: each bucket has ~n_obs/k samples
    se = factor_ret.std() / np.sqrt(n_obs / k)
    print(f"  mod {k:>2}: bucket-mean spread {spread:.6f} "
          f"(bucket std err ~{se:.6f}) "
          f"{'<-- LOOK' if spread > 4 * se else ''}")
