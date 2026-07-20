"""
lagTest.py — does the generator have memory beyond yesterday?
Run:  python3 lagTest.py     (expects prices.txt next to it, header row of tickers)

Two tests, cheapest first:

  TEST 1 (self-effects by lag): for each lag k = 1..5, does an instrument's
  relative move k days ago predict its relative move today? One correlation
  per lag, pooled across all instruments. Reads the diagonal's memory depth.

  TEST 2 (does lag k ADD anything?): fit the same ridge machine as the VAR,
  but stacking lags 1..k as inputs. Score each version by out-of-sample
  predictive correlation on held-out recent data. If lags 1..3 beats lag 1
  alone OUT OF SAMPLE, the VAR is genuinely deaf to something. If not,
  the generator is lag-1 and blind spot 3 is moot.
"""

import numpy as np
import pandas as pd

MAX_LAG   = 5
LAM       = 1.0        # same trace-scaled ridge skepticism as the VAR
TEST_FRAC = 0.3        # newest 30% of days held out for out-of-sample scoring

# ---------- load and build residuals (identical to the VAR's Block 2) ----------
prices = pd.read_csv("prices.txt", sep=None, engine="python")
px = prices.values.T                       # (nInst, nDays)
logrets = np.diff(np.log(px), axis=1)
res = logrets - logrets.mean(axis=0)       # moves-vs-the-pack, herd removed
nInst, T = res.shape

# ---------- TEST 1: self-effect strength at each lag ----------
print("TEST 1 — self-effect (diagonal) by lag")
print("  lag   corr(move k days ago, move today)   [pooled over instruments]")
for k in range(1, MAX_LAG + 1):
    past = res[:, :-k].ravel()             # every instrument's move at t-k
    now  = res[:, k:].ravel()              # same instrument's move at t
    c = np.corrcoef(past, now)[0, 1]
    n = past.size
    noise = 2 / np.sqrt(n)                 # rough 2-sigma band for pure noise
    flag = "  <- real" if abs(c) > noise else ""
    print(f"   {k}      {c:+.4f}   (noise band ±{noise:.4f}){flag}")

# ---------- TEST 2: full-matrix, incremental lags, out-of-sample ----------
def build_xy(n_lags):
    """Stack lags 1..n_lags as inputs. Row t: [res(t-1), res(t-2), ...] -> res(t)."""
    Y = res[:, n_lags:].T                              # (T-n_lags, nInst)
    X = np.hstack([res[:, n_lags - k : T - k].T        # lag-k block
                   for k in range(1, n_lags + 1)])     # (T-n_lags, nInst*n_lags)
    return X, Y

def ridge_fit(X, Y):
    G = X.T @ X
    G += LAM * np.eye(X.shape[1]) * np.trace(G) / X.shape[1]
    return np.linalg.solve(G, X.T @ Y)

print("\nTEST 2 — does adding lags beat lag-1 alone, out of sample?")
print("  lags used   in-sample corr   OUT-OF-SAMPLE corr")
split = int(T * (1 - TEST_FRAC))
for n_lags in range(1, MAX_LAG + 1):
    X, Y = build_xy(n_lags)
    cut = split - n_lags                   # align the split across versions
    Xtr, Ytr = X[:cut], Y[:cut]
    Xte, Yte = X[cut:], Y[cut:]

    B = ridge_fit(Xtr, Ytr)                # fit on old days only

    def pred_corr(Xs, Ys):
        P = Xs @ B
        return np.corrcoef(P.ravel(), Ys.ravel())[0, 1]

    print(f"   1..{n_lags}        {pred_corr(Xtr, Ytr):+.4f}          "
          f"{pred_corr(Xte, Yte):+.4f}")

print("""
How to read this:
  TEST 1: bars outside the noise band = that lag carries real self-signal.
          Sign tells you reversion (-) vs momentum (+) at that horizon.
  TEST 2: in-sample corr ALWAYS rises with more lags (more knobs = better
          memorization) — ignore it. Only the OUT-OF-SAMPLE column counts.
          If it peaks at lag 1, the generator has no memory beyond
          yesterday and the VAR already sees everything. If it keeps
          climbing to 2-3, a multi-lag model is worth building.
""")