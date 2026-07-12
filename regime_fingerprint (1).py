import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# REGIME FINGERPRINT
# For a given prices file, sweeps lookback horizons and measures, at each:
#   RAW:   does the trailing h-day return predict the next day's return?
#          (factor-exposed: mostly measures the factor's behaviour)
#   XSEC:  same, but cross-sectionally demeaned each day
#          (factor-neutral: measures idiosyncratic/relative behaviour)
# Positive = continuation (trend), negative = reversion, ~0 = nothing.
# Run on any chunk:  python regime_fingerprint.py [prices.txt] [start] [end]
# ---------------------------------------------------------------------------

START = 250      # <-- change these two lines per run
END = 500

fn = "prices.txt"
df = pd.read_csv(fn, sep=r"\s+", header=0)
prices = df.values.T
prices = prices[:, START:END]

nInst, nt = prices.shape
lr = np.diff(np.log(prices), axis=1)
T = lr.shape[1]
print(f"{fn}: {nInst} instruments, {nt} days (analysing {T} returns)\n")

lr_x = lr - lr.mean(axis=0, keepdims=True)   # cross-sectionally demeaned
cum = np.cumsum(lr, axis=1)
cum_x = np.cumsum(lr_x, axis=1)

def edge_xsec(h):
    """Cross-sectional: mean daily cross-instrument corr between trailing-h
    return and next return. Factor-neutral by construction (Pearson demeans)."""
    ics = []
    for d in range(h, T - 1):
        trail = cum[:, d] - (cum[:, d - h] if d - h >= 0 else 0)
        nxt = lr[:, d + 1]
        if trail.std() > 1e-12 and nxt.std() > 1e-12:
            ics.append(np.corrcoef(trail, nxt)[0, 1])
    ics = np.array(ics)
    if len(ics) < 20:
        return np.nan, np.nan
    se = ics.std() / np.sqrt(len(ics)) + 1e-12
    return ics.mean(), ics.mean() / se


def edge_raw(h):
    """Time-series (factor-exposed): per instrument, corr of its own
    trailing-h return with its own next-day return; averaged across
    instruments, t-stat from the cross-instrument spread."""
    per_inst = []
    for i in range(nInst):
        trail = cum[i, h:T - 1] - cum[i, :T - 1 - h]
        nxt = lr[i, h + 1:]
        if trail.std() > 1e-12 and nxt.std() > 1e-12:
            per_inst.append(np.corrcoef(trail, nxt)[0, 1])
    per_inst = np.array(per_inst)
    se = per_inst.std() / np.sqrt(len(per_inst)) + 1e-12
    return per_inst.mean(), per_inst.mean() / se

horizons = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 80, 100]
print(f"{'h':>4} {'RAW edge':>10} {'t':>6}   {'XSEC edge':>10} {'t':>6}")
for h in horizons:
    if h >= T - 30:
        break
    e_raw, t_raw = edge_raw(h)
    e_x, t_x = edge_xsec(h)
    m_raw = " *" if abs(t_raw) > 2 else "  "
    m_x = " *" if abs(t_x) > 2 else "  "
    print(f"{h:>4} {e_raw:>10.4f} {t_raw:>6.1f}{m_raw} {e_x:>10.4f} {t_x:>6.1f}{m_x}")

print("\n* = |t| > 2. Sign: + continuation / - reversion.")
print("Compare fingerprints across data chunks to see regime changes directly.")
