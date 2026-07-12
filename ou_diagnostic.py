import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# OU DIAGNOSTIC
# Hypothesis: each instrument's residual (vs the common factor) is a
# level-reverting OU process. If true:
#   1. residual cumulative series have AR(1) < 1 with a meaningful half-life
#   2. the CURRENT DEVIATION of the residual level predicts the next residual
#      return (this is the optimal signal, superior to trailing returns)
#   3. the parameters are STABLE across dataset halves (transfer test)
# ---------------------------------------------------------------------------

def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T

prices = load_prices()
nInst, nt = prices.shape
lr = np.diff(np.log(prices), axis=1)
T = lr.shape[1]

# betas and residual returns (full-sample, descriptive diagnostic)
f = lr.mean(axis=0)
fc = f - f.mean()
beta = (lr @ fc) / (fc @ fc)
resid = lr - beta[:, None] * f[None, :]
e = np.cumsum(resid, axis=1)               # residual "price" series

print(f"{nInst} instruments, {nt} days\n")

def ou_stats(series):
    """AR(1) coeff and implied half-life of a level series."""
    s = series - series.mean()
    denom = (s[:-1] @ s[:-1])
    if denom < 1e-12:
        return np.nan, np.nan
    phi = (s[:-1] @ s[1:]) / denom
    hl = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.inf
    return phi, hl

# ---------------------------------------------------------------------------
# 1. OU character of each residual series (full sample)
# ---------------------------------------------------------------------------
phis, hls = [], []
for i in range(nInst):
    phi, hl = ou_stats(e[i])
    phis.append(phi); hls.append(hl)
phis, hls = np.array(phis), np.array(hls)
finite = np.isfinite(hls)
print("=== 1. Residual OU character (full sample) ===")
print(f"AR(1): median {np.median(phis):.4f}, range [{phis.min():.4f}, {phis.max():.4f}]")
print(f"Fraction with AR(1) < 1 (reverting): {(phis < 1).mean():.2f}")
print(f"Half-life (finite only, {finite.sum()}/{nInst}): "
      f"median {np.median(hls[finite]):.1f}d, "
      f"IQR [{np.percentile(hls[finite], 25):.1f}, {np.percentile(hls[finite], 75):.1f}]")

# ---------------------------------------------------------------------------
# 2. Does the deviation predict the next residual return? (the optimal signal)
#    Signal at day d: -(e[i,d] - mean of e[i] up to d) using ONLY past data.
# ---------------------------------------------------------------------------
print("\n=== 2. Predictive power of the level deviation (no lookahead) ===")
BURN = 100
ics = []
run_mean = np.cumsum(e, axis=1) / np.arange(1, T + 1)[None, :]
for d in range(BURN, T - 1):
    devi = -(e[:, d] - run_mean[:, d])
    nxt = resid[:, d + 1]
    if devi.std() > 1e-12 and nxt.std() > 1e-12:
        ics.append(np.corrcoef(devi, nxt)[0, 1])
ics = np.array(ics)
se = ics.std() / np.sqrt(len(ics))
print(f"Mean daily cross-sectional IC: {ics.mean():.4f}   t-stat: {ics.mean()/se:.1f}")
print(f"(compare: two-band trailing-return signal had IC magnitude ~0.02, t ~2.7)")

# ---------------------------------------------------------------------------
# 3. Stability across halves: the transfer test
# ---------------------------------------------------------------------------
print("\n=== 3. Parameter stability: first half vs second half ===")
half = T // 2
hl1, hl2 = [], []
for i in range(nInst):
    _, h1 = ou_stats(e[i, :half])
    _, h2 = ou_stats(e[i, half:])
    if np.isfinite(h1) and np.isfinite(h2):
        hl1.append(h1); hl2.append(h2)
hl1, hl2 = np.array(hl1), np.array(hl2)
print(f"Instruments with finite half-life in BOTH halves: {len(hl1)}/{nInst}")
if len(hl1) > 5:
    print(f"Half-life medians: 1st {np.median(hl1):.1f}d, 2nd {np.median(hl2):.1f}d")
    print(f"Cross-instrument correlation of half-lives: "
          f"{np.corrcoef(hl1, hl2)[0,1]:.3f}")

# per-half predictive IC of the deviation signal
for name, a, b in [("1st half", BURN, half), ("2nd half", half + BURN, T - 1)]:
    sub = []
    for d in range(a, b):
        if d >= T - 1:
            break
        devi = -(e[:, d] - run_mean[:, d])
        nxt = resid[:, d + 1]
        if devi.std() > 1e-12 and nxt.std() > 1e-12:
            sub.append(np.corrcoef(devi, nxt)[0, 1])
    sub = np.array(sub)
    if len(sub) > 20:
        print(f"Deviation-signal IC, {name}: {sub.mean():.4f} "
              f"(t {sub.mean()/(sub.std()/np.sqrt(len(sub))):.1f})")
