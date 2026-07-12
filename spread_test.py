import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T

prices_all = load_prices()
nInst, nt = prices_all.shape

log_algo = np.log(prices_all[0])
# basket = equal-weighted average of synthetic LOG prices (geometric basket)
log_basket = np.log(prices_all[1:]).mean(axis=0)

print(f"Loaded {nInst} instruments, {nt} days\n")

# ---------------------------------------------------------------------------
# 1. Is the ALGO-basket spread stationary? (cointegration + ADF on the spread)
# ---------------------------------------------------------------------------
print("=== 1. Stationarity of the ALGO-basket relationship ===")
_, coint_p, _ = coint(log_algo, log_basket)
print(f"Engle-Granger cointegration p-value: {coint_p:.4f}"
      f"{'  <-- significant' if coint_p < 0.05 else '  (not significant)'}")

# fit the hedge ratio on log prices, then test the residual spread directly
X = np.column_stack([np.ones(nt), log_basket])
coeffs, _, _, _ = np.linalg.lstsq(X, log_algo, rcond=None)
spread = log_algo - X.dot(coeffs)
adf_stat, adf_p, *_ = adfuller(spread)
print(f"ADF test on fitted spread: stat {adf_stat:.3f}, p-value {adf_p:.4f}"
      f"{'  <-- stationary' if adf_p < 0.05 else '  (not stationary)'}")
print(f"Fitted hedge ratio (ALGO ~ basket): {coeffs[1]:.4f}")
print(f"Spread std: {spread.std():.5f}, spread range: [{spread.min():.4f}, {spread.max():.4f}]")

# how quickly does the spread revert? (AR(1) coefficient -> half-life)
s = spread - spread.mean()
ar1 = np.dot(s[:-1], s[1:]) / np.dot(s[:-1], s[:-1])
halflife = -np.log(2) / np.log(ar1) if 0 < ar1 < 1 else np.inf
print(f"Spread AR(1) coefficient: {ar1:.4f}   implied half-life: {halflife:.1f} days")

# ---------------------------------------------------------------------------
# 2. Walk-forward trading test of the spread
#    Strategy: compute rolling z-score of the spread using ONLY past data.
#    When z is high (ALGO rich vs basket): short ALGO, long basket. Vice versa.
#    Sized within eval.py's real caps: ALGO up to $100k, each synthetic $10k.
#    Includes real commissions (0.2bp ALGO, 1bp synthetics).
# ---------------------------------------------------------------------------
print("\n=== 2. Walk-forward trading test of the spread ===")

ROLL = 60            # rolling window for spread mean/std (uses only past data)
Z_CLIP = 2.0         # z-score saturation
ALGO_MAX = 100_000
SYNTH_MAX_EACH = 10_000
SYNTH_BOOK = 50 * SYNTH_MAX_EACH * 0.2   # use 20% of synthetic caps for the basket leg

scoreDefaultParam = 1.0
def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

commRate = np.full(nInst, 0.0001)
commRate[0] = 0.00002
dlrPosLimit = np.full(nInst, 10_000.0)
dlrPosLimit[0] = 100_000.0

WINDOW_LEN = 100
STEP = 50
MIN_START = 150

def get_spread_positions(prcSoFar):
    """Positions for the spread trade using only history up to today."""
    _, t = prcSoFar.shape
    la = np.log(prcSoFar[0])
    lb = np.log(prcSoFar[1:]).mean(axis=0)

    if t < ROLL + 5:
        return np.zeros(nInst, dtype=int)

    # rolling hedge ratio + z-score from the last ROLL days only
    Xw = np.column_stack([np.ones(ROLL), lb[-ROLL:]])
    cw, _, _, _ = np.linalg.lstsq(Xw, la[-ROLL:], rcond=None)
    spr = la[-ROLL:] - Xw.dot(cw)
    z = (spr[-1] - spr.mean()) / (spr.std() + 1e-9)
    strength = np.clip(z / Z_CLIP, -1, 1)

    # z > 0: ALGO rich -> short ALGO, long basket. z < 0: opposite.
    curPrices = prcSoFar[:, -1]
    pos = np.zeros(nInst)
    pos[0] = -strength * ALGO_MAX / curPrices[0]
    per_synth_dollars = strength * SYNTH_BOOK / 50
    pos[1:] = per_synth_dollars / curPrices[1:]
    return pos.astype(int)

def calcPL(prcHist, getPosition, testStart, testEnd):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0
    todayPLL = []
    for t in range(testStart, testEnd + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]
        if t < testEnd:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)
        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        totDVolume += np.sum(dvolumes)
        comm = np.sum(dvolumes * commRate)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        if t > testStart:
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    return np.mean(pll), np.std(pll), totDVolume

windows = []
start = MIN_START
while start + WINDOW_LEN <= nt:
    windows.append((start, start + WINDOW_LEN))
    start += STEP

print(f"{'window':>15} {'mean(PL)':>10} {'StdDev':>10} {'Score':>10} {'$vol':>12}")
scores = []
for ts, te in windows:
    mu, sd, dv = calcPL(prices_all, get_spread_positions, ts, te)
    s = score(mu, sd)
    scores.append(s)
    print(f"({ts:>4},{te:>4}) {mu:>10.2f} {sd:>10.2f} {s:>10.2f} {dv:>12.0f}")
scores = np.array(scores)
print(f"\n  mean {scores.mean():.2f} | std {scores.std():.2f} | "
      f"min {scores.min():.2f} | frac+ {(scores > 0).mean():.2f}")
