import numpy as np
import pandas as pd

def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T

prices_all = load_prices()
nInst, nt = prices_all.shape
logrets_all = np.diff(np.log(prices_all), axis=1)
n_obs = logrets_all.shape[1]

print(f"Loaded {nInst} instruments, {nt} days\n")

# ---------------------------------------------------------------------------
# 1. THE KEY TEST: does drift persist across halves?
#    If drift is a generator constant, first-half mean return should predict
#    second-half mean return across instruments. This is the within-visible
#    transfer test that reversion would have FAILED.
# ---------------------------------------------------------------------------
half = n_obs // 2
drift_1st = logrets_all[:, :half].mean(axis=1)
drift_2nd = logrets_all[:, half:].mean(axis=1)

corr = np.corrcoef(drift_1st, drift_2nd)[0, 1]
sign_agree = np.mean(np.sign(drift_1st) == np.sign(drift_2nd))

print("=== 1. Drift persistence: first half vs second half ===")
print(f"Cross-instrument correlation of drifts: {corr:.4f}")
print(f"Sign agreement (frac of instruments): {sign_agree:.2f}")
print(f"Drift spread across instruments (annualised): "
      f"min {drift_1st.min()*250:.1%}, max {drift_1st.max()*250:.1%}")
print(f"Mean |drift| (annualised): {np.abs(logrets_all.mean(axis=1)).mean()*250:.1%}")

# same test on quarters for a finer view
q = n_obs // 4
print("\nQuarter-to-quarter drift sign agreement:")
for a in range(3):
    d_a = logrets_all[:, a*q:(a+1)*q].mean(axis=1)
    d_b = logrets_all[:, (a+1)*q:(a+2)*q].mean(axis=1)
    print(f"  Q{a+1} vs Q{a+2}: corr {np.corrcoef(d_a, d_b)[0,1]:.3f}, "
          f"sign agree {np.mean(np.sign(d_a)==np.sign(d_b)):.2f}")

# ---------------------------------------------------------------------------
# 2. Backtest: near-static drift-following at full size.
#    Position = sign(historical mean return) sized to a fraction of each cap,
#    vol-scaled, recomputed daily but changing rarely (drift estimates are
#    stable), so commission is minimal.
# ---------------------------------------------------------------------------
print("\n=== 2. Walk-forward: drift-following strategy ===")

CAP_FRACTION = 0.9
ALGO_MAX, SYNTH_MAX = 100_000.0, 10_000.0
scoreDefaultParam = 1.0

def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

commRate = np.full(nInst, 0.0001); commRate[0] = 0.00002
dlrPosLimit = np.full(nInst, SYNTH_MAX); dlrPosLimit[0] = ALGO_MAX

def get_drift_positions(prcSoFar):
    _, t = prcSoFar.shape
    if t < 60:
        return np.zeros(nInst, dtype=int)
    lr = np.diff(np.log(prcSoFar), axis=1)
    drift = lr.mean(axis=1)                       # mean return over ALL history
    vol = lr.std(axis=1) + 1e-9
    conviction = drift / (vol / np.sqrt(lr.shape[1]))   # t-stat of the drift
    strength = np.clip(conviction / 2.0, -1, 1)         # saturate at |t|=2
    curPrices = prcSoFar[:, -1]
    dollars = strength * CAP_FRACTION * dlrPosLimit
    return (dollars / curPrices).astype(int)

def calcPL(prcHist, getPosition, testStart, testEnd):
    cash = 0; curPos = np.zeros(nInst); totDV = 0; value = 0; comm = 0
    pll = []
    for t in range(testStart, testEnd + 1):
        ph = prcHist[:, :t]; cp = ph[:, -1]
        if t < testEnd:
            npos = getPosition(ph)
            lim = (dlrPosLimit / cp).astype(int)
            npos = np.clip(npos, -lim, lim).astype(int)
        else:
            npos = np.array(curPos)
        dp = npos - curPos
        cash -= cp.dot(dp) + comm
        dv = cp * np.abs(dp); totDV += np.sum(dv); comm = np.sum(dv * commRate)
        curPos = np.array(npos)
        pv = curPos.dot(cp)
        pll_t = cash + pv - value; value = cash + pv
        if t > testStart:
            pll.append(pll_t)
    pll = np.array(pll)
    return np.mean(pll), np.std(pll), totDV

WINDOW_LEN, STEP, MIN_START = 100, 50, 150
windows = []
s = MIN_START
while s + WINDOW_LEN <= nt:
    windows.append((s, s + WINDOW_LEN)); s += STEP

print(f"{'window':>15} {'mean(PL)':>10} {'StdDev':>10} {'Score':>10} {'$vol':>12}")
scores = []
for ts, te in windows:
    mu, sd, dv = calcPL(prices_all, get_drift_positions, ts, te)
    sc = score(mu, sd); scores.append(sc)
    print(f"({ts:>4},{te:>4}) {mu:>10.2f} {sd:>10.2f} {sc:>10.2f} {dv:>12.0f}")
scores = np.array(scores)
print(f"\n  mean {scores.mean():.2f} | std {scores.std():.2f} | "
      f"min {scores.min():.2f} | frac+ {(scores > 0).mean():.2f}")

# also the standard full 250-day eval window for comparability
mu, sd, dv = calcPL(prices_all, get_drift_positions, nt - 250, nt)
print(f"\nStandard 250-day eval: mean(PL) {mu:.1f}, StdDev {sd:.1f}, "
      f"Score {score(mu, sd):.2f}, $vol {dv:.0f}")
