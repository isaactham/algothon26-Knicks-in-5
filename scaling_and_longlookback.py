import numpy as np
import pandas as pd

pricesFile = "./prices.txt"
scoreDefaultParam = 1.0
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

WINDOW_LEN = 100
STEP = 50
MIN_START = 150


def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def make_strategy(lags, vol_window, dollar_target, min_history):
    state = {"pos": None}

    def getPosition(prcSoFar):
        nInst, nt = prcSoFar.shape
        if state["pos"] is None or state["pos"].shape[0] != nInst:
            state["pos"] = np.zeros(nInst)
        if nt < min_history:
            return state["pos"]
        logrets = np.diff(np.log(prcSoFar), axis=1)
        combined = np.mean([logrets[:, -lag] for lag in lags], axis=0)
        signal = -combined
        window = logrets[:, -vol_window:]
        vol = np.std(window, axis=1)
        vol = np.where(vol < 1e-6, 1e-6, vol)
        signalStrength = np.clip(signal / vol, -1, 1)
        dollarTarget = signalStrength * dollar_target
        curPrices = prcSoFar[:, -1]
        newPos = (dollarTarget / curPrices).astype(int)
        state["pos"] = newPos
        return newPos

    def reset():
        state["pos"] = None
    return getPosition, reset


def make_strategy_longlookback(short_lags, long_lookback, long_weight, vol_window, dollar_target, min_history):
    """Multi-lag short reversion plus a longer-lookback reversion component."""
    state = {"pos": None}

    def getPosition(prcSoFar):
        nInst, nt = prcSoFar.shape
        if state["pos"] is None or state["pos"].shape[0] != nInst:
            state["pos"] = np.zeros(nInst)
        if nt < min_history:
            return state["pos"]
        logrets = np.diff(np.log(prcSoFar), axis=1)
        short_sig = -np.mean([logrets[:, -lag] for lag in short_lags], axis=0)
        long_sig = -logrets[:, -long_lookback:].sum(axis=1)  # trailing long-window return

        # standardise each component by its own cross-sectional scale before combining
        short_n = short_sig / (np.std(short_sig) + 1e-9)
        long_n = long_sig / (np.std(long_sig) + 1e-9)
        signal = (1 - long_weight) * short_n + long_weight * long_n

        window = logrets[:, -vol_window:]
        vol = np.std(window, axis=1)
        vol = np.where(vol < 1e-6, 1e-6, vol)
        signalStrength = np.clip(signal / (vol / np.mean(vol)), -1, 1)
        dollarTarget = signalStrength * dollar_target
        curPrices = prcSoFar[:, -1]
        newPos = (dollarTarget / curPrices).astype(int)
        state["pos"] = newPos
        return newPos

    def reset():
        state["pos"] = None
    return getPosition, reset


def calcPL(prcHist, nInst, commRate, dlrPosLimit, getPosition, testStart, testEnd):
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


prcAll = loadPrices(pricesFile)
nInst, nt = prcAll.shape
commRate = np.full(nInst, defaultCommRate)
commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit

windows = []
start = MIN_START
while start + WINDOW_LEN <= nt:
    windows.append((start, start + WINDOW_LEN))
    start += STEP


def evaluate(make_fn, params):
    getPosition, reset = make_fn(**params)
    scores, dvols = [], []
    for testStart, testEnd in windows:
        reset()
        mu, sd, dvol = calcPL(prcAll, nInst, commRate, dlrPosLimit, getPosition, testStart, testEnd)
        scores.append(score(mu, sd))
        dvols.append(dvol)
    scores = np.array(scores)
    return scores.mean(), scores.std(), scores.min(), (scores > 0).mean(), np.mean(dvols)


print(f"Loaded {nInst} instruments, {nt} days, {len(windows)} windows\n")

# ---- PART 1: scale dollar target up through the clipping regime ----
print("=== PART 1: dollar_target sweep (baseline lags 1,2,5) ===")
print(f"{'dollar_target':>14} {'mean score':>11} {'std':>8} {'min':>8} {'frac+':>7} {'avg $vol':>12}")
for dt in [2000, 5000, 10000, 20000, 40000, 80000]:
    m, s, mn, fp, dv = evaluate(make_strategy,
                                dict(lags=(1, 2, 5), vol_window=10, dollar_target=dt, min_history=15))
    print(f"{dt:>14} {m:>11.2f} {s:>8.2f} {mn:>8.2f} {fp:>7.2f} {dv:>12.0f}")

# ---- PART 2: add a longer-lookback reversion component ----
print("\n=== PART 2: adding long-lookback reversion (dollar_target=20000) ===")
print(f"{'long_lb':>8} {'long_wt':>8} {'mean score':>11} {'std':>8} {'min':>8} {'frac+':>7}")
for long_lb in [10, 20, 40]:
    for long_wt in [0.0, 0.3, 0.5, 0.7]:
        m, s, mn, fp, dv = evaluate(
            make_strategy_longlookback,
            dict(short_lags=(1, 2, 5), long_lookback=long_lb, long_weight=long_wt,
                 vol_window=10, dollar_target=20000, min_history=45))
        print(f"{long_lb:>8} {long_wt:>8.1f} {m:>11.2f} {s:>8.2f} {mn:>8.2f} {fp:>7.2f}")
