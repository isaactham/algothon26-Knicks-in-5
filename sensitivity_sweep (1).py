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
    """Builds a stateful getPosition function for a given set of parameters."""
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
    plmu, plstd = np.mean(pll), np.std(pll)
    return plmu, plstd


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

# --- variants to test: baseline plus small nearby tweaks ---
variants = {
    "baseline (lags 1,2,5 / vol10 / $2000)": dict(lags=(1, 2, 5), vol_window=10, dollar_target=2000, min_history=15),
    "lags (1,2,4)":                          dict(lags=(1, 2, 4), vol_window=10, dollar_target=2000, min_history=15),
    "lags (1,3,5)":                          dict(lags=(1, 3, 5), vol_window=10, dollar_target=2000, min_history=15),
    "vol_window=8":                          dict(lags=(1, 2, 5), vol_window=8,  dollar_target=2000, min_history=15),
    "vol_window=12":                         dict(lags=(1, 2, 5), vol_window=12, dollar_target=2000, min_history=15),
    "dollar_target=1000":                    dict(lags=(1, 2, 5), vol_window=10, dollar_target=1000, min_history=15),
    "dollar_target=4000":                    dict(lags=(1, 2, 5), vol_window=10, dollar_target=4000, min_history=15),
    "lags (2,5) only":                       dict(lags=(2, 5),    vol_window=10, dollar_target=2000, min_history=15),
}

print(f"Loaded {nInst} instruments, {nt} days, {len(windows)} windows\n")
print(f"{'variant':<40} {'mean score':>10} {'std score':>10} {'min score':>10} {'frac +':>8}")

for name, params in variants.items():
    getPosition, reset = make_strategy(**params)
    scores = []
    for testStart, testEnd in windows:
        reset()
        plmu, plstd = calcPL(prcAll, nInst, commRate, dlrPosLimit, getPosition, testStart, testEnd)
        scores.append(score(plmu, plstd))
    scores = np.array(scores)
    print(f"{name:<40} {scores.mean():>10.2f} {scores.std():>10.2f} {scores.min():>10.2f} {(scores > 0).mean():>8.2f}")
