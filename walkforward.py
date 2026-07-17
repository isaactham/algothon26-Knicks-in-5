import importlib
import numpy as np
import pandas as pd
import teamNameBlend as teamName

# --- same constants as the official eval.py ---
pricesFile = "./prices.txt"
scoreDefaultParam = 1.0
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

# --- walk-forward window settings ---
WINDOW_LEN = 100   # number of days scored per window
STEP = 50          # how far each window slides forward
MIN_START = 150    # earliest a test window can start, leaves burn-in history


def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T  # nInst x nt


def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


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
    annSharpe = np.sqrt(250) * plmu / plstd if plstd > 0 else 0.0
    return plmu, plstd, annSharpe, totDVolume


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

print(f"Loaded {nInst} instruments, {nt} days")
print(f"Testing {len(windows)} windows of {WINDOW_LEN} days, stepping by {STEP}\n")
print(f"{'window':>15} {'mean(PL)':>10} {'StdDev(PL)':>12} {'Sharpe':>8} {'Score':>10}")

scores = []
for testStart, testEnd in windows:
    importlib.reload(teamName)   # reset the strategy's internal state fresh for each window
    getPosition = teamName.getMyPosition

    plmu, plstd, sharpe, dvol = calcPL(
        prcAll, nInst, commRate, dlrPosLimit, getPosition, testStart, testEnd
    )
    s = score(plmu, plstd, scoreDefaultParam)
    scores.append(s)
    print(f"({testStart:>4},{testEnd:>4}) {plmu:>10.2f} {plstd:>12.2f} {sharpe:>8.2f} {s:>10.2f}")

scores = np.array(scores)
print("\n=== Summary across windows ===")
print(f"Mean score:     {scores.mean():.2f}")
print(f"StdDev of score: {scores.std():.2f}")
print(f"Fraction of windows with positive score: {(scores > 0).mean():.2f}")
print(f"Worst window score: {scores.min():.2f}")