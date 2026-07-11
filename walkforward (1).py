import numpy as np
import pandas as pd
import teamName

# --- same constants as the official eval.py ---
pricesFile = "./prices.txt"
scoreDefaultParam = 1.0
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

# --- walk-forward window settings ---
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


def run_walkforward(overrides=None, label="teamName as-is"):
    """
    Run walk-forward on teamName.getMyPosition.
    overrides: optional dict of {PARAM_NAME: value} to set on the teamName module
               before running, e.g. {"DOLLAR_TARGET": 80000}. None uses the file's values.
    """
    if overrides:
        for k, v in overrides.items():
            setattr(teamName, k, v)

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

    print(f"=== {label} ===")
    print(f"{'window':>15} {'mean(PL)':>10} {'StdDev':>10} {'Sharpe':>8} {'Score':>10} {'$vol':>12}")

    scores = []
    for testStart, testEnd in windows:
        teamName._reset()   # fresh internal state for each window
        plmu, plstd, sharpe, dvol = calcPL(
            prcAll, nInst, commRate, dlrPosLimit, teamName.getMyPosition, testStart, testEnd
        )
        s = score(plmu, plstd)
        scores.append(s)
        print(f"({testStart:>4},{testEnd:>4}) {plmu:>10.2f} {plstd:>10.2f} {sharpe:>8.2f} {s:>10.2f} {dvol:>12.0f}")

    scores = np.array(scores)
    print(f"  mean {scores.mean():.2f} | std {scores.std():.2f} | "
          f"min {scores.min():.2f} | frac+ {(scores > 0).mean():.2f}\n")
    return scores


if __name__ == "__main__":
    # Run whatever is currently in teamName.py
    run_walkforward(label="teamName.py as-is")

    # Compare conservative vs maxed sizing without editing the file
    run_walkforward({"DOLLAR_TARGET": 20000}, label="CONSERVATIVE ($20k)")
    run_walkforward({"DOLLAR_TARGET": 80000}, label="MAXED ($80k)")
