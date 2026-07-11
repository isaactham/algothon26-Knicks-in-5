import numpy as np

# ---------------------------------------------------------------------------
# CONSERVATIVE version: multi-lag + 20-day reversion, moderate sizing ($20k).
# Lower risk, StdDev lands near the competitive band. Use as an information-
# gathering submission before committing to larger size.
# ---------------------------------------------------------------------------
SHORT_LAGS = (1, 2, 5)
LONG_LOOKBACK = 20
LONG_WEIGHT = 0.5
VOL_WINDOW = 10
DOLLAR_TARGET = 20000
MIN_HISTORY = 45

currentPos = None


def _reset():
    """Clear internal state. Called by the walk-forward harness between windows."""
    global currentPos
    currentPos = None


def getMyPosition(prcSoFar):
    global currentPos

    nInst, nt = prcSoFar.shape
    if currentPos is None or currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst)

    if nt < MIN_HISTORY:
        return currentPos

    logrets = np.diff(np.log(prcSoFar), axis=1)

    short_sig = -np.mean([logrets[:, -lag] for lag in SHORT_LAGS], axis=0)
    long_sig = -logrets[:, -LONG_LOOKBACK:].sum(axis=1)

    short_n = short_sig / (np.std(short_sig) + 1e-9)
    long_n = long_sig / (np.std(long_sig) + 1e-9)
    signal = (1 - LONG_WEIGHT) * short_n + LONG_WEIGHT * long_n

    window = logrets[:, -VOL_WINDOW:]
    vol = np.std(window, axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    signalStrength = np.clip(signal / (vol / np.mean(vol)), -1, 1)

    dollarTarget = signalStrength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    newPos = (dollarTarget / curPrices).astype(int)

    currentPos = newPos
    return currentPos
