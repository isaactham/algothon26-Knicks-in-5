import numpy as np

nInst = 0
currentPos = np.zeros(nInst)

DOLLAR_TARGET = 2000
VOL_WINDOW = 10
MIN_HISTORY = 15


def getMyPosition(prcSoFar):
    global currentPos, nInst

    nInst, nt = prcSoFar.shape
    if currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst)

    if nt < MIN_HISTORY:
        return currentPos

    logrets = np.diff(np.log(prcSoFar), axis=1)  # nInst x (nt-1)
    marketFactor = logrets.mean(axis=0)           # simple average across instruments each day

    # closed-form per-instrument beta against the market factor (vectorised, no loop)
    mfc = marketFactor - marketFactor.mean()
    denom = mfc.dot(mfc)
    beta = (logrets @ mfc) / denom
    alpha = logrets.mean(axis=1) - beta * marketFactor.mean()

    fitted = alpha[:, None] + beta[:, None] * marketFactor[None, :]
    residuals = logrets - fitted                  # idiosyncratic part, factor removed

    signal = -residuals[:, -1]                    # mean-reversion tilt on the residual only

    window = residuals[:, -VOL_WINDOW:]
    vol = np.std(window, axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    signalStrength = np.clip(signal / vol, -1, 1)
    dollarTarget = signalStrength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    newPos = (dollarTarget / curPrices).astype(int)

    currentPos = newPos
    return currentPos
