import numpy as np

nInst = 0
currentPos = np.zeros(nInst)

DOLLAR_TARGET = 2000     # base dollar exposure per instrument before signal scaling
VOL_WINDOW = 10          # days used to estimate recent volatility
MIN_HISTORY = 12


def getMyPosition(prcSoFar):
    global currentPos, nInst

    nInst, nt = prcSoFar.shape
    if currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst)

    if nt < MIN_HISTORY:
        return currentPos

    logrets = np.diff(np.log(prcSoFar), axis=1)  # nInst x (nt-1)

    lastRet = logrets[:, -1]                                  # yesterday's return
    window = logrets[:, -VOL_WINDOW:]
    vol = np.std(window, axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)                     # avoid divide-by-zero

    signal = -lastRet                                         # mean-reversion tilt
    signalStrength = np.clip(signal / vol, -1, 1)             # -1..1, vol-normalised confidence

    dollarTarget = signalStrength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    newPos = (dollarTarget / curPrices).astype(int)

    currentPos = newPos
    return currentPos
