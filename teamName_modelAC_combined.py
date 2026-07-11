import numpy as np

nInst = 0
currentPos = np.zeros(nInst)

DOLLAR_TARGET = 2000
VOL_WINDOW = 10
MIN_HISTORY = 15
TRADE_THRESHOLD = 0.3


def getMyPosition(prcSoFar):
    global currentPos, nInst

    nInst, nt = prcSoFar.shape
    if currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst)

    if nt < MIN_HISTORY:
        return currentPos

    logrets = np.diff(np.log(prcSoFar), axis=1)

    ret1 = logrets[:, -1]
    ret2 = logrets[:, -2]
    ret5 = logrets[:, -5]
    combinedRet = (ret1 + ret2 + ret5) / 3.0
    signal = -combinedRet

    window = logrets[:, -VOL_WINDOW:]
    vol = np.std(window, axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    signalStrength = np.clip(signal / vol, -1, 1)
    dollarTarget = signalStrength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    proposedPos = (dollarTarget / curPrices).astype(int)

    tradeMask = np.abs(signalStrength) > TRADE_THRESHOLD
    newPos = np.where(tradeMask, proposedPos, currentPos)

    currentPos = newPos
    return currentPos
