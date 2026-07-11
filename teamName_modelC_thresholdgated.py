import numpy as np

nInst = 0
currentPos = np.zeros(nInst)

DOLLAR_TARGET = 2000
VOL_WINDOW = 10
MIN_HISTORY = 12
TRADE_THRESHOLD = 0.3   # only trade when |signalStrength| exceeds this


def getMyPosition(prcSoFar):
    global currentPos, nInst

    nInst, nt = prcSoFar.shape
    if currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst)

    if nt < MIN_HISTORY:
        return currentPos

    logrets = np.diff(np.log(prcSoFar), axis=1)

    lastRet = logrets[:, -1]
    window = logrets[:, -VOL_WINDOW:]
    vol = np.std(window, axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    signal = -lastRet
    signalStrength = np.clip(signal / vol, -1, 1)

    dollarTarget = signalStrength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    proposedPos = (dollarTarget / curPrices).astype(int)

    # only update positions where the signal is strong enough to be worth trading,
    # otherwise hold whatever position we already had
    tradeMask = np.abs(signalStrength) > TRADE_THRESHOLD
    newPos = np.where(tradeMask, proposedPos, currentPos)

    currentPos = newPos
    return currentPos
