import numpy as np

nInst=51
currentPos = np.zeros(nInst)
def getMyPosition (prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape
    if (nt < 21):
        return np.zeros(nins)
    
    ret5 = np.log(prcSoFar[:, -1] / prcSoFar[:, -6])
    ret10 = np.log(prcSoFar[:, -1] / prcSoFar[:, -11])
    ret20 = np.log(prcSoFar[:, -1] / prcSoFar[:, -21])

    lastRet = 0.5 * ret5 + 0.3 * ret10 + 0.2 * ret20

    returns = np.diff(np.log(prcSoFar[:, -21:]), axis=1)
    vol = np.std(returns, axis=1)
    lastRet = lastRet / (vol + 1e-8)

    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 4300 * lastRet / prcSoFar[:,-1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos


