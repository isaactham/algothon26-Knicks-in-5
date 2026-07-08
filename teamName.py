import numpy as np

nInst=51
currentPos = np.zeros(nInst)
def getMyPosition (prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape

    if (nt < 20):
        return np.zeros(nins)

    #todays prices are the last column of the price matrix
    today = prcSoFar[:,-1]

    #takes the last 21 days of prices and calculates the returns and volatility for each instrument
    returns = np.diff(np.log(prcSoFar[:, -20:]), axis=1)
    volatility = np.std(returns, axis=1) + 1e-10  # add a small value to avoid division by zero

    
    lastRet = np.log(today / prcSoFar[:,-10])

    lNorm = np.sqrt(lastRet.dot(lastRet))

    lastRet /= lNorm

    rpos = np.array([int(x) for x in 5000 * lastRet / (today * volatility)])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

    


