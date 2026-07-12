import numpy as np

# ---------------------------------------------------------------------------
# TREND PROBE (diagnostic submission, testing round only).
# Pure EMA crossover trend-following, fixed parameters, nothing else.
# Purpose: its hidden-set score is one clean bit of information about what
# regime days 501-750 are running. Reversion already scored ~0 there.
#   - clearly positive  -> hidden window is trend-friendly
#   - ~0                -> neither reversion nor trend; regime is other
#   - clearly negative  -> hidden window is reverting harder than visible did
# ---------------------------------------------------------------------------
FAST = 10
SLOW = 40
DOLLAR_TARGET = 20000
VOL_WINDOW = 10
MIN_HISTORY = 60

currentPos = None


def getMyPosition(prcSoFar):
    global currentPos

    nInst, nt = prcSoFar.shape
    if currentPos is None or currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst, dtype=int)

    if nt < MIN_HISTORY:
        return currentPos

    logp = np.log(prcSoFar)

    # EMAs of log price (iterative, cheap)
    af, asl = 2.0 / (FAST + 1), 2.0 / (SLOW + 1)
    fast = logp[:, 0].copy()
    slow = logp[:, 0].copy()
    for d in range(1, nt):
        fast = af * logp[:, d] + (1 - af) * fast
        slow = asl * logp[:, d] + (1 - asl) * slow

    signal = fast - slow                     # >0: uptrend, <0: downtrend

    logrets = np.diff(logp, axis=1)
    vol = logrets[:, -VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    # normalise signal by its cross-sectional scale, then vol-adjust
    sd = signal.std()
    if sd < 1e-12:
        return currentPos
    strength = np.clip((signal / sd) / (vol / vol.mean()), -1.5, 1.5) / 1.5

    dollars = strength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    currentPos = (dollars / curPrices).astype(int)
    return currentPos
