import numpy as np

# ============================================================
# VAR(1) — ONE CHANGE ONLY: correlation-space fitting
#
# CORR_SPACE = False  ->  exact current champion, untouched.
# CORR_SPACE = True   ->  before learning the 51x51 web, each
#                          instrument's daily move is divided by
#                          its own typical size, so loud and quiet
#                          instruments get an equal voice. Converted
#                          back to real dollars afterward.
#
# Nothing else in the model changes.
# ============================================================

LAM = 1.0
GROSS_SCALE = 40000.0
MIN_HISTORY = 110
CORR_SPACE = 10        # <-- the one switch. Flip to False to get the old model back.

CAP = np.full(51, 10_000.0)
CAP[0] = 100_000.0

currentPos = None


def getMyPosition(prcSoFar):
    global currentPos

    nInst, nt = prcSoFar.shape
    if currentPos is None or currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst, dtype=int)

    if nt < MIN_HISTORY:
        return currentPos

    cap = CAP[:nInst]

    logrets = np.diff(np.log(prcSoFar), axis=1)
    res = logrets - logrets.mean(axis=0)        # moves vs. the pack

    # ---- the one change lives here ----
    if CORR_SPACE:
        sigma = res.std(axis=1, keepdims=True) + 1e-12   # each instrument's typical size
        work = res / sigma                                 # equal voice for everyone
    else:
        sigma = None
        work = res
    # ------------------------------------

    X = work[:, :-1].T
    Y = work[:, 1:].T
    G = X.T @ X
    G += LAM * np.eye(nInst) * np.trace(G) / nInst
    B = np.linalg.solve(G, X.T @ Y)

    x_today = work[:, -1]
    pred = x_today @ B
    if CORR_SPACE:
        pred = pred * sigma.ravel()          # convert back to real dollar-scale

    s = pred / (np.abs(pred).mean() + 1e-12)
    dollars = np.clip(s * GROSS_SCALE, -cap, cap)
    dollars = dollars - dollars.mean()
    dollars = np.clip(dollars, -cap, cap)

    currentPos = (dollars / prcSoFar[:, -1]).astype(int)
    return currentPos