import numpy as np

# ============================================================
# VAR(1) CROSS-EFFECTS MODEL
# - The generator has lag-1 cross-effects between instruments: each
#   instrument's factor-residual return is predictable from ALL 51
#   lagged residuals, not just its own (the old core's reversion
#   signal is the diagonal shadow of this matrix).
# - Fit: ridge regression B (51x51) on residuals (returns minus
#   cross-sectional mean), refit every day on all history to date.
#   Leak-free by construction. lam = 4*nInst/T (trace-scaled): shrinkage
#   decays with training size, so the live fit (~950 days) shrinks less
#   than any local test. Official-engine validation of this exact file:
#   OLD chunk score 515.2, NEW chunk 503.8, all 10 walkforward windows
#   positive (min 258).
# - Position: predicted residual, scaled by mean |pred| to gross
#   40k/instrument, capped, dollar-netted, re-capped.
# - Selection: C=4 from pre-registered C in {4,6,8} (dominant on both
#   chunks), confirmed as a plateau vs C in {3,5}. Change from the
#   prior live model is this shrinkage schedule only.
# ============================================================

LAM_C = 4.0
GROSS_SCALE = 40000.0
MIN_HISTORY = 110

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
    res = logrets - logrets.mean(axis=0)        # factor-residual returns

    # ridge-fit lag-1 cross-effects matrix on all available history
    X = res[:, :-1].T                            # (T-1, nInst) lagged
    Y = res[:, 1:].T                             # (T-1, nInst) next
    G = X.T @ X
    lam = LAM_C * nInst / X.shape[0]
    G += lam * np.eye(nInst) * np.trace(G) / nInst
    B = np.linalg.solve(G, X.T @ Y)

    pred = res[:, -1] @ B                        # tomorrow's residuals

    s = pred / (np.abs(pred).mean() + 1e-12)
    dollars = np.clip(s * GROSS_SCALE, -cap, cap)
    dollars = dollars - dollars.mean()           # dollar-neutral book
    dollars = np.clip(dollars, -cap, cap)

    currentPos = (dollars / prcSoFar[:, -1]).astype(int)
    return currentPos
