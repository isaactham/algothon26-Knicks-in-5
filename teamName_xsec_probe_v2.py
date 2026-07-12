import numpy as np

# ---------------------------------------------------------------------------
# CROSS-SECTIONAL REVERSION PROBE v2: beta-neutral.
# Same hypothesis and signal as v1 (relative reversion at ~15d and ~90d),
# but the dollar book is projected orthogonal to BOTH the ones-vector
# (dollar-neutral) and the estimated beta vector (beta-neutral), removing the
# systematic high-beta tilt that leaks factor risk into a dollar-neutral book.
# Purpose of the change: lower StdDev at equal mean -> smaller SE -> a
# statistically readable hidden result.
# ---------------------------------------------------------------------------
BAND_SHORT = 15
BAND_LONG = 90
DOLLAR_TARGET = 20000
VOL_WINDOW = 10
MIN_HISTORY = 110

currentPos = None


def _xsec_standardize(x):
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)


def getMyPosition(prcSoFar):
    global currentPos

    nInst, nt = prcSoFar.shape
    if currentPos is None or currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst, dtype=int)

    if nt < MIN_HISTORY:
        return currentPos

    logrets = np.diff(np.log(prcSoFar), axis=1)

    trail_s = logrets[:, -BAND_SHORT:].sum(axis=1)
    trail_l = logrets[:, -BAND_LONG:].sum(axis=1)
    sig = -(_xsec_standardize(trail_s) + _xsec_standardize(trail_l)) / 2.0

    vol = logrets[:, -VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
    dollars = strength * DOLLAR_TARGET

    # estimate betas against the equal-weight factor over full history
    f = logrets.mean(axis=0)
    fc = f - f.mean()
    beta = (logrets @ fc) / (fc @ fc)

    # project the dollar book orthogonal to ones AND beta (Gram-Schmidt)
    ones = np.ones(nInst)
    dollars = dollars - (dollars @ ones) / (ones @ ones) * ones
    b_perp = beta - (beta @ ones) / (ones @ ones) * ones
    if b_perp @ b_perp > 1e-12:
        dollars = dollars - (dollars @ b_perp) / (b_perp @ b_perp) * b_perp

    curPrices = prcSoFar[:, -1]
    currentPos = (dollars / curPrices).astype(int)
    return currentPos
