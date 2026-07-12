import numpy as np

# ---------------------------------------------------------------------------
# CROSS-SECTIONAL REVERSION PROBE (diagnostic submission, testing round).
# Hypothesis: residual (factor-neutral) reversion is a stationary property of
# the generator and transfers across regime changes, unlike factor behaviour.
# Bets the two bands significant in visible data: ~15d and ~90d relative
# reversion, demeaned so the book carries ~zero factor exposure.
# Reading: hedged construction -> low StdDev -> small SE -> readable result.
#   - clearly positive -> XSEC reversion transfers; core of General Round model
#   - ~0               -> residual layer is also regime-dependent; rethink
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

    # relative losers get bought, relative winners get shorted, in each band
    sig = -(_xsec_standardize(trail_s) + _xsec_standardize(trail_l)) / 2.0

    vol = logrets[:, -VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    strength = np.clip((sig / (vol / vol.mean())), -1.5, 1.5) / 1.5

    dollars = strength * DOLLAR_TARGET
    # enforce dollar-neutrality so the book carries ~no net factor exposure
    dollars = dollars - dollars.mean()

    curPrices = prcSoFar[:, -1]
    currentPos = (dollars / curPrices).astype(int)
    return currentPos
