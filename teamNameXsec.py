import numpy as np

# ============================================================
# STRATEGY: cross-sectional mean reversion (Cole's leg only)
#   Relative losers bought, relative winners shorted, across
#   two bands (15d + 90d). Dollar-neutral book.
#
# Fix vs blend v2: positions are sized so the +/-10k limit
# almost never binds AFTER dollar-neutralisation, so clipping
# no longer breaks neutrality.
# ============================================================

# ---------- knobs ----------
BAND_SHORT = 15
BAND_LONG = 90
XSEC_VOL_WINDOW = 10
XSEC_MIN_HISTORY = 110

POS_LIMIT_DOLLARS = 10000     # competition hard limit per instrument
XSEC_DOLLAR_TARGET = 9000     # sized BELOW the limit so demeaning
                              # rarely pushes anything past 10k
MIN_TRADE_DOLLARS = 500       # churn control

currentPos = None             # submitted share positions (state)


def resetState():
    """Call between backtest runs / sweep iterations."""
    global currentPos
    currentPos = None


def _standardize(x):
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)


def xsecReversionDollars(prcSoFar):
    """Cross-sectional reversion signal -> dollar-neutral dollar targets."""
    nins, nt = prcSoFar.shape
    if nt < XSEC_MIN_HISTORY:
        return np.zeros(nins)

    logrets = np.diff(np.log(prcSoFar), axis=1)
    trail_s = logrets[:, -BAND_SHORT:].sum(axis=1)
    trail_l = logrets[:, -BAND_LONG:].sum(axis=1)

    # relative losers bought, relative winners shorted, in each band
    sig = -(_standardize(trail_s) + _standardize(trail_l)) / 2.0

    # dampen the wildest instruments so no single name dominates daily swings
    vol = logrets[:, -XSEC_VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5   # in [-1, 1]
    dollars = strength * XSEC_DOLLAR_TARGET

    # dollar-neutral book: longs and shorts cancel
    dollars = dollars - dollars.mean()
    return dollars


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if currentPos is None or currentPos.shape[0] != nins:
        currentPos = np.zeros(nins, dtype=int)

    today = prcSoFar[:, -1]

    dollars = xsecReversionDollars(prcSoFar)
    target = (dollars / today).astype(int)

    # churn control: skip trades that move less than MIN_TRADE_DOLLARS
    delta_dollars = np.abs(target - currentPos) * today
    small = delta_dollars < MIN_TRADE_DOLLARS
    target[small] = currentPos[small]

    # safety clip at the hard limit. Because XSEC_DOLLAR_TARGET (9000)
    # sits below POS_LIMIT_DOLLARS (10000), this almost never binds,
    # so neutrality survives. It exists only as a guarantee against
    # edge cases (price gaps between decision and clip).
    max_shares = (POS_LIMIT_DOLLARS / today).astype(int)
    target = np.clip(target, -max_shares, max_shares)

    currentPos = target
    return currentPos