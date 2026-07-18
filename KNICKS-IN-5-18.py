import numpy as np

# ============================================================
# GENERAL ROUND OPENER: cross-sectional reversion, 30/70 banded
# - Signal: relative trailing returns at 15d (w=0.3) and 90d (w=0.7),
#   weighted toward the long band because it held its magnitude across
#   the day-500 regime boundary while the short band weakened.
# - Sizing: FULL-HISTORY vol per instrument (vols are generator
#   constants, 0.978 correlated across chunks; a full-sample estimate
#   beats a noisy 10-day rolling one).
# - Book netted to zero dollars: factor-neutral, the hedge that held
#   out-of-sample (hidden StdDev 1192 vs local 1248).
# Validated at DOLLAR_TARGET = 120000 (core_sizing_sweep, pre-committed
# rule: largest size with Sharpe within 15% of the 20k baseline on both
# chunks): OLD chunk score ~194 (Sharpe 2.32), NEW chunk score ~132
# (Sharpe 1.75). Same code as the live $20k opener; only the size changed.
# ============================================================

W_SHORT = 0.3
W_LONG = 0.7
BAND_SHORT = 15
BAND_LONG = 90
DOLLAR_TARGET = 120000
MIN_HISTORY = 110

currentPos = None


def _std(x):
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

    sig = (W_SHORT * -_std(logrets[:, -BAND_SHORT:].sum(axis=1))
           + W_LONG * -_std(logrets[:, -BAND_LONG:].sum(axis=1)))

    # full-history vol: instruments' vols are permanent constants here
    vol = logrets.std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
    dollars = strength * DOLLAR_TARGET
    dollars = dollars - dollars.mean()          # factor-neutral book

    curPrices = prcSoFar[:, -1]
    currentPos = (dollars / curPrices).astype(int)
    return currentPos
