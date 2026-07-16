import numpy as np

# ============================================================
# MULTI-HORIZON CROSS-SECTIONAL REVERSION
#
# Idea (per teammate's writeup): don't ask "did X fall recently?"
# (that's really just "did the market fall?", since one common factor
# drives ~20% of every instrument's move). Instead ask "did X fall
# MORE than the other 50 did?" -> strip out the daily cross-sectional
# mean (the market factor) and z-score each instrument's leftover
# ("relative") move against its own recent history. Go long relative
# losers, short relative winners, in equal dollars -> market-neutral
# book, ~50 independent bets instead of one big correlated one.
#
# Two horizons (15d, 90d) are blended because the writeup says the
# effect shows up at both timescales in their data.
# ============================================================

LOOKBACK_FAST = 15
LOOKBACK_SLOW = 90

WEIGHT_FAST = 0.5
WEIGHT_SLOW = 0.5

ENTRY_Z = 1.0
EXIT_Z = 0.5

MAX_DOLLARS = 9000
POS_LIMIT_DOLLARS = 10000
MIN_TRADE_DOLLARS = 1500

MIN_HISTORY = LOOKBACK_SLOW + 1

currentPos = None


def resetState():
    global currentPos
    currentPos = None


def _zscore(rel, lookback):
    """Z-score of today's relative value vs its own trailing `lookback`-day history."""
    window = rel[:, -lookback:]
    mean = window.mean(axis=1)
    std = window.std(axis=1)
    std = np.where(std < 1e-8, 1e-8, std)
    return (rel[:, -1] - mean) / std


def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    if currentPos is None or currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst, dtype=int)

    if nt < MIN_HISTORY:
        return currentPos

    today_price = prcSoFar[:, -1]

    # ---- strip out the market factor ----
    logp = np.log(prcSoFar)
    index = logp.mean(axis=0, keepdims=True)   # daily cross-sectional mean = "market"
    rel = logp - index                          # each instrument's move net of the market

    # ---- z-score at both horizons, blend ----
    z_fast = _zscore(rel, LOOKBACK_FAST)
    z_slow = _zscore(rel, LOOKBACK_SLOW)
    z = WEIGHT_FAST * z_fast + WEIGHT_SLOW * z_slow

    signal = -z   # fade the relative move: relative losers -> long, relative winners -> short

    # ---- hysteresis: avoid flipping on noise near the thresholds ----
    had_position = currentPos != 0
    weak = np.abs(z) < ENTRY_Z
    dead = np.abs(z) < EXIT_Z
    flipped = had_position & (np.sign(currentPos) == np.sign(z)) & (z != 0)
    signal[~had_position & weak] = 0.0
    signal[had_position & dead] = 0.0
    signal[flipped & weak] = 0.0
    hold = had_position & weak & ~dead & ~flipped

    # ---- sizing ----
    signal = np.clip(signal, -2.0, 2.0) / 2.0
    dollar_pos = np.clip(signal * MAX_DOLLARS, -MAX_DOLLARS, MAX_DOLLARS)
    dollar_pos = dollar_pos - dollar_pos.mean()   # enforce market (dollar) neutrality

    share_pos = (dollar_pos / today_price).astype(int)
    share_pos[hold] = currentPos[hold].astype(int)

    # ---- churn control ----
    delta_dollars = np.abs(share_pos - currentPos) * today_price
    small = delta_dollars < MIN_TRADE_DOLLARS
    share_pos = share_pos.astype(int)
    share_pos[small] = currentPos[small].astype(int)

    # ---- hard position limit ----
    max_shares = (POS_LIMIT_DOLLARS / today_price).astype(int)
    share_pos = np.clip(share_pos, -max_shares, max_shares)

    currentPos = share_pos
    return currentPos