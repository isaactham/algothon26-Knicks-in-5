import numpy as np

# ============================================================
# CROSS-SECTIONAL time-series reversion (Isaac)
# Step 1: cut the market out -> rel = log(price) - log-index of all 51
# Step 2: z-score each instrument's rel series vs its own 15d history
# NOTE: this is the Test-1 signal (scored 8.68 locally). Rebuilt for
# iteration, NOT validated as better than the own-history original.
# ============================================================

currentPos = None

LOOKBACK = 15
ENTRY_Z = 1.7
EXIT_Z = 0.5
MAX_DOLLARS = 9000
POS_LIMIT_DOLLARS = 10000
MIN_TRADE_DOLLARS = 1500     # raised: this signal churns (24M vol in Test 1)

USE_HYSTERESIS = True


def resetState():
    global currentPos
    currentPos = None


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if currentPos is None or currentPos.shape[0] != nins:
        currentPos = np.zeros(nins)

    if nt < LOOKBACK + 1:
        return np.zeros(nins, dtype=int)

    today_price = prcSoFar[:, -1]          # REAL prices: sizing only

    # ---------- STEP 1: remove the market ----------
    logp = np.log(prcSoFar)
    index = logp.mean(axis=0, keepdims=True)   # the "51-instrument index", per day
    rel = logp - index                          # % above/below the index

    # ---------- STEP 2: z-score on the relative series ----------
    window = rel[:, -LOOKBACK:]                # last 15 days of RELATIVE position
    mean = window.mean(axis=1)                 # each instrument's own relative norm
    std = window.std(axis=1)
    std[std < 1e-8] = 1e-8
    z = (rel[:, -1] - mean) / std

    signal = -z

    # ---------- hysteresis (with sign-flip fix) ----------
    if USE_HYSTERESIS:
        had_position = currentPos != 0
        weak = np.abs(z) < ENTRY_Z
        dead = np.abs(z) < EXIT_Z
        flipped = had_position & (np.sign(currentPos) == np.sign(z)) & (z != 0)
        signal[~had_position & weak] = 0.0
        signal[had_position & dead] = 0.0
        signal[flipped & weak] = 0.0
        hold = had_position & weak & ~dead & ~flipped
    else:
        signal[np.abs(z) < ENTRY_Z] = 0.0
        hold = np.zeros(nins, dtype=bool)

    # ---------- sizing (REAL prices from here down) ----------
    signal = np.clip(signal, -2.0, 2.0) / 2.0
    dollar_pos = np.clip(signal * MAX_DOLLARS, -MAX_DOLLARS, MAX_DOLLARS)
    dollar_pos = dollar_pos - dollar_pos.mean()

    share_pos = (dollar_pos / today_price).astype(int)
    share_pos[hold] = currentPos[hold].astype(int)

    # ---------- churn control ----------
    delta_dollars = np.abs(share_pos - currentPos) * today_price
    small = delta_dollars < MIN_TRADE_DOLLARS
    share_pos = share_pos.astype(int)
    share_pos[small] = currentPos[small].astype(int)

    # ---------- hard limit LAST ----------
    max_shares = (POS_LIMIT_DOLLARS / today_price).astype(int)
    share_pos = np.clip(share_pos, -max_shares, max_shares)

    currentPos = share_pos
    return currentPos