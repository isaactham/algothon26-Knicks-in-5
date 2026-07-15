import numpy as np

# ============================================================
# BLEND: cross-sectional mean reversion (Isaac) + trend (Cole)
# final position = W_REVERSION * reversion + W_TREND * trend
#
# Reversion signal = z-score of log(price) - log(index),
# i.e. "how stretched is this instrument vs its peers, in %"
# ============================================================

W_REVERSION = 0.3
W_TREND = 0.7

# ---------- shared state ----------
currentPos = None            # actual submitted positions (shares)

# ---------- reversion knobs ----------
LOOKBACK = 15
ENTRY_Z = 1.0
EXIT_Z = 0.5
REV_MAX_DOLLARS = 9000
rev_pos = None               # reversion leg's own share positions (for hysteresis)

# ---------- trend knobs ----------
FAST = 10
SLOW = 40
TREND_MAX_DOLLARS = 9000
VOL_WINDOW = 20
MIN_HISTORY = 60

# ---------- churn / limits ----------
MIN_TRADE_DOLLARS = 500      # ignore position changes smaller than this
POS_LIMIT_DOLLARS = 10000    # hard per-instrument limit


def resetState():
    """Call between backtest runs so state doesn't leak across sweeps."""
    global currentPos, rev_pos
    currentPos = None
    rev_pos = None


# ============================================================
# LEG 1: cross-sectional mean reversion (returns DOLLAR positions)
# ============================================================
def reversionDollars(prcSoFar):
    global rev_pos
    nins, nt = prcSoFar.shape

    if rev_pos is None:
        rev_pos = np.zeros(nins)

    if nt < LOOKBACK + 1:
        return np.zeros(nins)

    today_price = prcSoFar[:, -1]          # REAL prices, used for sizing only

    # ---- signal lives in log-relative space ----
    logp = np.log(prcSoFar)
    index = logp.mean(axis=0, keepdims=True)   # log-space "market" each day
    rel = logp - index                          # ~ % above/below the index

    today_rel = rel[:, -1]
    window = rel[:, -LOOKBACK:]
    mean = window.mean(axis=1)
    std = window.std(axis=1)
    std[std < 1e-8] = 1e-8
    z = (today_rel - mean) / std

    signal = -z

    # ---- hysteresis (tracked on this leg's own positions) ----
    had_position = rev_pos != 0
    weak = np.abs(z) < ENTRY_Z
    dead = np.abs(z) < EXIT_Z

    # position on the wrong side (sign flip skipped the dead band) -> close, don't hold
    flipped = had_position & (np.sign(rev_pos) == np.sign(z)) & (z != 0)

    signal[~had_position & weak] = 0.0
    signal[had_position & dead] = 0.0
    signal[flipped & weak] = 0.0
    hold = had_position & weak & ~dead & ~flipped

    signal = np.clip(signal, -2.0, 2.0) / 2.0
    dollars = signal * REV_MAX_DOLLARS
    dollars = np.clip(dollars, -REV_MAX_DOLLARS, REV_MAX_DOLLARS)

    # held instruments keep their previous dollar value (REAL prices here!)
    dollars[hold] = rev_pos[hold] * today_price[hold]

    rev_pos = dollars / today_price        # remember as shares for next call
    return dollars


# ============================================================
# LEG 2: trend following / EMA crossover (returns DOLLAR positions)
# ============================================================
def trendDollars(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < MIN_HISTORY:
        return np.zeros(nins)

    logp = np.log(prcSoFar)

    af, asl = 2.0 / (FAST + 1), 2.0 / (SLOW + 1)
    fast = logp[:, 0].copy()
    slow = logp[:, 0].copy()
    for d in range(1, nt):
        fast = af * logp[:, d] + (1 - af) * fast
        slow = asl * logp[:, d] + (1 - asl) * slow

    signal = fast - slow               # >0 uptrend, <0 downtrend

    logrets = np.diff(logp, axis=1)
    vol = logrets[:, -VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    sd = signal.std()
    if sd < 1e-12:
        return np.zeros(nins)

    strength = np.clip((signal / sd) / (vol / vol.mean()), -1.5, 1.5) / 1.5
    dollars = strength * TREND_MAX_DOLLARS
    return dollars


# ============================================================
# BLENDER
# ============================================================
def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if currentPos is None:
        currentPos = np.zeros(nins, dtype=int)

    today = prcSoFar[:, -1]

    blended_dollars = (W_REVERSION * reversionDollars(prcSoFar)
                       + W_TREND * trendDollars(prcSoFar))

    target = (blended_dollars / today).astype(int)

    # ---- churn control: only trade meaningful changes ----
    delta_dollars = np.abs(target - currentPos) * today
    small = delta_dollars < MIN_TRADE_DOLLARS
    target[small] = currentPos[small]

    # ---- hard limit LAST, so held positions can't drift over $10k ----
    max_shares = (POS_LIMIT_DOLLARS / today).astype(int)
    target = np.clip(target, -max_shares, max_shares)

    currentPos = target
    return currentPos