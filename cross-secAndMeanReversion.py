import numpy as np

# ============================================================
# BLEND v2: two flavours of mean reversion
#   Leg A (Isaac): time-series  - stretched vs own 15d history
#   Leg B (Cole):  cross-sectional - stretched vs peers (15d + 90d bands),
#                  dollar-neutral (no net market exposure)
# final dollars = W_TS * legA + W_XSEC * legB
# ============================================================

W_TS = 0.5
W_XSEC = 0.5

# ---------- shared ----------
currentPos = None                 # actual submitted share positions
POS_LIMIT_DOLLARS = 10000
MIN_TRADE_DOLLARS = 500

# ---------- Leg A knobs (Isaac) ----------
LOOKBACK = 15
ENTRY_Z = 1.0
EXIT_Z = 0.5
TS_MAX_DOLLARS = 9000
USE_VOL_SCALE = False
USE_HYSTERESIS = True
ts_pos_shares = None              # this leg's own positions (hysteresis state)

# ---------- Leg B knobs (Cole) ----------
BAND_SHORT = 15
BAND_LONG = 90
XSEC_DOLLAR_TARGET = 20000        # NOTE: >10k solo; safe only via blend + final clip
XSEC_VOL_WINDOW = 10
XSEC_MIN_HISTORY = 110


def resetState():
    """Call between backtest runs / sweep iterations."""
    global currentPos, ts_pos_shares
    currentPos = None
    ts_pos_shares = None


# ============================================================
# LEG A: time-series mean reversion  (returns DOLLAR positions)
# ============================================================
def tsReversionDollars(prcSoFar):
    global ts_pos_shares
    nins, nt = prcSoFar.shape

    if ts_pos_shares is None:
        ts_pos_shares = np.zeros(nins)

    if nt < LOOKBACK + 1:
        return np.zeros(nins)

    today = prcSoFar[:, -1]
    window = prcSoFar[:, -LOOKBACK:]
    mean = window.mean(axis=1)
    std = window.std(axis=1)
    std[std < 1e-8] = 1e-8
    z = (today - mean) / std

    signal = -z

    if USE_HYSTERESIS:
        had_position = ts_pos_shares != 0
        weak = np.abs(z) < ENTRY_Z
        dead = np.abs(z) < EXIT_Z
        # wrong-side check: position sign should be -sign(z)
        flipped = had_position & (np.sign(ts_pos_shares) == np.sign(z)) & (z != 0)
        signal[~had_position & weak] = 0.0
        signal[had_position & dead] = 0.0
        signal[flipped & weak] = 0.0
        hold = had_position & weak & ~dead & ~flipped
    else:
        signal[np.abs(z) < ENTRY_Z] = 0.0
        hold = np.zeros(nins, dtype=bool)

    signal = np.clip(signal, -2.0, 2.0) / 2.0

    if USE_VOL_SCALE:
        rets = np.diff(np.log(window), axis=1)
        vol = rets.std(axis=1)
        vol[vol < 1e-8] = 1e-8
        risk_scale = np.clip(np.median(vol) / vol, 0.25, 2.0)
    else:
        risk_scale = 1.0

    dollars = np.clip(signal * TS_MAX_DOLLARS * risk_scale,
                      -TS_MAX_DOLLARS, TS_MAX_DOLLARS)

    # held instruments keep previous dollar value (real prices!)
    dollars[hold] = ts_pos_shares[hold] * today[hold]

    ts_pos_shares = dollars / today
    return dollars


# ============================================================
# LEG B: cross-sectional reversion, dollar-neutral (returns DOLLARS)
# ============================================================
def _xsec_standardize(x):
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)


def xsecReversionDollars(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < XSEC_MIN_HISTORY:
        return np.zeros(nins)

    logrets = np.diff(np.log(prcSoFar), axis=1)
    trail_s = logrets[:, -BAND_SHORT:].sum(axis=1)
    trail_l = logrets[:, -BAND_LONG:].sum(axis=1)

    # relative losers bought, relative winners shorted, in each band
    sig = -(_xsec_standardize(trail_s) + _xsec_standardize(trail_l)) / 2.0

    vol = logrets[:, -XSEC_VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)

    strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
    dollars = strength * XSEC_DOLLAR_TARGET
    dollars = dollars - dollars.mean()      # dollar-neutral book
    return dollars


# ============================================================
# BLENDER
# ============================================================
def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if currentPos is None or currentPos.shape[0] != nins:
        currentPos = np.zeros(nins, dtype=int)

    today = prcSoFar[:, -1]

    blended = (W_TS * tsReversionDollars(prcSoFar)
               + W_XSEC * xsecReversionDollars(prcSoFar))

    target = (blended / today).astype(int)

    # churn control: skip trades that move less than MIN_TRADE_DOLLARS
    delta_dollars = np.abs(target - currentPos) * today
    small = delta_dollars < MIN_TRADE_DOLLARS
    target[small] = currentPos[small]

    # hard per-instrument limit LAST (catches drift and Cole's 20k target)
    max_shares = (POS_LIMIT_DOLLARS / today).astype(int)
    target = np.clip(target, -max_shares, max_shares)

    currentPos = target
    return currentPos