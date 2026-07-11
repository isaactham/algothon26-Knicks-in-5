import numpy as np


nInst = 51
currentPos = None

# --- knobs ---
LOOKBACK = 15
ENTRY_Z = 1.0
EXIT_Z = 0.5
MAX_DOLLARS = 9000

# --- experiment toggles ---
USE_VOL_SCALE = False
USE_HYSTERESIS = True

# --- regime awareness knobs & state (currently OFF) ---
USE_REGIME = False
REGIME_WINDOW = 20       # judge the regime on the last 20 days of our own P&L
REGIME_SCALE = 0.75      # bet at reduced size during bad regimes
pnl_history = []         # running record of our daily P&L
prevPrices = None        # yesterday's prices, to compute yesterday's P&L


def getMyPosition(prcSoFar):
    global currentPos, pnl_history, prevPrices
    nins, nt = prcSoFar.shape

    # lazy init: adapts to however many instruments the eval provides
    if currentPos is None:
        currentPos = np.zeros(nins)

    if nt < LOOKBACK + 1:
        return np.zeros(nins, dtype=int)

    today = prcSoFar[:, -1]

    # record what yesterday's positions just earned
    if USE_REGIME and prevPrices is not None:
        daily_pnl = np.sum(currentPos * (today - prevPrices))
        pnl_history.append(daily_pnl)
    prevPrices = today.copy()

    # decide the regime scale from our trailing P&L
    if USE_REGIME and len(pnl_history) >= REGIME_WINDOW:
        trailing = sum(pnl_history[-REGIME_WINDOW:])
        regime_scale = REGIME_SCALE if trailing < 0 else 1.0
    else:
        regime_scale = 1.0

    # ---------- STEP 1: signal ----------
    window = prcSoFar[:, -LOOKBACK:]
    mean = window.mean(axis=1)
    std = window.std(axis=1)
    std[std < 1e-8] = 1e-8

    z = (today - mean) / std

    # ---------- STEP 2: direction ----------
    signal = -z

    if USE_HYSTERESIS:
        had_position = currentPos != 0
        weak = np.abs(z) < ENTRY_Z
        dead = np.abs(z) < EXIT_Z
        signal[~had_position & weak] = 0.0
        signal[had_position & dead] = 0.0
        hold = had_position & weak & ~dead
    else:
        signal[np.abs(z) < ENTRY_Z] = 0.0
        hold = np.zeros(nins, dtype=bool)

    # ---------- STEP 3: sizing ----------
    signal = np.clip(signal, -2.0, 2.0) / 2.0

    if USE_VOL_SCALE:
        rets = np.diff(np.log(window), axis=1)
        vol = rets.std(axis=1)
        vol[vol < 1e-8] = 1e-8
        risk_scale = np.median(vol) / vol
        risk_scale = np.clip(risk_scale, 0.25, 2.0)
    else:
        risk_scale = 1.0

    dollar_pos = signal * MAX_DOLLARS * risk_scale * regime_scale
    dollar_pos = np.clip(dollar_pos, -MAX_DOLLARS, MAX_DOLLARS)

    share_pos = (dollar_pos / today).astype(int)
    share_pos[hold] = currentPos[hold].astype(int)

    currentPos = share_pos
    return currentPos