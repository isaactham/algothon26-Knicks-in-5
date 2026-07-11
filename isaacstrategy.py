import numpy as np


nInst = 51
currentPos = np.zeros(nInst)

# --- knobs ---
LOOKBACK = 15
ENTRY_Z = 1.0
EXIT_Z = 0.5
MAX_DOLLARS = 9000

# --- experiment toggles (2x2) ---
USE_VOL_SCALE = False     # Run C: OFF
USE_HYSTERESIS = True   # Run C: ON

def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if nt < LOOKBACK + 1:
        return np.zeros(nins, dtype=int)

    # ---------- STEP 1: signal ----------
    window = prcSoFar[:, -LOOKBACK:]
    mean = window.mean(axis=1)
    std = window.std(axis=1)
    std[std < 1e-8] = 1e-8

    today = prcSoFar[:, -1]
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
        # original behaviour: one threshold, no holding zone
        signal[np.abs(z) < ENTRY_Z] = 0.0
        hold = np.zeros(nins, dtype=bool)   # nothing gets held

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

    dollar_pos = signal * MAX_DOLLARS * risk_scale
    dollar_pos = np.clip(dollar_pos, -MAX_DOLLARS, MAX_DOLLARS)

    share_pos = (dollar_pos / today).astype(int)
    share_pos[hold] = currentPos[hold].astype(int)

    currentPos = share_pos
    return currentPos