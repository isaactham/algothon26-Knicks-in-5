import numpy as np

nInst=51
currentPos = np.zeros(nInst)

# knobs
LOOKBACK = 20 #how many days define "recent average"
ENTRY_Z = 1.0 #only trade when |z| exceeds this
EXIT_Z = 0.3 #exit threshold, lower than entry
MAX_DOLLARS = 9000 #stay under the $10k limit for the competition



def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    # Not enough history yet? Do nothing.
    if nt < LOOKBACK + 1:
        return np.zeros(nins, dtype=int)

    # STEP 1: the signal — z-score of today's price vs its recent past
    window = prcSoFar[:, -LOOKBACK:]          # last 20 days, all instruments
    mean = window.mean(axis=1)                # each instrument's recent average
    std = window.std(axis=1)
    std[std < 1e-8] = 1e-8                    # avoid divide-by-zero on flat series

    today = prcSoFar[:, -1]
    z = (today - mean) / std                  # how stretched is each price?

    # STEP 2: signal -> desired direction
    # High z = price is stretched UP = we expect fall = go SHORT (negative)
    # So desired position is proportional to MINUS z
    signal = -z

    # two thresholds instead of one
    had_position = currentPos != 0
    # exit if |z| < EXIT_z
    weak = np.abs(z) < ENTRY_Z
    dead = np.abs(z) < EXIT_Z

    # no position + weak signal -> stay out
    signal[~had_position & weak] = 0.0
    signal[had_position & dead] = 0.0
    hold = had_position & weak & ~dead

    # STEP 3: sizing with VOL SCALING
    # Cap signal so one crazy z-score doesn't max everything
    signal = np.clip(signal, -2.0, 2.0) / 2.0   # now in [-1, 1]

    rets = np.diff(np.log(window), axis=1)
    vol = rets.std(axis=1)
    vol[vol < 1e-8] = 1e-8
    risk_scale = np.median(vol) / vol
    risk_scale = np.clip(risk_scale, 0.25, 2.0)          # don't go crazy on vol scaling     

    dollar_pos = signal * MAX_DOLLARS * risk_scale           # dollars to hold per instrument
    dollar_pos = np.clip(dollar_pos, -MAX_DOLLARS, MAX_DOLLARS)  # stay under the $10k limit

    share_pos = (dollar_pos / today).astype(int)               # dollars -> number of shares
    share_pos[hold] = currentPos[hold].astype(int) # hold if weak signal

    currentPos = share_pos

    return currentPos

