import numpy as np

# ---------------------------------------------------------------------------
# ADAPTIVE ENSEMBLE v2. Three signal families weighted by the t-statistic of
# their trailing edge over 125 days (the reoptimisation window documented by
# a 2025 Algothon team). A family gets zero weight until its trailing edge is
# 2 standard errors above noise, full weight at 4. If nothing clears the bar,
# the book stands down. All thresholds are principled (significance-based),
# not tuned to the visible data.
# ---------------------------------------------------------------------------
EDGE_WINDOW = 125
T_FLOOR = 2.0          # t-stat where a family starts earning weight
T_FULL = 4.0           # t-stat where it reaches full weight
DOLLAR_TARGET = 30000
VOL_WINDOW = 10
MIN_HISTORY = 160

currentPos = None

def _reset():
    """Clear internal state. Called by the walk-forward harness between windows."""
    global currentPos
    currentPos = None

def _signal_histories(logrets):
    """Each family's signal series, aligned so column d predicts return d+1."""
    nInst, T = logrets.shape
    sigs = {}

    # 1. short-lag reversion (lags 1, 2, 5)
    s = np.zeros((nInst, T))
    s[:, 5:] = -(logrets[:, 5:] + logrets[:, 4:-1] + logrets[:, 1:-4]) / 3.0
    sigs["rev_short"] = s

    # 2. 20-day reversion
    c = np.cumsum(logrets, axis=1)
    s = np.zeros((nInst, T))
    s[:, 20:] = -(c[:, 20:] - c[:, :-20])
    sigs["rev_20d"] = s

    # 3. EMA crossover trend (fast 10 vs slow 40 on cumulative log price)
    fast = np.zeros((nInst, T))
    slow = np.zeros((nInst, T))
    af, asl = 2 / 11, 2 / 41
    fast[:, 0] = slow[:, 0] = c[:, 0]
    for d in range(1, T):
        fast[:, d] = af * c[:, d] + (1 - af) * fast[:, d - 1]
        slow[:, d] = asl * c[:, d] + (1 - asl) * slow[:, d - 1]
    sigs["trend_ema"] = fast - slow

    return sigs


def _standardize(x):
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
    T = logrets.shape[1]
    sigs = _signal_histories(logrets)

    lo = max(60, T - EDGE_WINDOW - 1)
    weights, todays = {}, {}
    for name, S in sigs.items():
        ics = []
        for d in range(lo, T - 1):
            s_d, r_next = S[:, d], logrets[:, d + 1]
            if s_d.std() > 1e-12 and r_next.std() > 1e-12:
                ics.append(np.corrcoef(s_d, r_next)[0, 1])
        ics = np.array(ics)
        if len(ics) < 30:
            weights[name] = 0.0
        else:
            se = ics.std() / np.sqrt(len(ics)) + 1e-12
            tstat = ics.mean() / se
            weights[name] = float(np.clip((tstat - T_FLOOR) / (T_FULL - T_FLOOR), 0.0, 1.0))
        todays[name] = _standardize(S[:, -1])

    total_w = sum(weights.values())
    if total_w < 1e-9:
        currentPos = np.zeros(nInst, dtype=int)
        return currentPos

    combined = np.zeros(nInst)
    for name in sigs:
        combined += weights[name] * todays[name]
    combined /= total_w

    exposure = min(total_w, 1.0)

    vol = logrets[:, -VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    strength = np.clip(combined / (vol / vol.mean()), -1.5, 1.5) / 1.5

    dollars = strength * exposure * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    currentPos = (dollars / curPrices).astype(int)
    return currentPos
