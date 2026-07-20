import numpy as np

# ============================================================
# VAR(1) CROSS-EFFECTS MODEL  + POSITION INERTIA
# - The generator has lag-1 cross-effects between instruments: each
#   instrument's factor-residual return is predictable from ALL 51
#   lagged residuals, not just its own (the old core's reversion
#   signal is the diagonal shadow of this matrix).
# - Fit: ridge regression B (51x51) on residuals (returns minus
#   cross-sectional mean), refit every day on all history to date.
#   Leak-free by construction. lam = 1.0 (trace-scaled).
# - Position: predicted residual, scaled by mean |pred| to gross
#   40k/instrument, capped, dollar-netted, re-capped.
# - NEW (SMOOTH): instead of jumping to each day's target book, move
#   only SMOOTH of the way from yesterday's book toward the target.
#   Persistent signals still get expressed within a few days; one-day
#   noise flickers only drag positions partway, cutting round-trip
#   commissions. SMOOTH = 1.0 reproduces the old behaviour exactly.
# - Validated pipeline/selection notes: see team branch. SMOOTH to be
#   selected on the newest chunk from {0.3, 0.5, 0.7, 1.0} before
#   this file is submitted; placeholder below until that race runs.
# ============================================================

LAM = 1.0
GROSS_SCALE = 40000.0
MIN_HISTORY = 110
SMOOTH = 1          # 1.0 = old twitchy behaviour; race on newest chunk before submitting

CAP = np.full(51, 10_000.0)
CAP[0] = 100_000.0

currentPos = None


def getMyPosition(prcSoFar):
    global currentPos

    nInst, nt = prcSoFar.shape
    if currentPos is None or currentPos.shape[0] != nInst:
        currentPos = np.zeros(nInst, dtype=int)

    if nt < MIN_HISTORY:
        return currentPos

    cap = CAP[:nInst]

    logrets = np.diff(np.log(prcSoFar), axis=1)
    res = logrets - logrets.mean(axis=0)        # factor-residual returns

    # ridge-fit lag-1 cross-effects matrix on all available history
    X = res[:, :-1].T                            # (T-1, nInst) lagged
    Y = res[:, 1:].T                             # (T-1, nInst) next
    G = X.T @ X
    G += LAM * np.eye(nInst) * np.trace(G) / nInst
    B = np.linalg.solve(G, X.T @ Y)

    pred = res[:, -1] @ B                        # tomorrow's residuals

    s = pred / (np.abs(pred).mean() + 1e-12)
    dollars = np.clip(s * GROSS_SCALE, -cap, cap)
    dollars = dollars - dollars.mean()           # dollar-neutral target book
    dollars = np.clip(dollars, -cap, cap)

    # ---- position inertia: glide toward the target instead of jumping ----
    prev_dollars = currentPos * prcSoFar[:, -1]  # yesterday's book at today's prices
    dollars = prev_dollars + SMOOTH * (dollars - prev_dollars)
    dollars = dollars - dollars.mean()           # re-net (blend can leave a lean)
    dollars = np.clip(dollars, -cap, cap)        # re-cap (netting can nudge over)

    currentPos = (dollars / prcSoFar[:, -1]).astype(int)
    return currentPos