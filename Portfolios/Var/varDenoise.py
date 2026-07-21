import numpy as np

# ============================================================
# VAR(1) — ONE CHANGE ONLY: denoise the learned matrix
#
# DENOISE_K = None  ->  exact current champion, untouched.
# DENOISE_K = 15    ->  after learning the 51x51 web (B), keep only
#                        its 15 strongest patterns and discard the
#                        rest as noise. Lower K = more aggressive.
#
# Nothing else in the model changes.
# ============================================================

LAM = 1.0
GROSS_SCALE = 40000.0
MIN_HISTORY = 110
DENOISE_K = 15          # <-- the one switch. Set to None to get the old model back.

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
    res = logrets - logrets.mean(axis=0)

    X = res[:, :-1].T
    Y = res[:, 1:].T
    G = X.T @ X
    G += LAM * np.eye(nInst) * np.trace(G) / nInst
    B = np.linalg.solve(G, X.T @ Y)

    # ---- the one change lives here ----
    if DENOISE_K is not None:
        U, S, Vt = np.linalg.svd(B, full_matrices=False)   # break B into its patterns,
        B = (U[:, :DENOISE_K] * S[:DENOISE_K]) @ Vt[:DENOISE_K, :]  # keep only the strongest K
    # ------------------------------------

    pred = res[:, -1] @ B

    s = pred / (np.abs(pred).mean() + 1e-12)
    dollars = np.clip(s * GROSS_SCALE, -cap, cap)
    dollars = dollars - dollars.mean()
    dollars = np.clip(dollars, -cap, cap)

    currentPos = (dollars / prcSoFar[:, -1]).astype(int)
    return currentPos