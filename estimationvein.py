import numpy as np

# ============================================================
# VAR(1) — ESTIMATION VEIN VARIANTS
# Same model family as the champion (lag-1, ridge, dollar-neutral,
# capped). This file changes ONLY how B is fit, via three toggles
# that can be combined:
#
#   CORR_SPACE : fit on each instrument's return divided by its own
#                trailing volatility, instead of raw residuals.
#                Stops wild instruments from dominating the fit.
#   DENOISE_K  : if set, keep only the K strongest components of the
#                fitted B (truncated SVD) and discard the rest as noise.
#   LAM_MODE   : 'fixed' uses LAM below. 'cv' picks LAM from LAM_GRID
#                by inner train/val split on the training data itself
#                (no peeking at the true test days).
# ============================================================

LAM        = 1.0
LAM_MODE   = "fixed"          # "fixed" or "cv"
LAM_GRID   = [0.3, 1.0, 3.0, 10.0]
CORR_SPACE = False
DENOISE_K  = None             # e.g. 15, or None to disable

GROSS_SCALE = 40000.0
MIN_HISTORY = 110

CAP = np.full(51, 10_000.0)
CAP[0] = 100_000.0

currentPos = None


# ---------- core fit, reused everywhere ----------
def _ridge_fit(X, Y, lam):
    """Plain ridge fit: predicts Y from X, shrinkage strength lam."""
    G = X.T @ X
    G += lam * np.eye(X.shape[1]) * np.trace(G) / X.shape[1]
    return np.linalg.solve(G, X.T @ Y)


def _denoise(B, k):
    """Keep only the k strongest singular components of B."""
    if k is None or k >= min(B.shape):
        return B
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def fit_B(res, lam=LAM, lam_mode=LAM_MODE, lam_grid=LAM_GRID,
          corr_space=CORR_SPACE, denoise_k=DENOISE_K):
    """
    res: (nInst, T) residual history (moves vs. the pack).
    Returns B (nInst, nInst) and sigma (per-instrument std, or None
    if CORR_SPACE is off — needed to un-scale predictions later).
    """
    nInst, T = res.shape
    sigma = None

    work = res
    if corr_space:
        sigma = res.std(axis=1, keepdims=True) + 1e-12   # (nInst, 1)
        work = res / sigma                                # scale each instrument

    X = work[:, :-1].T
    Y = work[:, 1:].T

    if lam_mode == "cv":
        # inner split: oldest 70% of the TRAINING window fits candidates,
        # newest 30% of the TRAINING window scores them. The true test
        # days are never touched here.
        cut = int(len(X) * 0.7)
        Xtr, Ytr, Xval, Yval = X[:cut], Y[:cut], X[cut:], Y[cut:]
        best_lam, best_score = lam_grid[0], -np.inf
        for cand in lam_grid:
            Bc = _ridge_fit(Xtr, Ytr, cand)
            pred = Xval @ Bc
            score = np.corrcoef(pred.ravel(), Yval.ravel())[0, 1]
            if score > best_score:
                best_score, best_lam = score, cand
        lam = best_lam                      # chosen lam
        B = _ridge_fit(X, Y, lam)            # refit on FULL training window
    else:
        B = _ridge_fit(X, Y, lam)

    B = _denoise(B, denoise_k)
    return B, sigma, lam


# ---------- live trading entry point ----------
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

    B, sigma, _ = fit_B(res)

    x_today = res[:, -1]
    if sigma is not None:                    # CORR_SPACE: scale in, scale out
        pred = (x_today / sigma.ravel()) @ B * sigma.ravel()
    else:
        pred = x_today @ B

    s = pred / (np.abs(pred).mean() + 1e-12)
    dollars = np.clip(s * GROSS_SCALE, -cap, cap)
    dollars = dollars - dollars.mean()
    dollars = np.clip(dollars, -cap, cap)

    currentPos = (dollars / prcSoFar[:, -1]).astype(int)
    return currentPos