import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FACTOR LEVEL REVERSION TEST
# The new chunk shows RAW reversion of -0.107 (t=-14.5) at h=80-100, roughly
# 3x the old chunk. Question: is the FACTOR's LEVEL reverting to an anchor?
# Includes the small-sample AR(1) bias correction that killed the OU idea, so
# we don't fool ourselves the same way twice.
# ---------------------------------------------------------------------------

FILE = "prices17.txt"          # now days 1-750

df = pd.read_csv(FILE, sep=r"\s+", header=0)
prices = df.values.T
nInst, nt = prices.shape
print(f"{nInst} instruments, {nt} days")

lr = np.diff(np.log(prices), axis=1)
T = lr.shape[1]
factor_ret = lr.mean(axis=0)
factor_lvl = np.cumsum(factor_ret)          # the factor's "price" level

BOUNDARY = 499                               # day-500 split in return-space


def ar1_with_bias(x):
    """AR(1) of a level series, plus the null-hypothesis expectation for a
    pure random walk of the same length. If measured ~= random-walk expectation,
    the 'reversion' is a small-sample artifact, not real."""
    s = x - x.mean()
    n = len(s)
    denom = s[:-1] @ s[:-1]
    if denom < 1e-12:
        return np.nan, np.nan, np.nan
    phi = (s[:-1] @ s[1:]) / denom
    rw_expected = 1 - 4.0 / n               # approx bias for a random walk
    hl = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.inf
    return phi, rw_expected, hl


print("\n=== 1. Factor level AR(1), bias-aware ===")
for label, seg in [("OLD (1-500)", factor_lvl[:BOUNDARY]),
                   ("NEW (501-750)", factor_lvl[BOUNDARY:]),
                   ("FULL (1-750)", factor_lvl)]:
    phi, rw, hl = ar1_with_bias(seg)
    verdict = "artifact-like" if phi >= rw - 0.002 else "MORE reverting than a random walk"
    print(f"{label:>15}: AR(1) {phi:.4f}   random-walk expectation {rw:.4f}   "
          f"half-life {hl:>6.1f}d   -> {verdict}")

# ---------------------------------------------------------------------------
# 2. Is there a persistent ANCHOR? If the factor reverts to a fixed level,
#    the deviation from a long trailing mean should predict the next return.
#    Uses only past data at every step.
# ---------------------------------------------------------------------------
print("\n=== 2. Does deviation-from-anchor predict the factor's next return? ===")
for anchor_win in [60, 90, 120, 200]:
    for label, lo, hi in [("OLD", anchor_win, BOUNDARY),
                          ("NEW", max(BOUNDARY, anchor_win), T - 1)]:
        devs, nxts = [], []
        for d in range(lo, hi):
            anchor = factor_lvl[d - anchor_win:d].mean()
            devs.append(factor_lvl[d] - anchor)
            nxts.append(factor_ret[d + 1])
        devs, nxts = np.array(devs), np.array(nxts)
        if len(devs) < 30 or devs.std() < 1e-12:
            continue
        c = np.corrcoef(devs, nxts)[0, 1]
        t = c * np.sqrt(len(devs) - 2) / np.sqrt(1 - c**2 + 1e-12)
        flag = " *" if abs(t) > 2 else ""
        print(f"  anchor={anchor_win:>3}d  {label:>3}: corr(dev, next ret) "
              f"{c:>7.4f}  t {t:>5.1f}{flag}   (n={len(devs)})")

# ---------------------------------------------------------------------------
# 3. THE REAL TEST: fit the anchor on the OLD chunk only, then check whether
#    it predicts in the NEW chunk. Anchors fitted in-sample always look good;
#    only out-of-sample matters.
# ---------------------------------------------------------------------------
print("\n=== 3. Out-of-sample: old-chunk anchor applied to new chunk ===")
old_anchor = factor_lvl[:BOUNDARY].mean()
print(f"factor level: old-chunk mean {old_anchor:.4f}, "
      f"new-chunk mean {factor_lvl[BOUNDARY:].mean():.4f}, "
      f"final value {factor_lvl[-1]:.4f}")
devs, nxts = [], []
for d in range(BOUNDARY, T - 1):
    devs.append(factor_lvl[d] - old_anchor)
    nxts.append(factor_ret[d + 1])
devs, nxts = np.array(devs), np.array(nxts)
c = np.corrcoef(devs, nxts)[0, 1]
t = c * np.sqrt(len(devs) - 2) / np.sqrt(1 - c**2 + 1e-12)
print(f"corr(deviation from OLD anchor, next new-chunk return): {c:.4f}  t {t:.1f}")
print("(if the anchor is real and persistent, this should be clearly negative)")

# ---------------------------------------------------------------------------
# 4. Cross-check: is the h=80-100 RAW reversion factor-driven or instrument-
#    driven? Compare the factor's own long-horizon reversion to the average
#    instrument's AFTER removing the factor.
# ---------------------------------------------------------------------------
print("\n=== 4. Where does the h=90 reversion live? ===")
for label, s, e in [("OLD", 0, BOUNDARY), ("NEW", BOUNDARY, T)]:
    seg_lr = lr[:, s:e]
    f = seg_lr.mean(axis=0)
    fc = f - f.mean()
    beta = (seg_lr @ fc) / (fc @ fc)
    resid = seg_lr - beta[:, None] * f[None, :]
    fl = np.cumsum(f)
    rl = np.cumsum(resid, axis=1)
    n = len(f)
    h = 90
    if n <= h + 5:
        continue
    # factor's own h=90 reversion
    trail_f = fl[h:n - 1] - fl[:n - 1 - h]
    nxt_f = f[h + 1:]
    cf = np.corrcoef(trail_f, nxt_f)[0, 1]
    # average instrument's residual h=90 reversion
    cs = []
    for i in range(nInst):
        trail_r = rl[i, h:n - 1] - rl[i, :n - 1 - h]
        nxt_r = resid[i, h + 1:]
        if trail_r.std() > 1e-12 and nxt_r.std() > 1e-12:
            cs.append(np.corrcoef(trail_r, nxt_r)[0, 1])
    print(f"{label:>4}: factor h=90 self-reversion {cf:>7.4f}   "
          f"mean residual h=90 reversion {np.mean(cs):>7.4f}")
