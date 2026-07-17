import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Evaluate band weightings + vol sizing on BOTH chunks separately.
# Selection rule (pre-committed): pick the variant that is solidly positive
# in BOTH chunks, NOT the one that maximises the new chunk. A variant that
# only works in one chunk is regime-dependent and will not transfer.
# ---------------------------------------------------------------------------

pricesFile = "./prices17.txt"          # now days 1-750
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000
BOUNDARY = 500


def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu, sigma):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + 1.0)


def _std(x):
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)


def make_strategy(w_short, w_long, vol_mode, dollar_target=20000,
                  band_short=15, band_long=90, min_history=110):
    state = {"pos": None}

    def getPosition(prcSoFar):
        nInst, nt = prcSoFar.shape
        if state["pos"] is None or state["pos"].shape[0] != nInst:
            state["pos"] = np.zeros(nInst, dtype=int)
        if nt < min_history:
            return state["pos"]

        lr = np.diff(np.log(prcSoFar), axis=1)
        sig = np.zeros(nInst)
        if w_short > 0:
            sig += w_short * -_std(lr[:, -band_short:].sum(axis=1))
        if w_long > 0:
            sig += w_long * -_std(lr[:, -band_long:].sum(axis=1))
        sig /= (w_short + w_long)

        if vol_mode == "rolling10":
            vol = lr[:, -10:].std(axis=1)
        else:                                   # "full": vols are constants
            vol = lr.std(axis=1)
        vol = np.where(vol < 1e-6, 1e-6, vol)

        strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
        dollars = strength * dollar_target
        dollars = dollars - dollars.mean()      # net-zero book

        curPrices = prcSoFar[:, -1]
        state["pos"] = (dollars / curPrices).astype(int)
        return state["pos"]

    def reset():
        state["pos"] = None
    return getPosition, reset


def calcPL(prcHist, nInst, commRate, dlrPosLimit, getPosition, ts, te):
    cash = 0; curPos = np.zeros(nInst); value = 0; comm = 0
    pll = []
    for t in range(ts, te + 1):
        ph = prcHist[:, :t]; cp = ph[:, -1]
        if t < te:
            npos = getPosition(ph)
            lim = (dlrPosLimit / cp).astype(int)
            npos = np.clip(npos, -lim, lim).astype(int)
        else:
            npos = np.array(curPos)
        dp = npos - curPos
        cash -= cp.dot(dp) + comm
        dv = cp * np.abs(dp); comm = np.sum(dv * commRate)
        curPos = np.array(npos)
        pv = curPos.dot(cp)
        pll.append(cash + pv - value) if t > ts else None
        value = cash + pv
    pll = np.array(pll)
    return np.mean(pll), np.std(pll)


prcAll = loadPrices(pricesFile)
nInst, nt = prcAll.shape
commRate = np.full(nInst, defaultCommRate); commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, float(defaultDlrPosLimit)); dlrPosLimit[0] = inst0DlrPosLimit
print(f"{nInst} instruments, {nt} days\n")

variants = {
    "50/50 short+long (current)":  dict(w_short=0.5, w_long=0.5, vol_mode="rolling10"),
    "30/70 toward long":           dict(w_short=0.3, w_long=0.7, vol_mode="rolling10"),
    "0/100 long only":             dict(w_short=0.0, w_long=1.0, vol_mode="rolling10"),
    "100/0 short only":            dict(w_short=1.0, w_long=0.0, vol_mode="rolling10"),
    "30/70 + full-history vol":    dict(w_short=0.3, w_long=0.7, vol_mode="full"),
    "0/100 + full-history vol":    dict(w_short=0.0, w_long=1.0, vol_mode="full"),
    "50/50 + full-history vol":    dict(w_short=0.5, w_long=0.5, vol_mode="full"),
}

print("Selection rule: solidly positive in BOTH chunks. Not the new-chunk max.\n")
print(f"{'variant':<30} {'OLD mean':>9} {'OLD score':>10} {'NEW mean':>9} {'NEW score':>10}")
for name, params in variants.items():
    gp, reset = make_strategy(**params)
    reset()
    mu_o, sd_o = calcPL(prcAll, nInst, commRate, dlrPosLimit, gp, 150, BOUNDARY)
    reset()
    mu_n, sd_n = calcPL(prcAll, nInst, commRate, dlrPosLimit, gp, BOUNDARY, nt)
    print(f"{name:<30} {mu_o:>9.1f} {score(mu_o, sd_o):>10.1f} "
          f"{mu_n:>9.1f} {score(mu_n, sd_n):>10.1f}")

print("\nNote: NEW-chunk numbers are the closest thing we have to a true")
print("out-of-sample read (the model's design predates seeing this data),")
print("but the band weights are now informed by it, so treat with care.")
