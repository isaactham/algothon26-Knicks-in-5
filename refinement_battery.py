import numpy as np
import pandas as pd
from scipy.stats import rankdata

pricesFile = "./prices.txt"
scoreDefaultParam = 1.0
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000
WINDOW_LEN, STEP, MIN_START = 100, 50, 150

BAND_SHORT, BAND_LONG = 15, 90
MIN_HISTORY = 110


def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def _std(x):
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)


def make_strategy(use_rank, clip_reproject, vol_window, dollar_target=20000):
    state = {"pos": None}

    def transform(x):
        return _std(rankdata(x)) if use_rank else _std(x)

    def getPosition(prcSoFar):
        nInst, nt = prcSoFar.shape
        if state["pos"] is None or state["pos"].shape[0] != nInst:
            state["pos"] = np.zeros(nInst, dtype=int)
        if nt < MIN_HISTORY:
            return state["pos"]

        lr = np.diff(np.log(prcSoFar), axis=1)
        sig = -(transform(lr[:, -BAND_SHORT:].sum(axis=1))
                + transform(lr[:, -BAND_LONG:].sum(axis=1))) / 2.0

        vol = lr[:, -vol_window:].std(axis=1)
        vol = np.where(vol < 1e-6, 1e-6, vol)
        strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
        dollars = strength * dollar_target
        dollars = dollars - dollars.mean()          # dollar-neutral

        if clip_reproject:
            cap = np.full(nInst, float(defaultDlrPosLimit))
            cap[0] = inst0DlrPosLimit
            for _ in range(3):                       # clip, re-neutralise, repeat
                dollars = np.clip(dollars, -cap, cap)
                dollars = dollars - dollars.mean()

        curPrices = prcSoFar[:, -1]
        state["pos"] = (dollars / curPrices).astype(int)
        return state["pos"]

    def reset():
        state["pos"] = None
    return getPosition, reset


def calcPL(prcHist, nInst, commRate, dlrPosLimit, getPosition, ts, te):
    cash = 0; curPos = np.zeros(nInst); totDV = 0; value = 0; comm = 0
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
        dv = cp * np.abs(dp); totDV += np.sum(dv); comm = np.sum(dv * commRate)
        curPos = np.array(npos)
        pv = curPos.dot(cp)
        pll_t = cash + pv - value; value = cash + pv
        if t > ts:
            pll.append(pll_t)
    pll = np.array(pll)
    return np.mean(pll), np.std(pll)


prcAll = loadPrices(pricesFile)
nInst, nt = prcAll.shape
commRate = np.full(nInst, defaultCommRate); commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, float(defaultDlrPosLimit)); dlrPosLimit[0] = inst0DlrPosLimit

windows = []
s = MIN_START
while s + WINDOW_LEN <= nt:
    windows.append((s, s + WINDOW_LEN)); s += STEP

variants = {
    "BASELINE (v1 as submitted)":    dict(use_rank=False, clip_reproject=False, vol_window=10),
    "rank transform":                dict(use_rank=True,  clip_reproject=False, vol_window=10),
    "clip-then-reproject":           dict(use_rank=False, clip_reproject=True,  vol_window=10),
    "rank + clip-reproject":         dict(use_rank=True,  clip_reproject=True,  vol_window=10),
    "vol_window=8":                  dict(use_rank=False, clip_reproject=False, vol_window=8),
    "vol_window=12":                 dict(use_rank=False, clip_reproject=False, vol_window=12),
}

print(f"{nInst} instruments, {nt} days, {len(windows)} windows")
print("Dominance rule: adopt a change only if it beats BASELINE on mean AND min.\n")
print(f"{'variant':<30} {'mean':>8} {'std':>8} {'min':>8} {'frac+':>7}")
for name, params in variants.items():
    getPosition, reset = make_strategy(**params)
    scores = []
    for ts, te in windows:
        reset()
        mu, sd = calcPL(prcAll, nInst, commRate, dlrPosLimit, getPosition, ts, te)
        scores.append(score(mu, sd))
    scores = np.array(scores)
    print(f"{name:<30} {scores.mean():>8.2f} {scores.std():>8.2f} "
          f"{scores.min():>8.2f} {(scores > 0).mean():>7.2f}")
