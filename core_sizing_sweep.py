import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CORE SIZING SWEEP: 30/70 fullvol XSEC model, both chunks.
# Rule (pre-committed): largest size whose Sharpe stays within 15% of the
# 20k baseline ON BOTH CHUNKS. clip_reproject=True re-nets the book after
# internal clipping so neutrality survives at large size.
# ---------------------------------------------------------------------------

pricesFile = "./prices.txt"
BOUNDARY = 500
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000


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


def make_core(dollar_target, clip_reproject):
    state = {"pos": None}
    cap = np.full(51, float(defaultDlrPosLimit)); cap[0] = inst0DlrPosLimit

    def getPosition(prcSoFar):
        nInst, nt = prcSoFar.shape
        if state["pos"] is None or state["pos"].shape[0] != nInst:
            state["pos"] = np.zeros(nInst, dtype=int)
        if nt < 110:
            return state["pos"]
        lr = np.diff(np.log(prcSoFar), axis=1)
        sig = (0.3 * -_std(lr[:, -15:].sum(axis=1))
               + 0.7 * -_std(lr[:, -90:].sum(axis=1)))
        vol = lr.std(axis=1)
        vol = np.where(vol < 1e-6, 1e-6, vol)
        strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
        dollars = strength * dollar_target
        dollars = dollars - dollars.mean()
        if clip_reproject:
            for _ in range(4):
                dollars = np.clip(dollars, -cap[:nInst], cap[:nInst])
                dollars = dollars - dollars.mean()
            dollars = np.clip(dollars, -cap[:nInst], cap[:nInst])
        state["pos"] = (dollars / prcSoFar[:, -1]).astype(int)
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
        if t > ts:
            pll.append(cash + pv - value)
        value = cash + pv
    pll = np.array(pll)
    return np.mean(pll), np.std(pll)


prcAll = loadPrices(pricesFile)
nInst, nt = prcAll.shape
commRate = np.full(nInst, defaultCommRate); commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, float(defaultDlrPosLimit)); dlrPosLimit[0] = inst0DlrPosLimit
print(f"{nInst} instruments, {nt} days")
print("Rule: largest size with Sharpe within 15% of 20k baseline on BOTH chunks\n")
print(f"{'target':>8} {'reproj':>7} {'OLD mean':>9} {'OLD Shp':>8} {'OLD scr':>8} "
      f"{'NEW mean':>9} {'NEW Shp':>8} {'NEW scr':>8}")

for dt in [20000, 40000, 60000, 80000, 120000]:
    for cr in ([False] if dt <= 20000 else [False, True]):
        gp, reset = make_core(dt, cr)
        reset()
        mu_o, sd_o = calcPL(prcAll, nInst, commRate, dlrPosLimit, gp, 150, BOUNDARY)
        reset()
        mu_n, sd_n = calcPL(prcAll, nInst, commRate, dlrPosLimit, gp, BOUNDARY, nt)
        shp_o = np.sqrt(250) * mu_o / sd_o if sd_o > 0 else 0
        shp_n = np.sqrt(250) * mu_n / sd_n if sd_n > 0 else 0
        print(f"{dt:>8} {str(cr):>7} {mu_o:>9.1f} {shp_o:>8.2f} {score(mu_o, sd_o):>8.1f} "
              f"{mu_n:>9.1f} {shp_n:>8.2f} {score(mu_n, sd_n):>8.1f}")
