import numpy as np
import pandas as pd

pricesFile = "./prices17.txt"          # days 1-750
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


# ---------------------------------------------------------------------------
# COLE: banded XSEC reversion
# ---------------------------------------------------------------------------
def make_cole(w_short, w_long, vol_mode, dollar_target=20000,
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
        vol = lr[:, -10:].std(axis=1) if vol_mode == "rolling10" else lr.std(axis=1)
        vol = np.where(vol < 1e-6, 1e-6, vol)
        strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
        dollars = strength * dollar_target
        dollars = dollars - dollars.mean()
        state["pos"] = (dollars / prcSoFar[:, -1]).astype(int)
        return state["pos"]

    def reset():
        state["pos"] = None
    return getPosition, reset


# ---------------------------------------------------------------------------
# ISAAC: relative z-score with hysteresis (verbatim logic, wrapped for state)
# ---------------------------------------------------------------------------
def make_isaac(LOOKBACK=15, ENTRY_Z=1.0, EXIT_Z=0.5, MAX_DOLLARS=9000,
               POS_LIMIT_DOLLARS=10000, MIN_TRADE_DOLLARS=1500):
    state = {"pos": None}

    def getPosition(prcSoFar):
        nins, nt = prcSoFar.shape
        if state["pos"] is None or state["pos"].shape[0] != nins:
            state["pos"] = np.zeros(nins)
        currentPos = state["pos"]
        if nt < LOOKBACK + 1:
            return np.zeros(nins, dtype=int)

        today_price = prcSoFar[:, -1]
        logp = np.log(prcSoFar)
        index = logp.mean(axis=0, keepdims=True)
        rel = logp - index

        window = rel[:, -LOOKBACK:]
        mean = window.mean(axis=1)
        std = window.std(axis=1)
        std[std < 1e-8] = 1e-8
        z = (rel[:, -1] - mean) / std
        signal = -z

        had_position = currentPos != 0
        weak = np.abs(z) < ENTRY_Z
        dead = np.abs(z) < EXIT_Z
        flipped = had_position & (np.sign(currentPos) == np.sign(z)) & (z != 0)
        signal[~had_position & weak] = 0.0
        signal[had_position & dead] = 0.0
        signal[flipped & weak] = 0.0
        hold = had_position & weak & ~dead & ~flipped

        signal = np.clip(signal, -2.0, 2.0) / 2.0
        dollar_pos = np.clip(signal * MAX_DOLLARS, -MAX_DOLLARS, MAX_DOLLARS)
        dollar_pos = dollar_pos - dollar_pos.mean()
        share_pos = (dollar_pos / today_price).astype(int)
        share_pos[hold] = currentPos[hold].astype(int)

        delta_dollars = np.abs(share_pos - currentPos) * today_price
        small = delta_dollars < MIN_TRADE_DOLLARS
        share_pos = share_pos.astype(int)
        share_pos[small] = currentPos[small].astype(int)

        max_shares = (POS_LIMIT_DOLLARS / today_price).astype(int)
        share_pos = np.clip(share_pos, -max_shares, max_shares)
        state["pos"] = share_pos
        return share_pos

    def reset():
        state["pos"] = None
    return getPosition, reset


def calcPL(prcHist, nInst, commRate, dlrPosLimit, getPosition, ts, te):
    """Returns (mean, std, daily P&L series)."""
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
    return np.mean(pll), np.std(pll), pll


prcAll = loadPrices(pricesFile)
nInst, nt = prcAll.shape
commRate = np.full(nInst, defaultCommRate); commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, float(defaultDlrPosLimit)); dlrPosLimit[0] = inst0DlrPosLimit
print(f"{nInst} instruments, {nt} days\n")

models = {
    "cole 30/70 fullvol": make_cole(0.3, 0.7, "full"),
    "cole 30/70 roll10":  make_cole(0.3, 0.7, "rolling10"),
    "cole 50/50 roll10":  make_cole(0.5, 0.5, "rolling10"),
    "isaac zscore 15d":   make_isaac(),
}

pnls = {"OLD": {}, "NEW": {}}
print(f"{'model':<22} {'OLD mean':>9} {'OLD score':>10} {'NEW mean':>9} {'NEW score':>10}")
for name, (gp, reset) in models.items():
    reset()
    mu_o, sd_o, pl_o = calcPL(prcAll, nInst, commRate, dlrPosLimit, gp, 150, BOUNDARY)
    reset()
    mu_n, sd_n, pl_n = calcPL(prcAll, nInst, commRate, dlrPosLimit, gp, BOUNDARY, nt)
    pnls["OLD"][name] = pl_o
    pnls["NEW"][name] = pl_n
    print(f"{name:<22} {mu_o:>9.1f} {score(mu_o, sd_o):>10.1f} "
          f"{mu_n:>9.1f} {score(mu_n, sd_n):>10.1f}")

# ---------------------------------------------------------------------------
# Daily P&L correlations: is there anything to blend, or are they the same bet?
# ---------------------------------------------------------------------------
for chunk in ["OLD", "NEW"]:
    print(f"\n=== Daily P&L correlation, {chunk} chunk ===")
    names = list(pnls[chunk].keys())
    print(f"{'':<22}" + "".join(f"{n[:10]:>12}" for n in names))
    for a in names:
        row = f"{a:<22}"
        for b in names:
            pa, pb = pnls[chunk][a], pnls[chunk][b]
            n = min(len(pa), len(pb))
            c = np.corrcoef(pa[-n:], pb[-n:])[0, 1]
            row += f"{c:>12.2f}"
        print(row)

# ---------------------------------------------------------------------------
# Blend check: does cole+isaac beat either alone on the NEW chunk?
# ---------------------------------------------------------------------------
print("\n=== 50/50 P&L blend of cole 30/70 fullvol + isaac (NEW chunk) ===")
a = pnls["NEW"]["cole 30/70 fullvol"]
b = pnls["NEW"]["isaac zscore 15d"]
n = min(len(a), len(b))
blend = 0.5 * a[-n:] + 0.5 * b[-n:]
print(f"cole alone : mean {a[-n:].mean():7.1f}  std {a[-n:].std():7.1f}  "
      f"score {score(a[-n:].mean(), a[-n:].std()):7.1f}")
print(f"isaac alone: mean {b[-n:].mean():7.1f}  std {b[-n:].std():7.1f}  "
      f"score {score(b[-n:].mean(), b[-n:].std()):7.1f}")
print(f"50/50 blend: mean {blend.mean():7.1f}  std {blend.std():7.1f}  "
      f"score {score(blend.mean(), blend.std()):7.1f}")
print("\n(note: a P&L-level blend approximates but does not equal running a")
print("combined book; if the blend wins clearly, build it at signal level)")
