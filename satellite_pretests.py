import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GATED FACTOR-REVERSION SATELLITE: pre-build tests
# Chunks: OLD = days 1-500 (factor ~random walk -> gate should be OFF)
#         NEW = days 501-750 (factor reverting, hl ~22d -> gate should be ON)
# All gate signals use ONLY trailing data available at each day.
# ---------------------------------------------------------------------------

FILE = "prices.txt"
BOUNDARY = 499                    # in return-index space
GATE_WIN = 100                    # trailing days the gate looks at
DEV_WIN = 30                      # anchor window for the deviation signal
ALGO_CAP = 100_000
ALGO_COMM = 0.00002

df = pd.read_csv(FILE, sep=r"\s+", header=0)
prices = df.values.T
nInst, nt = prices.shape
lr = np.diff(np.log(prices), axis=1)
T = lr.shape[1]
f_ret = lr.mean(axis=0)
f_lvl = np.cumsum(f_ret)
algo_px = prices[0]

print(f"{nInst} instruments, {nt} days\n")

# ---------------------------------------------------------------------------
# Gate candidate A: trailing bias-corrected AR(1) of the factor level.
# gate strength = how far below the random-walk null the trailing AR(1) sits.
# ---------------------------------------------------------------------------
def gateA(d):
    seg = f_lvl[d - GATE_WIN:d]
    s = seg - seg.mean()
    den = s[:-1] @ s[:-1]
    if den < 1e-12:
        return 0.0
    phi = (s[:-1] @ s[1:]) / den
    null = 1 - 4.0 / GATE_WIN
    return max(0.0, (null - phi) / 0.02)      # 1.0 when phi is 0.02 below null

# ---------------------------------------------------------------------------
# Gate candidate B: trailing predictiveness of deviation-from-anchor.
# gate strength = t-stat of corr(deviation, next factor return), sign-flipped.
# ---------------------------------------------------------------------------
def gateB(d):
    devs, nxts = [], []
    for k in range(d - GATE_WIN, d - 1):
        if k - DEV_WIN < 0:
            return 0.0
        devs.append(f_lvl[k] - f_lvl[k - DEV_WIN:k].mean())
        nxts.append(f_ret[k + 1])
    devs, nxts = np.array(devs), np.array(nxts)
    if devs.std() < 1e-12:
        return 0.0
    c = np.corrcoef(devs, nxts)[0, 1]
    t = c * np.sqrt(len(devs) - 2) / np.sqrt(1 - c**2 + 1e-12)
    return max(0.0, -t / 2.0)                 # 1.0 at t = -2

START = GATE_WIN + DEV_WIN + 5
gA = np.zeros(T); gB = np.zeros(T)
for d in range(START, T):
    gA[d] = min(gateA(d), 1.0)
    gB[d] = min(gateB(d), 1.0)

print("=== TEST 1: gate behaviour per chunk (want OFF old, ON new) ===")
for name, g in [("A: AR(1) vs null", gA), ("B: deviation t-stat", gB)]:
    on_old = (g[START:BOUNDARY] > 0.25).mean()
    on_new = (g[BOUNDARY:] > 0.25).mean()
    m_old = g[START:BOUNDARY].mean()
    m_new = g[BOUNDARY:].mean()
    print(f"{name:<22} frac ON old: {on_old:.2f}  new: {on_new:.2f}   "
          f"mean strength old: {m_old:.2f}  new: {m_new:.2f}")

# first day the gate exceeds 0.25 after the boundary -> lag
for name, g in [("A", gA), ("B", gB)]:
    lag = next((d - BOUNDARY for d in range(BOUNDARY, T) if g[d] > 0.25), None)
    print(f"gate {name} switch-on lag after day-500 boundary: "
          f"{lag if lag is not None else 'never'} days")

# ---------------------------------------------------------------------------
# TEST 2+3: satellite P&L, gated vs ungated, per chunk.
# Satellite: z-score of factor level vs trailing DEV_WIN mean; bet reversion
# through ALGO alone (cleanest expression, cheap commission, $100k cap).
# ---------------------------------------------------------------------------
def run_satellite(gate, label):
    pll_old, pll_new = [], []
    pos = 0.0
    cash = 0.0
    value = 0.0
    comm = 0.0
    daily = np.zeros(T)
    for d in range(START, T - 1):
        px = algo_px[d + 1]                    # trade at next day's price index
        seg = f_lvl[d - DEV_WIN:d]
        z = (f_lvl[d] - seg.mean()) / (seg.std() + 1e-9)
        strength = np.clip(-z / 2.0, -1, 1) * gate[d]
        new_pos = int(strength * ALGO_CAP / px)

        dpos = new_pos - pos
        cash -= px * dpos + comm
        comm = abs(dpos) * px * ALGO_COMM
        pos = new_pos
        pv = pos * px
        pl = cash + pv - value
        value = cash + pv
        daily[d + 1] = pl
        (pll_old if d + 1 < BOUNDARY else pll_new).append(pl)

    po, pn = np.array(pll_old), np.array(pll_new)
    print(f"{label:<28} OLD mean {po.mean():>7.1f} (sd {po.std():>6.0f})   "
          f"NEW mean {pn.mean():>7.1f} (sd {pn.std():>6.0f})")
    return daily

print("\n=== TEST 2+3: satellite P&L per chunk ===")
ungated = run_satellite(np.ones(T), "UNGATED (always on)")
satA = run_satellite(gA, "gated by A")
satB = run_satellite(gB, "gated by B")

# ---------------------------------------------------------------------------
# TEST 4: correlation with the XSEC core's daily P&L (30/70 fullvol)
# ---------------------------------------------------------------------------
def _stdz(x):
    x = x - x.mean(); sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)

core_daily = np.zeros(T)
cpos = np.zeros(nInst); cash = 0.0; value = 0.0; comm = 0.0
commRate = np.full(nInst, 0.0001); commRate[0] = ALGO_COMM
for d in range(110, T - 1):
    sub = lr[:, :d]
    sig = 0.3 * -_stdz(sub[:, -15:].sum(axis=1)) + 0.7 * -_stdz(sub[:, -90:].sum(axis=1))
    vol = sub.std(axis=1); vol = np.where(vol < 1e-6, 1e-6, vol)
    strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
    dollars = strength * 20000
    dollars -= dollars.mean()
    px = prices[:, d + 1]
    npos = (dollars / prices[:, d]).astype(int)
    dpos = npos - cpos
    cash -= prices[:, d].dot(dpos) + comm
    comm = np.sum(np.abs(dpos) * prices[:, d] * commRate)
    cpos = npos
    pv = cpos.dot(px)
    core_daily[d + 1] = cash + pv - value
    value = cash + pv

print("\n=== TEST 4: satellite vs core daily P&L correlation ===")
mask = np.arange(T) > max(START + 1, 111)
for name, s in [("ungated", ungated), ("gated A", satA), ("gated B", satB)]:
    m = mask & (np.abs(s) + np.abs(core_daily) > 0)
    if m.sum() > 50:
        print(f"core vs {name:<9}: {np.corrcoef(core_daily[m], s[m])[0,1]:.3f}")
