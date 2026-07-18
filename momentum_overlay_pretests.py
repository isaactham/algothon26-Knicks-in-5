import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FAST FACTOR-MOMENTUM OVERLAY: pretests
# Hypothesis: days 501-750 developed day-scale factor momentum (lag-1 acf
# +0.018 -> +0.092) and this is what paid the pack in 751-850.
# Go/no-go (pre-committed):
#   T1: momentum strategy clearly positive on NEW chunk, ~flat on OLD
#   T2: trailing lag-1 gate ON for most of NEW, OFF for most of OLD
#   T3: stub-clone earnings per chunk (proxy for the pack's experience)
#   T4: overlay vs core correlation for combined-book risk
# ---------------------------------------------------------------------------

FILE = "prices.txt"
BOUNDARY = 499
GATE_WIN = 75
DOLLAR_SCALE = 200_000        # directional dollars across the book at full signal
STUB_ACCUM = 5000

df = pd.read_csv(FILE, sep=r"\s+", header=0)
prices = df.values.T
nInst, nt = prices.shape
lr = np.diff(np.log(prices), axis=1)
T = lr.shape[1]
f_ret = lr.mean(axis=0)

commRate = np.full(nInst, 0.0001); commRate[0] = 0.00002
capDlr = np.full(nInst, 10_000.0); capDlr[0] = 100_000.0

print(f"{nInst} instruments, {nt} days\n")

# ---------------------------------------------------------------------------
# T2 first (cheap): trailing lag-1 autocorrelation of the factor as the gate
# ---------------------------------------------------------------------------
gate = np.zeros(T)
for d in range(GATE_WIN + 2, T):
    seg = f_ret[d - GATE_WIN:d]
    c = np.corrcoef(seg[:-1], seg[1:])[0, 1]
    thr = 1.0 / np.sqrt(GATE_WIN)             # ~1 sigma of the estimator
    gate[d] = float(np.clip(c / (2 * thr), 0.0, 1.0))   # full at ~2 sigma

on_old = (gate[GATE_WIN + 2:BOUNDARY] > 0.25).mean()
on_new = (gate[BOUNDARY:] > 0.25).mean()
lag = next((d - BOUNDARY for d in range(BOUNDARY, T) if gate[d] > 0.25), None)
print("=== T2: trailing lag-1 momentum gate ===")
print(f"frac ON old: {on_old:.2f}   frac ON new: {on_new:.2f}   "
      f"switch-on lag after boundary: {lag} days")

# ---------------------------------------------------------------------------
# Generic backtest helper (positions decided at day d, priced day d -> d+1)
# ---------------------------------------------------------------------------
def run(position_fn, label):
    cash = 0.0; value = 0.0; comm = 0.0
    curPos = np.zeros(nInst)
    daily = np.zeros(T)
    for d in range(GATE_WIN + 5, T - 1):
        px = prices[:, d]                     # today's prices (decision time)
        npos = position_fn(d)
        lim = (capDlr / px).astype(int)
        npos = np.clip(npos, -lim, lim).astype(int)
        dpos = npos - curPos
        cash -= px.dot(dpos) + comm
        comm = np.sum(np.abs(dpos) * px * commRate)
        curPos = npos
        pv = curPos.dot(prices[:, d + 1])
        pl = cash + pv - value
        value = cash + pv
        daily[d + 1] = pl
    po = daily[GATE_WIN + 6:BOUNDARY]
    pn = daily[BOUNDARY:]
    print(f"{label:<30} OLD mean {po.mean():>8.1f} (sd {po.std():>6.0f})   "
          f"NEW mean {pn.mean():>8.1f} (sd {pn.std():>6.0f})")
    return daily

# ---------------------------------------------------------------------------
# T1: fast factor momentum, gated and ungated.
# Signal: yesterday's factor return direction, scaled by trailing vol.
# Expressed proportionally across all instruments (uses full directional cap).
# ---------------------------------------------------------------------------
def make_momentum(gated):
    def pos(d):
        fv = f_ret[d - GATE_WIN:d].std() + 1e-9
        strength = np.clip(f_ret[d - 1] / (2 * fv), -1, 1)
        g = gate[d] if gated else 1.0
        dollars_each = strength * g * DOLLAR_SCALE / nInst
        book = np.full(nInst, dollars_each)
        book[0] *= 10                          # ALGO gets 10x (its cap allows it)
        return (book / prices[:, d])
    return pos

print("\n=== T1: fast factor momentum (lag-1 direction) ===")
mom_un = run(make_momentum(False), "momentum UNGATED")
mom_g  = run(make_momentum(True),  "momentum GATED")

# ---------------------------------------------------------------------------
# T3: stub clone (the starter code's momentum accumulator), pack proxy
# ---------------------------------------------------------------------------
stub_state = {"pos": np.zeros(nInst)}
def stub(d):
    last = np.log(prices[:, d] / prices[:, d - 1])
    nrm = np.sqrt(last @ last) + 1e-9
    rpos = (STUB_ACCUM * last / nrm) / prices[:, d]
    stub_state["pos"] = stub_state["pos"] + rpos
    return stub_state["pos"]

print("\n=== T3: starter-stub clone (pack proxy) ===")
stub_daily = run(stub, "stub accumulator")

# ---------------------------------------------------------------------------
# T4: correlation with the XSEC core (30/70 fullvol, 120k)
# ---------------------------------------------------------------------------
def _stdz(x):
    x = x - x.mean(); sd = x.std()
    return x / sd if sd > 1e-12 else np.zeros_like(x)

core_state = {"pos": np.zeros(nInst)}
def core(d):
    sub = lr[:, :d]
    sig = 0.3 * -_stdz(sub[:, -15:].sum(axis=1)) + 0.7 * -_stdz(sub[:, -90:].sum(axis=1))
    vol = sub.std(axis=1); vol = np.where(vol < 1e-6, 1e-6, vol)
    strength = np.clip(sig / (vol / vol.mean()), -1.5, 1.5) / 1.5
    dollars = strength * 120000
    dollars -= dollars.mean()
    return dollars / prices[:, d]

print("\n=== T4: core (120k) per chunk + correlations ===")
core_daily = run(core, "XSEC core 120k")

m = np.arange(T) > GATE_WIN + 10
for name, s in [("momentum gated", mom_g), ("stub", stub_daily)]:
    print(f"core vs {name:<15}: {np.corrcoef(core_daily[m], s[m])[0,1]:.3f}")

# what would core + gated momentum have looked like combined?
comb = core_daily + mom_g
po, pn = comb[GATE_WIN + 10:BOUNDARY], comb[BOUNDARY:]
def score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr**2 / (sr**2 + 1.0)
print(f"\nCOMBINED core+gated-momentum: OLD mean {po.mean():.1f} score {score(po.mean(), po.std()):.1f}"
      f"   NEW mean {pn.mean():.1f} score {score(pn.mean(), pn.std()):.1f}")
