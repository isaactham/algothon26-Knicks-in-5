"""
pairs.py — pair discovery + spread reversion backtest for Algothon 2026
Run:  python3 pairs.py
Expects prices.txt in the same folder (header row of tickers, rows = days).

Pipeline:
  1. DISCOVER  - find highly correlated pairs (excluding ALGO)
  2. VALIDATE  - keep only pairs whose correlation holds in ALL sub-windows
  3. BACKTEST  - trade each surviving pair's spread with z-score hysteresis
                 (same ENTRY/EXIT logic as the existing reversion leg)
"""

import numpy as np
import pandas as pd
from itertools import combinations

# ---------------- knobs (match your existing setup) ----------------
LOOKBACK     = 15       # rolling window for spread z-score
ENTRY_Z      = 1.0      # open the trade when |z| exceeds this
EXIT_Z       = 0.5      # close when |z| falls back inside this (hysteresis)
MAX_DOLLARS  = 9000     # per leg, per pair (cap is 10k, leave headroom for price drift)
COMM_RATE    = 0.0005   # 5 bps commission on traded dollars — set to your comp's rate
MIN_CORR     = 0.10     # full-sample correlation needed to shortlist a pair
MIN_SUB_CORR = 0.70     # correlation needed in EVERY sub-window to survive
N_WINDOWS    = 3        # split history into this many sub-windows for stability check
TOP_N_PAIRS  = 5        # trade at most this many pairs (best first)

# ---------------- load ----------------
prices = pd.read_csv("prices.txt", sep=None, engine="python")
if "ALGO" in prices.columns:
    universe = prices.drop(columns="ALGO")   # pairs among instruments 1-50 only
else:
    universe = prices.iloc[:, 1:]            # fallback: assume col 0 is ALGO

rets = universe.pct_change().dropna()
nDays = len(universe)

# ---------------- 1 + 2: discover and validate pairs ----------------
def stable_pairs():
    corr = rets.corr()
    cols = corr.columns
    # sub-window boundaries for the stability check
    edges = np.linspace(0, len(rets), N_WINDOWS + 1, dtype=int)

    candidates = []
    for a, b in combinations(cols, 2):
        c_full = corr.loc[a, b]
        if c_full < MIN_CORR:
            continue
        # must hold up in every sub-window, not just overall
        ok = all(
            rets[a].iloc[s:e].corr(rets[b].iloc[s:e]) >= MIN_SUB_CORR
            for s, e in zip(edges[:-1], edges[1:])
        )
        if ok:
            candidates.append((c_full, a, b))

    candidates.sort(reverse=True)
    return candidates[:TOP_N_PAIRS]

pairs = stable_pairs()
print(f"Surviving pairs (corr, A, B): ")
for c, a, b in pairs:
    print(f"  {c:.3f}  {a} / {b}")
if not pairs:
    print("  none — try lowering MIN_CORR, or the universe may lack tight pairs")
    raise SystemExit

# ---------------- 3: backtest the spread on each pair ----------------
def backtest_pair(a, b):
    """Trade the log-price spread of one pair. Returns daily P&L series."""
    pa, pb = universe[a].values, universe[b].values

    # hedge ratio: how many dollars of B per dollar of A keeps the spread flat.
    # estimated on log prices over the whole history (simple, stable choice)
    la, lb = np.log(pa), np.log(pb)
    beta = np.polyfit(lb, la, 1)[0]
    spread = la - beta * lb

    spr = pd.Series(spread)
    z = (spr - spr.rolling(LOOKBACK).mean()) / spr.rolling(LOOKBACK).std()

    # state machine with hysteresis: -1 = short spread, +1 = long spread, 0 = flat
    state = np.zeros(len(z))
    s = 0
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            state[t] = 0
            continue
        if s == 0:
            if zt >  ENTRY_Z: s = -1          # spread rich: short A, long B
            elif zt < -ENTRY_Z: s = +1        # spread cheap: long A, short B
        else:
            if abs(zt) < EXIT_Z: s = 0        # snap-back done, go flat
        state[t] = s

    # dollar positions per leg (state decided at close t, held over day t+1)
    dollars_a =  state * MAX_DOLLARS
    dollars_b = -state * MAX_DOLLARS          # dollar-neutral: equal $ opposite legs

    ra = np.diff(pa) / pa[:-1]
    rb = np.diff(pb) / pb[:-1]
    pnl = dollars_a[:-1] * ra + dollars_b[:-1] * rb

    # commission on the dollars traded each day the state changes
    turnover = (np.abs(np.diff(dollars_a, prepend=0)) +
                np.abs(np.diff(dollars_b, prepend=0)))
    pnl -= turnover[:-1] * COMM_RATE

    return pd.Series(pnl)

all_pnl = []
print("\nPer-pair results:")
for c, a, b in pairs:
    pnl = backtest_pair(a, b)
    score = pnl.mean() - 0.1 * pnl.std()
    print(f"  {a}/{b}:  mean {pnl.mean():8.2f}   std {pnl.std():8.2f}   score {score:8.2f}")
    all_pnl.append(pnl)

# ---------------- combined book ----------------
book = pd.concat(all_pnl, axis=1).sum(axis=1)
score = book.mean() - 0.1 * book.std()
print(f"\nCombined book:  mean {book.mean():.2f}   std {book.std():.2f}   score {score:.2f}")

# stability: same stats on the recent regime only (closest to hidden window)
recent = book.iloc[len(book) * 2 // 3:]
r_score = recent.mean() - 0.1 * recent.std()
print(f"Last third only: mean {recent.mean():.2f}   std {recent.std():.2f}   score {r_score:.2f}")