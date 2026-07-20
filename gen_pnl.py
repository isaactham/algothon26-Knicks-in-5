"""
gen_pnl.py
Runs your strategy day-by-day through prices.txt and saves per-instrument P&L.
Run this FIRST: python3 gen_pnl.py
Outputs: pnl_per_instrument.npy  (shape: days x instruments)

IMPORTANT: check that COMM_RATE and the P&L timing convention below match
your existing backtester. The printed total P&L should match your
backtester's total for the same strategy — if it doesn't, fix this file
before trusting any charts.
"""

import numpy as np
from coleCode import getMyPosition   # <-- change to your strategy file's name

COMM_RATE = 0.0005   # competition commission rate — match your backtester

prices = np.loadtxt("prices.txt", skiprows=1)   # skip ticker header row
nDays, nInst = prices.shape
print(f"Loaded {nDays} days x {nInst} instruments")

pnl = np.zeros((nDays, nInst))
prevPos = np.zeros(nInst)

for t in range(1, nDays):
    # Positions decided using data up to and including day t-1 (no look-ahead)
    prcSoFar = prices[:t].T          # getMyPosition expects (nInst, days)
    newPos = np.array(getMyPosition(prcSoFar))

    # P&L from holding prevPos over the day t-1 -> t price move
    priceMove = prices[t] - prices[t - 1]
    pnl[t] = prevPos * priceMove

    # Commission on trades made to move from prevPos to newPos
    tradeValue = np.abs(newPos - prevPos) * prices[t - 1]
    pnl[t] -= tradeValue * COMM_RATE

    prevPos = newPos

    if t % 100 == 0:
        print(f"  day {t}/{nDays}...")

np.save("pnl_per_instrument.npy", pnl)

daily = pnl.sum(axis=1)
print("-" * 40)
print(f"Saved pnl_per_instrument.npy")
print(f"Total P&L:  {daily.sum():.0f}")
print(f"Mean daily: {daily.mean():.2f}")
print(f"Std daily:  {daily.std():.2f}")
print(f"Score (mean - 0.1*std): {daily.mean() - 0.1 * daily.std():.2f}")