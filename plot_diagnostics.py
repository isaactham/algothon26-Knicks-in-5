"""
plot_diagnostics.py
Strategy performance dashboard. Requires pnl_per_instrument.npy
(run gen_pnl.py first).

    python3 gen_pnl.py
    python3 plot_diagnostics.py

Produces two figures:
  Figure 1 - main dashboard (7 charts, full history)
  Figure 2 - old window (0-500) vs new window (500-750) comparison
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- load ----------
with open("prices.txt") as f:
    tickers = f.readline().split()

pnl = np.load("pnl_per_instrument.npy")   # (days, nInst)
daily = pnl.sum(axis=1)
equity = daily.cumsum()
nDays = len(daily)

NEW_DATA_START = 500   # where days 500-750 begin
ROLL_WINDOW = 60       # rolling Sharpe window


def running_score(d, warmup=20):
    """Competition score (mean - 0.1*std) computed on expanding history."""
    out = np.full(len(d), np.nan)
    for t in range(warmup, len(d)):
        out[t] = d[:t].mean() - 0.1 * d[:t].std()
    return out


def rolling_sharpe(d, w):
    out = np.full(len(d), np.nan)
    for i in range(w, len(d)):
        s = d[i - w:i].std()
        out[i] = d[i - w:i].mean() / s * np.sqrt(252) if s > 0 else 0
    return out


# ================= FIGURE 1: main dashboard =================
fig, ax = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle("Strategy Diagnostics (full history)", fontsize=14)

# 1. Equity curve + running competition score
ax[0, 0].plot(equity, color="tab:blue")
ax[0, 0].axvline(NEW_DATA_START, color="red", ls="--", alpha=0.5)
ax[0, 0].set_title("Equity curve (blue) / running score (green, right axis)")
ax2 = ax[0, 0].twinx()
ax2.plot(running_score(daily), color="tab:green", alpha=0.6)
ax2.tick_params(axis="y", labelcolor="tab:green")

# 2. Drawdown
dd = equity - np.maximum.accumulate(equity)
ax[0, 1].fill_between(range(nDays), dd, 0, color="red", alpha=0.4)
ax[0, 1].axvline(NEW_DATA_START, color="red", ls="--", alpha=0.5)
ax[0, 1].set_title(f"Drawdown (max {dd.min():.0f})")

# 3. Rolling Sharpe
rs = rolling_sharpe(daily, ROLL_WINDOW)
ax[1, 0].plot(rs, color="tab:purple")
ax[1, 0].axhline(0, color="k", lw=0.5)
ax[1, 0].axvline(NEW_DATA_START, color="red", ls="--", alpha=0.5)
ax[1, 0].set_title(f"Rolling Sharpe ({ROLL_WINDOW}d, annualised)")

# 4. Per-instrument P&L, sorted
inst_pnl = pnl.sum(axis=0)
order = np.argsort(inst_pnl)[::-1]
colors = ["tab:green" if v > 0 else "tab:red" for v in inst_pnl[order]]
ax[1, 1].bar(range(len(inst_pnl)), inst_pnl[order], color=colors)
ax[1, 1].set_title("Total P&L by instrument (sorted)")
# label the top and bottom 3
for rank in [0, 1, 2, len(order) - 3, len(order) - 2, len(order) - 1]:
    ax[1, 1].annotate(tickers[order[rank]], (rank, inst_pnl[order[rank]]),
                      fontsize=7, ha="center", rotation=90)

# 5. Profit concentration curve
total = inst_pnl.sum()
if total != 0:
    conc = inst_pnl[order].cumsum() / total
    ax[2, 0].plot(conc, marker=".", ms=3)
    ax[2, 0].axhline(0.8, color="r", ls="--", alpha=0.6)
    n80 = int(np.argmax(conc >= 0.8)) + 1 if (conc >= 0.8).any() else len(conc)
    ax[2, 0].set_title(f"Profit concentration (top {n80} instruments = 80%)")
else:
    ax[2, 0].set_title("Profit concentration (total P&L is zero)")
ax[2, 0].set_xlabel("Instruments (sorted by profit)")

# 6. Daily P&L histogram
ax[2, 1].hist(daily[1:], bins=50, color="tab:blue", alpha=0.7)
ax[2, 1].axvline(daily.max(), color="g", ls="--")
ax[2, 1].axvline(daily.min(), color="r", ls="--")
best_share = 100 * daily.max() / equity[-1] if equity[-1] != 0 else float("nan")
ax[2, 1].set_title(f"Daily P&L distribution (best day = {best_share:.0f}% of total)")

# 7. Long vs short leg P&L (needs positions -> approximated by sign of pnl
#    contribution direction; if you save positions in gen_pnl.py you can do
#    this exactly). Here: split per-instrument daily pnl into +/- days.
pos_days = np.where(pnl > 0, pnl, 0).sum(axis=1).cumsum()
neg_days = np.where(pnl < 0, pnl, 0).sum(axis=1).cumsum()
ax[3, 0].plot(pos_days, color="tab:green", label="cumulative gains")
ax[3, 0].plot(neg_days, color="tab:red", label="cumulative losses")
ax[3, 0].axvline(NEW_DATA_START, color="red", ls="--", alpha=0.5)
ax[3, 0].set_title("Gross gains vs gross losses (cumulative)")
ax[3, 0].legend()

# 8. Rolling mean daily P&L (smoother view of edge stability)
w = ROLL_WINDOW
rm = np.convolve(daily, np.ones(w) / w, mode="valid")
ax[3, 1].plot(range(w - 1, nDays), rm, color="tab:orange")
ax[3, 1].axhline(0, color="k", lw=0.5)
ax[3, 1].axvline(NEW_DATA_START, color="red", ls="--", alpha=0.5)
ax[3, 1].set_title(f"Rolling mean daily P&L ({w}d)")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ================= FIGURE 2: old vs new window =================
fig2, bx = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle("Old window (0-500) vs New window (500-750)", fontsize=14)

old, new = daily[:NEW_DATA_START], daily[NEW_DATA_START:]
old_pnl, new_pnl = pnl[:NEW_DATA_START], pnl[NEW_DATA_START:]

# Equity curves rebased to 0 at each window start
bx[0, 0].plot(old.cumsum(), label="old (0-500)")
bx[0, 0].plot(new.cumsum(), label="new (500-750)")
bx[0, 0].set_title("Equity per window (rebased)")
bx[0, 0].legend()

# Score comparison
def score(d):
    return d.mean() - 0.1 * d.std()

bars = bx[0, 1].bar(["old", "new"], [score(old), score(new)],
                    color=["tab:blue", "tab:orange"])
bx[0, 1].axhline(0, color="k", lw=0.5)
bx[0, 1].set_title(f"Score: old={score(old):.2f}  new={score(new):.2f}")

# Per-instrument P&L: old vs new scatter -> is the edge in the SAME names?
bx[1, 0].scatter(old_pnl.sum(axis=0), new_pnl.sum(axis=0), s=15)
bx[1, 0].axhline(0, color="k", lw=0.5)
bx[1, 0].axvline(0, color="k", lw=0.5)
corr = np.corrcoef(old_pnl.sum(axis=0), new_pnl.sum(axis=0))[0, 1]
bx[1, 0].set_title(f"Instrument P&L: old vs new (corr={corr:.2f})")
bx[1, 0].set_xlabel("P&L in old window")
bx[1, 0].set_ylabel("P&L in new window")

# Daily P&L distributions overlaid
bx[1, 1].hist(old, bins=40, alpha=0.5, density=True, label="old")
bx[1, 1].hist(new, bins=40, alpha=0.5, density=True, label="new")
bx[1, 1].set_title("Daily P&L distributions")
bx[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------- console summary ----------
print("=" * 50)
print(f"Full:  total={daily.sum():.0f}  score={score(daily):.2f}")
print(f"Old:   total={old.sum():.0f}  score={score(old):.2f}")
print(f"New:   total={new.sum():.0f}  score={score(new):.2f}")
print(f"Instrument P&L correlation old vs new: {corr:.2f}")
print("  (near 0 => per-instrument differences are noise; don't cherry-pick tickers)")