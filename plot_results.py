import numpy as np
import matplotlib.pyplot as plt

pnl = np.load("daily_pnl.npy")
equity = np.cumsum(pnl)                      # running total = your 'value' column
days = np.arange(len(pnl))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. Equity curve — the story of the whole run
ax = axes[0, 0]
ax.plot(days, equity)
ax.axhline(0, color='grey', lw=0.5)
ax.set_title("Equity curve (cumulative P&L)")

# 2. Drawdown — how far below the previous peak you are
ax = axes[0, 1]
peak = np.maximum.accumulate(equity)
drawdown = equity - peak
ax.fill_between(days, drawdown, 0, color='red', alpha=0.4)
ax.set_title("Drawdown from peak")

# 3. Daily P&L distribution — where the variance lives
ax = axes[1, 0]
ax.hist(pnl, bins=40)
ax.axvline(pnl.mean(), color='red', ls='--', label=f"mean ${pnl.mean():.0f}")
ax.legend()
ax.set_title("Daily P&L histogram")

# 4. Rolling Sharpe — is performance consistent or regime-dependent?
ax = axes[1, 1]
w = 40
roll_mu = np.array([pnl[i-w:i].mean() for i in range(w, len(pnl))])
roll_sd = np.array([pnl[i-w:i].std() for i in range(w, len(pnl))])
roll_sharpe = np.sqrt(250) * roll_mu / roll_sd
ax.plot(np.arange(w, len(pnl)), roll_sharpe)
ax.axhline(0, color='grey', lw=0.5)
ax.set_title(f"Rolling {w}-day Sharpe")

plt.tight_layout()
plt.savefig("results.png", dpi=120)
plt.show()