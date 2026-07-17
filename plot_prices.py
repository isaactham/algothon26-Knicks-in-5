"""
plot_prices.py
Visualise the raw price data. Independent of the strategy — run anytime:
    python3 plot_prices.py
"""

import numpy as np
import matplotlib.pyplot as plt

with open("prices.txt") as f:
    tickers = f.readline().split()

prices = np.loadtxt("prices.txt", skiprows=1)
print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} instruments")
print(f"Tickers: {tickers[:5]}... ({len(tickers)} total)")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: all instruments, normalised to day 0 so scales are comparable
axes[0].plot(prices / prices[0], linewidth=0.7, alpha=0.6)
axes[0].set_title("All instruments (normalised to day 0)")
axes[0].axvline(500, color="red", linestyle="--", label="New data (day 500)")
axes[0].legend()

# Bottom: a few named instruments
for t in ["ALGO", "SRTX", "MMBT", "EAFC"]:
    if t in tickers:
        i = tickers.index(t)
        axes[1].plot(prices[:, i], label=t)
axes[1].set_title("Selected instruments")
axes[1].set_xlabel("Day")
axes[1].legend()

plt.tight_layout()
plt.show()