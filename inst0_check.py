import numpy as np
import pandas as pd

df = pd.read_csv("prices.txt", sep=r"\s+", header=0, index_col=None)
prices = df.values.T                       # (51, 500)

rets = np.diff(np.log(prices), axis=1)

inst0 = rets[0]                            # instrument 0's daily returns
basket = rets[1:].mean(axis=0)             # average return of the other 50

corr = np.corrcoef(inst0, basket)[0, 1]
print(f"correlation of inst0 vs basket returns: {corr:+.3f}")

spread = np.cumsum(inst0 - basket)
ds = np.diff(spread)
ac = np.corrcoef(spread[:-1], ds)[0, 1]
print(f"spread level vs next-day change: {ac:+.3f}  (negative = spread reverts)")

# ============ lead-lag check: who moves first? ============
print()
print("lead-lag (lightning -> thunder?):")
for lag in [1, 2, 3]:
    b_leads = np.corrcoef(basket[:-lag], inst0[lag:])[0, 1]
    i_leads = np.corrcoef(inst0[:-lag], basket[lag:])[0, 1]
    print(f"lag {lag}:  basket->inst0 {b_leads:+.3f}    inst0->basket {i_leads:+.3f}")