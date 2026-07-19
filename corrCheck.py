import pandas as pd

prices = pd.read_csv("prices.txt", sep=None, engine="python")  # auto-detects comma/tab/space delimiter

rets = prices.pct_change().dropna()

algo_ret = rets["ALGO"]
market_ret = rets.drop(columns="ALGO").mean(axis=1)

corr = algo_ret.corr(market_ret)
beta = algo_ret.cov(market_ret) / market_ret.var()

print(f"Correlation: {corr:.3f}")
print(f"Beta:        {beta:.3f}")