import numpy as np
import pandas as pd


def load_prices(fn="prices.txt"):
    df = pd.read_csv(fn, sep=r"\s+", header=0)
    return df.values.T  # nInst x nt


prices_all = load_prices()
nInst, nt = prices_all.shape
logrets_all = np.diff(np.log(prices_all), axis=1)
n_obs = logrets_all.shape[1]
sig_threshold = 2 / np.sqrt(n_obs)

print(f"Loaded {nInst} instruments, {nt} days\n")

# ===========================================================================
# DIAGNOSTIC 1: Does trend-following have any signal?
# For a range of lookbacks, check whether the SIGN of the trailing return
# predicts the NEXT return. Positive mean = trend (winners keep winning),
# negative = reversion, near zero = neither.
# ===========================================================================
print("=== Trend vs reversion by lookback ===")
print("(positive = trend-following works, negative = reversion works)\n")
print(f"{'lookback':>9} {'mean next-ret * sign(trail)':>28} {'frac instruments positive':>27}")

cum = np.cumsum(logrets_all, axis=1)
for lb in [1, 2, 3, 5, 10, 20, 40]:
    per_inst = []
    for i in range(nInst):
        r = logrets_all[i]
        if len(r) <= lb + 1:
            continue
        trailing = cum[i, lb:] - np.concatenate([[0], cum[i, :len(cum[i]) - lb - 0]])[:len(cum[i]) - lb]
        # simpler: trailing return over window ending at t, next return at t+1
        trail = np.array([r[t - lb:t].sum() for t in range(lb, len(r) - 1)])
        nxt = np.array([r[t] for t in range(lb, len(r) - 1)])
        # does sign of trailing predict next return?
        signed = np.sign(trail) * nxt
        per_inst.append(signed.mean())
    per_inst = np.array(per_inst)
    frac_pos = (per_inst > 0).mean()
    print(f"{lb:>9} {per_inst.mean():>28.6f} {frac_pos:>27.2f}")

# ===========================================================================
# DIAGNOSTIC 2: How much of the position cap does Model A actually use?
# Re-run Model A's logic over the test window and record, each day, what
# fraction of the dollar cap each position occupies.
# ===========================================================================
print("\n=== Model A position utilisation vs caps ===")

DOLLAR_TARGET = 2000
VOL_WINDOW = 10
MIN_HISTORY = 15
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit

numTestDays = 250
startDay = nt - numTestDays

utilisations = []  # fraction of cap used, per instrument per day
for t in range(startDay, nt):
    prcSoFar = prices_all[:, :t]
    if t < MIN_HISTORY:
        continue
    logrets = np.diff(np.log(prcSoFar), axis=1)
    combined = (logrets[:, -1] + logrets[:, -2] + logrets[:, -5]) / 3.0
    signal = -combined
    window = logrets[:, -VOL_WINDOW:]
    vol = np.std(window, axis=1)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    signalStrength = np.clip(signal / vol, -1, 1)
    dollarTarget = signalStrength * DOLLAR_TARGET
    curPrices = prcSoFar[:, -1]
    pos = (dollarTarget / curPrices).astype(int)
    dollarPos = np.abs(pos * curPrices)
    utilisations.append(dollarPos / dlrPosLimit)

utilisations = np.array(utilisations)  # nDays x nInst
print(f"Mean cap utilisation across all positions: {utilisations.mean():.1%}")
print(f"Median cap utilisation: {np.median(utilisations):.1%}")
print(f"Max cap utilisation ever reached: {utilisations.max():.1%}")
print(f"Fraction of positions using > 50% of cap: {(utilisations > 0.5).mean():.1%}")
print(f"Fraction of positions using > 90% of cap: {(utilisations > 0.9).mean():.1%}")
