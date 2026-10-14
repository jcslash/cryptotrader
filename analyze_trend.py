#!/usr/bin/env python3
"""Continuation test: buy the first reclaim after a pullback in an already-strong coin (the BAT-type week). Parameters fixed before running."""
import sys, glob, numpy as np, pandas as pd
import backtest as bt
SRC = sys.argv[1]; bt.SOURCE = SRC; bt.QV_SCALE = 3.0 if SRC == "spot" else 1.0
gate = bt.BtcGate(bt.fetch_klines("BTCUSDT" if SRC == "spot" else "BTC-USDT-SWAP", 180))
P = bt.Params(trail="atr", fail_bars=0, time_bars=12, cap=99)
rows = []
for path in sorted(glob.glob(f"data/{SRC}_*_180d.csv")):
    sym = path.split(f"{SRC}_")[1].split("_180d")[0]
    if sym.startswith("BTC"): continue
    d = bt.prep(bt.load_csv(path))
    if len(d) < 400: continue
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["hi24"] = d.high.rolling(24).max()
    O, H, L, C, E = (d[x].to_numpy() for x in ("open", "high", "low", "close", "ema20"))
    last = -999
    for j in range(170, len(d) - 2):
        if j - last < 24 or d.qv24.iloc[j] < 20e6 or np.isnan(d.ret7d.iloc[j]): continue
        if d.ret7d.iloc[j] < 0.4: continue                                  # already a strong coin
        if d.hi24.iloc[j - 1] < d.hh48.iloc[j - 1] * 0.999: continue         # made a fresh 48h high within the last 24h
        pull = L[j - 3:j + 1].min() <= E[j - 3:j + 1].max() * 1.01          # pulled back to the 1H EMA20 in the last 4 bars
        reclaim = C[j] > E[j] and C[j] > H[j - 1] and C[j] > O[j]           # closes back up through prior bar high
        if not (pull and reclaim): continue
        e = C[j]; stop = min(L[j - 5:j + 1].min() * 0.998, e * 0.98); sd = e / stop - 1
        if sd > 0.08: continue
        last = j
        c = bt.Cand(sym, j, j, d.index[j] + bt.H1, e, 0.0, stop, "T", 0, 0, 1.0)
        g = gate.state(c.t)
        rows.append(dict(t=c.t, sym=sym, R=bt.simulate(d, c, P).R, btc=(g[0] if g else -1), nl=(g[1] if g else False)))
df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
mid = df.t.iloc[len(df) // 2]; df["h"] = np.where(df.t < mid, 1, 2)
def rep(n, g):
    if not len(g): print(n, "n=0"); return
    pf = g.R[g.R > 0].sum() / max(-g.R[g.R < 0].sum(), 1e-9)
    print(f"{n:<28} n={len(g):>4} avgR={g.R.mean():+.2f}±{g.R.std()/np.sqrt(len(g)):.2f} PF={pf:.2f} win={(g.R>0).mean():.0%} P(R>=3)={(g.R>=3).mean():.0%} | h1 {g[g.h==1].R.mean():+.2f}(n{(g.h==1).sum()}) h2 {g[g.h==2].R.mean():+.2f}(n{(g.h==2).sum()})")
print(f"=== trend-continuation long on {SRC} ===")
rep("ALL", df); rep("BTC gate (>=3, no newlow)", df[(df.btc >= 3) & ~df.nl]); rep("BTC not gate", df[~((df.btc >= 3) & ~df.nl)])
