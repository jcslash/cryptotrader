#!/usr/bin/env python3
"""Which ignition-bar features actually predict R?  Every ignition event, immediate entry at close, v2 exits.
Rule of trust: a feature bucket counts only if its edge has the SAME SIGN in both time halves."""
import sys, glob, numpy as np, pandas as pd
import backtest as bt

bt.SOURCE, bt.QV_SCALE = "spot", float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
days = 180
btc = bt.fetch_klines("BTCUSDT", days)
gate = bt.BtcGate(btc)
rows = []
for path in sorted(glob.glob(f"data/spot_*_{days}d.csv")):
    sym = path.split("spot_")[1].split(f"_{days}d")[0]
    if sym == "BTCUSDT": continue
    d = bt.prep(bt.load_csv(path))
    if len(d) < 400: continue
    last = -999
    for j in bt.events(d):
        if j - last < 48 or j + 1 >= len(d): continue
        L = d.hh48.iloc[j]; e = d.close.iloc[j]; a = d.atr.iloc[j]
        stop = min(max(d.low.iloc[j], L - a), e * 0.98)
        sd = e / stop - 1
        if sd > 0.08: continue
        c = bt.Cand(sym, j, j, d.index[j] + bt.H1, e, L, stop, "X", 0, d.mult.iloc[j], 1.0)
        tr = bt.simulate(d, c, bt.P_V2)
        rng = d.high.iloc[j] - d.low.iloc[j]
        g = gate.state(c.t)
        rows.append(dict(sym=sym, t=c.t, R=tr.R,
            mult=d.mult.iloc[j], loc=(e - d.low.iloc[j]) / rng if rng > 0 else .5,
            dist=e / L - 1, width=(L - d.ll48.iloc[j]) / d.ll48.iloc[j], qv24=d.qv24.iloc[j],
            ret7d=d.ret7d.iloc[j], ret24=e / d.close.iloc[j - 24] - 1, body=e / d.open.iloc[j] - 1,
            atrp=a / e, sd=sd, qvshare=d.qv.iloc[j] / d.qv24.iloc[j],
            vol_prev12=d.qv.iloc[j - 12:j].mean() / d.base.iloc[j], hour=(c.t.hour),
            btc_cnt=(g[0] if g else -1), btc_newlow=(g[1] if g else False)))
df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
df.to_csv("events.csv", index=False)
mid = df.t.iloc[len(df) // 2]
df["half"] = np.where(df.t < mid, 1, 2)
print(f"events={len(df)}  avgR all={df.R.mean():+.3f}  half1={df[df.half==1].R.mean():+.3f}  half2={df[df.half==2].R.mean():+.3f}")
print(f"split at {mid}\n")
def se(x): return x.std() / np.sqrt(max(len(x), 1))
feats = ["mult", "loc", "dist", "width", "qv24", "ret7d", "ret24", "body", "atrp", "sd", "qvshare", "vol_prev12"]
for f in feats:
    # quartile edges from FULL sample (features are not outcome-dependent, so no leakage)
    q = pd.qcut(df[f], 4, duplicates="drop"); df["b"] = q
    print(f"--- {f}")
    for b, g in df.groupby("b", observed=True):
        a, c = g[g.half == 1].R, g[g.half == 2].R
        print(f"  {str(b):<28} n={len(g):>4}  avgR={g.R.mean():+.2f}±{se(g.R):.2f}   h1={a.mean():+.2f}(n{len(a)})  h2={c.mean():+.2f}(n{len(c)})")
print("--- btc_cnt (consecutive 4H closes above EMA20)")
for k, g in df.groupby("btc_cnt"):
    print(f"  cnt={k}  n={len(g)}  avgR={g.R.mean():+.2f}±{se(g.R):.2f}  h1={g[g.half==1].R.mean():+.2f} h2={g[g.half==2].R.mean():+.2f}")
print("--- btc_newlow")
for k, g in df.groupby("btc_newlow"):
    print(f"  newlow={k}  n={len(g)}  avgR={g.R.mean():+.2f}±{se(g.R):.2f}  h1={g[g.half==1].R.mean():+.2f} h2={g[g.half==2].R.mean():+.2f}")
