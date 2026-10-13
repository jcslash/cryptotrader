#!/usr/bin/env python3
"""Do any short setups have edge? Shorts simulated by mirroring prices linearly around the entry (exact P&L for fixed entry)."""
import sys, glob, numpy as np, pandas as pd
import backtest as bt

SRC = sys.argv[1]; bt.SOURCE = SRC; bt.QV_SCALE = 3.0 if SRC == "spot" else 1.0
pat = f"data/{SRC}_*_180d.csv"
btcname = "BTCUSDT" if SRC == "spot" else "BTC-USDT-SWAP"
gate = bt.BtcGate(bt.fetch_klines(btcname, 180))
PS = bt.Params(trail="atr", fail_bars=0, time_bars=12, cap=99)

def short_R(d, j, e, stop):
    m = lambda p: e * (2 - p / e)
    dm = d.copy()
    o, h, l, c = m(d.open), m(d.low), m(d.high), m(d.close)   # high/low swap
    dm["open"], dm["high"], dm["low"], dm["close"] = o, h, l, c
    cand = bt.Cand("s", j, j, d.index[j] + bt.H1, e, 0.0, m(stop), "S", 0, 0, 1.0)
    return bt.simulate(dm, cand, PS).R

rows = []
for path in sorted(glob.glob(pat)):
    d = bt.prep(bt.load_csv(path))
    if len(d) < 400 or "BTC" == path.split(f"{SRC}_")[1][:3] and False: continue
    sym = path.split(f"{SRC}_")[1].split("_180d")[0]
    if sym.startswith("BTC") and "USDT" in sym[:8] and sym.split("-")[0] == "BTC" or sym == "BTCUSDT": continue
    O, H, L, C = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    last = {"A": -999, "B": -999, "C": -999}
    ign = set(bt.events(d))
    for j in range(170, len(d) - 2):
        e = C[j]; rng = H[j] - L[j]
        if rng <= 0 or d.qv24.iloc[j] < 20e6 or np.isnan(d.mult.iloc[j]): continue
        g = gate.state(d.index[j] + bt.H1)
        tags = []
        # A: blow-off top: extended (7d>=+60%), volume spike, long upper wick, bearish close
        if (d.mult.iloc[j] >= 5 and d.ret7d.iloc[j] >= 0.6 and (H[j] - max(O[j], e)) / rng >= 0.5 and e < O[j]):
            tags.append(("A", H[j] * 1.002))
        # B: failed breakout: ignition within last 3 bars, now first close back under its level
        for k in range(1, 4):
            jj = j - k
            if jj in ign and C[jj] > d.hh48.iloc[jj] and e < d.hh48.iloc[jj] and all(C[x] >= d.hh48.iloc[jj] for x in range(jj, j)):
                tags.append(("B", max(H[jj:j + 1]) * 1.002)); break
        # C: breakdown below 48h low with volume
        if d.mult.iloc[j] >= 5 and e < d.ll48.iloc[j]:
            tags.append(("C", min(d.ll48.iloc[j] + d.atr.iloc[j], e * 1.08)))
        for tg, stop in tags:
            if j - last[tg] < 48: continue
            stop = max(stop, e * 1.02); sd = stop / e - 1
            if sd > 0.10: continue
            last[tg] = j
            rows.append(dict(sym=sym, t=d.index[j] + bt.H1, tag=tg, sd=sd, btc=(g[0] if g else -1), R=short_R(d, j, e, stop)))
df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
mid = df.t.iloc[len(df) // 2]; df["h"] = np.where(df.t < mid, 1, 2)
def rep(name, g):
    if len(g) == 0: print(f"{name:<34} n=0"); return
    pf = g.R[g.R > 0].sum() / max(-g.R[g.R < 0].sum(), 1e-9)
    print(f"{name:<34} n={len(g):>4} avgR={g.R.mean():+.2f}±{g.R.std() / np.sqrt(len(g)):.2f} PF={pf:.2f} win={(g.R > 0).mean():.0%} "
          f"P(R>=2)={(g.R >= 2).mean():.0%} avgSD={g.sd.mean():.1%} | h1 {g[g.h == 1].R.mean():+.2f}(n{(g.h == 1).sum()}) h2 {g[g.h == 2].R.mean():+.2f}(n{(g.h == 2).sum()})")
print(f"=== SHORT setups on {SRC} ===")
for tg in "ABC":
    rep(f"{tg} all", df[df.tag == tg])
    rep(f"{tg} & BTC not >=3 closes above EMA", df[(df.tag == tg) & (df.btc < 3)])
    rep(f"{tg} & BTC >=3 closes above EMA", df[(df.tag == tg) & (df.btc >= 3)])
