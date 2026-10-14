#!/usr/bin/env python3
"""H4: delta-neutral funding carry (long spot / short perp). Weekly rebalance, pick top-N by trailing-7d mean funding among liquid names.
Models: realised funding, spot+perp fees and slippage on entry+exit each rebalance. Does NOT model: basis moves, margin/liquidation, spot availability, transfer limits."""
import numpy as np, pandas as pd
import research1 as r1

close, qv, tb, fund = r1.load()
# per-settlement funding (value just after each settlement)
liq50 = qv.rolling(24).sum() >= 50e6
# settlement rows: hours 1,9,17 hold the value settled at 0,8,16 (see research2)
fs = fund[fund.index.hour.isin([1, 9, 17])]
def run(N=5, hold_days=7, trail=21, rt_cost=0.005, minpers=0.7, liq=liq50):
    idx = fs.index; out = {}
    pos = list(range(trail, len(idx) - hold_days * 3, hold_days * 3))
    for i in pos:
        t = idx[i]
        past = fs.iloc[i - trail:i]                       # settlements strictly before entry
        fut = fs.iloc[i:i + hold_days * 3]
        ok = liq.loc[t].reindex(past.columns).fillna(False) & past.notna().all()
        m = past.mean()[ok]
        pers = (past[m.index] > 0).mean()                 # share of positive settlements (persistence)
        m = m[pers >= minpers].nlargest(N)
        if len(m) < max(2, N // 2): continue
        earned = fut[m.index].sum().mean()                # equal weight, funding received by the short perp leg
        out[t] = dict(gross=earned, net=earned - rt_cost, picked=m.mean(), n=len(m))
    return pd.DataFrame(out).T

def line(name, d):
    s = d.net; yrs = " ".join(f"{y}:{s[s.index.year==y].sum()*100:+.1f}%" for y in sorted(set(s.index.year)))
    ann = s.mean() * 365 / 7
    cum = (1 + s).cumprod(); mdd = (1 - cum / cum.cummax()).max()
    neg = (s < 0).mean()
    return (f"{name:<34} weeks={len(s):>3} gross/wk={d.gross.mean()*100:+.2f}% net/wk={s.mean()*100:+.2f}% annualised(net,simple)={ann*100:+.0f}% "
            f"weeks<0={neg:.0%} maxDD={mdd:.1%} | yearly net sum[{yrs}]")
print(line("top5, 7d, cost 0.5% RT", run()))
print(line("top5, 7d, cost 1.0% RT", run(rt_cost=0.01)))
print(line("top10, 7d, cost 0.5% RT", run(N=10)))
print(line("top3, 7d, cost 0.5% RT", run(N=3)))
print(line("top5, 14d hold, 0.5% RT", run(hold_days=14)))
print(line("top5, 7d, persistence>=0.9", run(minpers=0.9)))
big = (qv.rolling(24).sum() >= 300e6)
print(line("liquid>=300M top5 7d 0.5%", run(liq=big)))
for s in ("BTCUSDT", "ETHUSDT"):
    f = fs[s].dropna(); print(f"{s} avg funding per year (gross, unlevered, per 3 settlements/day): " + " ".join(f"{y}:{f[f.index.year==y].sum()*100*365/ (len(f[f.index.year==y])/3 if len(f[f.index.year==y]) else 1):+.1f}%" for y in sorted(set(f.index.year))))
