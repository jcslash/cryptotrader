#!/usr/bin/env python3
"""Is the 'negative-funding coins keep underperforming' effect tradable after REAL funding payments and costs?
Daily rebalance at 00:00 UTC bar, hold 24h (bar close to bar close). Shorts PAY funding when rate<0 and RECEIVE when >0.
Costs: COST round trip on the traded leg(s). Hedge legs: BTC or equal-weight liquid universe."""
import sys, numpy as np, pandas as pd
import research1 as r1

close, qv, tb, fund = r1.load()
liquid = qv.rolling(24).sum() >= 20e6
H = 24
fwd = close.shift(-H) / close - 1
idx = close.index
def realised_funding(t):
    # settlements at t+9h, t+17h, t+25h (value of ffilled series just after settlement)
    ts = [t + pd.Timedelta(hours=h) for h in (9, 17, 25)]
    ts = [x for x in ts if x in fund.index]
    return fund.loc[ts].sum() if len(ts) == 3 else None

def run(q=0.10, cost=0.002, lo=True, hours=(0,), absthr=None):
    rows = {}
    for t in idx[idx.hour.isin(hours)]:
        if t + pd.Timedelta(hours=25) > idx[-1]: break
        l = liquid.loc[t]; f = fund.loc[t][l].dropna(); r = fwd.loc[t][f.index].dropna(); f = f[r.index]
        if len(f) < 30: continue
        rf = realised_funding(t)
        if rf is None: continue
        sel = (f <= absthr) if absthr is not None else (f <= f.quantile(q)) if lo else (f >= f.quantile(1 - q))
        if sel.sum() < 3: continue
        names = f.index[sel]; mkt = r.mean(); btc = fwd.loc[t].get("BTCUSDT", np.nan)
        px = r[names].mean(); fnd = rf.reindex(names).mean()
        short_pnl = -px + fnd - cost       # short: -price move, + funding (positive rate received, negative paid)
        rows[t] = dict(n=len(names), px=px, mkt=mkt, btc=btc, fund=fnd,
                       short_raw=short_pnl, short_btc_hedged=short_pnl + btc - cost * 0.5 if not np.isnan(btc) else np.nan,
                       short_mkt_hedged=short_pnl + mkt - cost * 0.5,
                       long_raw=px - fnd - cost, long_mkt_hedged=px - fnd - cost - mkt - cost * 0.5)
    return pd.DataFrame(rows).T

def line(name, s):
    s = s.dropna(); d = s.groupby(s.index.floor("1D")).mean()
    yrs = " ".join(f"{y}:{s[s.index.year==y].mean()*1e4:+.0f}" for y in sorted(set(s.index.year)))
    ex, va = s[s.index < r1.SPLIT], s[s.index >= r1.SPLIT]
    sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else np.nan
    cum = (1 + s.fillna(0)).cumprod(); mdd = (1 - cum / cum.cummax()).max()
    return (f"{name:<34} n={len(s):>4} mean/day={s.mean()*1e4:+6.1f}bp t={r1.block_t(d):+5.1f} Sharpe={sh:+.2f} maxDD(unlevered)={mdd:.0%} | "
            f"explore={ex.mean()*1e4:+.1f} valid={va.mean()*1e4:+.1f} | yr[{yrs}]")

if __name__ == "__main__":
    out = []
    base = run(0.10)
    out.append(f"avg funding paid per day by bottom-decile (short pays if negative): {base.fund.mean()*1e4:+.1f}bp ; avg #names {base.n.mean():.1f}")
    for k in ("short_raw", "short_btc_hedged", "short_mkt_hedged"):
        out.append(line(f"SHORT bottom10% funding  {k}", base[k]))
    out.append("-- robustness (short_mkt_hedged): decile width / cost / entry hour")
    for q in (0.05, 0.20):
        out.append(line(f"  bottom {int(q*100)}%", run(q).short_mkt_hedged))
    out.append(line("  bottom10% cost 0.4% RT", run(0.10, cost=0.004).short_mkt_hedged))
    out.append(line("  bottom10% enter 08:00 bar", run(0.10, hours=(8,)).short_mkt_hedged))
    out.append(line("  bottom10% enter 16:00 bar", run(0.10, hours=(16,)).short_mkt_hedged))
    out.append(line("  abs funding <= -0.03%/8h", run(absthr=-0.0003).short_mkt_hedged))
    out.append("-- long-side filter value: long bottom10% vs market (should be NEGATIVE if filter useful)")
    out.append(line("LONG bottom10% mkt-hedged", base.long_mkt_hedged))
    top = run(0.10, lo=False)
    out.append(f"top10% funding avg paid by longs per day: {top.fund.mean()*1e4:+.1f}bp")
    out.append(line("LONG top10% funding raw", top.long_raw)); out.append(line("LONG top10% funding mkt-hedged", top.long_mkt_hedged))
    print("\n".join(out))
