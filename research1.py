#!/usr/bin/env python3
"""Stage 1 tests (H1 funding crowding, H2 taker-flow continuation, H4 funding carry) per PREREG.md. Liquidity is point-in-time."""
import glob, sys, numpy as np, pandas as pd
CACHE = "data_vis"; COST = 0.002          # 0.20% round trip for directional trades
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")

def load():
    C, Q, TB, F = {}, {}, {}, {}
    for p in glob.glob(f"{CACHE}/k_*.pkl"):
        s = p.split("k_")[1][:-4]; k, f = pd.read_pickle(p)
        C[s], Q[s], TB[s] = k.close, k.qv, k.tbqv
        if f is not None: F[s] = f.last_funding_rate
    end = max(x.index.max() for x in C.values())
    idx = pd.date_range(pd.Timestamp("2023-01-01", tz="UTC"), end, freq="1h")
    mk = lambda d: pd.DataFrame(d).reindex(idx)
    close, qv, tb = mk(C), mk(Q), mk(TB)
    fund = pd.DataFrame(F).sort_index().reindex(idx.union(pd.DataFrame(F).index)).ffill().reindex(idx).reindex(columns=close.columns)
    return close, qv, tb, fund

def block_t(daily):
    d = daily.dropna()
    return (d.mean() / (d.std() / np.sqrt(len(d)))) if len(d) > 10 and d.std() > 0 else np.nan

def summarize(name, s):
    """s: Series of per-rebalance excess returns indexed by time (UTC)."""
    s = s.dropna(); d = s.groupby(s.index.floor("1D")).mean()
    out = [f"{name:<44} n={len(s):>6} mean={s.mean() * 1e4:+7.1f}bp t={block_t(d):+5.1f}"]
    for lab, m in (("explore", s.index < SPLIT), ("valid", s.index >= SPLIT)):
        x = s[m]; dd = x.groupby(x.index.floor("1D")).mean()
        out.append(f"{lab}={x.mean() * 1e4:+7.1f}bp(t{block_t(dd):+.1f})")
    yrs = " ".join(f"{y}:{s[s.index.year == y].mean() * 1e4:+.0f}" for y in sorted(set(s.index.year)))
    return "  ".join(out) + "   yr[" + yrs + "]"

def decile_excess(sig, fwd, liquid, hours, q=0.1, mask=None):
    """At each rebalance hour: mean excess fwd return of bottom/top decile of `sig` among liquid names."""
    rows = {}
    for t in sig.index[sig.index.hour.isin(hours)]:
        l = liquid.loc[t]; x = sig.loc[t][l].dropna(); r = fwd.loc[t][x.index].dropna(); x = x[r.index]
        if len(x) < 30: continue
        mkt = r.mean(); lo = x <= x.quantile(q); hi = x >= x.quantile(1 - q)
        rows[t] = (r[lo].mean() - mkt, r[hi].mean() - mkt, mkt)
    return pd.DataFrame(rows, index=["bottom", "top", "mkt"]).T

if __name__ == "__main__":
    close, qv, tb, fund = load()
    print("symbols:", close.shape[1], "hours:", len(close), "period", close.index[0], "->", close.index[-1])
    liquid = qv.rolling(24).sum() >= 20e6
    print("avg liquid names per hour:", liquid.sum(axis=1).mean().round(1))
    R = {h: close.shift(-h) / close - 1 for h in (4, 24, 72)}
    res = []
    # ---- H1 funding crowding (rebalance every 8h at funding settlement hours)
    d1 = {h: decile_excess(fund, R[h], liquid, [0, 8, 16]) for h in (24, 72)}
    for h, d in d1.items():
        res += [summarize(f"H1 fund bottom-decile (long) fwd{h}h excess", d.bottom),
                summarize(f"H1 fund top-decile  (avoid) fwd{h}h excess", d.top),
                summarize(f"H1 spread bottom-top fwd{h}h", d.bottom - d.top)]
    # ---- H2 taker-flow continuation (every 4h)
    tb4 = tb.rolling(4).sum() / qv.rolling(4).sum()
    surge = qv / qv.rolling(48).median().shift(1) >= 3
    sig2 = tb4.where(surge & liquid)
    for h in (4, 24):
        d = decile_excess(sig2, R[h], liquid, [0, 4, 8, 12, 16, 20])
        res += [summarize(f"H2 taker-buy top-decile fwd{h}h excess", d.top),
                summarize(f"H2 taker-buy bottom-decile fwd{h}h excess", d.bottom)]
    print("\n".join(res))
    pd.to_pickle(dict(res=res), "/tmp/research1_res.pkl")
