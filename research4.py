#!/usr/bin/env python3
"""H3 (price x open-interest structure) and H5 (market crowding timing) per PREREG.md. Stage-2 metrics start 2024-01."""
import glob, numpy as np, pandas as pd
import research1 as r1

close, qv, tb, fund = r1.load()
liquid = qv.rolling(24).sum() >= 20e6
M = {}
for p in glob.glob("data_vis/m_*.pkl"):
    M[p.split("m_")[1][:-4]] = pd.read_pickle(p)
cols = list(M)
def panel(c): return pd.DataFrame({s: M[s][c] for s in cols}).reindex(close.index)
oi, oiv = panel("sum_open_interest"), panel("sum_open_interest_value")
ls_g, ls_top, taker = panel("count_long_short_ratio"), panel("sum_toptrader_long_short_ratio"), panel("sum_taker_long_short_vol_ratio")
print("metric symbols:", len(cols), "| with OI data hours:", int(oi.notna().sum().sum()))
fwd24 = close.shift(-24) / close - 1
fwd72 = close.shift(-72) / close - 1
SPL = r1.SPLIT

def blk(s, k=1):
    s = s.dropna(); s = s.groupby(s.index.floor(f"{k}D")).mean(); return r1.block_t(s)
def report(name, s):
    s = s.dropna()
    if len(s) < 20: return f"{name:<52} n={len(s)} (too few)"
    ex, va = s[s.index < SPL], s[s.index >= SPL]
    yrs = " ".join(f"{y}:{s[s.index.year==y].mean()*1e4:+.0f}" for y in sorted(set(s.index.year)))
    return (f"{name:<52} n={len(s):>4} mean={s.mean()*1e4:+7.1f}bp t={blk(s):+5.1f} | explore={ex.mean()*1e4:+7.1f}(t{blk(ex):+.1f},n{len(ex)}) "
            f"valid={va.mean()*1e4:+7.1f}(t{blk(va):+.1f},n{len(va)}) yr[{yrs}]")

# ------------------------------------------------------------------ H3
px24 = close / close.shift(24) - 1
oi24 = oi / oi.shift(24) - 1
H3 = {"A new-longs (px>0, OI>0, funding<median)": [], "B short-cover (px>0, OI<0)": [], "C px<0 & OI>0 (new shorts)": [], "D px<0 & OI<0 (long liquidation)": []}
rows = {k: {} for k in H3}; allr = {}
for t in close.index[close.index.hour == 0]:
    l = liquid.loc[t].reindex(cols).fillna(False)
    names = [s for s in cols if l[s]]
    if len(names) < 20: continue
    d = pd.DataFrame({"px": px24.loc[t, names], "oi": oi24.loc[t, names], "f": fund.loc[t, names], "r": fwd24.loc[t, names]}).dropna()
    if len(d) < 20: continue
    mkt = d.r.mean(); fm = d.f.median(); allr[t] = mkt
    sel = {"A new-longs (px>0, OI>0, funding<median)": (d.px > 0) & (d.oi > 0) & (d.f < fm),
           "B short-cover (px>0, OI<0)": (d.px > 0) & (d.oi < 0),
           "C px<0 & OI>0 (new shorts)": (d.px < 0) & (d.oi > 0),
           "D px<0 & OI<0 (long liquidation)": (d.px < 0) & (d.oi < 0)}
    for k, m in sel.items():
        if m.sum() >= 3: rows[k][t] = d.r[m].mean() - mkt
print("\n=== H3: next-24h EXCESS return vs liquid-universe mean (pre-registered: A > 0, B < 0) ===")
for k in rows: print(report("H3 " + k, pd.Series(rows[k])))
# long-only tradability of A, net of 20bp cost and funding ignored (conservative check on price alone)
A = pd.Series(rows["A new-longs (px>0, OI>0, funding<median)"]); mk = pd.Series(allr)
print(report("H3 A raw long net of 20bp cost (market beta incl.)", (A + mk.reindex(A.index)) - 0.002))
print(report("H3 market (equal-weight liquid) fwd24h, for reference", mk))

# ------------------------------------------------------------------ H5
print("\n=== H5: market-crowding timing. outcome = equal-weight liquid universe fwd72h (non-overlapping 3-day blocks) ===")
mfund = fund.where(liquid).mean(axis=1)                                   # market mean funding (2023+)
ls_btc = ls_g["BTCUSDT"] if "BTCUSDT" in ls_g else pd.Series(index=close.index, dtype=float)
oi_btc = oi24["BTCUSDT"] if "BTCUSDT" in oi24 else pd.Series(index=close.index, dtype=float)
mk72 = pd.DataFrame({s: fwd72[s] for s in close.columns}).where(liquid).mean(axis=1)
day0 = close.index[close.index.hour == 0]
feat = pd.DataFrame({"mfund": mfund, "btcfund": fund["BTCUSDT"], "ls_btc": ls_btc, "oi24_btc": oi_btc, "ls_top_btc": ls_top.get("BTCUSDT")}).loc[day0]
out = mk72.loc[day0]
def tertile_test(f, label, start=None):
    x = feat[f].dropna(); 
    if start: x = x[x.index >= start]
    y = out.reindex(x.index).dropna(); x = x[y.index]
    # rolling (expanding) percentile rank to avoid look-ahead: rank vs prior 180 days
    rk = x.rolling(180, min_periods=60).apply(lambda w: (w.iloc[-1] > w.iloc[:-1]).mean(), raw=False)
    hi, lo = y[rk >= 0.8], y[rk <= 0.2]; rest = y[(rk > 0.2) & (rk < 0.8)]
    # non-overlapping: take every 3rd day
    def nov(s): return s.iloc[::3]
    for nm, g in (("top-20% (crowded)", hi), ("bottom-20%", lo)):
        g3 = nov(g - y.mean())
        print(report(f"H5 {label} {nm} fwd72h excess", g3))
    print(report(f"H5 {label} top20 - bottom20 spread", nov(hi).mean() - nov(lo).mean() * 0 + pd.Series([0.0])) if False else "", end="")
for f, lab in (("mfund", "market mean funding"), ("btcfund", "BTC funding"), ("ls_btc", "BTC global long/short ratio"), ("oi24_btc", "BTC OI 24h change"), ("ls_top_btc", "BTC top-trader L/S")):
    tertile_test(f, lab)
