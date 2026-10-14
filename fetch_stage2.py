import glob, os, sys, time, numpy as np, pandas as pd, vision as v
from concurrent.futures import ThreadPoolExecutor
# universe: top 120 by average rolling-24h turnover since 2024
vol = {}
for p in glob.glob(f"{v.CACHE}/k_*.pkl"):
    s = p.split("k_")[1][:-4]; k, _ = pd.read_pickle(p); k = k[k.index >= "2024-01-01"]
    if len(k) > 2000: vol[s] = k.qv.rolling(24).sum().mean()
syms = [s for s, _ in sorted(vol.items(), key=lambda x: -x[1])[:120]]
print("universe", len(syms), syms[:8], flush=True)
days = pd.date_range("2024-01-01", pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta("1D"), freq="D")
def one_day(a):
    s, d = a
    return v._csv(f"{v.HOST}/data/futures/um/daily/metrics/{s}/{s}-metrics-{d:%Y-%m-%d}.zip")
def hourly(df):
    df.index = pd.to_datetime(df.create_time, utc=True)
    df = df.drop(columns=["create_time", "symbol"]).astype(float).sort_index()
    return df.resample("1h", label="left", closed="left").last()
t0 = time.time()
with ThreadPoolExecutor(48) as ex:
    for i, s in enumerate(syms):
        out = f"{v.CACHE}/m_{s}.pkl"
        if os.path.exists(out): continue
        parts = [d for d in ex.map(one_day, [(s, d) for d in days]) if d is not None]
        if parts: pd.to_pickle(hourly(pd.concat(parts)), out)
        print(f"{i+1}/{len(syms)} {s} files={len(parts)} {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
