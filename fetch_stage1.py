import os, sys, time, pandas as pd, vision as v
from concurrent.futures import ThreadPoolExecutor
os.makedirs(v.CACHE, exist_ok=True)
syms = v.usdt_perp_symbols()
def one(s):
    out = f"{v.CACHE}/k_{s}.pkl"
    if os.path.exists(out): return s, "cached"
    try:
        k = v.klines_1h(s, "2023-01"); f = v.funding(s, "2023-01")
        if k is None or len(k) < 500: return s, "skip"
        pd.to_pickle((k, f), out); return s, f"{len(k)}"
    except Exception as e:
        return s, f"ERR {e}"
t0 = time.time(); done = 0
with ThreadPoolExecutor(32) as ex:
    for s, r in ex.map(one, syms):
        done += 1
        if done % 25 == 0: print(f"{done}/{len(syms)} {time.time()-t0:.0f}s last={s}:{r}", flush=True)
print("DONE", time.time() - t0, flush=True)
