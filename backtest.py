#!/usr/bin/env python3
"""Binance USDT-perp long-only ignition strategy: v1 (group's current rules) vs v2 (conviction-tiered).

Usage:
  python3 backtest.py --days 120                 # real data from fapi.binance.com (needs network)
  python3 backtest.py --days 120 --top 150       # limit to 150 most liquid symbols
  python3 backtest.py --synthetic                # engine smoke test only; results are MEANINGLESS

All bars are 1H, signals are evaluated at bar CLOSE, orders fill at that close (+slippage).
Intrabar ambiguity is resolved pessimistically (stop before target in the same bar).
"""
import argparse, json, os, sys, time, urllib.request
from dataclasses import dataclass
import numpy as np
import pandas as pd

FAPI = "https://fapi.binance.com"
SPOT = "https://data-api.binance.vision"
SOURCE = "perp"   # perp | spot | okx
QV_SCALE = 1.0    # spot volume understates perp volume; multiply to approximate (assumption, see README)
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDE","EUR","EURI","AEUR","USD1","XUSD","PYUSD","UST","USDT"}
H1 = pd.Timedelta("1h")


# ----------------------------------------------------------------------------- data
def http_json(url, retries=4):
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as r:
                return json.loads(r.read())
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 ** a)


def universe(min_qv=20e6, top=None):
    if SOURCE == "spot":
        tk = http_json(f"{SPOT}/api/v3/ticker/24hr")
        ok = []
        for t in tk:
            sy = t["symbol"]
            if not sy.endswith("USDT"): continue
            b = sy[:-4]
            if b in STABLE or b.endswith(("UP", "DOWN", "BULL", "BEAR")): continue
            ok.append((float(t["quoteVolume"]), sy))
        ok = [sy for q, sy in sorted(ok, reverse=True) if q * QV_SCALE >= min_qv * 0.25]
        return ok[:top] if top else ok
    if SOURCE == "okx":
        tk = http_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")["data"]
        ok = [(float(t["volCcy24h"]) * float(t["last"]), t["instId"]) for t in tk if t["instId"].endswith("-USDT-SWAP")]
        ok = [i for q, i in sorted(ok, reverse=True) if q >= min_qv * 0.25]
        return ok[:top] if top else ok
    info = http_json(f"{FAPI}/fapi/v1/exchangeInfo")["symbols"]
    ok = {s["symbol"] for s in info if s["contractType"] == "PERPETUAL"
          and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    tk = [t for t in http_json(f"{FAPI}/fapi/v1/ticker/24hr") if t["symbol"] in ok]
    tk = sorted(tk, key=lambda t: -float(t["quoteVolume"]))
    # NOTE: survivorship bias - only symbols listed today (delisted rugs are absent).
    out = [t["symbol"] for t in tk if float(t["quoteVolume"]) >= min_qv * 0.25]
    return out[:top] if top else out


def fetch_klines(sym, days, cache="data"):
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/{SOURCE}_{sym}_{days}d.csv"
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
        return load_csv(path)
    end = int(time.time() * 1000)
    start = end - days * 86400_000
    rows = []
    if SOURCE == "okx":
        rows, after = [], end
        while after > start:
            r = http_json(f"https://www.okx.com/api/v5/market/history-candles?instId={sym}&bar=1H&limit=100&after={after}")["data"]
            if not r: break
            rows += [[int(x[0]), x[1], x[2], x[3], x[4], 0, 0, x[7]] for x in r if x[8] == "1"]
            after = int(r[-1][0]); time.sleep(0.9)
        rows = [x for x in rows if x[0] >= start][::-1]
        df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 7]]
        df.columns = ["t", "open", "high", "low", "close", "qv"]
        df.to_csv(path, index=False)
        return load_csv(path)
    base, ep, lim = (SPOT, "/api/v3/klines", 1000) if SOURCE == "spot" else (FAPI, "/fapi/v1/klines", 1500)
    while start < end:
        r = http_json(f"{base}{ep}?symbol={sym}&interval=1h&limit={lim}&startTime={start}")
        if not r:
            break
        rows += r
        start = r[-1][0] + 3600_000
        if len(r) < lim:
            break
        time.sleep(0.05)
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 7]]
    df.columns = ["t", "open", "high", "low", "close", "qv"]
    df = df.iloc[:-1]  # drop the still-forming bar
    df.to_csv(path, index=False)
    return load_csv(path)


def load_csv(path):
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    df = df.astype(float)
    df["qv"] *= QV_SCALE
    return df


def synthetic(n_sym=40, days=120, seed=7):
    """Random walks with injected ignition events. Only for testing the engine."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days * 24, freq="1h", tz="UTC")
    def walk(drift_sd, vol_scale, n=len(idx)):
        r = rng.normal(0, drift_sd, n)
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, drift_sd / 2, n)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, drift_sd / 2, n)))
        qv = vol_scale * rng.lognormal(0, 0.4, n)
        return pd.DataFrame(dict(open=o, high=hi, low=lo, close=c, qv=qv), index=idx), r
    btc, _ = walk(0.004, 5e8)
    out = {}
    for s in range(n_sym):
        d, r = walk(0.008, 2e6)
        for _ in range(6):
            j = int(rng.integers(100, len(idx) - 80))
            cont = rng.random() < 0.35
            r2 = r.copy()
            r2[j] = 0.06
            r2[j + 1:j + 40] = (0.006 if cont else -0.004) + rng.normal(0, 0.006, 39)
            c = 100 * np.exp(np.cumsum(r2)); o = np.r_[c[0], c[:-1]]
            d["close"], d["open"] = c, o
            d["high"] = np.maximum(o, c) * 1.002; d["low"] = np.minimum(o, c) * 0.998
            d.iloc[j, d.columns.get_loc("qv")] *= 14
            d.iloc[j + 1, d.columns.get_loc("qv")] *= 6
            r = r2
        out[f"SYN{s}USDT"] = d
    return btc, out


# ----------------------------------------------------------------------------- indicators
def prep(d):
    d = d.copy()
    d["hh48"] = d.high.rolling(48).max().shift(1)
    d["ll48"] = d.low.rolling(48).min().shift(1)
    d["base"] = d.qv.rolling(48).median().shift(1)
    d["mult"] = d.qv / d.base
    d["qv24"] = d.qv.rolling(24).sum()
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["ret7d"] = d.close / d.close.shift(168) - 1
    return d


QV24_MIN = 20e6
def events(d, min_mult=5.0, min_qv24=None):
    min_qv24 = QV24_MIN if min_qv24 is None else min_qv24
    m = (d.mult >= min_mult) & (d.close > d.hh48) & (d.qv24 >= min_qv24)
    return np.flatnonzero(m.fillna(False).to_numpy())


# ----------------------------------------------------------------------------- BTC gate
class BtcGate:
    def __init__(self, btc):
        self.btc = btc
        h4 = btc.resample("4h", label="left", closed="left").agg(
            dict(open="first", high="max", low="min", close="last")).dropna()
        self.h4 = h4
        self.above = (h4.close > h4.close.ewm(span=20, adjust=False).mean()).to_numpy()
        self.end = h4.index + pd.Timedelta("4h")

    def state(self, t):
        """t = close time of signal bar. Returns (consecutive 4H closes above EMA20, forming bar making new low)."""
        m = int(self.end.searchsorted(t, side="right"))
        if m < 25:
            return None
        cnt = 0
        for q in range(m - 1, max(m - 4, -1), -1):
            if not self.above[q]:
                break
            cnt += 1
        f = self.btc.loc[(self.btc.index >= self.end[m - 1]) & (self.btc.index < t)]
        newlow = len(f) > 0 and f.low.min() < self.h4.low.iloc[m - 6:m].min()
        return cnt, bool(newlow)


# ----------------------------------------------------------------------------- candidates
@dataclass
class Cand:
    sym: str; ev: int; i: int; t: pd.Timestamp; entry: float; level: float; stop: float
    tier: str; score: float; mult: float; base_size: float = 0.0


def _confirm(d, j, k, lim):
    L = d.hh48.iloc[j]
    return d.close.iloc[k] > L and d.qv.iloc[k] >= 0.5 * d.qv.iloc[j] and d.close.iloc[k] / L - 1 <= lim


def cands_v1(sym, d):
    """Group's current rules, literally. Stop is not specified in the spec: assumed = lowest low of ignition/confirm bars."""
    out, last = [], -999
    n = len(d)
    for j in events(d):
        if j - last < 48 or j + 1 >= n:
            continue
        L = d.hh48.iloc[j]; c = d.close.iloc[j]; k = None
        if d.mult.iloc[j] >= 10 and d.qv.iloc[j] > 5e6 and c / L - 1 <= 0.06:
            k = j
        else:
            k1 = j + 1
            consec = (d.mult.iloc[k1] >= 5 and d.close.iloc[k1] > L and d.qv.iloc[j] + d.qv.iloc[k1] > 10e6
                      and d.close.iloc[k1] / L - 1 <= 0.12)
            if consec or _confirm(d, j, k1, 0.08):
                k = k1
        if k is None:
            continue
        stop = d.low.iloc[j:k + 1].min()
        e = d.close.iloc[k]
        if not (0.01 <= e / stop - 1 <= 0.25):
            continue
        out.append(Cand(sym, j, k, d.index[k] + H1, e, L, stop, "v1", d.mult.iloc[j], d.mult.iloc[j]))
    return out


def score_v2(d, j):
    L = d.hh48.iloc[j]; c = d.close.iloc[j]; o = d.open.iloc[j]
    rng = d.high.iloc[j] - d.low.iloc[j]
    s = 2 if d.mult.iloc[j] >= 10 else 1                                  # volume expansion
    s += int(rng > 0 and (c - d.low.iloc[j]) / rng >= 0.75 and c > o)    # closes near high, no fat upper wick
    s += int(c / L - 1 <= 0.03)                                           # not extended past the level
    s += int((L - d.ll48.iloc[j]) / d.ll48.iloc[j] <= 0.20)               # tight 48h base (compression -> expansion)
    s += int(d.qv24.iloc[j] >= 50e6)                                      # liquidity / slippage
    s += int(not (d.ret7d.iloc[j] > 0.5))                                 # not already parabolic (hypothesis)
    return s


def cands_v2(sym, d, tA=6, tB=4, tC=3):
    out, last = [], -999
    n = len(d)
    for j in events(d):
        if j - last < 48:
            continue
        s = score_v2(d, j)
        if s < tC:
            continue
        tier = "A" if s >= tA else "B" if s >= tB else "C"
        L = d.hh48.iloc[j]; c = d.close.iloc[j]
        if tier == "A" and c / L - 1 <= 0.08:
            k = j
        else:
            k1 = j + 1
            if k1 >= n:
                continue
            consec = d.mult.iloc[k1] >= 5 and d.close.iloc[k1] > L
            if not _confirm(d, j, k1, 0.12 if consec else 0.08):
                continue
            k = k1
            if consec:  # persistent volume = conviction
                tier = {"C": "B", "B": "A", "A": "A"}[tier]
        e = d.close.iloc[k]
        a = d.atr.iloc[k]
        stop = max(d.low.iloc[j:k + 1].min(), L - a)   # never wider than 1 ATR under the level
        stop = min(stop, e * 0.98)                      # but at least 2% away (avoid noise stops)
        if e / stop - 1 > 0.08:
            continue                                    # too extended / too wide: skip
        out.append(Cand(sym, j, k, d.index[k] + H1, e, L, stop, tier, s, d.mult.iloc[j]))
    return out


def size_v1(g):
    if g is None or g[1] or g[0] == 0: return 0.0
    return 0.25 if g[0] == 1 else 0.5


V2_TIER = {"A": 1.0, "B": 0.5, "C": 0.25}
def size_v2(g, tier):
    if g is None or g[1] or g[0] == 0: return 0.0
    return V2_TIER[tier] * (0.5 if g[0] == 1 else 1.0)


def cands_v3(sym, d):
    """v3: immediate entry at ignition close; quality tag from the 2 features that held up in analysis.py."""
    out, last = [], -999
    for j in events(d):
        if j - last < 48: continue
        L = d.hh48.iloc[j]; e = d.close.iloc[j]; a = d.atr.iloc[j]
        if e / L - 1 > 0.08: continue
        stop = min(max(d.low.iloc[j], L - a), e * 0.98)
        if e / stop - 1 > 0.08: continue
        last = j
        q = (L - d.ll48.iloc[j]) / d.ll48.iloc[j] <= 0.13 and d.ret7d.iloc[j] <= 0.12
        out.append(Cand(sym, j, j, d.index[j] + H1, e, L, stop, "Q" if q else "N", float(q), d.mult.iloc[j]))
    return out


HEAT_MIN = 6
def size_v3(g, heat, tier, tilt):
    if g is None or g[1] or g[0] < 3 or heat < HEAT_MIN: return 0.0   # BTC held >=3 closes above EMA20, no new-low, market not quiet
    return 1.0 if (tilt and tier == "Q") else 0.5


# ----------------------------------------------------------------------------- trade simulation
@dataclass
class Params:
    tp_r: float = 1.5; tp_frac: float = 1 / 3
    trail: str = "fixed8"          # "fixed8" | "atr"
    fail_bars: int = 0             # exit if a close is back under the level within N bars
    time_bars: int = 0             # exit if < 0.5R of progress after N bars
    fee: float = 0.0005; slip: float = 0.001
    cap: float = 1.0               # max concurrent open risk, in R


P_V1 = Params()
P_V2 = Params(trail="atr", fail_bars=3, time_bars=12, cap=2.0)


@dataclass
class Trade:
    sym: str; tier: str; size: float; t0: pd.Timestamp; t1: pd.Timestamp; tbe: pd.Timestamp
    R: float; bars: int; ev: int


def simulate(d, c, P):
    e = c.entry * (1 + P.slip); stop = c.stop
    sd = e / stop - 1
    tp1 = e * (1 + P.tp_r * sd)
    w = 0.08 if P.trail == "fixed8" else float(np.clip(3 * d.atr.iloc[c.i] / e, 0.05, 0.10))
    O, Hh, Ll, Cc = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    idx = d.index
    pnl, rem, done, hi, mfe, tbe = 0.0, 1.0, False, e, 0.0, None
    k_exit = len(d) - 1; px_exit = Cc[-1]
    for k in range(c.i + 1, len(d)):
        age = k - c.i; tclose = idx[k] + H1
        if not done:
            if O[k] <= stop: px_exit, k_exit = O[k], k; break
            if Ll[k] <= stop: px_exit, k_exit = stop, k; break
            if Hh[k] >= tp1:
                pnl += P.tp_frac * (tp1 * (1 - P.slip) / e - 1); rem -= P.tp_frac
                done, stop, hi, tbe = True, e, Hh[k], tclose
                continue
            mfe = max(mfe, Hh[k] / e - 1)
            if P.fail_bars and age <= P.fail_bars and Cc[k] < c.level: px_exit, k_exit = Cc[k], k; break
            if P.time_bars and age == P.time_bars and mfe < 0.5 * sd: px_exit, k_exit = Cc[k], k; break
        else:
            eff = max(e, hi * (1 - w))
            if O[k] <= eff: px_exit, k_exit = O[k], k; break
            if Ll[k] <= eff: px_exit, k_exit = eff, k; break
            hi = max(hi, Hh[k])
    pnl += rem * (px_exit * (1 - P.slip) / e - 1) - 2 * P.fee
    t1 = idx[k_exit] + H1
    return Trade(c.sym, c.tier, c.base_size, c.t, t1, tbe or t1, pnl / sd, k_exit - c.i, c.ev)


# ----------------------------------------------------------------------------- portfolio + stats
def run_strategy(name, data, gate, r_pct):
    P = P_V1 if name == "v1" else P_V2
    pre = {s: prep(d) for s, d in data.items()}
    ev_t = np.sort(np.array([(d.index[j] + H1).value for d in pre.values() for j in events(d, min_qv24=20e6)], dtype="int64"))
    day = pd.Timedelta("24h").value
    raw = []
    for s, d in pre.items():
        cs = cands_v1(s, d) if name == "v1" else cands_v3(s, d) if name.startswith("v3") else cands_v2(s, d)
        for c in cs:
            g = gate.state(c.t)
            if name.startswith("v3"):
                heat = int(np.searchsorted(ev_t, c.t.value, "left") - np.searchsorted(ev_t, c.t.value - day, "left"))
                c.base_size = size_v3(g, heat, c.tier, name == "v3tilt")
            else:
                c.base_size = size_v1(g) if name == "v1" else size_v2(g, c.tier)
            raw.append((c, d))
    sig_all = len(raw)
    live = sorted([x for x in raw if x[0].base_size > 0], key=lambda x: (x[0].t, -x[0].score))
    taken, open_ = [], []
    for c, d in live:
        open_ = [t for t in open_ if t.tbe > c.t]               # risk is released once stop is at breakeven
        if any(t.sym == c.sym and t.t1 > c.t for t in taken): continue
        if sum(t.size for t in open_) + c.base_size > P.cap + 1e-9: continue
        tr = simulate(d, c, P)
        taken.append(tr); open_.append(tr)
    return taken, sig_all, [c for c, _ in raw], pre


def equity(trades, r_pct):
    eq, peak, dd, curve = 1.0, 1.0, 0.0, []
    for t in sorted(trades, key=lambda t: t.t1):
        eq *= 1 + t.R * t.size * r_pct
        peak = max(peak, eq); dd = max(dd, 1 - eq / peak); curve.append(eq)
    return eq, dd


def summarize(trades, r_pct):
    if not trades: return dict(n=0)
    R = np.array([t.R for t in trades]); W = np.array([t.R * t.size for t in trades])
    eq, dd = equity(trades, r_pct)
    return dict(n=len(R), win=(R > 0).mean(), avgR=R.mean(), avgWinR=R[R > 0].mean() if (R > 0).any() else 0,
                avgLossR=R[R <= 0].mean() if (R <= 0).any() else 0, sumRw=W.sum(),
                pf=W[W > 0].sum() / max(-W[W < 0].sum(), 1e-9), ret=eq - 1, mdd=dd)


def recall(pre, cands, hold=72, thr=0.20):
    """Of ignition events that went on to run >= thr within `hold` bars, how many did the strategy produce a signal for?"""
    got = {(c.sym, c.ev) for c in cands}
    win = hit = 0
    for s, d in pre.items():
        for j in events(d):
            if j + hold >= len(d): continue
            if d.high.iloc[j + 1:j + 1 + hold].max() / d.close.iloc[j] - 1 >= thr:
                win += 1; hit += (s, j) in got
    return hit, win


def fmt(label, s):
    if s["n"] == 0: return f"{label:<22} no trades"
    return (f"{label:<22} n={s['n']:>4}  win={s['win']:.0%}  avgR={s['avgR']:+.2f}  avgWin={s['avgWinR']:+.2f}  "
            f"avgLoss={s['avgLossR']:+.2f}  PF={s['pf']:.2f}  ret={s['ret']:+.1%}  maxDD={s['mdd']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120); ap.add_argument("--top", type=int)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-qv24", default="20", help="comma list in $M, e.g. 20,5"); ap.add_argument("--strategies", default="v1,v2,v3flat,v3tilt")
    ap.add_argument("--source", default="perp", choices=["perp", "spot", "okx"]); ap.add_argument("--qv-scale", type=float, default=1.0)
    ap.add_argument("--r-pct", type=float, default=0.01, help="1R as fraction of equity")
    a = ap.parse_args()
    global SOURCE, QV_SCALE
    SOURCE, QV_SCALE = a.source, a.qv_scale
    if a.synthetic:
        print("*** SYNTHETIC DATA: validates the engine only. Numbers say NOTHING about real markets. ***")
        btc, data = synthetic(days=a.days)
    else:
        btcsym = "BTC-USDT-SWAP" if SOURCE == "okx" else "BTCUSDT"
        btc = fetch_klines(btcsym, a.days)
        data = {}
        from concurrent.futures import ThreadPoolExecutor
        syms = [s for s in universe(min_qv=4e6, top=a.top) if s != btcsym]
        def one(s):
            try: return s, fetch_klines(s, a.days)
            except Exception as ex: print("skip", s, ex, file=sys.stderr); return s, None
        with ThreadPoolExecutor(8) as ex:
            for s, d in ex.map(one, syms):
                if d is not None and len(d) > 300: data[s] = d
        print(f"loaded {len(data)} symbols", flush=True)
    gate = BtcGate(btc)
    res = {}
    for qm in [float(x) * 1e6 for x in a.min_qv24.split(",")]:
      global QV24_MIN
      QV24_MIN = qm
      print(f"\n########## entry liquidity threshold: 24h turnover >= ${qm / 1e6:.0f}M (scaled) ##########")
      for name in a.strategies.split(","):
          trades, nsig, cands, pre = run_strategy(name, data, gate, a.r_pct)
          res[name] = trades
          h, w = recall(pre, cands)
          print(f"\n===== {name} | raw signals {nsig} | winner-event recall {h}/{w} ({h / max(w, 1):.0%}) =====")
          print(fmt("ALL", summarize(trades, a.r_pct)))
          if trades:
              tmid = sorted(t.t0 for t in trades)[len(trades) // 2]
              print(fmt("first half", summarize([t for t in trades if t.t0 < tmid], a.r_pct)))
              print(fmt("second half (OOS)", summarize([t for t in trades if t.t0 >= tmid], a.r_pct)))
          if name.startswith("v3"):
              for tier in "QN":
                  print(fmt(f"tag {tier}", summarize([t for t in trades if t.tier == tier], a.r_pct)))
          if name == "v2":
              for tier in "ABC":
                  print(fmt(f"tier {tier}", summarize([t for t in trades if t.tier == tier], a.r_pct)))
    print("\nDecision rule: keep tier sizing only if avgR(A) > avgR(B) > avgR(C) on BOTH halves; else flatten sizes.")


if __name__ == "__main__":
    main()
