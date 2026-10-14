#!/usr/bin/env python3
"""Ignition-breakout long strategies (v1 original, v3 flat) on REAL Binance USDT-perp data, 2023-2026, point-in-time liquidity."""
import glob, sys, numpy as np, pandas as pd
import backtest as bt

data = {}
for p in glob.glob("data_vis/k_*.pkl"):
    s = p.split("k_")[1][:-4]; k, _ = pd.read_pickle(p)
    k = k[["open", "high", "low", "close", "qv"]].copy()
    if (k.qv.rolling(24).sum() >= 20e6).any() and len(k) > 400: data[s] = k
btc = data.pop("BTCUSDT")
print("symbols with any liquid period:", len(data), "| BTC bars", len(btc), btc.index[0], "->", btc.index[-1])
gate = bt.BtcGate(btc)
for name in sys.argv[1].split(","):
    trades, nsig, cands, pre = bt.run_strategy(name, data, gate, 0.01)
    print(f"\n===== {name} | raw signals {nsig} | trades {len(trades)} =====")
    print(bt.fmt("ALL", bt.summarize(trades, 0.01)))
    for y in sorted({t.t0.year for t in trades}):
        print(bt.fmt(str(y), bt.summarize([t for t in trades if t.t0.year == y], 0.01)))
    if name.startswith("v3"):
        for tg in "QN": print(bt.fmt(f"tag {tg}", bt.summarize([t for t in trades if t.tier == tg], 0.01)))
