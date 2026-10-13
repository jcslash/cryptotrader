#!/usr/bin/env python3
"""Hourly live scanner (v3): evaluates the latest CLOSED 1H bar and prints/pushes every actionable card.

  python3 scan.py --equity 20000 [--source spot|perp]   # spot = Binance public mirror (works from restricted regions)
  TG_TOKEN=... TG_CHAT=... python3 scan.py --equity 20000   # also push to Telegram
Cron (minute 1 of every hour):  1 * * * * cd /path/cryptotrader && python3 scan.py --equity 20000 >> scan.log 2>&1
Every signal (DO / DON'T) is appended to signals.csv with the reason.
"""
import argparse, csv, os, sys, shutil, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import backtest as bt


def card(c, equity, r_pct, size, heat):
    risk = equity * r_pct * size
    sd = c.entry / c.stop - 1
    e = c.entry
    tag = "品質標記Q" if c.tier == "Q" else "一般"
    return (f"✅ 做多 {c.sym}  [{tag}]\n"
            f"量 {c.mult:.1f}x · 突破位 {c.level:.6g} · 現價 {e:.6g} ({e / c.level - 1:+.1%}) · 市場熱度 {heat}\n"
            f"進場區間 {e:.6g} ~ {e * 1.005:.6g}（高於上緣不追）\n"
            f"停損 {c.stop:.6g} (-{sd:.1%})  |  風險 {size:.2f}R = ${risk:,.0f}\n"
            f"名目 = ${risk:,.0f} ÷ {sd:.2%} = ${risk / sd:,.0f}\n"
            f"+1.5R @ {e * (1 + 1.5 * sd):.6g} 出1/3 → 停損移到成本；其餘 ATR 移動止損（回落 5~10%）\n"
            f"出場：3 根內收回 {c.level:.6g} 下方→市價出；12 根沒走出 +0.5R→出")


def push(msg):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=15)


def log(rows):
    new = not os.path.exists("signals.csv")
    with open("signals.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["ts", "sym", "action", "tag", "reason"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, required=True); ap.add_argument("--r-pct", type=float, default=0.01)
    ap.add_argument("--top", type=int, default=200); ap.add_argument("--source", default="spot", choices=["spot", "perp"])
    ap.add_argument("--qv-scale", type=float, default=3.0)
    a = ap.parse_args()
    bt.SOURCE, bt.QV_SCALE = a.source, (a.qv_scale if a.source == "spot" else 1.0)
    shutil.rmtree("data_live", ignore_errors=True)          # always fresh bars
    btc = bt.fetch_klines("BTCUSDT", 20, cache="data_live")
    gate = bt.BtcGate(btc)
    g = gate.state(btc.index[-1] + bt.H1)
    syms = [s for s in bt.universe(top=a.top) if s != "BTCUSDT"]
    def one(s):
        try: return s, bt.prep(bt.fetch_klines(s, 20, cache="data_live"))
        except Exception as ex: print("skip", s, ex, file=sys.stderr); return s, None
    with ThreadPoolExecutor(8) as ex:
        data = {s: d for s, d in ex.map(one, syms) if d is not None and len(d) > 250}
    t_now = btc.index[-1] + bt.H1
    # market heat = ignition bars across the universe in the 24h before the latest close
    heat = sum(int(((d.index + bt.H1 > t_now - pd.Timedelta("24h")) & (d.index + bt.H1 < t_now))[bt.events(d)].sum())
               for d in data.values())
    gtxt = (f"{t_now:%Y-%m-%d %H:%M} UTC | BTC: 連續{g[0]}根4H收在EMA20上, 4H創新低={g[1]} | 市場熱度(24h點火數)={heat}"
            if g else "BTC資料不足")
    why = None
    if g is None: why = "BTC資料不足"
    elif g[1]: why = "BTC 4H 正在創新低"
    elif g[0] < 3: why = f"BTC 僅連續 {g[0]} 根 4H 收在 EMA20 上（需 ≥3）"
    elif heat < bt.HEAT_MIN: why = f"市場冷清（熱度 {heat} < {bt.HEAT_MIN}）"
    rows, cards = [], []
    for s, d in data.items():
        if d.index[-1] + bt.H1 != t_now: continue
        for c in bt.cands_v3(s, d):
            if c.t != t_now: continue
            if why: rows.append([str(t_now), s, "DONT", c.tier, why]); continue
            cards.append((c.mult, c, card(c, a.equity, a.r_pct, 0.5, heat)))
            rows.append([str(t_now), s, "DO", c.tier, ""])
    cards.sort(key=lambda x: -x[0])
    keep = cards[:4]                                          # total open risk cap 2R at 0.5R each
    for _, c, _ in cards[4:]:
        rows.append([str(t_now), c.sym, "DONT", c.tier, "風險上限2R，已有更強的"])
    msg = gtxt + "\n" + ("\n\n".join(x[2] for x in keep) if keep else ("本小時不發卡：" + why if why else "本小時無符合條件的訊號"))
    print(msg); log(rows)
    if keep: push(msg)


if __name__ == "__main__":
    main()
