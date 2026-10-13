#!/usr/bin/env python3
"""Hourly live scanner: evaluates v2 on the latest CLOSED 1H bar and PUSHES every actionable card (no need to ask).

  python3 scan.py --equity 20000                 # print cards
  TG_TOKEN=... TG_CHAT=... python3 scan.py --equity 20000   # also push to Telegram
Run at minute 01 of every hour:  1 * * * * cd /path/cryptotrader && python3 scan.py --equity 20000 >> scan.log 2>&1
Every signal (DO / DON'T / WATCH) is appended to signals.csv - discipline rule: log all, review every 20.
"""
import argparse, csv, os, sys, time, urllib.parse, urllib.request
import pandas as pd
import backtest as bt

TIER_R = bt.V2_TIER


def card(c, g, equity, r_pct):
    size = bt.size_v2(g, c.tier)
    risk = equity * r_pct * size
    sd = c.entry / c.stop - 1
    notional = risk / sd
    e = c.entry
    return dict(sym=c.sym, tier=c.tier, score=c.score, risk=risk, notional=notional, sd=sd, size=size,
                text=(f"✅ 做多 {c.sym}  [{c.tier}級 分數{c.score:.0f}/7]\n"
                      f"量 {c.mult:.1f}x · 突破位 {c.level:.6g} · 現價 {e:.6g} ({e / c.level - 1:+.1%})\n"
                      f"進場區間 {e:.6g} ~ {e * 1.005:.6g}（超過上緣不追）\n"
                      f"停損 {c.stop:.6g} (-{sd:.1%})  |  R={size:.2f}R = ${risk:,.0f}\n"
                      f"名目 = ${risk:,.0f} ÷ {sd:.2%} = ${notional:,.0f}\n"
                      f"+1.5R @ {e * (1 + 1.5 * sd):.6g} 出1/3，停損移到成本；其餘 ATR 移動止損（5~10%）\n"
                      f"3 根內收回 {c.level:.6g} 下方→市價出；12 根沒走出 0.5R→出"))


def push(msg):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if not (tok and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=15)


def log(rows):
    new = not os.path.exists("signals.csv")
    with open("signals.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["ts", "sym", "action", "tier", "score", "reason"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, required=True); ap.add_argument("--r-pct", type=float, default=0.01)
    ap.add_argument("--top", type=int, default=200)
    a = ap.parse_args()
    now = pd.Timestamp.utcnow().floor("1h")
    # live: always bypass cache
    import shutil; shutil.rmtree("data_live", ignore_errors=True)
    btc = bt.fetch_klines("BTCUSDT", 15, cache="data_live")
    gate = bt.BtcGate(btc)
    g = gate.state(btc.index[-1] + bt.H1)
    gtxt = f"BTC闸门: 連續{g[0]}根4H收在EMA20上, 4H創新低={g[1]}" if g else "BTC資料不足"
    rows, cards = [], []
    for s in bt.universe(top=a.top):
        if s == "BTCUSDT": continue
        try: d = bt.prep(bt.fetch_klines(s, 15, cache="data_live"))
        except Exception as ex: print("skip", s, ex, file=sys.stderr); continue
        last_t = d.index[-1] + bt.H1
        for c in bt.cands_v2(s, d):
            if c.t != last_t:        # only signals completing on the latest closed bar
                continue
            if bt.size_v2(g, c.tier) == 0:
                rows.append([str(now), s, "DONT", c.tier, c.score, "BTC gate closed"]); continue
            cd = card(c, g, a.equity, a.r_pct)
            cards.append(cd); rows.append([str(now), s, "DO", c.tier, c.score, ""])
        # B/C ignition on the latest bar waiting for the confirm bar
        for j in bt.events(d):
            if j == len(d) - 1 and bt.score_v2(d, j) >= 3 and not any(r[1] == s for r in rows):
                msg = f"👀 WATCH {s} 點火棒 量{d.mult.iloc[j]:.1f}x 分數{bt.score_v2(d, j)}，下一根收在 {d.hh48.iloc[j]:.6g} 上且量≥一半才進"
                cards.append(dict(text=msg, score=bt.score_v2(d, j) - 10, risk=0))
                rows.append([str(now), s, "WATCH", "", bt.score_v2(d, j), "await confirm"])
    cards.sort(key=lambda x: -x["score"])
    cap = 2.0  # total open risk cap (R); one wave -> best first
    used = 0.0
    out = [gtxt]
    for cd in cards:
        r = cd.get("size", 0)
        if r and used + r > cap:
            rows.append([str(now), cd.get("sym", "?"), "DONT", cd.get("tier", ""), cd["score"], "risk cap"]); continue
        used += r; out.append(cd["text"])
    msg = "\n\n".join(out) if len(out) > 1 else gtxt + "\n本小時無可做訊號"
    print(msg); log(rows)
    if len(out) > 1: push(msg)


if __name__ == "__main__":
    main()
