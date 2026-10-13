#!/usr/bin/env python3
"""Minimal trade journal: log what you ACTUALLY did with each alert, get R-based stats.

  python3 journal.py open  SOLUSDT --entry 142.5 --stop 138.2 --risk 100 --tier A
  python3 journal.py close 1 --exit 150.3 --frac 0.33      # partial exit (1/3)
  python3 journal.py close 1 --exit 158.0                  # rest
  python3 journal.py skip  WIFUSDT --note "alert said DO, I was afraid / too late"
  python3 journal.py report
R = (exit/entry - 1) / ((entry - stop)/entry), weighted by fraction closed. Fees/slippage not included - add via --note.
"""
import argparse, csv, os, datetime as dt, collections

F = "trades.csv"
COLS = ["id", "ts", "sym", "kind", "price", "frac", "stop", "risk", "tier", "note"]


def rows():
    if not os.path.exists(F): return []
    return list(csv.DictReader(open(F)))


def add(r):
    new = not os.path.exists(F)
    with open(F, "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new: w.writeheader()
        w.writerow({c: r.get(c, "") for c in COLS})


def next_id():
    return max([int(r["id"]) for r in rows() if r["kind"] == "open"] + [0]) + 1


def trades():
    T = {}
    for r in rows():
        if r["kind"] == "open":
            T[r["id"]] = dict(sym=r["sym"], e=float(r["price"]), s=float(r["stop"]), risk=float(r["risk"] or 0),
                              tier=r["tier"] or "?", closed=0.0, pnl=0.0)
        elif r["kind"] == "close" and r["id"] in T:
            t = T[r["id"]]; fr = float(r["frac"])
            t["pnl"] += fr * (float(r["price"]) / t["e"] - 1); t["closed"] += fr
    out = []
    for i, t in T.items():
        if t["closed"] >= 0.999:
            out.append(dict(id=i, sym=t["sym"], tier=t["tier"], R=t["pnl"] / ((t["e"] - t["s"]) / t["e"]), usd=t["pnl"] * t["risk"] / ((t["e"] - t["s"]) / t["e"])))
    return out


def stat(label, L):
    if not L: return f"{label:<8} 無"
    R = [x["R"] for x in L]; w = [r for r in R if r > 0]; l = [r for r in R if r <= 0]
    pf = sum(w) / -sum(l) if l and sum(l) else float("inf")
    return (f"{label:<8} n={len(R):>3} 勝率={len(w) / len(R):.0%} 平均R={sum(R) / len(R):+.2f} "
            f"均贏={sum(w) / len(w) if w else 0:+.2f} 均輸={sum(l) / len(l) if l else 0:+.2f} 總R={sum(R):+.2f} PF={pf:.2f}")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("open"); o.add_argument("sym"); o.add_argument("--entry", type=float, required=True)
    o.add_argument("--stop", type=float, required=True); o.add_argument("--risk", type=float, required=True)
    o.add_argument("--tier", default="?"); o.add_argument("--note", default="")
    c = sp.add_parser("close"); c.add_argument("id"); c.add_argument("--exit", type=float, required=True)
    c.add_argument("--frac", type=float, default=None); c.add_argument("--note", default="")
    k = sp.add_parser("skip"); k.add_argument("sym"); k.add_argument("--note", default="")
    sp.add_parser("report")
    a = ap.parse_args(); now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    if a.cmd == "open":
        assert a.stop < a.entry, "做多：停損必須低於進場價"
        i = next_id(); add(dict(id=i, ts=now, sym=a.sym, kind="open", price=a.entry, stop=a.stop, risk=a.risk, tier=a.tier, note=a.note))
        print(f"記錄 #{i}  名目 ≈ ${a.risk / ((a.entry - a.stop) / a.entry):,.0f}")
    elif a.cmd == "close":
        left = 1 - sum(float(r["frac"]) for r in rows() if r["kind"] == "close" and r["id"] == a.id)
        fr = a.frac if a.frac is not None else left
        assert 0 < fr <= left + 1e-9, f"剩餘只有 {left:.2f}"
        add(dict(id=a.id, ts=now, kind="close", price=a.exit, frac=fr, note=a.note)); print(f"#{a.id} 已出 {fr:.2f}，剩 {left - fr:.2f}")
    elif a.cmd == "skip":
        add(dict(id=0, ts=now, sym=a.sym, kind="skip", note=a.note)); print("已記錄未做")
    else:
        T = trades(); sk = [r for r in rows() if r["kind"] == "skip"]
        print(stat("全部", T))
        for t in sorted({x["tier"] for x in T}): print(stat(f"級別{t}", [x for x in T if x["tier"] == t]))
        print(f"未做的報單: {len(sk)}")
        if len(T) < 20: print(f"⚠ 僅 {len(T)} 筆，未滿 20，不改規則。")


if __name__ == "__main__":
    main()
