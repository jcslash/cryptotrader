"""Binance public data archive (data.binance.vision) loader for USD-M perpetuals. Free, no key. Includes delisted symbols."""
import io, os, re, time, zipfile, urllib.request, urllib.parse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

HOST = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UA = {"User-Agent": "Mozilla/5.0"}
CACHE = "data_vis"


def _get(url, retries=4):
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(1.5 * (a + 1))
        except Exception:
            time.sleep(1.5 * (a + 1))
    return None


def list_prefixes(prefix):
    out, marker = [], ""
    while True:
        u = f"{S3}?delimiter=/&prefix={urllib.parse.quote(prefix)}&max-keys=1000" + (f"&marker={urllib.parse.quote(marker)}" if marker else "")
        x = _get(u)
        if not x: break
        t = x.decode()
        ps = re.findall(r"<Prefix>([^<]*)</Prefix>", t)
        ps = [p for p in ps if p != prefix]
        out += ps
        if "<IsTruncated>true</IsTruncated>" in t:
            nm = re.findall(r"<NextMarker>([^<]*)</NextMarker>", t)
            marker = nm[0] if nm else ps[-1]
        else:
            break
    return out


def usdt_perp_symbols():
    ps = list_prefixes("data/futures/um/monthly/klines/")
    return sorted(p.rstrip("/").split("/")[-1] for p in ps if re.fullmatch(r"[A-Z0-9]+USDT", p.rstrip("/").split("/")[-1]))


def _csv(url, names=None):
    b = _get(url)
    if b is None: return None
    try:
        z = zipfile.ZipFile(io.BytesIO(b))
        raw = z.read(z.namelist()[0])
        return pd.read_csv(io.BytesIO(raw), header=0 if raw[:1].isalpha() else None, names=None if raw[:1].isalpha() else names)
    except Exception:
        return None


KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "n", "tbv", "tbqv", "ignore"]


def months(start, end):
    return [f"{p.year}-{p.month:02d}" for p in pd.period_range(start, end, freq="M")]


def klines_1h(sym, start="2023-01", end=None):
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m")
    ms = months(start, end)
    cur = ms[-1]
    parts = []
    for m in ms[:-1]:
        d = _csv(f"{HOST}/data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{m}.zip", KCOLS)
        if d is not None: parts.append(d)
    # current month: daily files
    days = pd.date_range(f"{cur}-01", pd.Timestamp.utcnow().tz_localize(None).normalize(), freq="D")
    for dt in days[:-1]:
        d = _csv(f"{HOST}/data/futures/um/daily/klines/{sym}/1h/{sym}-1h-{dt:%Y-%m-%d}.zip", KCOLS)
        if d is not None: parts.append(d)
    if not parts: return None
    parts = [p.set_axis(KCOLS, axis=1) for p in parts]          # files with / without header row -> same names
    df = pd.concat(parts)
    df = df[pd.to_numeric(df.open_time, errors="coerce").notna()].astype({"open_time": "int64"})
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df[["open", "high", "low", "close", "qv", "n", "tbqv"]].astype(float).sort_index().loc[lambda x: ~x.index.duplicated()]


def funding(sym, start="2023-01", end=None):
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m")
    parts = [d for m in months(start, end) if (d := _csv(f"{HOST}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")) is not None]
    if not parts: return None
    df = pd.concat(parts)
    df.index = pd.to_datetime(df.calc_time, unit="ms", utc=True)
    return df[["last_funding_rate"]].astype(float).sort_index()


def metrics_days(sym, days):
    def one(dt):
        return _csv(f"{HOST}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{dt:%Y-%m-%d}.zip")
    parts = [d for d in map(one, days) if d is not None]
    if not parts: return None
    df = pd.concat(parts)
    df.index = pd.to_datetime(df.create_time, utc=True)
    return df.drop(columns=["create_time", "symbol"]).astype(float).sort_index()
