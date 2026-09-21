"""Free market-data access for NIFTY 50 with layered fallbacks.

Sources, in the order they are attempted:
  1. Yahoo Finance chart API  -- no key, generous limits, OHLCV history.
  2. Stooq CSV endpoint       -- no key, daily OHLC, used if Yahoo is blocked.
  3. Repo-side cache          -- state/ohlc_cache.json, written on every run so
                                 the pipeline keeps working through an outage
                                 and slowly accumulates its own dataset.

The NSE option chain is fetched on a strictly best-effort basis: NSE blocks
most datacenter IPs, so a failure there degrades the signal (OI-based
strategies abstain) rather than breaking the run.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

IST = timezone(timedelta(hours=5, minutes=30))

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"
NSE_HOME = "https://www.nseindia.com"
NSE_OPTION_CHAIN = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class Candles:
    """Daily OHLCV series ordered oldest -> newest."""

    dates: List[str] = field(default_factory=list)
    open: List[float] = field(default_factory=list)
    high: List[float] = field(default_factory=list)
    low: List[float] = field(default_factory=list)
    close: List[float] = field(default_factory=list)
    volume: List[float] = field(default_factory=list)
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.close)

    def to_dict(self) -> Dict:
        return {
            "dates": self.dates,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "Candles":
        return cls(
            dates=list(payload.get("dates", [])),
            open=[float(x) for x in payload.get("open", [])],
            high=[float(x) for x in payload.get("high", [])],
            low=[float(x) for x in payload.get("low", [])],
            close=[float(x) for x in payload.get("close", [])],
            volume=[float(x) for x in payload.get("volume", [])],
            source=payload.get("source", "cache"),
        )

    def merge(self, other: "Candles") -> "Candles":
        """Union of two series keyed by date; `other` wins on conflict."""
        rows: Dict[str, tuple] = {}
        for series in (self, other):
            for i, day in enumerate(series.dates):
                rows[day] = (
                    series.open[i],
                    series.high[i],
                    series.low[i],
                    series.close[i],
                    series.volume[i],
                )
        merged = Candles(source=other.source or self.source)
        for day in sorted(rows):
            o, h, l, c, v = rows[day]
            merged.dates.append(day)
            merged.open.append(o)
            merged.high.append(h)
            merged.low.append(l)
            merged.close.append(c)
            merged.volume.append(v)
        return merged

    def tail(self, n: int) -> "Candles":
        if n <= 0 or n >= len(self.close):
            return self
        return Candles(
            dates=self.dates[-n:],
            open=self.open[-n:],
            high=self.high[-n:],
            low=self.low[-n:],
            close=self.close[-n:],
            volume=self.volume[-n:],
            source=self.source,
        )


def _get(url: str, *, session: Optional[requests.Session] = None,
         timeout: int = 20, retries: int = 3) -> Optional[requests.Response]:
    client = session or requests
    for attempt in range(retries):
        try:
            response = client.get(url, headers=BROWSER_HEADERS, timeout=timeout)
            if response.status_code == 200:
                return response
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def fetch_yahoo(symbol: str, days: int = 400) -> Optional[Candles]:
    span = "2y" if days > 365 else "1y"
    url = YAHOO_CHART.format(symbol=requests.utils.quote(symbol)) + f"?range={span}&interval=1d"
    response = _get(url)
    if response is None:
        return None
    try:
        result = response.json()["chart"]["result"][0]
        stamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    candles = Candles(source="yahoo")
    for i, stamp in enumerate(stamps):
        o, h, l, c = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i]
        if None in (o, h, l, c):
            continue
        volume = quote.get("volume", [None] * len(stamps))[i] or 0.0
        candles.dates.append(datetime.fromtimestamp(stamp, IST).strftime("%Y-%m-%d"))
        candles.open.append(float(o))
        candles.high.append(float(h))
        candles.low.append(float(l))
        candles.close.append(float(c))
        candles.volume.append(float(volume))
    return candles if len(candles) else None


def fetch_stooq(symbol: str) -> Optional[Candles]:
    response = _get(STOOQ_CSV.format(symbol=symbol.lower()), retries=2)
    if response is None or "Date" not in response.text[:50]:
        return None
    candles = Candles(source="stooq")
    for row in csv.DictReader(io.StringIO(response.text)):
        try:
            candles.dates.append(row["Date"])
            candles.open.append(float(row["Open"]))
            candles.high.append(float(row["High"]))
            candles.low.append(float(row["Low"]))
            candles.close.append(float(row["Close"]))
            candles.volume.append(float(row.get("Volume") or 0.0))
        except (KeyError, TypeError, ValueError):
            continue
    return candles if len(candles) else None


def load_cache(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def get_history(yahoo_symbol: str, stooq_symbol: str, cache_path: str,
                cache_key: str, lookback_days: int = 400) -> tuple:
    """Return (Candles, notes). Never raises: falls back to cache."""
    notes: List[str] = []
    cache = load_cache(cache_path)
    cached = Candles.from_dict(cache[cache_key]) if cache_key in cache else None

    fresh = fetch_yahoo(yahoo_symbol, lookback_days)
    if fresh is None:
        notes.append(f"yahoo unavailable for {yahoo_symbol}")
        fresh = fetch_stooq(stooq_symbol)
        if fresh is None:
            notes.append(f"stooq unavailable for {stooq_symbol}")

    if fresh is None and cached is None:
        return None, notes + ["no data from any source and no cache"]

    if fresh is None:
        notes.append("serving cached history only")
        return cached.tail(lookback_days), notes

    combined = cached.merge(fresh) if cached else fresh
    combined = combined.tail(max(lookback_days, 500))
    cache[cache_key] = combined.to_dict()
    cache["updated_at"] = datetime.now(IST).isoformat()
    save_cache(cache_path, cache)
    return combined.tail(lookback_days), notes


def fetch_option_chain(nse_symbol: str = "NIFTY") -> tuple:
    """Best-effort NSE option chain. Returns (payload_or_None, notes)."""
    notes: List[str] = []
    session = requests.Session()
    try:
        session.get(NSE_HOME, headers=BROWSER_HEADERS, timeout=15)
        time.sleep(1.0)
        session.get(f"{NSE_HOME}/option-chain", headers=BROWSER_HEADERS, timeout=15)
        time.sleep(1.0)
        response = session.get(
            NSE_OPTION_CHAIN.format(symbol=nse_symbol),
            headers={**BROWSER_HEADERS, "Referer": f"{NSE_HOME}/option-chain"},
            timeout=20,
        )
        if response.status_code != 200:
            notes.append(f"nse option chain http {response.status_code}")
            return None, notes
        payload = response.json()
        if "records" not in payload:
            notes.append("nse option chain missing records")
            return None, notes
        return payload, notes
    except (requests.RequestException, ValueError) as exc:
        notes.append(f"nse option chain error: {type(exc).__name__}")
        return None, notes
    finally:
        session.close()


def summarise_option_chain(payload: Dict, spot: float, strike_step: int = 50,
                           window: int = 10) -> Optional[Dict]:
    """Condense the raw chain into the few numbers the strategies need."""
    try:
        records = payload["records"]
        rows = records["data"]
        expiries = records.get("expiryDates", [])
    except (KeyError, TypeError):
        return None

    near_expiry = expiries[0] if expiries else None
    atm = round(spot / strike_step) * strike_step
    low, high = atm - window * strike_step, atm + window * strike_step

    call_oi = put_oi = call_chg = put_chg = 0.0
    call_vol = put_vol = 0.0
    ivs: List[float] = []
    by_strike: Dict[float, Dict[str, float]] = {}

    for row in rows:
        strike = row.get("strikePrice")
        if strike is None or not (low <= strike <= high):
            continue
        if near_expiry and row.get("expiryDate") != near_expiry:
            continue
        bucket = by_strike.setdefault(float(strike), {"call_oi": 0.0, "put_oi": 0.0})
        ce, pe = row.get("CE"), row.get("PE")
        if ce:
            call_oi += ce.get("openInterest", 0) or 0
            call_chg += ce.get("changeinOpenInterest", 0) or 0
            call_vol += ce.get("totalTradedVolume", 0) or 0
            bucket["call_oi"] += ce.get("openInterest", 0) or 0
            if ce.get("impliedVolatility"):
                ivs.append(float(ce["impliedVolatility"]))
        if pe:
            put_oi += pe.get("openInterest", 0) or 0
            put_chg += pe.get("changeinOpenInterest", 0) or 0
            put_vol += pe.get("totalTradedVolume", 0) or 0
            bucket["put_oi"] += pe.get("openInterest", 0) or 0
            if pe.get("impliedVolatility"):
                ivs.append(float(pe["impliedVolatility"]))

    if not by_strike or call_oi <= 0 or put_oi <= 0:
        return None

    max_pain = _max_pain(by_strike)
    max_call_oi_strike = max(by_strike, key=lambda k: by_strike[k]["call_oi"])
    max_put_oi_strike = max(by_strike, key=lambda k: by_strike[k]["put_oi"])
    atm_ivs = [iv for iv in ivs if iv > 0]

    return {
        "expiry": near_expiry,
        "atm_strike": atm,
        "pcr_oi": round(put_oi / call_oi, 4),
        "pcr_volume": round(put_vol / call_vol, 4) if call_vol else None,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_oi_change": call_chg,
        "put_oi_change": put_chg,
        "max_pain": max_pain,
        "resistance_strike": max_call_oi_strike,
        "support_strike": max_put_oi_strike,
        "atm_iv": round(sum(atm_ivs) / len(atm_ivs), 2) if atm_ivs else None,
        "underlying": records.get("underlyingValue"),
        "expiry_dates": expiries[:6],
    }


def _max_pain(by_strike: Dict[float, Dict[str, float]]) -> float:
    """Strike where total option-writer payout is smallest."""
    strikes = sorted(by_strike)
    best_strike, best_pain = strikes[0], float("inf")
    for expiry_price in strikes:
        pain = 0.0
        for strike in strikes:
            pain += by_strike[strike]["call_oi"] * max(0.0, expiry_price - strike)
            pain += by_strike[strike]["put_oi"] * max(0.0, strike - expiry_price)
        if pain < best_pain:
            best_pain, best_strike = pain, expiry_price
    return best_strike
