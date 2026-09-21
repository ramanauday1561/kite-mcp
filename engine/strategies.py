"""Independent strategy voters.

Each strategy reads the shared feature dictionary and returns a StrategyVote
scoring the next-session direction on [-1, +1]:

    +1  strongly bullish  -> CALL
     0  no opinion
    -1  strongly bearish  -> PUT

A strategy that cannot see the data it needs (e.g. the OI strategies when the
NSE chain is unreachable) *abstains* rather than guessing, and the ensemble
renormalises around it. Every vote carries a plain-English rationale so the
published site can explain itself instead of being a black box.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from . import indicators as ind


@dataclass
class StrategyVote:
    name: str
    label: str
    score: float
    confidence: float
    rationale: str
    abstained: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _abstain(name: str, label: str, reason: str) -> StrategyVote:
    return StrategyVote(name, label, 0.0, 0.0, reason, abstained=True)


def build_features(candles, vix_candles=None, chain: Optional[Dict] = None) -> Dict:
    """Compute every indicator once and hand the results to the strategies."""
    close = np.asarray(candles.close, dtype=float)
    high = np.asarray(candles.high, dtype=float)
    low = np.asarray(candles.low, dtype=float)
    open_ = np.asarray(candles.open, dtype=float)
    volume = np.asarray(candles.volume, dtype=float)

    macd_line, macd_signal, macd_hist = ind.macd(close)
    bb_mid, bb_upper, bb_lower, percent_b, bb_width = ind.bollinger(close)
    adx_values, plus_di, minus_di = ind.adx(high, low, close)
    st_line, st_dir = ind.supertrend(high, low, close)
    dc_upper, dc_lower = ind.donchian(high, low, 20)
    atr_values = ind.atr(high, low, close)
    obv_values = ind.obv(close, volume)

    spot = float(close[-1])
    atr_value = ind.last(atr_values)

    features: Dict = {
        "spot": spot,
        "prev_close": float(close[-2]) if len(close) > 1 else spot,
        "date": candles.dates[-1] if candles.dates else None,
        "open": float(open_[-1]),
        "high": float(high[-1]),
        "low": float(low[-1]),
        "ema9": ind.last(ind.ema(close, 9)),
        "ema21": ind.last(ind.ema(close, 21)),
        "ema50": ind.last(ind.ema(close, 50)),
        "ema200": ind.last(ind.ema(close, 200)),
        "ema21_prev": ind.last(ind.ema(close, 21), 3),
        "rsi": ind.last(ind.rsi(close)),
        "rsi_prev": ind.last(ind.rsi(close), 1),
        "macd": ind.last(macd_line),
        "macd_signal": ind.last(macd_signal),
        "macd_hist": ind.last(macd_hist),
        "macd_hist_prev": ind.last(macd_hist, 1),
        "atr": atr_value,
        "atr_pct": (atr_value / spot * 100.0) if atr_value else None,
        "percent_b": ind.last(percent_b),
        "bb_width": ind.last(bb_width),
        "bb_width_rank": ind.percentile_rank(bb_width, 120),
        "adx": ind.last(adx_values),
        "plus_di": ind.last(plus_di),
        "minus_di": ind.last(minus_di),
        "supertrend": ind.last(st_line),
        "supertrend_dir": ind.last(st_dir),
        "supertrend_dir_prev": ind.last(st_dir, 1),
        "donchian_upper": ind.last(dc_upper, 1),
        "donchian_lower": ind.last(dc_lower, 1),
        "roc5": ind.last(ind.roc(close, 5)),
        "roc20": ind.last(ind.roc(close, 20)),
        "zscore20": ind.last(ind.zscore(close, 20)),
        "realized_vol": ind.last(ind.realized_vol(close, 20)),
        "obv_slope": _slope(obv_values[-10:]) if len(obv_values) >= 10 else None,
        "volume_ratio": _volume_ratio(volume),
        "close_position": _close_position(open_[-1], high[-1], low[-1], close[-1]),
        "weekday": _weekday(candles.dates[-1]) if candles.dates else None,
        "weekday_edge": _weekday_edge(candles),
        "chain": chain,
    }

    if vix_candles is not None and len(vix_candles):
        vix = np.asarray(vix_candles.close, dtype=float)
        features["vix"] = float(vix[-1])
        features["vix_prev"] = float(vix[-2]) if len(vix) > 1 else float(vix[-1])
        features["vix_change_pct"] = (
            (vix[-1] - vix[-2]) / vix[-2] * 100.0 if len(vix) > 1 and vix[-2] else 0.0
        )
        features["vix_percentile"] = ind.percentile_rank(vix, 252)
        features["vix_sma20"] = ind.last(ind.sma(vix, 20))
    else:
        features["vix"] = None
        features["vix_change_pct"] = None
        features["vix_percentile"] = None
        features["vix_sma20"] = None

    return features


def _slope(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3 or not np.isfinite(arr).all():
        return None
    x = np.arange(len(arr))
    scale = np.abs(arr).max() or 1.0
    return float(np.polyfit(x, arr / scale, 1)[0])


def _volume_ratio(volume: np.ndarray) -> Optional[float]:
    if len(volume) < 21:
        return None
    baseline = float(np.mean(volume[-21:-1]))
    if baseline <= 0:
        return None
    return float(volume[-1] / baseline)


def _close_position(open_v: float, high_v: float, low_v: float, close_v: float) -> Optional[float]:
    """Where the close sits in the day's range: 1.0 = at the high, 0.0 = at the low."""
    span = high_v - low_v
    if span <= 0:
        return None
    return float((close_v - low_v) / span)


def _weekday(date_str: str) -> Optional[int]:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday()
    except (TypeError, ValueError):
        return None


def _weekday_edge(candles) -> Optional[Dict]:
    """Mean next-day return grouped by weekday, measured on this index's own history."""
    if len(candles) < 60:
        return None
    close = np.asarray(candles.close, dtype=float)
    returns = np.diff(close) / close[:-1] * 100.0
    buckets: Dict[int, List[float]] = {}
    for i, value in enumerate(returns):
        day = _weekday(candles.dates[i + 1])
        if day is None:
            continue
        buckets.setdefault(day, []).append(float(value))
    return {
        str(day): {"mean": float(np.mean(vals)), "n": len(vals)}
        for day, vals in buckets.items() if len(vals) >= 10
    }


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

def strat_ema_trend(f: Dict) -> StrategyVote:
    name, label = "ema_trend", "EMA Trend Stack"
    ema9, ema21, ema50 = f.get("ema9"), f.get("ema21"), f.get("ema50")
    if None in (ema9, ema21, ema50):
        return _abstain(name, label, "not enough history for the EMA stack")

    spot = f["spot"]
    score = 0.0
    score += 0.35 if ema9 > ema21 else -0.35
    score += 0.25 if ema21 > ema50 else -0.25
    score += 0.20 if spot > ema21 else -0.20
    ema200 = f.get("ema200")
    if ema200:
        score += 0.20 if spot > ema200 else -0.20
    else:
        score *= 1.25

    slope = None
    if f.get("ema21_prev"):
        slope = (ema21 - f["ema21_prev"]) / f["ema21_prev"] * 100.0
        score += _clamp(slope * 0.6, -0.25, 0.25)

    score = _clamp(score)
    direction = "above" if spot > ema21 else "below"
    rationale = (
        f"Spot {spot:,.0f} is {direction} the 21-EMA ({ema21:,.0f}); "
        f"9/21/50 stack reads {'bullish' if ema9 > ema21 > ema50 else 'mixed' if ema9 > ema21 or ema21 > ema50 else 'bearish'}"
    )
    if slope is not None:
        rationale += f", 21-EMA slope {slope:+.2f}% over 3 sessions"
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.25), rationale + ".")


def strat_macd(f: Dict) -> StrategyVote:
    name, label = "macd_momentum", "MACD Momentum"
    hist, prev = f.get("macd_hist"), f.get("macd_hist_prev")
    macd_line, signal = f.get("macd"), f.get("macd_signal")
    if None in (hist, macd_line, signal):
        return _abstain(name, label, "MACD needs more history")

    spot = f["spot"]
    normalised = hist / (spot * 0.004) if spot else 0.0
    score = _clamp(normalised * 0.6)
    score += 0.25 if macd_line > signal else -0.25
    if prev is not None:
        score += 0.15 if hist > prev else -0.15
    score = _clamp(score)

    expanding = "expanding" if prev is not None and abs(hist) > abs(prev) else "contracting"
    rationale = (
        f"MACD {macd_line:+.1f} vs signal {signal:+.1f}; histogram {hist:+.1f} and "
        f"{expanding} — momentum is {'building to the upside' if score > 0 else 'leaning down'}."
    )
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), rationale)


def strat_rsi_reversion(f: Dict) -> StrategyVote:
    name, label = "rsi_reversion", "RSI Mean Reversion"
    rsi_value = f.get("rsi")
    if rsi_value is None:
        return _abstain(name, label, "RSI needs at least 15 sessions")

    # Contrarian at the extremes, trend-following in the middle band.
    if rsi_value >= 70:
        score = -_clamp((rsi_value - 70) / 15.0 + 0.3)
        read = f"overbought at {rsi_value:.0f} — fade the move"
    elif rsi_value <= 30:
        score = _clamp((30 - rsi_value) / 15.0 + 0.3)
        read = f"oversold at {rsi_value:.0f} — bounce setup"
    else:
        score = _clamp((rsi_value - 50) / 40.0)
        read = f"neutral zone at {rsi_value:.0f}, mildly {'positive' if score > 0 else 'negative'}"

    prev = f.get("rsi_prev")
    if prev is not None:
        score += _clamp((rsi_value - prev) / 30.0, -0.2, 0.2)
        score = _clamp(score)
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), f"RSI(14) {read}.")


def strat_donchian_breakout(f: Dict) -> StrategyVote:
    name, label = "donchian_breakout", "20-Day Range Breakout"
    upper, lower, spot = f.get("donchian_upper"), f.get("donchian_lower"), f["spot"]
    if None in (upper, lower) or upper <= lower:
        return _abstain(name, label, "20-day range not established yet")

    span = upper - lower
    position = (spot - lower) / span
    if spot > upper:
        score = _clamp(0.6 + (spot - upper) / span * 2.0)
        read = f"broke above the 20-day high ({upper:,.0f})"
    elif spot < lower:
        score = -_clamp(0.6 + (lower - spot) / span * 2.0)
        read = f"broke below the 20-day low ({lower:,.0f})"
    else:
        score = _clamp((position - 0.5) * 1.2)
        read = f"inside the {lower:,.0f}-{upper:,.0f} range, {position * 100:.0f}% of the way up"

    volume_ratio = f.get("volume_ratio")
    if volume_ratio and abs(score) > 0.5:
        score *= 1.15 if volume_ratio > 1.2 else 0.85
        score = _clamp(score)
        read += f" on {volume_ratio:.1f}x average volume"
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), f"Price {read}.")


def strat_supertrend(f: Dict) -> StrategyVote:
    name, label = "supertrend", "Supertrend"
    direction, line = f.get("supertrend_dir"), f.get("supertrend")
    if direction is None or line is None:
        return _abstain(name, label, "Supertrend needs more history")

    spot = f["spot"]
    distance_pct = abs(spot - line) / spot * 100.0
    score = _clamp(direction * (0.55 + min(distance_pct / 3.0, 0.45)))
    flipped = f.get("supertrend_dir_prev") is not None and f["supertrend_dir_prev"] != direction
    if flipped:
        score = _clamp(score * 1.2)

    state = "bullish" if direction > 0 else "bearish"
    rationale = (
        f"Supertrend is {state} with the band at {line:,.0f} "
        f"({distance_pct:.1f}% from spot)"
    )
    rationale += " and it just flipped — fresh signal." if flipped else "."
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), rationale)


def strat_adx_di(f: Dict) -> StrategyVote:
    name, label = "adx_di", "ADX Directional Strength"
    adx_value, pdi, mdi = f.get("adx"), f.get("plus_di"), f.get("minus_di")
    if None in (adx_value, pdi, mdi):
        return _abstain(name, label, "ADX needs at least 30 sessions")

    total = pdi + mdi
    if total <= 0:
        return _abstain(name, label, "flat directional movement")

    direction = (pdi - mdi) / total
    strength = _clamp(adx_value / 40.0, 0.0, 1.0)
    score = _clamp(direction * (0.4 + 0.6 * strength))

    if adx_value < 20:
        regime = "choppy (ADX < 20) so this vote is deliberately small"
        score *= 0.5
    elif adx_value < 25:
        regime = "a developing trend"
    else:
        regime = "a strong trend"
    rationale = (
        f"ADX {adx_value:.0f} signals {regime}; +DI {pdi:.0f} vs -DI {mdi:.0f} "
        f"favours the {'upside' if direction > 0 else 'downside'}."
    )
    return StrategyVote(name, label, score, min(1.0, strength + 0.2), rationale)


def strat_bollinger(f: Dict) -> StrategyVote:
    name, label = "bollinger", "Bollinger Position & Squeeze"
    percent_b, width_rank = f.get("percent_b"), f.get("bb_width_rank")
    if percent_b is None:
        return _abstain(name, label, "Bollinger bands need 20 sessions")

    if percent_b > 1.0:
        score = -0.35
        read = "closed above the upper band — stretched"
    elif percent_b < 0.0:
        score = 0.35
        read = "closed below the lower band — stretched"
    else:
        score = _clamp((percent_b - 0.5) * 1.4)
        read = f"sits at {percent_b * 100:.0f}% of the band width"

    if width_rank is not None and width_rank < 20:
        score *= 0.6
        read += f"; bands are squeezed (width in the {width_rank:.0f}th percentile), so a breakout is pending"
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), f"Price {read}.")


def strat_volatility_regime(f: Dict) -> StrategyVote:
    name, label = "volatility_regime", "India VIX Regime"
    vix, change = f.get("vix"), f.get("vix_change_pct")
    if vix is None:
        return _abstain(name, label, "India VIX unavailable")

    score = 0.0
    parts = []
    if change is not None:
        score += _clamp(-change / 12.0, -0.5, 0.5)
        parts.append(f"VIX {'fell' if change < 0 else 'rose'} {abs(change):.1f}%")

    percentile = f.get("vix_percentile")
    if percentile is not None:
        if percentile > 80:
            score += 0.2
            parts.append(f"at the {percentile:.0f}th percentile — fear is elevated, mean reversion favours upside")
        elif percentile < 20:
            score -= 0.1
            parts.append(f"at the {percentile:.0f}th percentile — complacency, limited cushion")
        else:
            parts.append(f"at the {percentile:.0f}th percentile")

    sma20 = f.get("vix_sma20")
    if sma20:
        score += -0.2 if vix > sma20 * 1.08 else 0.1
    score = _clamp(score)
    rationale = f"India VIX {vix:.2f}: " + ", ".join(parts) + "."
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.25), rationale)


def strat_price_action(f: Dict) -> StrategyVote:
    name, label = "price_action", "Candle & Volume Structure"
    position = f.get("close_position")
    if position is None:
        return _abstain(name, label, "flat session, no range to read")

    score = _clamp((position - 0.5) * 1.6)
    gap = (f["open"] - f["prev_close"]) / f["prev_close"] * 100.0 if f.get("prev_close") else 0.0
    score += _clamp(gap / 1.5, -0.25, 0.25)

    obv_slope = f.get("obv_slope")
    if obv_slope is not None:
        score += _clamp(obv_slope * 3.0, -0.25, 0.25)
    score = _clamp(score)

    where = "near the high" if position > 0.7 else "near the low" if position < 0.3 else "mid-range"
    rationale = (
        f"Last close finished {where} of the day's range ({position * 100:.0f}%), "
        f"gap {gap:+.2f}%"
    )
    if obv_slope is not None:
        rationale += f", OBV slope {'positive' if obv_slope > 0 else 'negative'}"
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), rationale + ".")


def strat_momentum_roc(f: Dict) -> StrategyVote:
    name, label = "momentum_roc", "Rate of Change"
    roc5, roc20 = f.get("roc5"), f.get("roc20")
    if roc5 is None and roc20 is None:
        return _abstain(name, label, "not enough history for rate of change")

    score = 0.0
    if roc5 is not None:
        score += _clamp(roc5 / 2.5, -0.6, 0.6)
    if roc20 is not None:
        score += _clamp(roc20 / 6.0, -0.4, 0.4)

    zscore = f.get("zscore20")
    if zscore is not None and abs(zscore) > 2:
        score -= _clamp(zscore / 8.0, -0.3, 0.3)  # fade statistical extremes
    score = _clamp(score)
    rationale = (
        f"5-day ROC {roc5:+.2f}%" if roc5 is not None else "5-day ROC n/a"
    ) + (f", 20-day ROC {roc20:+.2f}%" if roc20 is not None else "")
    if zscore is not None:
        rationale += f", price z-score {zscore:+.1f}"
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), rationale + ".")


def strat_seasonality(f: Dict) -> StrategyVote:
    name, label = "seasonality", "Weekday Seasonality"
    edges, weekday = f.get("weekday_edge"), f.get("weekday")
    if not edges or weekday is None:
        return _abstain(name, label, "not enough history to measure weekday effects")

    # The signal is for the *next* session, so look up the next weekday.
    target = (weekday + 1) % 7
    if target > 4:
        target = 0
    stats = edges.get(str(target))
    if not stats:
        return _abstain(name, label, "no sample for the next session's weekday")

    mean, sample = stats["mean"], stats["n"]
    score = _clamp(mean / 0.45, -0.5, 0.5) * min(1.0, sample / 40.0)
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][target]
    rationale = (
        f"{day_name} has averaged {mean:+.2f}% over {sample} sessions of this index's "
        f"own history — a {'mild tailwind' if mean > 0 else 'mild headwind'}."
    )
    return StrategyVote(name, label, score, min(0.6, abs(score) + 0.15), rationale)


def strat_oi_pcr(f: Dict) -> StrategyVote:
    name, label = "oi_pcr", "Option Chain PCR & OI Shift"
    chain = f.get("chain")
    if not chain or chain.get("pcr_oi") is None:
        return _abstain(name, label, "NSE option chain unavailable this run")

    pcr = chain["pcr_oi"]
    # PCR above 1 = more puts written = support; below 0.7 = call writing = resistance.
    # Extremes flip contrarian.
    if pcr > 1.6:
        score = -0.3
        read = f"PCR {pcr:.2f} is extreme — crowded put writing, reversal risk"
    elif pcr > 1.0:
        score = _clamp((pcr - 1.0) * 1.2, 0.0, 0.6)
        read = f"PCR {pcr:.2f} shows put writers holding support"
    elif pcr < 0.5:
        score = 0.25
        read = f"PCR {pcr:.2f} is extreme — capitulation, bounce risk for shorts"
    else:
        score = -_clamp((1.0 - pcr) * 1.2, 0.0, 0.6)
        read = f"PCR {pcr:.2f} shows call writers capping upside"

    call_change, put_change = chain.get("call_oi_change"), chain.get("put_oi_change")
    if call_change is not None and put_change is not None:
        total = abs(call_change) + abs(put_change)
        if total > 0:
            score += _clamp((put_change - call_change) / total * 0.4, -0.35, 0.35)
            score = _clamp(score)
            read += (
                f"; today's OI build favours {'puts (bullish)' if put_change > call_change else 'calls (bearish)'}"
            )
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.25), read + ".")


def strat_max_pain(f: Dict) -> StrategyVote:
    name, label = "max_pain", "Max Pain Gravity"
    chain = f.get("chain")
    if not chain or chain.get("max_pain") is None:
        return _abstain(name, label, "NSE option chain unavailable this run")

    spot, max_pain = f["spot"], chain["max_pain"]
    distance_pct = (max_pain - spot) / spot * 100.0
    score = _clamp(distance_pct / 1.5, -0.6, 0.6)
    support, resistance = chain.get("support_strike"), chain.get("resistance_strike")
    rationale = (
        f"Max pain sits at {max_pain:,.0f}, {abs(distance_pct):.2f}% "
        f"{'above' if distance_pct > 0 else 'below'} spot — options writers are pulling "
        f"{'up' if distance_pct > 0 else 'down'} into expiry"
    )
    if support and resistance:
        rationale += f"; heaviest put OI at {support:,.0f}, heaviest call OI at {resistance:,.0f}"
    return StrategyVote(name, label, score, min(1.0, abs(score) + 0.2), rationale + ".")


STRATEGIES: List[Callable[[Dict], StrategyVote]] = [
    strat_ema_trend,
    strat_macd,
    strat_rsi_reversion,
    strat_donchian_breakout,
    strat_supertrend,
    strat_adx_di,
    strat_bollinger,
    strat_volatility_regime,
    strat_price_action,
    strat_momentum_roc,
    strat_seasonality,
    strat_oi_pcr,
    strat_max_pain,
]

STRATEGY_NAMES = [
    "ema_trend", "macd_momentum", "rsi_reversion", "donchian_breakout",
    "supertrend", "adx_di", "bollinger", "volatility_regime",
    "price_action", "momentum_roc", "seasonality", "oi_pcr", "max_pain",
]


def run_all(features: Dict) -> List[StrategyVote]:
    votes = []
    for strategy in STRATEGIES:
        try:
            votes.append(strategy(features))
        except Exception as exc:  # a broken voter must not kill the run
            votes.append(_abstain(
                strategy.__name__.replace("strat_", ""),
                strategy.__name__,
                f"strategy error: {type(exc).__name__}",
            ))
    return votes
