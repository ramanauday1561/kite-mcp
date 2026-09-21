"""Technical indicators implemented on plain numpy arrays.

No TA-Lib / pandas / scipy: keeps the GitHub Actions install to a few seconds
and avoids any wheel that needs compiling.

Every function takes 1-D float arrays ordered oldest -> newest and returns an
array of the same length, front-padded with NaN where the window is not yet
full. Callers should use `last(x)` which returns None instead of NaN.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def last(series: np.ndarray, offset: int = 0) -> Optional[float]:
    """Most recent finite value (offset=1 -> the one before it), or None."""
    arr = _as_array(series)
    idx = len(arr) - 1 - offset
    if idx < 0:
        return None
    value = arr[idx]
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def sma(values, period: int) -> np.ndarray:
    arr = _as_array(values)
    out = np.full(len(arr), np.nan)
    if len(arr) < period or period <= 0:
        return out
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def ema(values, period: int) -> np.ndarray:
    arr = _as_array(values)
    out = np.full(len(arr), np.nan)
    if len(arr) < period or period <= 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def wilder_smooth(values, period: int) -> np.ndarray:
    """Wilder's smoothing (used by RSI / ATR / ADX)."""
    arr = _as_array(values)
    out = np.full(len(arr), np.nan)
    if len(arr) < period or period <= 0:
        return out
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def rsi(close, period: int = 14) -> np.ndarray:
    arr = _as_array(close)
    out = np.full(len(arr), np.nan)
    if len(arr) <= period:
        return out
    delta = np.diff(arr, prepend=arr[0])
    delta[0] = 0.0
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gains[1:], period)
    avg_loss = wilder_smooth(losses[1:], period)
    for i in range(len(avg_gain)):
        if not np.isfinite(avg_gain[i]):
            continue
        if avg_loss[i] == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    arr = _as_array(close)
    fast_ema = ema(arr, fast)
    slow_ema = ema(arr, slow)
    macd_line = fast_ema - slow_ema
    valid = np.isfinite(macd_line)
    signal_line = np.full(len(arr), np.nan)
    if valid.sum() >= signal:
        start = int(np.argmax(valid))
        signal_line[start:] = ema(macd_line[start:], signal)
    return macd_line, signal_line, macd_line - signal_line


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))


def atr(high, low, close, period: int = 14) -> np.ndarray:
    tr = true_range(high, low, close)
    return wilder_smooth(tr, period)


def bollinger(close, period: int = 20, num_std: float = 2.0):
    arr = _as_array(close)
    mid = sma(arr, period)
    std = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        std[i] = arr[i - period + 1:i + 1].std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = np.where(np.isfinite(mid) & (mid != 0), (upper - lower) / mid, np.nan)
    span = upper - lower
    percent_b = np.where(np.isfinite(span) & (span != 0), (arr - lower) / span, np.nan)
    return mid, upper, lower, percent_b, width


def adx(high, low, close, period: int = 14):
    """Returns (adx, +DI, -DI)."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(h)
    out_adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    if n < period * 2 + 1:
        return out_adx, plus_di, minus_di

    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(h, l, c)[1:]

    smooth_tr = wilder_smooth(tr, period)
    smooth_plus = wilder_smooth(plus_dm, period)
    smooth_minus = wilder_smooth(minus_dm, period)

    dx = np.full(len(tr), np.nan)
    for i in range(len(tr)):
        if not np.isfinite(smooth_tr[i]) or smooth_tr[i] == 0:
            continue
        pdi = 100.0 * smooth_plus[i] / smooth_tr[i]
        mdi = 100.0 * smooth_minus[i] / smooth_tr[i]
        plus_di[i + 1] = pdi
        minus_di[i + 1] = mdi
        denom = pdi + mdi
        if denom > 0:
            dx[i] = 100.0 * abs(pdi - mdi) / denom
    smoothed_dx = wilder_smooth(dx[np.isfinite(dx)], period)
    first_valid = int(np.argmax(np.isfinite(dx))) if np.isfinite(dx).any() else 0
    for i, value in enumerate(smoothed_dx):
        target = first_valid + i + 1
        if target < n:
            out_adx[target] = value
    return out_adx, plus_di, minus_di


def supertrend(high, low, close, period: int = 10, multiplier: float = 3.0):
    """Returns (supertrend_line, direction) where direction is +1 up / -1 down."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    atr_values = atr(h, l, c, period)
    hl2 = (h + l) / 2.0

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(atr_values[i]):
            continue
        basic_upper = hl2[i] + multiplier * atr_values[i]
        basic_lower = hl2[i] - multiplier * atr_values[i]
        if i == 0 or not np.isfinite(final_upper[i - 1]):
            final_upper[i], final_lower[i] = basic_upper, basic_lower
            direction[i] = 1.0 if c[i] >= basic_lower else -1.0
        else:
            final_upper[i] = (basic_upper if basic_upper < final_upper[i - 1]
                              or c[i - 1] > final_upper[i - 1] else final_upper[i - 1])
            final_lower[i] = (basic_lower if basic_lower > final_lower[i - 1]
                              or c[i - 1] < final_lower[i - 1] else final_lower[i - 1])
            prev_dir = direction[i - 1]
            if prev_dir == 1.0:
                direction[i] = -1.0 if c[i] < final_lower[i] else 1.0
            else:
                direction[i] = 1.0 if c[i] > final_upper[i] else -1.0
        line[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]
    return line, direction


def donchian(high, low, period: int = 20):
    h, l = _as_array(high), _as_array(low)
    n = len(h)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        upper[i] = h[i - period + 1:i + 1].max()
        lower[i] = l[i - period + 1:i + 1].min()
    return upper, lower


def roc(close, period: int = 10) -> np.ndarray:
    arr = _as_array(close)
    out = np.full(len(arr), np.nan)
    for i in range(period, len(arr)):
        if arr[i - period] != 0:
            out[i] = 100.0 * (arr[i] - arr[i - period]) / arr[i - period]
    return out


def obv(close, volume) -> np.ndarray:
    c, v = _as_array(close), _as_array(volume)
    out = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - v[i]
        else:
            out[i] = out[i - 1]
    return out


def realized_vol(close, period: int = 20, annualise: bool = True) -> np.ndarray:
    """Annualised close-to-close volatility in percent."""
    arr = _as_array(close)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    log_returns = np.full(n, np.nan)
    log_returns[1:] = np.log(arr[1:] / arr[:-1])
    for i in range(period, n):
        window = log_returns[i - period + 1:i + 1]
        if np.isfinite(window).all():
            sigma = window.std(ddof=1)
            out[i] = sigma * (math.sqrt(252) if annualise else 1.0) * 100.0
    return out


def zscore(values, period: int = 20) -> np.ndarray:
    arr = _as_array(values)
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        sigma = window.std(ddof=0)
        if sigma > 0:
            out[i] = (arr[i] - window.mean()) / sigma
    return out


def percentile_rank(values, lookback: int = 252) -> Optional[float]:
    """Percentile (0-100) of the latest value within its trailing window."""
    arr = _as_array(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return None
    window = arr[-lookback:]
    return float(100.0 * (window < window[-1]).sum() / len(window))
