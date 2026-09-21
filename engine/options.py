"""Option pricing, expiry maths and trade construction for NIFTY weeklies.

Black-Scholes is implemented with math.erf so the project needs no scipy —
that keeps the GitHub Actions install to numpy + requests.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

TRADING_DAYS = 252.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(spot: float, strike: float, t_years: float, iv: float,
             option_type: str, rate: float = 0.065) -> Dict[str, float]:
    """Black-Scholes price and greeks. `iv` is a decimal (0.13 = 13%)."""
    option_type = option_type.upper()
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(0.0, spot - strike) if option_type == "CE" else max(0.0, strike - spot)
        return {"price": intrinsic, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    discount = math.exp(-rate * t_years)

    if option_type == "CE":
        price = spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = (-spot * norm_pdf(d1) * iv / (2 * sqrt_t)
                 - rate * strike * discount * norm_cdf(d2))
    else:
        price = strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1.0
        theta = (-spot * norm_pdf(d1) * iv / (2 * sqrt_t)
                 + rate * strike * discount * norm_cdf(-d2))

    return {
        "price": round(price, 2),
        "delta": round(delta, 4),
        "gamma": round(norm_pdf(d1) / (spot * iv * sqrt_t), 6),
        "theta": round(theta / 365.0, 2),          # per calendar day
        "vega": round(spot * norm_pdf(d1) * sqrt_t / 100.0, 2),  # per 1 IV point
    }


def next_weekly_expiry(from_date: date, weekday: int = 1) -> date:
    """Next weekly expiry on or after `from_date` (0=Mon ... 4=Fri)."""
    ahead = (weekday - from_date.weekday()) % 7
    return from_date + timedelta(days=ahead)


def monthly_expiry(year: int, month: int, weekday: int = 1) -> date:
    """Last `weekday` of the month — NSE's monthly expiry convention."""
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def parse_nse_expiry(text: str) -> Optional[date]:
    """Parse NSE's '30-Sep-2026' expiry format."""
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def resolve_expiry(today: date, weekday: int, chain: Optional[Dict]) -> Dict:
    """Prefer the live chain's expiry list; fall back to the configured weekday."""
    if chain and chain.get("expiry"):
        parsed = parse_nse_expiry(chain["expiry"])
        if parsed and parsed >= today:
            return {
                "date": parsed.isoformat(),
                "days_to_expiry": max((parsed - today).days, 0),
                "source": "nse_option_chain",
            }
    expiry = next_weekly_expiry(today, weekday)
    if expiry == today:  # expiry day itself -> roll to the next week
        expiry = next_weekly_expiry(today + timedelta(days=1), weekday)
    return {
        "date": expiry.isoformat(),
        "days_to_expiry": max((expiry - today).days, 0),
        "source": "calculated",
    }


def pick_strike(spot: float, option_type: str, target_delta: float, iv: float,
                t_years: float, strike_step: int = 50) -> Dict:
    """Closest listed strike to the requested delta."""
    atm = round(spot / strike_step) * strike_step
    best = None
    for offset in range(-16, 17):
        strike = atm + offset * strike_step
        if strike <= 0:
            continue
        greeks = bs_price(spot, strike, t_years, iv, option_type)
        gap = abs(abs(greeks["delta"]) - target_delta)
        if best is None or gap < best["gap"]:
            best = {"strike": float(strike), "gap": gap, **greeks}
    return best


def build_trade(spot: float, direction: str, iv_pct: float, expiry: Dict,
                risk_cfg: Dict, lot_size: int, strike_step: int,
                iv_percentile: Optional[float] = None) -> Dict:
    """Turn a CALL/PUT call into a concrete, sized trade plan."""
    option_type = "CE" if direction == "CALL" else "PE"
    iv = max(iv_pct, 5.0) / 100.0
    days = max(expiry["days_to_expiry"], 1)
    t_years = days / 365.0

    atm_strike = round(spot / strike_step) * strike_step
    atm = bs_price(spot, atm_strike, t_years, iv, option_type)
    primary = pick_strike(spot, option_type, risk_cfg["target_delta"], iv, t_years, strike_step)

    # In a high-IV tape a naked long option bleeds; a debit spread caps that.
    high_iv = iv_percentile is not None and iv_percentile >= risk_cfg["high_iv_percentile"]
    structure = "debit_spread" if high_iv else "long_option"

    entry = primary["price"]
    stop_loss = round(entry * (1 - risk_cfg["stop_loss_pct"] / 100.0), 2)
    target = round(entry * (1 + risk_cfg["target_pct"] / 100.0), 2)

    risk_amount = risk_cfg["capital"] * risk_cfg["risk_per_trade_pct"] / 100.0
    risk_per_lot = max((entry - stop_loss) * lot_size, 1.0)
    lots = max(0, int(risk_amount // risk_per_lot))

    plan = {
        "structure": structure,
        "option_type": option_type,
        "direction": direction,
        "strike": primary["strike"],
        "atm_strike": float(atm_strike),
        "expiry": expiry["date"],
        "days_to_expiry": days,
        "iv_used_pct": round(iv * 100, 2),
        "theoretical_premium": entry,
        "delta": primary["delta"],
        "gamma": primary["gamma"],
        "theta_per_day": primary["theta"],
        "vega": primary["vega"],
        "atm_premium": atm["price"],
        "stop_loss": stop_loss,
        "target": target,
        "lot_size": lot_size,
        "suggested_lots": lots,
        "capital_at_risk": round(min(lots * risk_per_lot, risk_amount), 2),
        "premium_outlay": round(lots * entry * lot_size, 2),
        "capital_deployed_pct": round(
            lots * entry * lot_size / risk_cfg["capital"] * 100.0, 3
        ) if risk_cfg.get("capital") else None,
        "breakeven": round(
            primary["strike"] + entry if option_type == "CE" else primary["strike"] - entry, 2
        ),
        "notes": [],
    }

    if structure == "debit_spread":
        wing_offset = 4 * strike_step if direction == "CALL" else -4 * strike_step
        short_strike = primary["strike"] + wing_offset
        short_leg = bs_price(spot, short_strike, t_years, iv, option_type)
        net_debit = round(entry - short_leg["price"], 2)
        plan["short_leg"] = {
            "strike": float(short_strike),
            "premium": short_leg["price"],
            "delta": short_leg["delta"],
        }
        plan["net_debit"] = net_debit
        plan["max_profit"] = round(abs(wing_offset) - net_debit, 2)
        plan["notes"].append(
            f"IV is rich (percentile {iv_percentile:.0f}) — a {int(primary['strike'])}/"
            f"{int(short_strike)} debit spread cuts the vega bleed versus a naked long."
        )

    plan["notes"].append(
        f"Theta burns about Rs {abs(primary['theta']):.1f} per day per unit "
        f"(Rs {abs(primary['theta']) * lot_size:,.0f} per lot) — this is a {days}-day view, not a hold."
    )
    if days <= 1:
        plan["notes"].append(
            "Expiry is within a day: gamma is extreme and premium decays fast. Size down or roll to the next series."
        )
    return plan


def simulate_pnl(trade: Dict, spot_then: float, spot_now: float,
                 iv_pct: float, days_elapsed: int = 1) -> Optional[Dict]:
    """Reprice the recommended option a day later to score the call honestly."""
    if not trade or not trade.get("strike"):
        return None
    days_left = max(trade["days_to_expiry"] - days_elapsed, 0)
    iv = max(iv_pct, 5.0) / 100.0
    entry = trade.get("theoretical_premium") or 0.0
    exit_leg = bs_price(spot_now, trade["strike"], days_left / 365.0, iv, trade["option_type"])
    exit_price = exit_leg["price"]
    if entry <= 0:
        return None
    pct = (exit_price - entry) / entry * 100.0
    return {
        "entry_premium": entry,
        "exit_premium": exit_price,
        "pnl_pct": round(pct, 2),
        "pnl_per_lot": round((exit_price - entry) * trade.get("lot_size", 75), 2),
        "hit_target": pct >= 0,
    }


# Fields the history file needs to reprice a trade and render it. Storing the
# full plan for every session bloats the published JSON that the site fetches.
COMPACT_TRADE_FIELDS = (
    "strike", "option_type", "direction", "expiry", "days_to_expiry",
    "theoretical_premium", "lot_size", "iv_used_pct", "capital_deployed_pct",
    "structure", "stop_loss", "target",
)


def compact_trade(trade: Optional[Dict]) -> Optional[Dict]:
    """Trim a trade plan down to the fields needed for scoring and display."""
    if not trade:
        return None
    return {key: trade[key] for key in COMPACT_TRADE_FIELDS if key in trade}
