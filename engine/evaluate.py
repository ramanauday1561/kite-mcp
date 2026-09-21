"""Resolve past predictions against what the market actually did.

This is the feedback half of the loop. Every run:
  1. finds history records whose forecast session has now printed a close,
  2. scores them (direction + a repriced option P&L),
  3. folds the result back into the model via learner.apply_outcome,
  4. recomputes the published performance metrics.

Honesty rules baked in here: a NO_TRADE record is scored for its *bias* but
never counted in traded P&L, and a record is only resolved against a session
that is genuinely later than the one it was built from.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import learner, options


def _close_map(candles) -> Dict[str, float]:
    return {day: candles.close[i] for i, day in enumerate(candles.dates)}


def _next_session(dates: List[str], after: str) -> Optional[str]:
    for day in dates:
        if day > after:
            return day
    return None


def resolve_pending(history: List[Dict], candles, vix_candles, state: Dict,
                    cfg: Dict) -> Tuple[int, List[str]]:
    """Score every record that can now be resolved. Returns (count, notes)."""
    closes = _close_map(candles)
    dates = candles.dates
    vix_closes = _close_map(vix_candles) if vix_candles is not None else {}
    learn_cfg = {**cfg["learning"], "flat_band": cfg["signal"]["move_threshold_pct"]}

    resolved = 0
    notes: List[str] = []

    for record in history:
        if record.get("outcome") is not None:
            continue
        base_date = record.get("session_date")
        if not base_date or base_date not in closes:
            continue
        forecast_date = _next_session(dates, base_date)
        if forecast_date is None:
            continue  # the market has not printed the next session yet

        entry_close = closes[base_date]
        exit_close = closes[forecast_date]
        move_pct = (exit_close - entry_close) / entry_close * 100.0

        trade = record.get("trade")
        pnl = None
        if trade and trade.get("strike"):
            iv_now = vix_closes.get(forecast_date) or trade.get("iv_used_pct", 13.0)
            pnl = options.simulate_pnl(trade, entry_close, exit_close, iv_now, days_elapsed=1)

        action = record.get("action", "NO_TRADE")
        bias = record.get("bias", "CALL")
        flat = abs(move_pct) < cfg["signal"]["move_threshold_pct"]
        direction_right = (move_pct > 0) == (bias == "CALL")

        record["outcome"] = {
            "forecast_date": forecast_date,
            "entry_close": round(entry_close, 2),
            "exit_close": round(exit_close, 2),
            "move_pct": round(move_pct, 3),
            "flat": flat,
            "bias_correct": None if flat else direction_right,
            "traded": action in ("CALL", "PUT"),
            "option_pnl_pct": pnl["pnl_pct"] if pnl else None,
            "option_pnl_per_lot": pnl["pnl_per_lot"] if pnl else None,
        }

        learner.apply_outcome(
            state, record, move_pct, pnl["pnl_pct"] if pnl else None, learn_cfg
        )
        resolved += 1

    if resolved:
        notes.append(f"resolved {resolved} prediction(s) and updated the model")
    return resolved, notes


def performance(history: List[Dict]) -> Dict:
    """Published scoreboard. Only resolved, non-flat records count for accuracy."""
    resolved = [r for r in history if r.get("outcome")]
    scored = [r for r in resolved if r["outcome"].get("bias_correct") is not None]
    traded = [r for r in scored if r["outcome"].get("traded")]

    def accuracy(records: List[Dict]) -> Optional[float]:
        if not records:
            return None
        hits = sum(1 for r in records if r["outcome"]["bias_correct"])
        return round(hits / len(records) * 100.0, 2)

    # Portfolio curve, not a sum of premium returns. Each trade only deploys the
    # premium the sizing rule allows (a couple of percent of capital), so the
    # account return is the option return scaled by the capital actually at work
    # and then compounded. Summing raw premium percentages would imply betting
    # the whole account on every signal and would wildly overstate results.
    equity: List[Dict] = []
    account = 100.0
    wins = losses = 0
    for record in traded:
        pnl = record["outcome"].get("option_pnl_pct")
        if pnl is None:
            continue
        trade = record.get("trade") or {}
        deployed = trade.get("capital_deployed_pct")
        if deployed is None:
            deployed = 3.0  # conservative default for older records
        account *= (1.0 + (pnl / 100.0) * (deployed / 100.0))
        wins += 1 if pnl > 0 else 0
        losses += 1 if pnl <= 0 else 0
        equity.append({
            "date": record["outcome"]["forecast_date"],
            "pnl_pct": pnl,
            "capital_deployed_pct": round(deployed, 2),
            "account_return_pct": round((pnl / 100.0) * deployed, 3),
            "cumulative_pct": round(account - 100.0, 2),
        })
    cumulative = account - 100.0

    streak, best_streak = 0, 0
    for record in scored:
        if record["outcome"]["bias_correct"]:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0

    by_regime: Dict[str, Dict] = {}
    for record in scored:
        bucket = by_regime.setdefault(record.get("regime", "mid_vol"), {"n": 0, "hits": 0})
        bucket["n"] += 1
        bucket["hits"] += 1 if record["outcome"]["bias_correct"] else 0
    for bucket in by_regime.values():
        bucket["accuracy"] = round(bucket["hits"] / bucket["n"] * 100.0, 2) if bucket["n"] else None

    by_action: Dict[str, Dict] = {}
    for record in scored:
        bucket = by_action.setdefault(record.get("action", "NO_TRADE"), {"n": 0, "hits": 0})
        bucket["n"] += 1
        bucket["hits"] += 1 if record["outcome"]["bias_correct"] else 0
    for bucket in by_action.values():
        bucket["accuracy"] = round(bucket["hits"] / bucket["n"] * 100.0, 2) if bucket["n"] else None

    returns = [e["pnl_pct"] for e in equity]
    avg_return = sum(returns) / len(returns) if returns else None
    account_returns = [e["account_return_pct"] for e in equity]
    avg_account_return = (sum(account_returns) / len(account_returns)
                          if account_returns else None)
    # Drawdown as a percentage of the peak account value, not in raw percentage
    # points, so it stays meaningful once the curve compounds.
    drawdown = 0.0
    peak = 100.0
    for point in equity:
        value = 100.0 + point["cumulative_pct"]
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value / peak - 1.0) * 100.0)

    return {
        "total_signals": len(history),
        "resolved": len(resolved),
        "scored": len(scored),
        "pending": len(history) - len(resolved),
        "accuracy_all": accuracy(scored),
        "accuracy_last_20": accuracy(scored[-20:]),
        "accuracy_last_50": accuracy(scored[-50:]),
        "accuracy_traded_only": accuracy(traded),
        "current_streak": streak,
        "best_streak": best_streak,
        "trades_simulated": len(equity),
        "win_rate_pct": round(wins / (wins + losses) * 100.0, 2) if (wins + losses) else None,
        "avg_trade_pnl_pct": round(avg_return, 2) if avg_return is not None else None,
        "avg_account_return_pct": round(avg_account_return, 3) if avg_account_return is not None else None,
        "cumulative_pnl_pct": round(cumulative, 2) if equity else None,
        "max_drawdown_pct": round(drawdown, 2) if equity else None,
        "by_regime": by_regime,
        "by_action": by_action,
        "equity_curve": equity[-120:],
    }
