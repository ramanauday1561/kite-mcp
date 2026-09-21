"""Replay the full signal -> outcome -> learn loop over historical candles.

This is how you check that the adaptive layer is doing something real: it
walks forward one session at a time, generates a signal from only the data
available at that point, resolves it against the following session, and feeds
the result back into the model -- exactly what the scheduled job does in
production, only compressed.

    python -m tools.simulate --synthetic 400
    python -m tools.simulate --cache          # replay the cached real history

It never touches the live model in state/model.json unless you pass --persist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import datafeed, evaluate, learner, options, strategies  # noqa: E402
from engine.run import CACHE_PATH, load_config  # noqa: E402


def synthetic_candles(n: int = 400, seed: int = 42) -> datafeed.Candles:
    """A series with a genuine, learnable momentum component plus noise.

    Trend strategies *should* beat coin-flipping here; if the learner cannot
    find that edge, the learner is broken.
    """
    rng = np.random.default_rng(seed)
    price = 24000.0
    drift = 0.0
    closes, highs, lows, opens, volumes = [], [], [], [], []
    for _ in range(n):
        drift = 0.90 * drift + rng.normal(0, 0.28)       # persistent momentum
        shock = rng.normal(0, 0.55)                       # unpredictable noise
        ret = (drift + shock) / 100.0
        open_price = price
        price *= (1 + ret)
        intraday = abs(rng.normal(0, 0.45)) / 100.0 * price
        opens.append(open_price)
        closes.append(price)
        highs.append(max(open_price, price) + intraday)
        lows.append(min(open_price, price) - intraday)
        volumes.append(float(rng.normal(2.0e6, 4.0e5)))

    dates = _business_dates(n)
    return datafeed.Candles(dates=dates, open=opens, high=highs, low=lows,
                            close=closes, volume=volumes, source="synthetic")


def synthetic_vix(candles: datafeed.Candles, seed: int = 7) -> datafeed.Candles:
    """VIX that rises when the index falls, as it does in the real world."""
    rng = np.random.default_rng(seed)
    closes = np.asarray(candles.close, dtype=float)
    returns = np.diff(closes, prepend=closes[0]) / closes
    level, values = 14.0, []
    for ret in returns:
        level = max(8.0, 0.88 * level + 0.12 * 14.0 - ret * 220.0 + rng.normal(0, 0.35))
        values.append(level)
    return datafeed.Candles(dates=list(candles.dates), open=values, high=values,
                            low=values, close=values, volume=[0.0] * len(values),
                            source="synthetic")


def _business_dates(n: int) -> List[str]:
    from datetime import date, timedelta
    out, day = [], date(2024, 1, 1)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _slice(candles: datafeed.Candles, upto: int) -> datafeed.Candles:
    return datafeed.Candles(
        dates=candles.dates[:upto], open=candles.open[:upto], high=candles.high[:upto],
        low=candles.low[:upto], close=candles.close[:upto], volume=candles.volume[:upto],
        source=candles.source,
    )


def replay(candles: datafeed.Candles, vix: datafeed.Candles, cfg: Dict,
           warmup: int = 220, adaptive: bool = True) -> Dict:
    """Walk forward. `adaptive=False` freezes the weights as a control group."""
    state = learner.new_state()
    history: List[Dict] = []
    weight_trace: List[Dict] = []

    for i in range(warmup, len(candles) - 1):
        window = _slice(candles, i + 1)
        vix_window = _slice(vix, i + 1)
        features = strategies.build_features(window, vix_window, None)
        regime = learner.regime_of(features, cfg["learning"]["vix_regime_bounds"])

        votes = strategies.run_all(features)
        names = [v.name for v in votes]
        weights = (learner.strategy_weights(state, regime, names, cfg["learning"])
                   if adaptive else {n: 1.0 / len(names) for n in names})
        ensemble = learner.combine(votes, weights)
        calibration = learner.calibrate(state, ensemble["score"]) if adaptive else {
            "probability_up": 0.5 + ensemble["score"] * 0.25,
            "confidence": abs(ensemble["score"]) * 0.5, "basis": "fixed", "sample": 0,
            "bin": 0,
        }
        decision = learner.decide(ensemble["score"], calibration, cfg["signal"])

        expiry = {"date": window.dates[-1], "days_to_expiry": 3, "source": "sim"}
        trade = options.build_trade(
            features["spot"], decision["bias"], features.get("vix") or 13.0, expiry,
            cfg["risk"], cfg["index"]["lot_size"], cfg["index"]["strike_step"],
            features.get("vix_percentile"),
        )

        history.append({
            "id": window.dates[-1], "session_date": window.dates[-1],
            "spot": features["spot"], "regime": regime, "score": ensemble["score"],
            "probability_up": calibration["probability_up"], "action": decision["action"],
            "bias": decision["bias"], "confidence": decision["confidence"],
            "trade": options.compact_trade(trade) if decision["action"] != "NO_TRADE" else None,
            "votes": {v.name: {"score": v.score, "abstained": v.abstained} for v in votes},
            "outcome": None,
        })

        # Resolve against everything that has now printed, then learn from it.
        resolvable = _slice(candles, i + 2)
        if adaptive:
            evaluate.resolve_pending(history, resolvable, _slice(vix, i + 2), state, cfg)
        else:
            frozen = learner.new_state()
            evaluate.resolve_pending(history, resolvable, _slice(vix, i + 2), frozen, cfg)

        if adaptive and i % 20 == 0:
            weight_trace.append({"step": i - warmup, **{
                k: round(v, 4) for k, v in
                learner.strategy_weights(state, regime, names, cfg["learning"]).items()}})

    return {
        "history": history,
        "state": state,
        "performance": evaluate.performance(history),
        "weight_trace": weight_trace,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the learning loop")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="replay N synthetic sessions")
    parser.add_argument("--cache", action="store_true",
                        help="replay the real history cached in state/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=220)
    parser.add_argument("--control", action="store_true",
                        help="also run a fixed-weight control for comparison")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.cache:
        cache = datafeed.load_cache(CACHE_PATH)
        if "nifty" not in cache:
            print("no cached history yet - run the pipeline once first", file=sys.stderr)
            return 1
        candles = datafeed.Candles.from_dict(cache["nifty"])
        vix = (datafeed.Candles.from_dict(cache["vix"]) if "vix" in cache
               else synthetic_vix(candles))
    else:
        n = args.synthetic or 400
        candles = synthetic_candles(n, args.seed)
        vix = synthetic_vix(candles, args.seed)

    result = replay(candles, vix, cfg, warmup=args.warmup, adaptive=True)
    perf = result["performance"]

    if args.json:
        print(json.dumps({"adaptive": perf}, indent=2, default=str))
        return 0

    print(f"Replayed {len(result['history'])} sessions on {candles.source} data")
    print(f"  direction accuracy : {perf['accuracy_all']}%  "
          f"(last 20: {perf['accuracy_last_20']}%)")
    print(f"  signals scored     : {perf['scored']}  traded: {perf['trades_simulated']}")
    print(f"  option return/trade: avg {perf['avg_trade_pnl_pct']}% of premium  "
          f"win rate {perf['win_rate_pct']}%")
    print(f"  sized account curve: {perf['cumulative_pnl_pct']}% cumulative  "
          f"(avg {perf['avg_account_return_pct']}% of account per trade, "
          f"max drawdown {perf['max_drawdown_pct']}%)")
    if candles.source == "synthetic":
        print("\n  NOTE: synthetic data carries a deliberately learnable momentum\n"
              "  pattern, so these numbers are a test of the machinery, NOT an\n"
              "  estimate of live performance. Real accuracy will be far lower.")

    if args.control:
        control = replay(candles, vix, cfg, warmup=args.warmup, adaptive=False)
        c_perf = control["performance"]
        print(f"\nFixed-weight control accuracy: {c_perf['accuracy_all']}% "
              f"vs adaptive {perf['accuracy_all']}%")

    print("\nLearned weights (final):")
    final = result["weight_trace"][-1] if result["weight_trace"] else {}
    for name, weight in sorted(
            ((k, v) for k, v in final.items() if k != "step"),
            key=lambda kv: -kv[1]):
        bucket = result["state"]["strategies"].get(name, {})
        total = bucket.get("wins", 0) + bucket.get("losses", 0)
        hit = bucket.get("wins", 0) / total * 100 if total else float("nan")
        print(f"  {name:>20}  weight {weight:.3f}   hit {hit:5.1f}%  (n={total:5.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
