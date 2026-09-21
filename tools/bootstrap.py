"""Warm-start the model by replaying real history through the learning loop.

Without this, the model ships cold: equal weights, no calibration, and months
of live running before it knows anything. Bootstrapping walks forward through
the last year or so of real NIFTY data exactly as the scheduled job would,
so day one starts with learned weights and a populated calibration curve.

    python -m tools.bootstrap --sessions 180

Walk-forward discipline: each simulated signal only ever sees candles up to
that session, so nothing leaks backwards from the future. The LLM voters are
deliberately excluded -- they cannot be replayed historically without paying
for hundreds of calls, so they start cold and earn their weight live.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import datafeed, learner  # noqa: E402
from engine.run import (CACHE_PATH, DATA_DIR, HISTORY_PATH, MODEL_PATH,  # noqa: E402
                        PUBLISHED_HISTORY_PATH, load_config,
                        publishable_history, write_json)
from tools.simulate import replay  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Warm-start the model on real history")
    parser.add_argument("--sessions", type=int, default=180,
                        help="how many past sessions to replay (default 180)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing trained model")
    args = parser.parse_args(argv)

    cfg = load_config()
    index = cfg["index"]

    if os.path.exists(MODEL_PATH) and not args.force:
        state = learner.load_state(MODEL_PATH)
        if state.get("resolved", 0) > 0:
            print(f"model already trained ({state['resolved']} resolved signals) - "
                  f"nothing to do. Use --force to rebuild.")
            return 0

    print("fetching real history for the warm start ...")
    candles, notes = datafeed.get_history(
        index["yahoo_symbol"], index["nse_symbol"].lower(), CACHE_PATH, "nifty",
        cfg["history"]["lookback_days"])
    for note in notes:
        print(f"  note: {note}")
    if candles is None or len(candles) < 260:
        print(f"FATAL: need at least 260 sessions to bootstrap, got "
              f"{len(candles) if candles else 0}", file=sys.stderr)
        return 1

    vix, vix_notes = datafeed.get_history(
        index["vix_yahoo_symbol"], "^vix", CACHE_PATH, "vix",
        cfg["history"]["lookback_days"])
    for note in vix_notes:
        print(f"  note: {note}")
    if vix is None:
        print("  note: no VIX history - the volatility voter will abstain in replay")
        vix = datafeed.Candles(dates=list(candles.dates), open=[], high=[], low=[],
                               close=[], volume=[], source="missing")

    warmup = max(220, len(candles) - args.sessions)
    if warmup >= len(candles) - 5:
        warmup = max(220, len(candles) - 30)

    print(f"replaying {len(candles) - warmup - 1} sessions "
          f"({candles.dates[warmup]} -> {candles.dates[-1]}) ...")
    result = replay(candles, vix, cfg, warmup=warmup, adaptive=True)

    state, history, perf = result["state"], result["history"], result["performance"]
    for record in history:
        record["bootstrapped"] = True

    learner.save_state(MODEL_PATH, state)
    write_json(HISTORY_PATH, history[-cfg["history"]["max_history_records"]:], compact=True)
    write_json(PUBLISHED_HISTORY_PATH, publishable_history(history), compact=True)
    write_json(os.path.join(DATA_DIR, "performance.json"), perf, compact=True)

    print(f"\nwarm start complete on real {candles.source} data")
    print(f"  sessions replayed : {len(history)}")
    print(f"  signals resolved  : {state['resolved']}")
    print(f"  direction accuracy: {perf['accuracy_all']}% "
          f"(last 50: {perf['accuracy_last_50']}%)")
    print(f"  calibration bins  : {len(state.get('calibration', {}))}")

    names = list(history[-1]["votes"].keys())
    weights = learner.strategy_weights(state, "mid_vol", names, cfg["learning"])
    print("\n  learned weights (mid_vol regime):")
    for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        bucket = state["strategies"].get(name, {})
        total = bucket.get("wins", 0.0) + bucket.get("losses", 0.0)
        hit = f"{bucket.get('wins', 0.0) / total * 100:5.1f}%" if total else "    -"
        print(f"    {name:>20}  {weight:.3f}   hit {hit}  (n={total:5.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
