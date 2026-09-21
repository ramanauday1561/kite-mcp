"""Pipeline entry point: fetch -> analyse -> learn -> publish.

Run it with `python -m engine.run`. Everything it writes under docs/data/ is
what the GitHub Pages site reads, and everything it writes under state/ is the
memory that makes the next run smarter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from . import archive, datafeed, envfile, evaluate, learner, llm, options, strategies

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(ROOT, "config.json")
STATE_DIR = os.path.join(ROOT, "state")
MODEL_PATH = os.path.join(STATE_DIR, "model.json")
CACHE_PATH = os.path.join(STATE_DIR, "ohlc_cache.json")
DATA_DIR = os.path.join(ROOT, "docs", "data")
# The authoritative signal store lives in state/ (it is the record the resolver
# works against). docs/data/history.json is a trimmed copy for the site, so
# every visitor is not made to download years of records to draw one table.
HISTORY_PATH = os.path.join(STATE_DIR, "history.json")
# Append-only permanent record; the rolling files above are bounded and
# eventually drop old rows, this never does.
ARCHIVE_DIR = os.path.join(STATE_DIR, "archive")
CSV_PATH = os.path.join(DATA_DIR, "signals.csv")
ENV_PATH = os.path.join(ROOT, ".env")
PUBLISHED_HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
PUBLISHED_HISTORY_RECORDS = 80


def publishable_history(history: List[Dict]) -> List[Dict]:
    """The slice the site actually renders, without the per-strategy vote detail.

    The dashboard's history table shows the verdict and the outcome; the full
    vote breakdown is only needed by the resolver, which reads state/history.json.
    Dropping it here roughly halves what every visitor downloads.
    """
    trimmed = []
    for record in history[-PUBLISHED_HISTORY_RECORDS:]:
        trimmed.append({k: v for k, v in record.items() if k != "votes"})
    return trimmed


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json(path: str, fallback):
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path: str, payload, compact: bool = False) -> None:
    """Write JSON for the site. `compact` drops indentation on the big files -
    history.json is fetched by every visitor, so its size is worth minding."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, separators=(",", ":"), default=str)
        else:
            json.dump(payload, handle, indent=2, default=str)


# The scheduled run times, in UTC, exactly as .github/workflows/signal.yml
# declares them. Deriving the IST display slots from this one list keeps the
# site's "next update" from drifting away from what actually runs.
SCHEDULE_UTC = (
    [(2, 45)]                                          # pre-open
    + [(h, m) for h in range(4, 10) for m in (7, 37)]  # intraday, off-peak
    + [(10, 45)]                                       # post-close
)


def _ist_slots() -> List[tuple]:
    """Convert the UTC cron slots to IST (UTC+5:30) wall-clock times."""
    slots = []
    for hour, minute in SCHEDULE_UTC:
        total = hour * 60 + minute + 330
        slots.append(((total // 60) % 24, total % 60))
    return sorted(set(slots))


SCHEDULE_IST = _ist_slots()


def _next_run_ist(now: datetime) -> Dict:
    """When the next scheduled signal is due, for the site's freshness banner."""
    slots = sorted(set(SCHEDULE_IST))
    today = now.date()
    for hour, minute in slots:
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > now and candidate.weekday() < 5:
            return {"at": candidate.isoformat(),
                    "label": candidate.strftime("%d %b, %I:%M %p IST")}
    # Nothing left today: roll to the next weekday's first slot.
    ahead = 1
    while (today + timedelta(days=ahead)).weekday() >= 5:
        ahead += 1
    nxt = (now + timedelta(days=ahead)).replace(
        hour=slots[0][0], minute=slots[0][1], second=0, microsecond=0)
    return {"at": nxt.isoformat(), "label": nxt.strftime("%d %b, %I:%M %p IST")}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NIFTY 50 F&O signal pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="analyse and print, but do not write any file")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM panel even if a key is configured")
    parser.add_argument("--no-chain", action="store_true",
                        help="skip the NSE option chain fetch")
    args = parser.parse_args(argv)

    # A local .env is a convenience for running outside CI; real environment
    # variables (how the key arrives in Actions) always take precedence.
    env_applied = envfile.load(ENV_PATH)
    cfg = load_config()
    index = cfg["index"]
    notes: List[str] = []
    env_summary = envfile.describe(env_applied)
    if env_summary:
        notes.append(env_summary)
    now = datetime.now(IST)

    # ---------------------------------------------------------------- data
    candles, data_notes = datafeed.get_history(
        index["yahoo_symbol"], index["nse_symbol"].lower(), CACHE_PATH,
        "nifty", cfg["history"]["lookback_days"],
    )
    notes.extend(data_notes)
    if candles is None or len(candles) < 60:
        print("FATAL: could not obtain enough NIFTY history from any source", file=sys.stderr)
        for note in notes:
            print(f"  - {note}", file=sys.stderr)
        return 1

    vix_candles, vix_notes = datafeed.get_history(
        index["vix_yahoo_symbol"], "^vix", CACHE_PATH, "vix",
        cfg["history"]["lookback_days"],
    )
    notes.extend(vix_notes)

    chain_summary = None
    if not args.no_chain:
        raw_chain, chain_notes = datafeed.fetch_option_chain(index["nse_symbol"])
        notes.extend(chain_notes)
        if raw_chain:
            chain_summary = datafeed.summarise_option_chain(
                raw_chain, candles.close[-1], index["strike_step"])
            if chain_summary is None:
                notes.append("option chain returned but could not be summarised")
    else:
        notes.append("option chain skipped (--no-chain)")

    # ------------------------------------------------------------ features
    features = strategies.build_features(candles, vix_candles, chain_summary)
    state = learner.load_state(MODEL_PATH)
    regime = learner.regime_of(features, cfg["learning"]["vix_regime_bounds"])

    # ------------------------------------------ resolve what we can, first
    history: List[Dict] = read_json(HISTORY_PATH, [])
    if not history:   # migrate from the pre-split location if it is still there
        history = read_json(PUBLISHED_HISTORY_PATH, [])
    resolved_count, resolve_notes = evaluate.resolve_pending(
        history, candles, vix_candles, state, cfg,
        today=now.date().isoformat())
    notes.extend(resolve_notes)

    # --------------------------------------------------------------- votes
    votes = strategies.run_all(features)
    if not args.no_llm:
        llm_votes, llm_notes = llm.run_llm_votes(features)
        votes.extend(llm_votes)
        notes.extend(llm_notes)
    else:
        notes.append("LLM panel skipped (--no-llm)")

    names = [vote.name for vote in votes]
    weights = learner.strategy_weights(state, regime, names, cfg["learning"])
    ensemble = learner.combine(votes, weights)
    calibration = learner.calibrate(state, ensemble["score"])
    decision = learner.decide(ensemble["score"], calibration, cfg["signal"])

    # ---------------------------------------------------------- trade plan
    today = now.date()
    expiry = options.resolve_expiry(today, cfg["expiry"]["weekly_weekday"], chain_summary)
    iv_pct = (chain_summary or {}).get("atm_iv") or features.get("vix") or 13.0
    trade = options.build_trade(
        features["spot"], decision["bias"], iv_pct, expiry, cfg["risk"],
        index["lot_size"], index["strike_step"], features.get("vix_percentile"),
    )

    stats = learner.strategy_stats(state, regime, names, cfg["learning"])
    state["runs"] = state.get("runs", 0) + 1

    # ------------------------------------------------------------- publish
    record = {
        "id": f"{features['date']}-{now.strftime('%H%M')}",
        "generated_at": now.isoformat(),
        "session_date": features["date"],
        "spot": round(features["spot"], 2),
        "regime": regime,
        "score": ensemble["score"],
        "probability_up": calibration["probability_up"],
        "action": decision["action"],
        "bias": decision["bias"],
        "confidence": decision["confidence"],
        "trade": options.compact_trade(trade) if decision["action"] != "NO_TRADE" else None,
        "votes": {v.name: {"score": round(v.score, 4), "abstained": v.abstained}
                  for v in votes},
        "outcome": None,
    }

    # One record per session: a re-run on the same session replaces the old one.
    history = [r for r in history if r.get("session_date") != record["session_date"]
               or r.get("outcome") is not None]
    history.append(record)
    history = history[-cfg["history"]["max_history_records"]:]

    perf = evaluate.performance(history)

    # Persist before publishing: the archive is append-only and must capture
    # every resolved record regardless of what the rolling files keep.
    archived_count = archive.archive_resolved(history, ARCHIVE_DIR)
    all_records = archive.merge_for_export(archive.load_archive(ARCHIVE_DIR), history)

    latest = {
        "generated_at": now.isoformat(),
        "generated_at_ist": now.strftime("%d %b %Y, %I:%M %p IST"),
        "index": index["name"],
        "session_date": features["date"],
        "spot": round(features["spot"], 2),
        "change_pct": round(
            (features["spot"] - features["prev_close"]) / features["prev_close"] * 100.0, 2
        ) if features.get("prev_close") else None,
        "regime": regime,
        "recommendation": {
            "action": decision["action"],
            "bias": decision["bias"],
            "confidence": decision["confidence"],
            "probability_up": calibration["probability_up"],
            "reason": decision["reason"],
            "calibration_basis": calibration["basis"],
        },
        "ensemble": ensemble,
        "trade": trade,
        "expiry": expiry,
        "votes": [
            {**v.to_dict(),
             "weight": round(weights.get(v.name, 0.0), 4),
             "contribution": ensemble["contributions"].get(v.name, 0.0),
             "hit_rate": stats.get(v.name, {}).get("hit_rate"),
             "sample": stats.get(v.name, {}).get("sample")}
            for v in sorted(votes, key=lambda x: -abs(ensemble["contributions"].get(x.name, 0.0)))
        ],
        "market_data": {
            "vix": features.get("vix"),
            "vix_change_pct": round(features["vix_change_pct"], 2) if features.get("vix_change_pct") is not None else None,
            "vix_percentile": features.get("vix_percentile"),
            "atr_pct": round(features["atr_pct"], 2) if features.get("atr_pct") else None,
            "rsi": round(features["rsi"], 1) if features.get("rsi") else None,
            "adx": round(features["adx"], 1) if features.get("adx") else None,
            "realized_vol": round(features["realized_vol"], 2) if features.get("realized_vol") else None,
            "source": candles.source,
        },
        "option_chain": chain_summary,
        "learning": {
            "runs": state["runs"],
            "resolved": state.get("resolved", 0),
            "resolved_this_run": resolved_count,
            "regime": regime,
            "strategy_stats": stats,
            "llm_enabled": llm.is_enabled() and not args.no_llm,
            "llm_models": llm.voter_names(),
        },
        "performance": perf,
        "storage": {
            "archived_this_run": archived_count,
            "total_archived": len(all_records),
            "csv": "data/signals.csv",
            "note": (
                "Every signal is appended to a permanent month-by-month archive "
                "in the repository and exported as CSV. The rolling JSON files "
                "are trimmed for page weight; the archive and CSV are not."
            ),
        },
        "next_update": _next_run_ist(now),
        "data_quality": {
            "candles": len(candles),
            "price_source": candles.source,
            "vix_available": features.get("vix") is not None,
            "option_chain_available": chain_summary is not None,
            "notes": notes,
        },
        "disclaimer": (
            "Educational research output, not investment advice. F&O trading carries "
            "a high risk of loss. Signals are model estimates with no guarantee of "
            "accuracy — verify independently and size positions responsibly."
        ),
    }

    if args.dry_run:
        print(json.dumps({
            "action": decision["action"], "bias": decision["bias"],
            "score": ensemble["score"], "probability_up": calibration["probability_up"],
            "confidence": decision["confidence"], "reason": decision["reason"],
            "regime": regime, "spot": record["spot"],
            "strike": trade["strike"], "expiry": expiry["date"],
            "notes": notes,
        }, indent=2))
        return 0

    write_json(os.path.join(DATA_DIR, "latest.json"), latest)
    write_json(HISTORY_PATH, history, compact=True)
    write_json(PUBLISHED_HISTORY_PATH, publishable_history(history), compact=True)
    write_json(os.path.join(DATA_DIR, "performance.json"), perf, compact=True)
    write_json(os.path.join(DATA_DIR, "model.json"), {
        "runs": state["runs"], "resolved": state.get("resolved", 0),
        "updated_at": state.get("updated_at"), "regime": regime,
        "strategy_stats": stats,
        "calibration": state.get("calibration", {}),
    })
    archive.write_csv(all_records, CSV_PATH)
    learner.save_state(MODEL_PATH, state)

    print(f"{latest['session_date']}  spot {record['spot']:,.2f}  "
          f"{decision['action']} ({decision['confidence'] * 100:.0f}%)  "
          f"score {ensemble['score']:+.3f}  regime {regime}  "
          f"resolved {resolved_count}  archived {archived_count}  "
          f"total stored {len(all_records)}")
    for note in notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
