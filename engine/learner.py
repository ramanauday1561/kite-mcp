"""The adaptive layer: how the system gets better the longer it runs.

Three things are learned from the signals' own track record, and all of them
live in state/model.json so they survive between GitHub Actions runs:

1. **Strategy weights.** Every strategy carries decayed win/loss counters.
   Its hit rate is a Beta posterior, and its weight is proportional to the
   edge over a coin flip. Strategies that stop working shrink toward the
   floor; strategies that work get more of the vote.

2. **Per-regime weights.** The same counters are kept separately for low /
   mid / high volatility tape, because a breakout strategy that shines in a
   trending market is often noise in a quiet one. Small regime samples are
   shrunk toward the global estimate (hierarchical pooling), so a regime
   needs real evidence before it diverges.

3. **Probability calibration.** The raw ensemble score is mapped to a
   probability using the observed frequency of up-moves in each score bucket,
   blended with a logistic prior. That is what turns "score +0.42" into
   "62% chance of an up-move" — and it is why early predictions are honest
   about being uncertain.

Decay (config learning.decay) means recent evidence outweighs old evidence,
so the model tracks a changing market instead of averaging over all history.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

MODEL_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_state() -> Dict:
    return {
        "version": MODEL_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "runs": 0,
        "resolved": 0,
        "strategies": {},
        "regimes": {},
        "calibration": {},
        "notes": "Counters are exponentially decayed; recent evidence dominates.",
    }


def load_state(path: str) -> Dict:
    if not os.path.exists(path):
        return new_state()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return new_state()
    if state.get("version") != MODEL_VERSION:
        state = {**new_state(), **{k: v for k, v in state.items() if k in ("created_at",)}}
    for key in ("strategies", "regimes", "calibration"):
        state.setdefault(key, {})
    state.setdefault("runs", 0)
    state.setdefault("resolved", 0)
    return state


def save_state(path: str, state: Dict) -> None:
    state["updated_at"] = _now()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def regime_of(features: Dict, bounds: List[float]) -> str:
    """Classify the tape by volatility: India VIX if we have it, else realised vol."""
    vix = features.get("vix")
    if vix is None:
        realised = features.get("realized_vol")
        vix = realised if realised is not None else bounds[0] + 0.1
    low, high = bounds
    if vix < low:
        return "low_vol"
    if vix < high:
        return "mid_vol"
    return "high_vol"


def _bucket(store: Dict, name: str) -> Dict:
    return store.setdefault(name, {"wins": 0.0, "losses": 0.0, "pnl": 0.0, "n": 0.0})


def _hit_rate(bucket: Dict, prior_strength: float, prior: float = 0.5) -> float:
    total = bucket["wins"] + bucket["losses"]
    return (bucket["wins"] + prior_strength * prior) / (total + prior_strength)


def strategy_weights(state: Dict, regime: str, names: List[str], cfg: Dict) -> Dict[str, float]:
    """Blend global and regime-specific hit rates into normalised weights."""
    prior_strength = cfg["prior_strength"]
    min_w, max_w = cfg["min_weight"], cfg["max_weight"]
    regime_store = state.get("regimes", {}).get(regime, {})

    raw: Dict[str, float] = {}
    for name in names:
        global_bucket = state.get("strategies", {}).get(
            name, {"wins": 0.0, "losses": 0.0, "pnl": 0.0, "n": 0.0})
        global_hit = _hit_rate(global_bucket, prior_strength)

        regime_bucket = regime_store.get(name)
        if regime_bucket:
            observed = regime_bucket["wins"] + regime_bucket["losses"]
            # Shrink the regime estimate toward the global one until it has evidence.
            hit = ((regime_bucket["wins"] + prior_strength * global_hit)
                   / (observed + prior_strength))
        else:
            hit = global_hit

        edge = max(0.0, hit - 0.5) * 2.0
        raw[name] = max(min_w, min(max_w, edge if edge > 0 else min_w))

    total = sum(raw.values()) or 1.0
    return {name: value / total for name, value in raw.items()}


def strategy_stats(state: Dict, regime: str, names: List[str], cfg: Dict) -> Dict[str, Dict]:
    """Per-strategy scoreboard for the published site."""
    weights = strategy_weights(state, regime, names, cfg)
    out = {}
    for name in names:
        bucket = state.get("strategies", {}).get(
            name, {"wins": 0.0, "losses": 0.0, "pnl": 0.0, "n": 0.0})
        total = bucket["wins"] + bucket["losses"]
        regime_bucket = state.get("regimes", {}).get(regime, {}).get(name)
        out[name] = {
            "weight": round(weights[name], 4),
            "hit_rate": round(_hit_rate(bucket, cfg["prior_strength"]), 4),
            "raw_hit_rate": round(bucket["wins"] / total, 4) if total > 0 else None,
            "sample": round(total, 2),
            "avg_pnl_pct": round(bucket["pnl"] / bucket["n"], 2) if bucket.get("n") else None,
            "regime_sample": round(
                regime_bucket["wins"] + regime_bucket["losses"], 2) if regime_bucket else 0.0,
        }
    return out


def combine(votes: List, weights: Dict[str, float]) -> Dict:
    """Weighted ensemble of the active (non-abstaining) voters."""
    active = [v for v in votes if not v.abstained]
    if not active:
        return {"score": 0.0, "participation": 0.0, "active": 0, "agreement": 0.0,
                "contributions": {}}

    total_weight = sum(weights.get(v.name, 0.0) for v in active) or 1.0
    contributions = {}
    score = 0.0
    for vote in active:
        weight = weights.get(vote.name, 0.0) / total_weight
        contribution = weight * vote.score
        contributions[vote.name] = round(contribution, 4)
        score += contribution

    bullish = sum(1 for v in active if v.score > 0.05)
    bearish = sum(1 for v in active if v.score < -0.05)
    dominant = max(bullish, bearish)
    return {
        "score": round(max(-1.0, min(1.0, score)), 4),
        "participation": round(len(active) / max(len(votes), 1), 3),
        "active": len(active),
        "bullish_votes": bullish,
        "bearish_votes": bearish,
        "neutral_votes": len(active) - bullish - bearish,
        "agreement": round(dominant / len(active), 3),
        "contributions": contributions,
    }


def _bin_key(score: float) -> int:
    return int(max(-10, min(10, round(score * 10))))


def calibrate(state: Dict, score: float, prior_strength: float = 8.0) -> Dict:
    """Map an ensemble score to P(up), using observed frequencies where we have them."""
    prior_probability = 1.0 / (1.0 + math.exp(-2.6 * score))

    key = _bin_key(score)
    calibration = state.get("calibration", {})
    up = total = 0.0
    for neighbour in (key - 1, key, key + 1):  # pool neighbours for a usable sample
        bucket = calibration.get(str(neighbour))
        if bucket:
            factor = 1.0 if neighbour == key else 0.5
            up += bucket.get("up", 0.0) * factor
            total += bucket.get("n", 0.0) * factor

    if total <= 0:
        probability = prior_probability
        basis = "logistic prior (no resolved history in this score band yet)"
    else:
        empirical = up / total
        probability = (total * empirical + prior_strength * prior_probability) / (total + prior_strength)
        basis = f"{total:.0f} resolved signals in this score band (observed {empirical * 100:.0f}% up)"

    probability = max(0.02, min(0.98, probability))
    return {
        "probability_up": round(probability, 4),
        "confidence": round(abs(probability - 0.5) * 2.0, 4),
        "basis": basis,
        "sample": round(total, 1),
        "bin": key,
    }


def decide(score: float, calibration: Dict, cfg: Dict) -> Dict:
    """Final CALL / PUT / NO-TRADE verdict."""
    probability = calibration["probability_up"]
    edge = max(probability, 1.0 - probability)
    bias = "CALL" if probability >= 0.5 else "PUT"

    reasons = []
    if abs(score) < cfg["no_trade_band"]:
        reasons.append(
            f"ensemble score {score:+.2f} is inside the ±{cfg['no_trade_band']:.2f} no-trade band"
        )
    if edge < cfg["min_confidence"]:
        reasons.append(
            f"calibrated edge {edge * 100:.0f}% is below the {cfg['min_confidence'] * 100:.0f}% minimum"
        )

    if reasons:
        return {
            "action": "NO_TRADE",
            "bias": bias,
            "confidence": round(edge, 4),
            "reason": "Standing aside: " + "; ".join(reasons) + ".",
        }

    strength = "high" if edge >= 0.65 else "moderate" if edge >= 0.58 else "low"
    return {
        "action": bias,
        "bias": bias,
        "confidence": round(edge, 4),
        "reason": (
            f"Ensemble score {score:+.2f} calibrates to a {edge * 100:.0f}% chance of "
            f"{'an up' if bias == 'CALL' else 'a down'} move — {strength} conviction."
        ),
    }


def apply_outcome(state: Dict, record: Dict, actual_return_pct: float,
                  option_pnl_pct: Optional[float], cfg: Dict) -> None:
    """Fold one resolved prediction back into the model."""
    decay = cfg["decay"]
    flat_band = cfg.get("flat_band", 0.0)
    if abs(actual_return_pct) < flat_band:
        direction = 0
    else:
        direction = 1 if actual_return_pct > 0 else -1

    regime = record.get("regime", "mid_vol")
    regime_store = state.setdefault("regimes", {}).setdefault(regime, {})

    # Decay every counter so stale evidence fades, then credit this outcome.
    for store in (state.setdefault("strategies", {}), regime_store):
        for bucket in store.values():
            for field in ("wins", "losses", "pnl", "n"):
                bucket[field] = bucket.get(field, 0.0) * decay

    for name, vote in (record.get("votes") or {}).items():
        score = vote.get("score", 0.0)
        if vote.get("abstained") or abs(score) < 0.05 or direction == 0:
            continue
        credit = min(1.0, abs(score))  # confident calls are rewarded and punished more
        for store in (state["strategies"], regime_store):
            bucket = _bucket(store, name)
            if (score > 0) == (direction > 0):
                bucket["wins"] += credit
            else:
                bucket["losses"] += credit
            bucket["n"] += 1.0
            if option_pnl_pct is not None:
                bucket["pnl"] += option_pnl_pct * (1.0 if (score > 0) == (direction > 0) else 0.0)

    if direction != 0:
        key = str(_bin_key(record.get("score", 0.0)))
        calibration = state.setdefault("calibration", {})
        for bucket in calibration.values():
            bucket["up"] = bucket.get("up", 0.0) * decay
            bucket["n"] = bucket.get("n", 0.0) * decay
        bin_bucket = calibration.setdefault(key, {"up": 0.0, "n": 0.0})
        bin_bucket["n"] += 1.0
        if direction > 0:
            bin_bucket["up"] += 1.0

    state["resolved"] = state.get("resolved", 0) + 1
