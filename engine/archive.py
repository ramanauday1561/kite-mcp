"""Durable, append-only storage of every signal this system has ever produced.

The rolling files are deliberately bounded -- docs/data/history.json is
trimmed so visitors do not download years of records, and state/history.json
is capped by `history.max_history_records`. Neither is a permanent record.

This module is the permanent record:

  state/archive/YYYY-MM.jsonl   append-only, one JSON object per line, never
                                rewritten or trimmed. Nothing that lands here
                                is ever dropped, so the full track record
                                survives even after the rolling files age out.
  docs/data/signals.csv         a flat export of the same data, published
                                alongside the site so it can be opened in a
                                spreadsheet or pulled into pandas directly.

Both are committed by the workflow, so the history lives in git with a full
audit trail of when each row appeared.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Dict, Iterable, List, Set

CSV_COLUMNS = [
    "session_date", "generated_at", "spot", "regime", "action", "bias",
    "score", "probability_up", "confidence",
    "structure", "option_type", "strike", "expiry", "days_to_expiry",
    "premium", "stop_loss", "target",
    "forecast_date", "exit_close", "move_pct", "flat", "bias_correct",
    "traded", "option_pnl_pct", "option_pnl_per_lot",
]


def _month_of(record: Dict) -> str:
    session = record.get("session_date") or ""
    return session[:7] if len(session) >= 7 else "unknown"


def _existing_keys(path: str) -> Set[str]:
    """Session dates already archived, so re-runs never duplicate a row."""
    keys: Set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line).get("session_date"))
            except json.JSONDecodeError:
                continue  # never let one bad line block the append
    return keys


def archive_resolved(history: List[Dict], archive_dir: str) -> int:
    """Append newly resolved records to their month's file. Returns how many."""
    resolved = [r for r in history if r.get("outcome")]
    if not resolved:
        return 0

    os.makedirs(archive_dir, exist_ok=True)
    by_month: Dict[str, List[Dict]] = {}
    for record in resolved:
        by_month.setdefault(_month_of(record), []).append(record)

    written = 0
    for month, records in by_month.items():
        path = os.path.join(archive_dir, f"{month}.jsonl")
        already = _existing_keys(path)
        fresh = [r for r in records if r.get("session_date") not in already]
        if not fresh:
            continue
        with open(path, "a", encoding="utf-8") as handle:
            for record in sorted(fresh, key=lambda r: r.get("session_date") or ""):
                handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
                written += 1
    return written


def load_archive(archive_dir: str) -> List[Dict]:
    """Every archived record, oldest first. The full permanent history."""
    if not os.path.isdir(archive_dir):
        return []
    records: List[Dict] = []
    for name in sorted(os.listdir(archive_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(archive_dir, name), "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records.sort(key=lambda r: r.get("session_date") or "")
    return records


def _row(record: Dict) -> Dict:
    trade = record.get("trade") or {}
    outcome = record.get("outcome") or {}
    return {
        "session_date": record.get("session_date"),
        "generated_at": record.get("generated_at"),
        "spot": record.get("spot"),
        "regime": record.get("regime"),
        "action": record.get("action"),
        "bias": record.get("bias"),
        "score": record.get("score"),
        "probability_up": record.get("probability_up"),
        "confidence": record.get("confidence"),
        "structure": trade.get("structure"),
        "option_type": trade.get("option_type"),
        "strike": trade.get("strike"),
        "expiry": trade.get("expiry"),
        "days_to_expiry": trade.get("days_to_expiry"),
        "premium": trade.get("theoretical_premium"),
        "stop_loss": trade.get("stop_loss"),
        "target": trade.get("target"),
        "forecast_date": outcome.get("forecast_date"),
        "exit_close": outcome.get("exit_close"),
        "move_pct": outcome.get("move_pct"),
        "flat": outcome.get("flat"),
        "bias_correct": outcome.get("bias_correct"),
        "traded": outcome.get("traded"),
        "option_pnl_pct": outcome.get("option_pnl_pct"),
        "option_pnl_per_lot": outcome.get("option_pnl_per_lot"),
    }


def to_csv(records: Iterable[Dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(_row(record))
    return buffer.getvalue()


def write_csv(records: Iterable[Dict], path: str) -> int:
    rows = list(records)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(to_csv(rows))
    return len(rows)


def merge_for_export(archived: List[Dict], live: List[Dict]) -> List[Dict]:
    """Archive plus anything still pending, de-duplicated by session date."""
    merged: Dict[str, Dict] = {}
    for record in list(archived) + list(live):
        key = record.get("session_date")
        if not key:
            continue
        # A resolved copy always beats a pending one for the same session.
        if key in merged and merged[key].get("outcome") and not record.get("outcome"):
            continue
        merged[key] = record
    return [merged[k] for k in sorted(merged)]
