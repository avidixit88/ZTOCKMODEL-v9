"""Payload normalization utilities.

Provides a single canonical mapping for payload fields used across UI, email, and auto-exec.

- Standardizes common keys while preserving the original payload in '__raw__' (debug only).
- Canonical keys are TitleCase.
- Never raises; returns best-effort values.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _first(payload: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in payload:
            v = payload.get(k)
            if v is not None:
                return v
    return None


def normalize_alert_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized payload with canonical TitleCase keys.

    This does not modify the input.
    """
    try:
        src = dict(payload) if payload is not None else {}
    except Exception:
        src = {}

    # Extras can arrive as dict or JSON-y string; keep best-effort dict.
    extras = _first(src, "Extras", "extras")
    if not isinstance(extras, dict):
        extras = {}

    out: Dict[str, Any] = {
        "__raw__": payload,
        "Symbol": _first(src, "Symbol", "symbol", "ticker", "Ticker"),
        "Engine": _first(src, "Engine", "engine"),
        "SignalFamily": _first(src, "SignalFamily", "signal_family", "family") or extras.get("family"),
        "Stage": _first(src, "Stage", "stage"),
        "Tier": _first(src, "Tier", "tier"),
        "Bias": _first(src, "Bias", "bias", "Direction", "direction"),
        "Score": _first(src, "Score", "score"),
        "Session": _first(src, "Session", "session"),
        "Last": _first(src, "Last", "last", "last_price", "LastPrice"),
        "Entry": _first(src, "Entry", "entry", "entry_price", "EntryPrice"),
        "Stop": _first(src, "Stop", "stop", "stop_price", "StopPrice"),
        "TP0": _first(src, "TP0", "tp0", "tp_0", "TakeProfit0"),
        "TP1": _first(src, "TP1", "tp1", "tp_1", "TakeProfit1"),
        "TP2": _first(src, "TP2", "tp2", "tp_2", "TakeProfit2"),
        "AsOf": _first(src, "AsOf", "as_of", "timestamp", "Timestamp", "ts"),
        "Why": _first(src, "Why", "why", "Reason", "reason"),
        "Extras": extras,
    }

    # Some feeds may put Stage in Tier or vice versa; keep both populated if one is missing.
    if not out.get("Stage") and out.get("Tier"):
        out["Stage"] = out.get("Tier")
    if not out.get("Tier") and out.get("Stage"):
        out["Tier"] = out.get("Stage")

    # Basic cleanup for symbol
    if isinstance(out.get("Symbol"), str):
        out["Symbol"] = out["Symbol"].strip().upper()

    return out
