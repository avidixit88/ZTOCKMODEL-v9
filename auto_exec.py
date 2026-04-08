"""Auto-execution manager (E*TRADE).

Design goals:
 - Zero impact on existing signal logic.
 - State survives Streamlit reruns via st.session_state.
 - No cached functions or unstable return shapes.
 - Conservative defaults: LONG-only, confirm-only optional.

This module owns:
 - eligibility gating (time windows, min score, engine selection)
 - lifecycle state machine per symbol
 - order placement + reconciliation (entry -> stop + TP0)
 - end-of-day liquidation (hard close by 15:55 ET)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields, MISSING
from payload_utils import normalize_alert_payload
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, Tuple

import math
import re
import hashlib
import time as pytime

import streamlit as st

from email_utils import send_email_alert
from etrade_client import ETradeClient
from sessions import classify_session
import pandas as pd


ET_TZ = "America/New_York"
ENTRY_TIMEOUT_MINUTES = 20  # default; can be overridden via AutoExecConfig.timeout_minutes


def _log(msg: str) -> None:
    """Lightweight logger.

    We intentionally keep logging simple (print) because Streamlit captures stdout
    and because we avoid introducing logging handlers that might interact with
    Streamlit reruns.
    """
    try:
        print(str(msg))
    except Exception:
        pass


def _append_note(lifecycle: "TradeLifecycle", note: str) -> None:
    """Append a short breadcrumb into lifecycle.notes (human-readable).

    To avoid unbounded Streamlit session-state growth, keep only the most recent
    note fragments.
    """
    try:
        n = str(note or "").strip()
        if not n:
            return
        cur = str(lifecycle.notes or "")
        parts = [p.strip() for p in cur.split(" | ") if str(p or "").strip()] if cur else []
        parts.append(n)
        max_parts = 25
        if len(parts) > max_parts:
            parts = parts[-max_parts:]
        lifecycle.notes = " | ".join(parts)
    except Exception:
        # Never let observability break the workflow
        pass


def _is_market_order_session_error(exc: Exception) -> bool:
    s = str(exc or "").lower()
    # E*TRADE preview errors often surface as: "PreviewOrder returned Error: code=... message=..."
    # We treat a few broad keyword patterns as "market order not allowed in this session".
    if "previeworder returned error" in s or "previeworder failed" in s:
        if "market" in s and ("session" in s or "hours" in s or "closed" in s or "not accepted" in s or "not permitted" in s or "not allowed" in s):
            return True
    return False


def _compute_marketable_limit_sell(last_px: float) -> float:
    # Marketable LIMIT for SELL: set limit slightly below last price so it will cross the bid.
    # Keep it deterministic and bounded; final rounding is handled by existing tick rounding downstream.
    px = float(last_px)
    if px <= 0:
        return 0.01
    # Conservative step-down by price regime.
    if px < 3.0:
        px2 = px - 0.02
    elif px < 10.0:
        px2 = px - 0.05
    else:
        px2 = px * 0.995  # ~0.5% under
    return max(0.01, round(px2, 2))


def _place_sell_close_best_effort(
    client: "ETradeClient",
    account_id_key: str,
    symbol: str,
    qty: int,
    client_order_id: str,
    last_px: float | None,
    lifecycle: "TradeLifecycle",
    note_prefix: str | None = None,
    reason_tag: str | None = None,
) -> tuple[str, int | None, int | None]:
    """Attempt to close via MARKET; if preview indicates session restriction, fallback to marketable LIMIT.

    Returns (mode, order_id, preview_id).
      - mode: "MARKET" or "LIMIT_FALLBACK"
    """
    note_prefix = str(note_prefix or reason_tag or "CLOSE")
    try:
        oid, pid = client.place_equity_market_order_ex(
            account_id_key=account_id_key,
            symbol=symbol,
            qty=int(qty),
            action="SELL",
            market_session="REGULAR",
            client_order_id=client_order_id,
        )
        _append_note(lifecycle, f"{note_prefix}_MK_PREVIEW_OK pid={pid}")
        _append_note(lifecycle, f"{note_prefix}_MK_PLACE_OK oid={oid}")
        return "MARKET", int(oid), int(pid)
    except Exception as e:
        _append_note(lifecycle, f"{note_prefix}_MK_ERR {str(e)[:300]}")
        # Optional fallback: if broker rejects MARKET due to session rules, try marketable LIMIT.
        if _is_market_order_session_error(e) and last_px is not None:
            try:
                lpx = _compute_marketable_limit_sell(float(last_px))
                coid2 = str(client_order_id)[:18] + "L"
                oid2, pid2 = client.place_equity_limit_order_ex(
                    account_id_key=account_id_key,
                    symbol=symbol,
                    qty=int(qty),
                    limit_price=float(lpx),
                    action="SELL",
                    market_session="REGULAR",
                    client_order_id=coid2,
                )
                _append_note(lifecycle, f"{note_prefix}_LIM_FALLBACK_PREVIEW_OK pid={pid2} px={lpx}")
                _append_note(lifecycle, f"{note_prefix}_LIM_FALLBACK_PLACE_OK oid={oid2}")
                return "LIMIT_FALLBACK", int(oid2), int(pid2)
            except Exception as e2:
                _append_note(lifecycle, f"{note_prefix}_LIM_FALLBACK_ERR {str(e2)[:300]}")
        raise

def _tick_round(price: Optional[float]) -> Optional[float]:
    """Round prices to a safe equity tick.

    Conservative rule:
      - price < 1.00 -> 4 decimals (0.0001)
      - price >= 1.00 -> 2 decimals (0.01)

    Keeps email display + broker order prices coherent and avoids ultra-granular floats.
    """
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    return round(p, 4) if p < 1.0 else round(p, 2)



def _recent_submit(ts: Optional[str], now: datetime, window_seconds: int = 90) -> bool:
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts))
    except Exception:
        return False
    try:
        return (now - dt).total_seconds() < float(window_seconds)
    except Exception:
        return False


def _close_inflight(lifecycle: "TradeLifecycle", now: datetime, client_order_id: str, window_seconds: int = 90) -> bool:
    try:
        if lifecycle.market_exit_order_id:
            return True
        if str(getattr(lifecycle, 'close_client_order_id', '') or '') == str(client_order_id) and _recent_submit(getattr(lifecycle, 'close_submit_started_at', None), now, window_seconds):
            _append_note(lifecycle, f"CLOSE_SUBMIT_INFLIGHT<{window_seconds}s")
            return True
    except Exception:
        return False
    return False


def _mark_close_submit_started(lifecycle: "TradeLifecycle", now: datetime, client_order_id: str) -> None:
    lifecycle.close_submit_started_at = now.isoformat()
    lifecycle.close_client_order_id = str(client_order_id)


def _mark_exit_sent(lifecycle: "TradeLifecycle", now: datetime, reason: str, *, order_id: Optional[str] = None, mode: Optional[str] = None) -> None:
    lifecycle.stage = "EXIT_SENT"
    if order_id:
        lifecycle.market_exit_order_id = str(order_id)
    note = f"EXIT_SENT reason={reason}"
    if mode:
        note += f" mode={mode}"
    _append_note(lifecycle, note)


def _mk_client_order_id(base_id: str, leg: str) -> str:
    """Deterministic E*TRADE clientOrderId (<=20 chars).

    base_id is typically lifecycle_id; leg is EN/ST/TP/MK.
    """
    base = ''.join(ch for ch in str(base_id) if ch.isalnum())
    leg = ''.join(ch for ch in str(leg) if ch.isalnum()).upper() or 'X'
    max_base = max(1, 20 - len(leg))
    return (base[:max_base] + leg)[:20]

def _fmt_price(price: Optional[float]) -> str:
    p = _tick_round(price)
    if p is None:
        return "—"
    try:
        return f"{p:.4f}" if p < 1.0 else f"{p:.2f}"
    except Exception:
        return str(p)


def _format_realized_today(state: Dict[str, Any]) -> str:
    try:
        today = _now_et().date().isoformat()
    except Exception:
        today = ""
    rows = []
    for r in (state.get("realized_trades") or []):
        if not isinstance(r, dict):
            continue
        ts = str(r.get("closed_ts", "") or "")
        if today and not ts.startswith(today):
            continue
        rows.append(r)
    if not rows:
        return "—"
    vals = [r.get("realized") for r in rows if isinstance(r.get("realized"), (int, float))]
    if vals:
        return f"${float(sum(vals)):,.2f} ({len(vals)} trades)"
    return "N/A (missing fill prices)"


def _record_activity(state: Dict[str, Any], kind: str, lifecycle: Optional["TradeLifecycle"]=None, details: str = "") -> None:
    """Append a lightweight activity event for hourly reporting (does not affect execution)."""
    try:
        log = state.setdefault("activity_log", [])
        if not isinstance(log, list):
            log = []
            state["activity_log"] = log
        evt = {
            "ts": _now_et().isoformat(),
            "kind": str(kind or "").upper().strip()[:40],
        }
        if lifecycle is not None:
            try:
                evt["symbol"] = str(getattr(lifecycle, "symbol", "") or "")
                evt["engine"] = str(getattr(lifecycle, "engine", "") or "")
                evt["lifecycle_id"] = str(getattr(lifecycle, "lifecycle_id", "") or "")
            except Exception:
                pass
        if details:
            evt["details"] = str(details)[:200]
        log.append(evt)
        # Keep bounded (Streamlit session_state)
        if len(log) > 200:
            del log[:-200]
    except Exception:
        pass


def _activity_since_last_report(state: Dict[str, Any]) -> Tuple[list[dict], str]:
    """Return (events, cutoff_ts_iso). cutoff is stored in state."""
    cutoff = str(state.get("activity_cutoff_ts") or "")
    events = state.get("activity_log") or []
    if not isinstance(events, list):
        return ([], cutoff)
    if not cutoff:
        # default cutoff = start of today ET
        try:
            now = _now_et()
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        except Exception:
            cutoff = ""
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = str(e.get("ts") or "")
        if cutoff and ts and ts <= cutoff:
            continue
        out.append(e)
    return (out, cutoff)



@dataclass
class AutoExecConfig:
    enabled: bool
    sandbox: bool
    engines: Tuple[str, ...]
    min_score: float
    max_dollars_per_trade: float
    max_pool_dollars: float
    max_concurrent_symbols: int
    lifecycles_per_symbol_per_day: int
    timeout_minutes: int
    tp0_deviation: float
    confirm_only: bool
    status_emails: bool
    hourly_pnl_emails: bool





    entry_mode: str  # 'touch_required' | 'early_band' | 'immediate_on_stage'

    early_entry_limit_orders: bool
    entry_distance_guard_bps: float

    enforce_entry_windows: bool
    entry_grace_minutes: int

    # Time-based profit capture (RIDE-only friendly): after X minutes in trade, if PnL >= threshold, exit.
    enable_time_profit_capture: bool = False
    time_profit_capture_minutes: int = 12
    time_profit_capture_profit_pct: float = 0.5

    # Threshold-trigger sell mode: skip broker stop and monitor price fences in reconcile.
    threshold_exit_enabled: bool = False
    threshold_exit_gain_pct: float = 1.0
    threshold_exit_loss_pct: float = 0.7
    threshold_exit_use_engine_specific: bool = False
    threshold_exit_gain_pct_scalp: float = 0.8
    threshold_exit_loss_pct_scalp: float = 0.5
    threshold_exit_gain_pct_ride: float = 1.8
    threshold_exit_loss_pct_ride: float = 1.2
    # Optional adaptive policy for engine-specific threshold-trigger sell mode.
    # When enabled, the threshold gain/loss percentages are computed after fill
    # from actual trade geometry (entry avg, TP0, stop / pullback band) instead
    # of using the configured static percentages.
    threshold_exit_use_adaptive_engine_policy: bool = False

    # Optional marketable-limit buffer for ENTRY (immediate_on_stage only).
    # When enabled, we nudge the entry limit upward by a small, tick-rounded amount to improve fills.
    use_entry_buffer: bool = False
    entry_buffer_max: float = 0.01

    # Optional execution-layer STOP buffer.
    # IMPORTANT: this does NOT change the signal/payload thesis stop; it only widens the
    # actual broker-submitted protective stop order to let valid trades breathe slightly.
    use_stop_buffer: bool = False
    stop_buffer_amount: float = 0.01

    # Execution-window controls for order submission.
    # IMPORTANT: these are intentionally DECOUPLED from the scanner session toggles.
    # If these fields are missing (older saved config), auto-exec will default to
    # allowing entries in both windows.
    exec_allow_preopen: bool = True
    exec_allow_opening: bool = True
    exec_allow_late_morning: bool = True
    exec_allow_midday: bool = True
    exec_allow_power: bool = True
    stage_only_within_exec_windows: bool = False

    # Broker ping / token validity check.
    # When enabled, auto-exec will periodically call a lightweight broker endpoint
    # to verify OAuth tokens are not just present, but actually working.
    broker_ping_enabled: bool = True
    broker_ping_interval_sec: int = 60

    # Entry price source: for entry placement we can either fetch a fresh quote (GLOBAL_QUOTE)
    # or strictly use the last cached price observed by the scanners.
    entry_use_last_price_cache_only: bool = False

    # If a lifecycle is STAGED but entry isn't sent, optionally email the skip reason once per lifecycle.
    email_on_entry_skip: bool = True

    # For reconciliation checks that may need last price (e.g., stop-breach cancel on unfilled entry),
    # prefer using the scanner-cached LAST to avoid additional quote requests.
    reconcile_use_last_price_cache_only: bool = True

    # Periodic digest email (observability) — does not affect execution.
    digest_emails_enabled: bool = False
    digest_interval_minutes: int = 15
    digest_rth_only: bool = True

@dataclass
class TradeLifecycle:
    symbol: str
    engine: str
    created_ts: str
    # Lifecycle stage contract:
    #   PRESTAGED: signal captured but NOT executable because broker is not "armed" (OAuth/account).
    #   STAGED: executable (subject to execution window + price gates).
    #   ENTRY_SENT, IN_POSITION, EXIT_SENT, CLOSED, CANCELED, CANCEL_PENDING
    stage: str
    desired_entry: float
    stop: float
    tp0: float  # already adjusted by cfg.tp0_deviation (exit limit)
    qty: int
    reserved_dollars: float
    broker_stop: Optional[float] = None
    stop_buffer_applied: float = 0.0
    # Order IDs stored as strings because state persists in st.session_state (JSON-ish)
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    tp0_order_id: Optional[str] = None
    market_exit_order_id: Optional[str] = None

    filled_qty: int = 0

    # --- Broker-observed entry facts (sticky across reruns) ---
    # The goal is to never "forget" that E*TRADE told us a fill happened.
    # These fields are additive; older JSON/state can hydrate safely.
    entry_status_cached: Optional[str] = None
    entry_filled_qty_cached: int = 0
    entry_avg_price_cached: Optional[float] = None
    entry_avg_price_locked: bool = False
    entry_avg_last_source: Optional[str] = None
    entry_avg_last_qty: int = 0
    entry_avg_stable_passes: int = 0
    entry_exec_detected_at: Optional[str] = None  # ISO string
    entry_last_checked_at: Optional[str] = None
    entry_poll_failures: int = 0

    # Bracket tracking (avoid "miss" + avoid spam)
    brackets_attempts: int = 0
    brackets_last_attempt_at: Optional[str] = None
    brackets_last_error: Optional[str] = None
    # Per-leg backoff + synthetic TP0 trigger persistence (must survive reruns)
    stop_last_attempt_at: Optional[str] = None
    tp0_last_attempt_at: Optional[str] = None
    tp0_triggered_at: Optional[str] = None
    stop_submit_started_at: Optional[str] = None
    stop_client_order_id: Optional[str] = None
    tp0_submit_started_at: Optional[str] = None
    tp0_client_order_id: Optional[str] = None
    threshold_exit_activated_at: Optional[str] = None
    # Optional RIDE pullback-band metadata used by threshold-exit logic. Additive and
    # safe for older persisted lifecycles that do not include these fields.
    ride_entry_mode: Optional[str] = None
    pullback_band_low: Optional[float] = None
    pullback_band_high: Optional[float] = None
    # Locked per-lifecycle exit architecture. THRESHOLD skips broker stop placement;
    # TIME_PROFIT / STOP keep normal protective-stop behavior.
    exit_mode: Optional[str] = None
    close_submit_started_at: Optional[str] = None
    close_client_order_id: Optional[str] = None
    # First timestamp when broker position was observed flat while lifecycle remained
    # IN_POSITION / EXIT_SENT. Used to debounce flat-close and avoid one-off stale reads.
    flat_detected_at: Optional[str] = None
    entry_sent_ts: Optional[str] = None
    # Entry submit guard metadata to prevent duplicate sends across Streamlit reruns
    # / transient broker-response timing.
    entry_submit_started_at: Optional[str] = None
    entry_client_order_id: Optional[str] = None
    cancel_requested_at: Optional[str] = None  # ISO timestamp when CANCEL_PENDING was entered
    cancel_attempts: int = 0  # number of cancel attempts issued for this lifecycle
    bracket_qty: int = 0
    emailed_events: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # Last evaluation breadcrumb for debugging why an entry did/didn't send.
    last_entry_eval: str = ""

    # How many times this lifecycle has been evaluated for entry while STAGED.
    # Stored as an int in session_state for observability (digest emails).
    entry_eval_count: int = 0

    # How many times this lifecycle has been evaluated by the reconcile loop (ENTRY_SENT/IN_POSITION/EXIT_SENT/CANCEL_PENDING).
    # This is separate from entry_eval_count (which is STAGED-only) and helps diagnose broker/read throttling.
    reconcile_eval_count: int = 0

    @property
    def lifecycle_id(self) -> str:
        """Canonical lifecycle identifier.

        Some downstream components (e.g., clientOrderId generation, order bookkeeping)
        expect a stable `lifecycle_id`. Older lifecycle objects did not include an
        explicit id field, so we derive a deterministic identifier from immutable
        attributes.

        This is intentionally a *property* (not a dataclass field) to avoid changing
        persisted session-state schema.
        """
        try:
            sym = ''.join(ch for ch in str(self.symbol) if ch.isalnum()).upper()[:6] or 'X'
            raw = f"{self.symbol}|{self.engine}|{self.created_ts}"
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
            return f"{sym}{digest}"
        except Exception:
            # Fallback: best-effort stable string
            return ''.join(ch for ch in f"{self.symbol}{self.engine}{self.created_ts}" if ch.isalnum())[:32]


# ----------------------------
# Session-state schema hardening
# ----------------------------


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return default


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return int(float(v))
    except Exception:
        return default


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return float(v)
    except Exception:
        return default


def _safe_dataclass_from_dict(cls, raw: Any, *, required_defaults: Optional[Dict[str, Any]] = None,
                              coercers: Optional[Dict[str, Any]] = None) -> Any:
    """Safely hydrate a dataclass from a dict.

    Goals:
      - ignore unknown keys (old/new versions)
      - backfill missing keys using dataclass defaults or required_defaults
      - optionally coerce types for stability across reruns and deployments
    """
    if not isinstance(raw, dict):
        raw = {} if raw is None else dict(getattr(raw, "__dict__", {}) or {})

    required_defaults = required_defaults or {}
    coercers = coercers or {}

    out: Dict[str, Any] = {}
    flds = list(fields(cls))
    allowed = {f.name for f in flds}

    # Filter unknown keys first (prevents unexpected keyword errors).
    filtered = {k: raw.get(k) for k in allowed if k in raw}

    for f in flds:
        name = f.name
        if name in filtered:
            val = filtered[name]
        else:
            if f.default is not MISSING:
                val = f.default
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore
                val = f.default_factory()  # type: ignore
            elif name in required_defaults:
                val = required_defaults[name]
            else:
                # Last-resort: keep None for missing required fields.
                val = None

        if name in coercers:
            try:
                val = coercers[name](val)
            except Exception:
                pass
        out[name] = val

    return cls(**out)


def autoexec_cfg_from_raw(raw: Any) -> AutoExecConfig:
    """Hydrate AutoExecConfig from session_state safely (schema tolerant)."""
    required = {
        # Conservative defaults so missing keys don't arm trading accidentally.
        "enabled": False,
        "sandbox": True,
        "engines": tuple(),
        "min_score": 0.0,
        "max_dollars_per_trade": 0.0,
        "max_pool_dollars": 0.0,
        "max_concurrent_symbols": 1,
        "lifecycles_per_symbol_per_day": 1,
        "timeout_minutes": ENTRY_TIMEOUT_MINUTES,
        "tp0_deviation": 0.0,
        "confirm_only": False,
        "status_emails": False,
        "hourly_pnl_emails": False,
        "entry_mode": "touch_required",
        "early_entry_limit_orders": False,
        "entry_distance_guard_bps": 0.0,
        "use_stop_buffer": False,
        "stop_buffer_amount": 0.01,
        "enforce_entry_windows": True,
        "entry_grace_minutes": 0,
        "reconcile_use_last_price_cache_only": True,
        "digest_emails_enabled": False,
        "digest_interval_minutes": 15,
        "digest_rth_only": True,
    }
    coercers = {
        "enabled": lambda v: _coerce_bool(v, False),
        "sandbox": lambda v: _coerce_bool(v, True),
        "min_score": lambda v: _coerce_float(v, 0.0),
        "max_dollars_per_trade": lambda v: _coerce_float(v, 0.0),
        "max_pool_dollars": lambda v: _coerce_float(v, 0.0),
        "max_concurrent_symbols": lambda v: _coerce_int(v, 1),
        "lifecycles_per_symbol_per_day": lambda v: _coerce_int(v, 1),
        "timeout_minutes": lambda v: _coerce_int(v, ENTRY_TIMEOUT_MINUTES),
        "tp0_deviation": lambda v: _coerce_float(v, 0.0),
        "confirm_only": lambda v: _coerce_bool(v, False),
        "status_emails": lambda v: _coerce_bool(v, False),
        "hourly_pnl_emails": lambda v: _coerce_bool(v, False),
        "early_entry_limit_orders": lambda v: _coerce_bool(v, False),
        "entry_distance_guard_bps": lambda v: _coerce_float(v, 0.0),
        "use_stop_buffer": lambda v: _coerce_bool(v, False),
        "stop_buffer_amount": lambda v: min(0.05, max(0.0, _coerce_float(v, 0.01))),
        "enforce_entry_windows": lambda v: _coerce_bool(v, True),
        "entry_grace_minutes": lambda v: _coerce_int(v, 0),
        "exec_allow_preopen": lambda v: _coerce_bool(v, True),
        "exec_allow_opening": lambda v: _coerce_bool(v, True),
        "exec_allow_late_morning": lambda v: _coerce_bool(v, True),
        "exec_allow_midday": lambda v: _coerce_bool(v, True),
        "exec_allow_power": lambda v: _coerce_bool(v, True),
        "stage_only_within_exec_windows": lambda v: _coerce_bool(v, False),
        "broker_ping_enabled": lambda v: _coerce_bool(v, True),
        "broker_ping_interval_sec": lambda v: max(15, min(300, _coerce_int(v, 60))),
        "reconcile_use_last_price_cache_only": lambda v: _coerce_bool(v, True),
        "digest_emails_enabled": lambda v: _coerce_bool(v, False),
        "digest_interval_minutes": lambda v: max(5, min(60, _coerce_int(v, 15))),
        "digest_rth_only": lambda v: _coerce_bool(v, True),
    }
    cfg = _safe_dataclass_from_dict(AutoExecConfig, raw, required_defaults=required, coercers=coercers)
    # Ensure engines is a tuple[str,...]
    try:
        if isinstance(cfg.engines, list):
            cfg.engines = tuple(str(x) for x in cfg.engines)  # type: ignore
        elif isinstance(cfg.engines, str):
            cfg.engines = tuple([cfg.engines])  # type: ignore
    except Exception:
        pass
    return cfg


def lifecycle_from_raw(raw: Any) -> TradeLifecycle:
    """Hydrate TradeLifecycle from session_state safely (schema tolerant)."""
    now_ts = _now_et().isoformat()
    required = {
        "symbol": str(getattr(raw, "get", lambda k, d=None: d)("symbol", "UNKNOWN")) if isinstance(raw, dict) else "UNKNOWN",
        "engine": (raw.get("engine") if isinstance(raw, dict) else "") or "",
        "created_ts": (raw.get("created_ts") if isinstance(raw, dict) else None) or now_ts,
        "stage": (raw.get("stage") if isinstance(raw, dict) else None) or "CANCELED",
        "desired_entry": 0.0,
        "stop": 0.0,
        "tp0": 0.0,
        "qty": 0,
        "reserved_dollars": 0.0,
    }
    coercers = {
        "symbol": lambda v: str(v or "UNKNOWN"),
        "engine": lambda v: str(v or ""),
        "created_ts": lambda v: str(v or now_ts),
        "stage": lambda v: str(v or "CANCELED"),
        "desired_entry": lambda v: _coerce_float(v, 0.0),
        "stop": lambda v: _coerce_float(v, 0.0),
        "tp0": lambda v: _coerce_float(v, 0.0),
        "broker_stop": lambda v: (_coerce_float(v, 0.0) if v not in (None, "") else None),
        "stop_buffer_applied": lambda v: max(0.0, _coerce_float(v, 0.0)),
        "qty": lambda v: max(0, _coerce_int(v, 0)),
        "reserved_dollars": lambda v: max(0.0, _coerce_float(v, 0.0)),
        "filled_qty": lambda v: max(0, _coerce_int(v, 0)),
        "bracket_qty": lambda v: max(0, _coerce_int(v, 0)),
        "emailed_events": lambda v: v if isinstance(v, dict) else {},
        "notes": lambda v: str(v or ""),
        "last_entry_eval": lambda v: str(v or ""),
        "entry_eval_count": lambda v: max(0, _coerce_int(v, 0)),
        "reconcile_eval_count": lambda v: max(0, _coerce_int(v, 0)),
        "entry_order_id": lambda v: (str(v) if v else None),
        "stop_order_id": lambda v: (str(v) if v else None),
        "tp0_order_id": lambda v: (str(v) if v else None),
        "market_exit_order_id": lambda v: (str(v) if v else None),
        "entry_sent_ts": lambda v: (str(v) if v else None),
        "entry_submit_started_at": lambda v: (str(v) if v else None),
        "entry_client_order_id": lambda v: (str(v) if v else None),
        "stop_last_attempt_at": lambda v: (str(v) if v else None),
        "tp0_last_attempt_at": lambda v: (str(v) if v else None),
        "tp0_triggered_at": lambda v: (str(v) if v else None),
        "stop_submit_started_at": lambda v: (str(v) if v else None),
        "stop_client_order_id": lambda v: (str(v) if v else None),
        "tp0_submit_started_at": lambda v: (str(v) if v else None),
        "tp0_client_order_id": lambda v: (str(v) if v else None),
        "close_submit_started_at": lambda v: (str(v) if v else None),
        "close_client_order_id": lambda v: (str(v) if v else None),
        "threshold_exit_activated_at": lambda v: (str(v) if v else None),
        "ride_entry_mode": lambda v: (str(v).upper().strip() if v not in (None, "") else None),
        "pullback_band_low": lambda v: (_coerce_float(v, None) if v not in (None, "") else None),
        "pullback_band_high": lambda v: (_coerce_float(v, None) if v not in (None, "") else None),
        "exit_mode": lambda v: (str(v).upper() if v else None),
    }
    lc = _safe_dataclass_from_dict(TradeLifecycle, raw, required_defaults=required, coercers=coercers)
    # Ensure stage is a known lifecycle stage string
    try:
        if lc.stage not in {"PRESTAGED", "STAGED", "ENTRY_SENT", "IN_POSITION", "EXIT_SENT", "CLOSED", "CANCELED", "CANCEL_PENDING"}:
            lc.notes = (lc.notes + " | bad_state:unknown_stage").strip(" |")
            lc.stage = "CANCELED"
    except Exception:
        pass
    return lc


def _effective_exit_mode(lifecycle: TradeLifecycle) -> str:
    """Return lifecycle-scoped exit mode without relying on current global toggles.

    THRESHOLD mode is sticky once activated. Older persisted lifecycles that predate
    the explicit field are inferred from threshold_exit_activated_at. All other
    lifecycles default to STOP/TIME_PROFIT-style protective stop behavior.
    """
    try:
        em = str(getattr(lifecycle, "exit_mode", "") or "").upper()
    except Exception:
        em = ""
    if em:
        return em
    try:
        if getattr(lifecycle, "threshold_exit_activated_at", None):
            return "THRESHOLD"
    except Exception:
        pass
    return "STOP"




def _threshold_trade_direction(lifecycle: TradeLifecycle) -> str:
    try:
        notes = str(getattr(lifecycle, "notes", "") or "").upper()
    except Exception:
        notes = ""
    if "BIAS=SHORT" in notes or "SHORT" in notes:
        return "SHORT"
    return "LONG"


def _adaptive_threshold_geometry(
    lifecycle: TradeLifecycle,
    entry_px: float,
) -> tuple[Optional[float], Optional[float], str, str]:
    """Return (reward_room_pct, failure_room_pct, failure_basis_label, direction).

    Values are expressed as percentages of actual filled entry price so the
    threshold policy can remain downstream-percent based while still reflecting
    real executed geometry.
    """
    if not entry_px or entry_px <= 0:
        return (None, None, "ENTRY", "LONG")
    direction = _threshold_trade_direction(lifecycle)
    try:
        engine = str(getattr(lifecycle, "engine", "") or "").upper()
    except Exception:
        engine = ""
    try:
        ride_mode = str(getattr(lifecycle, "ride_entry_mode", "") or "").upper()
    except Exception:
        ride_mode = ""

    tp0 = _coerce_float(getattr(lifecycle, "tp0", None), None)
    stop_px = _coerce_float(getattr(lifecycle, "stop", getattr(lifecycle, "stop_px", None)), None)
    pbl = _coerce_float(getattr(lifecycle, "pullback_band_low", None), None)
    pbh = _coerce_float(getattr(lifecycle, "pullback_band_high", None), None)

    reward_room_pct = None
    if tp0 is not None and tp0 > 0:
        if direction == "SHORT":
            reward_room_pct = (float(entry_px) - float(tp0)) / float(entry_px) * 100.0
        else:
            reward_room_pct = (float(tp0) - float(entry_px)) / float(entry_px) * 100.0
        if reward_room_pct is not None and reward_room_pct <= 0:
            reward_room_pct = None

    failure_basis_px = stop_px
    failure_basis_label = "stop_px"
    failure_room_pct = None
    if engine == "RIDE" and ride_mode == "PULLBACK" and pbl is not None and pbh is not None:
        if direction == "SHORT":
            failure_basis_px = float(max(pbl, pbh))
            failure_basis_label = "pullback_band_high"
            if stop_px is not None and stop_px > failure_basis_px:
                # Align adaptive threshold basis with downstream loss evaluation,
                # which measures adverse movement as a percent above PB-high.
                failure_room_pct = (float(stop_px) - float(failure_basis_px)) / float(failure_basis_px) * 100.0
        else:
            failure_basis_px = float(min(pbl, pbh))
            failure_basis_label = "pullback_band_low"
            if stop_px is not None and stop_px < failure_basis_px:
                # Align adaptive threshold basis with downstream loss evaluation,
                # which measures adverse movement as a percent below PB-low.
                failure_room_pct = (float(failure_basis_px) - float(stop_px)) / float(failure_basis_px) * 100.0

    if failure_room_pct is None and failure_basis_px is not None and failure_basis_px > 0:
        if direction == "SHORT":
            failure_room_pct = (float(failure_basis_px) - float(entry_px)) / float(entry_px) * 100.0
        else:
            failure_room_pct = (float(entry_px) - float(failure_basis_px)) / float(entry_px) * 100.0
    if failure_room_pct is not None and failure_room_pct <= 0:
        failure_room_pct = None

    return (reward_room_pct, failure_room_pct, failure_basis_label, direction)


def _adaptive_threshold_engine_trigger_pcts(
    cfg: AutoExecConfig,
    lifecycle: TradeLifecycle,
    entry_px: Optional[float],
) -> Optional[tuple[float, float, str]]:
    """Return adaptive (gain_pct, loss_pct, source_label) after fill.

    This policy intentionally derives its thresholds from actual post-fill trade
    geometry instead of static configured percentages:
      - reward room from real cost basis to TP0
      - failure room from real cost basis to structural invalidation

    For low-priced, high-volatility names we also scale the fractions using a
    *self-normalizing* tape-aware multiplier derived from:
      - executed price regime (cheap names naturally wiggle more)
      - the trade's own reward/failure geometry (wider rooms imply noisier tape)

    No manual caps or preset anchors are used; the equations stay bounded by the
    fractional geometry itself plus smooth asymptotic scaling functions.
    """
    try:
        enabled = bool(getattr(cfg, "threshold_exit_use_adaptive_engine_policy", False))
    except Exception:
        enabled = False
    if not enabled:
        return None
    try:
        use_engine_specific = bool(getattr(cfg, "threshold_exit_use_engine_specific", False))
    except Exception:
        use_engine_specific = False
    if not use_engine_specific:
        return None
    if entry_px is None or entry_px <= 0:
        return None

    try:
        engine = str(getattr(lifecycle, "engine", "") or "").upper()
    except Exception:
        engine = ""
    try:
        ride_mode = str(getattr(lifecycle, "ride_entry_mode", "") or "").upper()
    except Exception:
        ride_mode = ""

    reward_room_pct, failure_room_pct, failure_basis_label, _direction = _adaptive_threshold_geometry(lifecycle, float(entry_px))
    if reward_room_pct is None or failure_room_pct is None:
        return None

    # Tiny wiggle room so thresholds are not overly eager on exact fractions.
    wiggle = 1.04

    if engine == "SCALP":
        reward_eff = 0.42
        failure_tol = 0.46
        source = "ADAPTIVE_SCALP"
    elif engine == "RIDE" and ride_mode == "PULLBACK":
        reward_eff = 0.34
        failure_tol = 0.62
        source = f"ADAPTIVE_RIDE_PULLBACK[{failure_basis_label}]"
    elif engine == "RIDE":
        reward_eff = 0.47
        failure_tol = 0.52
        source = f"ADAPTIVE_RIDE_BREAKOUT[{failure_basis_label}]"
    else:
        reward_eff = 0.40
        failure_tol = 0.50
        source = f"ADAPTIVE_GENERIC[{failure_basis_label}]"

    # Self-normalizing tape factor.
    # Cheap names (especially $1-$5) need more movement before a threshold is
    # behaviorally meaningful; higher-priced names naturally compress toward 1x.
    # Geometry factor uses the trade's own room as a volatility proxy without
    # needing new payload fields (ATR / OHLC context) downstream.
    try:
        px = float(entry_px)
    except Exception:
        px = 0.0
    try:
        avg_room_pct = max(0.0, (float(reward_room_pct) + float(failure_room_pct)) / 2.0)
    except Exception:
        avg_room_pct = 0.0

    # Asymptotically bounded cheap-tape boost: ~1.22x near $1, tapering toward 1x.
    cheap_tape_mult = 1.0 + 0.22 * math.exp(-max(0.0, px - 1.0) / 2.8)
    # Geometry-derived volatility proxy centered around ~2.5%% average room.
    geom_mult = 1.0 + 0.18 * math.tanh((avg_room_pct - 2.5) / 2.0)
    tape_mult = cheap_tape_mult * geom_mult

    # Gain can respect the full tape multiplier; loss uses a moderated share so
    # volatile cheap names get needed wiggle room without becoming reckless.
    if engine == "SCALP":
        loss_tape_mult = 1.0 + 0.45 * (tape_mult - 1.0)
    elif engine == "RIDE" and ride_mode == "PULLBACK":
        loss_tape_mult = 1.0 + 0.70 * (tape_mult - 1.0)
    elif engine == "RIDE":
        loss_tape_mult = 1.0 + 0.55 * (tape_mult - 1.0)
    else:
        loss_tape_mult = 1.0 + 0.50 * (tape_mult - 1.0)

    # Slight gain-side uplift for cheap, volatile names only when the filled
    # trade still has healthy reward room. This avoids under-letting winners
    # develop in the $1-$5 tape without broadly loosening everything.
    cheap_gain_mult = 1.0
    try:
        reward_room_for_boost = max(0.0, float(reward_room_pct))
    except Exception:
        reward_room_for_boost = 0.0
    if px > 0:
        cheapness_boost = math.exp(-max(0.0, px - 1.0) / 2.3)
        reward_room_boost = max(0.0, math.tanh((reward_room_for_boost - 3.0) / 1.8))
        if px < 5.0 and reward_room_for_boost > 3.0:
            cheap_gain_mult = 1.0 + 0.12 * cheapness_boost * reward_room_boost

    # Fill-quality refinement: compare actual filled basis to desired entry.
    # Worse fills should make the policy a bit less generous; better fills can
    # earn slightly more patience. This remains smooth and bounded.
    fill_quality_mult_gain = 1.0
    fill_quality_mult_loss = 1.0
    desired_entry = _coerce_float(getattr(lifecycle, "desired_entry", None), None)
    if desired_entry is not None and desired_entry > 0:
        try:
            if _direction == "SHORT":
                fill_slip_pct = (float(desired_entry) - float(entry_px)) / float(desired_entry) * 100.0
            else:
                fill_slip_pct = (float(entry_px) - float(desired_entry)) / float(desired_entry) * 100.0
        except Exception:
            fill_slip_pct = 0.0
        fill_quality_mult_gain = 1.0 - 0.10 * math.tanh(fill_slip_pct / 1.25)
        fill_quality_mult_loss = 1.0 - 0.05 * math.tanh(fill_slip_pct / 1.50)

    # Reward/failure asymmetry refinement: when actual reward room is much
    # better than failure room, let the gain side be a bit more patient while
    # keeping the loss side only modestly affected.
    asymmetry_mult_gain = 1.0
    asymmetry_mult_loss = 1.0
    try:
        asym_ratio = float(reward_room_pct) / max(float(failure_room_pct), 1e-9)
        asym_ratio = max(asym_ratio, 1e-9)
        log_asym = math.log(asym_ratio)
        asymmetry_mult_gain = 1.0 + 0.12 * math.tanh(log_asym / 0.70)
        asymmetry_mult_loss = 1.0 + 0.04 * math.tanh(log_asym / 0.90)
    except Exception:
        pass

    gain_pct = (
        float(reward_room_pct)
        * float(reward_eff)
        * wiggle
        * float(tape_mult)
        * float(cheap_gain_mult)
        * float(fill_quality_mult_gain)
        * float(asymmetry_mult_gain)
    )
    loss_pct = (
        float(failure_room_pct)
        * float(failure_tol)
        * wiggle
        * float(loss_tape_mult)
        * float(fill_quality_mult_loss)
        * float(asymmetry_mult_loss)
    )

    if gain_pct <= 0 or loss_pct <= 0:
        return None
    return (gain_pct, loss_pct, source)


def _threshold_engine_trigger_pcts(cfg: AutoExecConfig, lifecycle: TradeLifecycle) -> tuple[float, float, str]:
    """Return (gain_pct, loss_pct, source_label) for threshold-trigger sell mode."""
    try:
        generic_gain = float(getattr(cfg, "threshold_exit_gain_pct", 1.0) or 1.0)
    except Exception:
        generic_gain = 1.0
    try:
        generic_loss = float(getattr(cfg, "threshold_exit_loss_pct", 0.7) or 0.7)
    except Exception:
        generic_loss = 0.7

    try:
        use_engine_specific = bool(getattr(cfg, "threshold_exit_use_engine_specific", False))
    except Exception:
        use_engine_specific = False
    if not use_engine_specific:
        return (generic_gain, generic_loss, "GENERIC")

    try:
        engine = str(getattr(lifecycle, "engine", "") or "").upper()
    except Exception:
        engine = ""

    if engine == "SCALP":
        return (
            float(getattr(cfg, "threshold_exit_gain_pct_scalp", generic_gain) or generic_gain),
            float(getattr(cfg, "threshold_exit_loss_pct_scalp", generic_loss) or generic_loss),
            "SCALP",
        )
    if engine == "RIDE":
        return (
            float(getattr(cfg, "threshold_exit_gain_pct_ride", generic_gain) or generic_gain),
            float(getattr(cfg, "threshold_exit_loss_pct_ride", generic_loss) or generic_loss),
            "RIDE",
        )
    return (generic_gain, generic_loss, "GENERIC")


def _normalize_state_schemas(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize durable state schemas in-place.

    Ensures:
      - lifecycles: dict[str, list[dict]]
      - each lifecycle dict matches current TradeLifecycle schema
      - pool_reserved remains numeric

    This prevents "bad session state" issues when upgrading builds where the
    persisted dict schema may have missing/extra keys.
    """
    try:
        if not isinstance(state, dict):
            return state

        # Normalize lifecycles container
        lifecycles = state.get("lifecycles")
        if not isinstance(lifecycles, dict):
            lifecycles = {}
            state["lifecycles"] = lifecycles

        for sym, lst in list(lifecycles.items()):
            if not isinstance(sym, str):
                # Drop non-string symbol keys
                lifecycles.pop(sym, None)
                continue
            if not isinstance(lst, list):
                lst = []
            normalized: list = []
            for raw in lst:
                try:
                    lc = lifecycle_from_raw(raw)
                    # If symbol is missing or malformed, make it explicit and cancel.
                    if not lc.symbol or lc.symbol == "UNKNOWN":
                        lc.stage = "CANCELED"
                        lc.notes = (lc.notes + " | bad_state:missing_symbol").strip(" |")
                    normalized.append(asdict(lc))
                except Exception:
                    # Worst-case: drop irrecoverable entries (prevents poisoning state)
                    continue
            lifecycles[sym] = normalized

        # Normalize pool_reserved
        try:
            state["pool_reserved"] = float(state.get("pool_reserved", 0.0) or 0.0)
        except Exception:
            state["pool_reserved"] = 0.0

    except Exception:
        return state
    return state

def _now_et() -> datetime:
    # Streamlit environment should have tzdata; fall back to naive local if needed.
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(ET_TZ))
    except Exception:
        return datetime.now()


def _exec_window_label(now: datetime) -> str:
    """Return which exec window the time falls into: PREOPEN / OPENING / LATE_MORNING / MIDDAY / POWER / OFF."""
    t = now.time()
    if time(9, 30) <= t <= time(9, 50):
        return "PREOPEN"
    if time(9, 50) <= t <= time(11, 0):
        return "OPENING"
    if time(11, 0) <= t <= time(12, 0):
        return "LATE_MORNING"
    if time(14, 0) <= t <= time(15, 30):
        return "MIDDAY" if t < time(15, 0) else "POWER"
    return "OFF"


def _in_exec_window(now: datetime, cfg: AutoExecConfig) -> bool:
    """Two windows: 09:50–11:00 and 14:00–15:30 ET.

    IMPORTANT: execution windows are controlled by AutoExecConfig.exec_allow_* and are
    intentionally decoupled from the scanner session toggles.
    """
    label = _exec_window_label(now)
    if label == "PREOPEN":
        return bool(getattr(cfg, "exec_allow_preopen", True))
    if label == "OPENING":
        return bool(getattr(cfg, "exec_allow_opening", True))
    if label == "LATE_MORNING":
        return bool(getattr(cfg, "exec_allow_late_morning", True))
    if label == "MIDDAY":
        return bool(getattr(cfg, "exec_allow_midday", True))
    if label == "POWER":
        return bool(getattr(cfg, "exec_allow_power", True))
    return False


def _is_liquidation_time(now: datetime) -> bool:
    return now.time() >= time(15, 55)


def _get_state() -> Dict[str, Any]:
    """Return the durable autoexec state stored in st.session_state.

    Important: app.py may create st.session_state['autoexec'] during OAuth
    before this function is ever called. In that case, we *must not* wipe
    the auth tokens when we "initialize" the rest of the state.
    """

    today = _now_et().date().isoformat()

    # Start from whatever is present (OAuth flow may have created a partial dict)
    existing: Dict[str, Any] = st.session_state.get("autoexec", {}) or {}
    existing_auth = existing.get("auth", {}) if isinstance(existing, dict) else {}

    # Initialize / backfill missing keys without losing auth
    state: Dict[str, Any] = {
        "pool_reserved": float(existing.get("pool_reserved", 0.0)) if isinstance(existing, dict) else 0.0,
        "lifecycles": existing.get("lifecycles", {}) if isinstance(existing, dict) else {},
        "auth": existing_auth if isinstance(existing_auth, dict) else {},
        "day": str(existing.get("day", today)) if isinstance(existing, dict) else today,
        "skip_notices": existing.get("skip_notices", {}) if isinstance(existing, dict) else {},
        "hourly_report_last": str(existing.get("hourly_report_last", "")) if isinstance(existing, dict) else "",
        # Observability / reporting (must persist across Streamlit reruns)
        # NOTE: Without these keys, digest / hourly emails can spam on every rerun.
        "activity_log": existing.get("activity_log", []) if isinstance(existing, dict) else [],
        "activity_cutoff_ts": str(existing.get("activity_cutoff_ts", "")) if isinstance(existing, dict) else "",
        "digest_last_ts": str(existing.get("digest_last_ts", "")) if isinstance(existing, dict) else "",
        "digest_cutoff_ts": str(existing.get("digest_cutoff_ts", "")) if isinstance(existing, dict) else "",
        "last_action": str(existing.get("last_action", "")) if isinstance(existing, dict) else "",
        "realized_trades": existing.get("realized_trades", []) if isinstance(existing, dict) else [],
        "broker_ping": existing.get("broker_ping", {}) if isinstance(existing, dict) else {},
    }

    # Daily reset (preserve auth so "auth before boot" remains valid)
    if state.get("day") != today:
        state = {
            "pool_reserved": 0.0,
            "lifecycles": {},
            "auth": state.get("auth", {}),
            "day": today,
            "skip_notices": {},
            "hourly_report_last": "",
            # Reset reporting cursors daily, but keep structure so we don't spam.
            "activity_log": [],
            "activity_cutoff_ts": "",
            "digest_last_ts": "",
            "digest_cutoff_ts": "",
            "last_action": "",
            "realized_trades": [],
            "broker_ping": {},
        }

    st.session_state["autoexec"] = state

    # Schema normalize on every access so upgrades can't poison state.
    try:
        state = _normalize_state_schemas(state)
        st.session_state["autoexec"] = state
    except Exception:
        pass

    # Self-heal pool_reserved drift:
    # pool_reserved is intended to reflect ONLY the dollars reserved by ACTIVE lifecycles.
    # In rare Streamlit rerun/cancel edge-cases, pool_reserved may drift from the lifecycle
    # truth. We recompute from active lifecycles on every state access.
    try:
        lifecycles = state.get("lifecycles", {}) or {}
        s = 0.0
        for _, lst in lifecycles.items():
            for raw in (lst or []):
                stg = str((raw or {}).get("stage", ""))
                if stg in {"STAGED", "ENTRY_SENT", "IN_POSITION", "EXIT_SENT"}:
                    s += float((raw or {}).get("reserved_dollars", 0.0) or 0.0)
        if abs(float(state.get("pool_reserved", 0.0) or 0.0) - s) > 0.01:
            state["pool_reserved"] = float(s)
            st.session_state["autoexec"] = state
    except Exception:
        pass

    return state



def _email_settings():
    """Read SMTP settings from Streamlit secrets. Returns tuple or None."""
    try:
        cfg = st.secrets.get("email", {}) or {}
    except Exception:
        return None
    smtp_server = cfg.get("smtp_server")
    smtp_port = cfg.get("smtp_port")
    smtp_user = cfg.get("smtp_user")
    smtp_password = cfg.get("smtp_password")

    # Accept to_emails (preferred) OR to_email (string). Normalize to list[str].
    to_emails = cfg.get("to_emails")
    if to_emails is None:
        to_email = cfg.get("to_email", "")
        if isinstance(to_email, str):
            # Support comma-separated lists in legacy config.
            parts = [p.strip() for p in to_email.split(",") if p.strip()]
            to_emails = parts
        elif to_email:
            to_emails = [str(to_email).strip()]
        else:
            to_emails = []
    if isinstance(to_emails, str):
        to_emails = [e.strip() for e in to_emails.split(",") if e.strip()]
    if not (smtp_server and smtp_port and smtp_user and smtp_password and to_emails):
        return None
    try:
        smtp_port_int = int(smtp_port)
    except Exception:
        return None
    return smtp_server, smtp_port_int, str(smtp_user), str(smtp_password), [str(e).strip() for e in to_emails if str(e).strip()]


def _send_status_email(cfg: AutoExecConfig, subject: str, body: str) -> None:
    """Send auto-exec lifecycle status emails (one email per recipient)."""
    if not getattr(cfg, "status_emails", False):
        return
    settings = _email_settings()
    if settings is None:
        return
    smtp_server, smtp_port, smtp_user, smtp_password, to_emails = settings
    try:
        send_email_alert(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_emails=to_emails,
            subject=subject,
            body=body,
        )
    except Exception:
        # Never crash the app due to email
        return


def _market_session_for_now(now: datetime) -> str:
    try:
        sess = classify_session(pd.Timestamp(now))
    except Exception:
        sess = "OPENING"
    return "REGULAR" if str(sess).upper() in {"OPENING", "MIDDAY", "POWER"} else "EXTENDED"


def _tp0_step_email(cfg: AutoExecConfig, lifecycle: "TradeLifecycle", event_key: str, subject_suffix: str, body: str, *, once: bool = True) -> None:
    try:
        subj = f"[AUTOEXEC][TP0] {lifecycle.symbol} {subject_suffix}"
        if once:
            _event_once(cfg, lifecycle, event_key, subj, body)
        else:
            _send_status_email(cfg, subj, body)
    except Exception:
        pass


def _timeout_profit_state_block(
    cfg: AutoExecConfig,
    lifecycle: "TradeLifecycle",
    now: datetime,
    *,
    shared_last_px: Optional[float],
    entry_px: Optional[float],
    profit_basis_source: str,
    gain_pct: Optional[float],
    mins_cfg: float,
    thresh_cfg: float,
    rem_qty: Optional[int] = None,
) -> str:
    try:
        last_src = "cache_only" if bool(getattr(cfg, "reconcile_use_last_price_cache_only", False)) else "quote_or_cache"
    except Exception:
        last_src = "unknown"
    try:
        broker_qty = _get_broker_position_qty(lifecycle.symbol, force_refresh=True)
    except Exception:
        broker_qty = None
    gain_str = f"{gain_pct:.2f}%" if gain_pct is not None else "N/A"
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Flow: TIMEOUT_PROFIT\n"
        f"Lifecycle ID: {getattr(lifecycle, 'lifecycle_id', '')}\n"
        f"Symbol: {lifecycle.symbol}\n"
        f"Engine: {lifecycle.engine}\n"
        f"Stage: {lifecycle.stage}\n"
        f"Desired entry: {_fmt_price(lifecycle.desired_entry)}\n"
        f"Profit basis: {_fmt_price(entry_px)}\n"
        f"Profit basis source: {profit_basis_source}\n"
        f"Last price: {_fmt_price(shared_last_px)}\n"
        f"Last price source: {last_src}\n"
        f"Gain %: {gain_str}\n"
        f"Timeout minutes: {mins_cfg:.1f}\n"
        f"Threshold %: {thresh_cfg:.2f}\n"
        f"Stop order id: {lifecycle.stop_order_id or 'N/A'}\n"
        f"Exit order id: {lifecycle.market_exit_order_id or 'N/A'}\n"
        f"Filled qty: {int(lifecycle.filled_qty or 0)}\n"
        f"Broker qty: {broker_qty if broker_qty is not None else 'N/A'}\n"
        f"Remaining qty considered: {rem_qty if rem_qty is not None else 'N/A'}\n"
        f"Close submit started at: {lifecycle.close_submit_started_at or 'N/A'}\n"
        f"Entry eval breadcrumb: {lifecycle.last_entry_eval or 'N/A'}\n"
    )
    return body



def _threshold_exit_state_block(
    cfg: AutoExecConfig,
    lifecycle: "TradeLifecycle",
    now: datetime,
    *,
    shared_last_px: Optional[float],
    entry_px: Optional[float],
    profit_basis_source: str,
    gain_pct: Optional[float],
    loss_pct_now: Optional[float],
    gain_trigger_pct: float,
    loss_trigger_pct: float,
    broker_qty: Optional[int] = None,
    rem_qty: Optional[int] = None,
) -> str:
    try:
        broker_qty = int(broker_qty) if broker_qty is not None else None
    except Exception:
        broker_qty = None
    try:
        rem_qty = int(rem_qty) if rem_qty is not None else None
    except Exception:
        rem_qty = None
    return (
        f"Time (ET): {now.isoformat()}\n"
        f"Lifecycle ID: {lifecycle.lifecycle_id}\n"
        f"Symbol: {lifecycle.symbol}\n"
        f"Engine: {lifecycle.engine}\n"
        f"Stage: {lifecycle.stage}\n"
        f"Desired entry: {_fmt_price(lifecycle.desired_entry)}\n"
        f"Profit basis: {_fmt_price(entry_px)}\n"
        f"Profit basis source: {profit_basis_source or 'N/A'}\n"
        f"Last price: {_fmt_price(shared_last_px)}\n"
        f"Last price source: reconcile_fetch_last\n"
        f"Gain % vs basis: {f'{gain_pct:.3f}%' if gain_pct is not None else 'N/A'}\n"
        f"Loss % vs basis: {f'{loss_pct_now:.3f}%' if loss_pct_now is not None else 'N/A'}\n"
        f"Gain trigger %: {gain_trigger_pct:.3f}%\n"
        f"Loss trigger %: {loss_trigger_pct:.3f}%\n"
        f"Threshold active at: {lifecycle.threshold_exit_activated_at or 'N/A'}\n"
        f"Stop order id: {lifecycle.stop_order_id or 'N/A'}\n"
        f"Exit order id: {lifecycle.market_exit_order_id or 'N/A'}\n"
        f"Filled qty: {int(lifecycle.filled_qty or 0)}\n"
        f"Broker qty: {broker_qty if broker_qty is not None else 'N/A'}\n"
        f"Remaining qty considered: {rem_qty if rem_qty is not None else 'N/A'}\n"
        f"Close submit started at: {lifecycle.close_submit_started_at or 'N/A'}\n"
        f"Entry eval breadcrumb: {lifecycle.last_entry_eval or 'N/A'}\n"
    )


def _threshold_step_email(cfg: AutoExecConfig, lifecycle: "TradeLifecycle", event_key: str, subject_suffix: str, body: str, *, once: bool = True) -> None:
    _tp0_step_email(cfg, lifecycle, event_key, subject_suffix, body, once=once)


def _place_timeout_profit_market_close_explicit(
    client: "ETradeClient",
    account_id_key: str,
    symbol: str,
    qty: int,
    client_order_id: str,
    lifecycle: "TradeLifecycle",
    now: datetime,
) -> tuple[int, int, str]:
    market_session = _market_session_for_now(now)
    oid, pid = client.place_equity_market_order_ex(
        account_id_key=account_id_key,
        symbol=symbol,
        qty=int(qty),
        action="SELL",
        client_order_id=client_order_id,
        market_session=market_session,
    )
    _append_note(lifecycle, f"TIMEOUT_EXIT_PREVIEW_OK pid={pid} session={market_session}")
    _append_note(lifecycle, f"TIMEOUT_EXIT_PLACE_OK oid={oid}")
    return int(oid), int(pid), str(market_session)


def _should_send_hourly(now: datetime) -> Optional[str]:
    """Return an hourly key (YYYY-MM-DD:HH) if we are within the report window.

    We aim for "every hour" during the regular session. Because Streamlit reruns
    on a timer, we allow a small minute window so we don't miss the top of the hour.
    """
    # Monday=0 ... Sunday=6
    if now.weekday() > 4:
        return None

    t = now.time()
    # Regular session 09:30–16:00 ET
    if t < time(9, 30) or t > time(16, 0):
        return None

    # Send at 10:00, 11:00, ... 16:00 (inclusive). Allow minute window [0, 7].
    if now.hour < 10 or now.hour > 16:
        return None
    if not (0 <= now.minute <= 7):
        return None

    return f"{now.date().isoformat()}:{now.hour:02d}"




def _digest_activity_since_last(state: Dict[str, Any]) -> Tuple[list[dict], str]:
    """Return (events, cutoff_ts_iso) for digest emails (separate from hourly)."""
    cutoff = str(state.get("digest_cutoff_ts") or "")
    events = state.get("activity_log") or []
    if not isinstance(events, list):
        return ([], cutoff)
    if not cutoff:
        # default cutoff = start of today ET
        try:
            now = _now_et()
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        except Exception:
            cutoff = ""
    out_events = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = str(e.get("ts") or "")
        if cutoff and ts and ts <= cutoff:
            continue
        out_events.append(e)
    return (out_events, cutoff)


def _maybe_send_autoexec_digest(cfg: AutoExecConfig, state: Dict[str, Any], now: datetime) -> None:
    """Send a periodic auto-exec digest email (observability only)."""
    if not getattr(cfg, "digest_emails_enabled", False):
        return

    # Respect RTH-only if enabled.
    if getattr(cfg, "digest_rth_only", True):
        try:
            ts = pd.Timestamp(now)
            rth = classify_session(ts, allow_opening=True, allow_midday=True, allow_power=True, allow_premarket=False, allow_afterhours=False)
            if rth == "OFF":
                return
        except Exception:
            pass

    interval_min = int(getattr(cfg, "digest_interval_minutes", 15) or 15)
    interval_sec = max(60, min(3600, interval_min * 60))

    last_ts = str(state.get("digest_last_ts") or "")
    if last_ts:
        try:
            prev = datetime.fromisoformat(last_ts)
            if (now - prev).total_seconds() < interval_sec:
                return
        except Exception:
            # if parse fails, send and reset
            pass

    # Summarize lifecycle state
    lifecycles = state.get("lifecycles", {}) or {}
    stage_counts: Dict[str, int] = {}
    active_rows = []
    for sym, lst in lifecycles.items():
        if not isinstance(lst, list):
            continue
        for raw in lst:
            if not isinstance(raw, dict):
                continue
            stg = str(raw.get("stage") or "")
            stage_counts[stg] = stage_counts.get(stg, 0) + 1
            if stg in {"STAGED", "PRESTAGED", "ENTRY_SENT", "IN_POSITION", "EXIT_SENT", "CANCEL_PENDING"}:
                active_rows.append(raw)

    # Activity since last digest
    events, _cutoff = _digest_activity_since_last(state)
    counts: Dict[str, int] = {}
    for e in events:
        k = str(e.get("kind") or "")
        counts[k] = counts.get(k, 0) + 1

    # Broker ping status
    bp = state.get("broker_ping") or {}
    bp_ok = None
    bp_err = ""
    bp_ts = ""
    if isinstance(bp, dict):
        if "ok" in bp:
            bp_ok = bool(bp.get("ok"))
        bp_err = str(bp.get("err") or "")
        bp_ts = str(bp.get("ts") or "")

    subj = f"[AUTOEXEC] Digest ({now.strftime('%H:%M')} ET)"

    lines: list[str] = []
    lines.append(f"Time (ET): {now.isoformat()}")
    lines.append(f"Env: {'SANDBOX' if getattr(cfg, 'sandbox', True) else 'LIVE'}")
    if bp_ok is not None:
        extra = f" ({bp_ts})" if bp_ts else ""
        if (not bp_ok) and bp_err:
            extra += f" — {bp_err}"
        lines.append(f"Broker ping: {'OK' if bp_ok else 'FAILED'}{extra}")

    lines.append("")
    lines.append("Lifecycle counts:")
    if stage_counts:
        for k in sorted(stage_counts.keys()):
            lines.append(f"  • {k}: {stage_counts[k]}")
    else:
        lines.append("  • —")

    lines.append("")
    lines.append("Activity since last digest:")
    if counts:
        for k in sorted(counts.keys()):
            lines.append(f"  • {k}: {counts[k]}")
    else:
        lines.append("  • —")

    if active_rows:
        lines.append("")
        lines.append("Active lifecycles (top 12):")
        def _cts(r):
            return str(r.get("created_ts") or "")
        for r in sorted(active_rows, key=_cts)[-12:]:
            sym = str(r.get("symbol") or "")
            stg = str(r.get("stage") or "")
            oid = str(r.get("entry_order_id") or "")
            evc = r.get("entry_eval_count")
            rcv = r.get("reconcile_eval_count")
            last_eval = str(r.get("last_entry_eval") or "")
            notes = str(r.get("notes") or "")
            evc_s = ""
            rcv_s = ""
            try:
                if evc is not None:
                    evc_s = f" | evals={int(evc)}"
            except Exception:
                evc_s = f" | evals={evc}"
            try:
                if rcv is not None:
                    rcv_s = f" | recon={int(rcv)}"
            except Exception:
                rcv_s = f" | recon={rcv}"
            lines.append(f"  • {sym} | {stg} | entry_oid={oid or '—'}{evc_s}{rcv_s}")
            if last_eval:
                lines.append(f"      last_entry_eval: {last_eval}")
            if notes:
                lines.append(f"      notes: {notes}")

    # Persist digest cursor
    state["digest_last_ts"] = now.isoformat()
    state["digest_cutoff_ts"] = now.isoformat()

    _send_status_email(cfg, subj, "\n".join(lines) + "\n")
def _extract_positions(portfolio_json: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Best-effort extraction of position objects from E*TRADE portfolio JSON."""

    positions: list[Dict[str, Any]] = []

    def _walk(obj):
        if isinstance(obj, dict):
            # E*TRADE commonly uses key 'Position' or 'position' for lists
            for k, v in obj.items():
                if str(k).lower() == "position":
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                positions.append(it)
                    elif isinstance(v, dict):
                        positions.append(v)
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it)

    _walk(portfolio_json)
    return positions


def _pos_symbol(pos: Dict[str, Any]) -> str:
    # Common: pos['Product']['symbol']
    try:
        sym = pos.get("Product", {}).get("symbol")
        if sym:
            return str(sym).upper().strip()
    except Exception:
        pass
    # Sometimes: pos['product']['symbol']
    try:
        sym = pos.get("product", {}).get("symbol")
        if sym:
            return str(sym).upper().strip()
    except Exception:
        pass
    # Fallback: traverse
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == "symbol" and isinstance(v, str):
                    return v
                r = _walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for it in obj:
                r = _walk(it)
                if r:
                    return r
        return ""
    sym = _walk(pos)
    return str(sym).upper().strip() if sym else ""



def _safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_position_qty(pos: Dict[str, Any]) -> int:
    for key in ("quantity", "qty", "positionQuantity", "positionQty"):
        try:
            if key in pos and pos.get(key) is not None:
                return int(float(pos.get(key)))
        except Exception:
            pass
    return 0


def _walk_numeric_candidates(obj: Any, wanted_keys: set[str]) -> list[float]:
    vals: list[float] = []

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower().strip()
                if lk in wanted_keys:
                    num = _safe_num(v)
                    if num is not None:
                        vals.append(float(num))
                _walk(v)
        elif isinstance(x, list):
            for it in x:
                _walk(it)

    _walk(obj)
    return vals


def _extract_position_entry_avg(pos: Dict[str, Any]) -> Optional[float]:
    """Best-effort extraction of avg entry/cost basis from an E*TRADE position row."""
    direct_keys = {
        "pricepaid", "averageprice", "avgprice", "costpershare", "averagecost",
        "averagecostbasis", "costbasispershare", "costbasisprice", "avgcost",
    }
    for val in _walk_numeric_candidates(pos, direct_keys):
        if val is not None and val > 0:
            return float(val)

    qty = _extract_position_qty(pos)
    if qty > 0:
        total_cost_keys = {"totalcost", "costbasis", "costbasisamount", "positioncost", "bookvalue"}
        for total in _walk_numeric_candidates(pos, total_cost_keys):
            try:
                total_f = float(total)
                if total_f > 0:
                    avg = total_f / float(qty)
                    if avg > 0:
                        return float(avg)
            except Exception:
                continue
    return None


def _maybe_send_hourly_pnl(cfg: AutoExecConfig, state: Dict[str, Any], client: Optional[ETradeClient]) -> None:
    """Send an hourly P&L + analytics email during the regular session.

    IMPORTANT: This function must never crash the app. It is a best-effort
    reporting feature and must not contain execution invariants.
    """
    if not getattr(cfg, "hourly_pnl_emails", False):
        return

    now = _now_et()
    key = _should_send_hourly(now)
    if not key:
        return

    last = str(state.get("hourly_report_last", "") or "")
    if last == key:
        return

    account_id_key = (state.get("auth", {}) or {}).get("account_id_key")
    if not (client and account_id_key):
        return

    lifecycles = state.get("lifecycles", {}) or {}
    staged = entry_sent = in_pos = closed = canceled = 0
    active_symbols = 0
    managed_symbols: set[str] = set()
    for sym, lst in lifecycles.items():
        managed_symbols.add(str(sym).upper())
        sym_active = False
        for raw in (lst or []):
            stg = str((raw or {}).get("stage", ""))
            if stg == "STAGED":
                staged += 1
            elif stg == "ENTRY_SENT":
                entry_sent += 1
                sym_active = True
            elif stg == "IN_POSITION":
                in_pos += 1
                sym_active = True
            elif stg == "EXIT_SENT":
                entry_sent += 1
                sym_active = True
            elif stg == "CLOSED":
                closed += 1
            elif stg == "CANCELED":
                canceled += 1
        if sym_active:
            active_symbols += 1

    port_lines: list[str] = []
    total_mkt = total_gl = 0.0
    try:
        pj = client.get_portfolio(str(account_id_key))
        pos_list = _extract_positions(pj)
        rows = []
        for p in pos_list:
            sym = _pos_symbol(p)
            if not sym:
                continue
            if managed_symbols and sym not in managed_symbols:
                continue
            qty = _safe_num(p.get("quantity") or p.get("Quantity"))
            mv = _safe_num(p.get("marketValue") or p.get("MarketValue"))
            gl = _safe_num(p.get("totalGainLoss") or p.get("TotalGainLoss") or p.get("unrealizedGainLoss") or p.get("UnrealizedGainLoss"))
            if mv is not None:
                total_mkt += mv
            if gl is not None:
                total_gl += gl
            rows.append((sym, qty, mv, gl))
        if rows:
            port_lines.append("Bot-managed positions (E*TRADE portfolio):")
            for sym, qty, mv, gl in sorted(rows, key=lambda x: x[0]):
                q = "—" if qty is None else f"{qty:.0f}"
                m = "—" if mv is None else f"${mv:,.2f}"
                g = "—" if gl is None else f"${gl:,.2f}"
                port_lines.append(f"  • {sym}: qty {q} | mkt {m} | P&L {g}")
            port_lines.append(f"Totals (managed): market ${total_mkt:,.2f} | P&L ${total_gl:,.2f}")
        else:
            port_lines.append("No bot-managed positions found in portfolio snapshot.")
    except Exception as e:
        port_lines.append(f"Portfolio snapshot unavailable: {e}")

    order_lines: list[str] = []
    try:
        oj = client.list_orders(str(account_id_key), status="OPEN", count=50)
        open_orders = 0
        def _walk_orders(obj):
            nonlocal open_orders
            if isinstance(obj, dict):
                if "Instrument" in obj and isinstance(obj.get("Instrument"), list):
                    sym = ""
                    try:
                        sym = str(obj.get("Instrument")[0].get("Product", {}).get("symbol", "") or "").upper().strip()
                    except Exception:
                        sym = ""
                    if not managed_symbols or (sym and sym in managed_symbols):
                        open_orders += 1
                for v in obj.values():
                    _walk_orders(v)
            elif isinstance(obj, list):
                for it in obj:
                    _walk_orders(it)
        _walk_orders(oj)
        order_lines.append(f"Open orders (managed): {open_orders}")
    except Exception as e:
        order_lines.append(f"Open orders snapshot unavailable: {e}")

    # --- Activity since last report (lightweight ledger) ---
    events, cutoff = _activity_since_last_report(state)
    activity_lines: list[str] = []
    if events:
        # Aggregate counts
        counts: Dict[str, int] = {}
        for e in events:
            k = str(e.get("kind") or "UNKNOWN").upper()
            counts[k] = counts.get(k, 0) + 1
        activity_lines.append("Activity since last report:")
        # ordered buckets
        for k in ("ENTRY_PLACED", "ENTRY_CANCELED", "ENTRY_TIMEOUT", "BRACKETS_SENT", "EXIT_EXECUTED", "CLOSE", "FLATTEN", "CLEANUP"):
            if k in counts:
                activity_lines.append(f"  • {k}: {counts[k]}")
        # show up to 12 most recent lines
        activity_lines.append("Recent events:")
        for e in events[-12:]:
            ts = str(e.get("ts") or "")
            sym = str(e.get("symbol") or "")
            kind = str(e.get("kind") or "")
            det = str(e.get("details") or "")
            activity_lines.append(f"  • {ts} | {sym} | {kind} {('- ' + det) if det else ''}".rstrip())
    else:
        activity_lines.append("Activity since last report: —")

    subj = f"[AUTOEXEC] Hourly P&L Update — {now.hour:02d}:00 ET"
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Environment: {'SANDBOX' if cfg.sandbox else 'LIVE'}\n\n"
        f"Last bot action: {str(state.get('last_action', '') or '—')}\n\n"
        + "\n".join(activity_lines)
        + "\n\n"
        f"Auto‑exec today:\n"
        f"  • Active symbols: {active_symbols}\n"
        f"  • Lifecycles — STAGED: {staged}, ENTRY_SENT/EXIT_SENT: {entry_sent}, IN_POSITION: {in_pos}, CLOSED: {closed}, CANCELED: {canceled}\n"
        f"  • Pool reserved: ${float(state.get('pool_reserved', 0.0) or 0.0):,.2f} / ${float(cfg.max_pool_dollars):,.2f}\n"
        f"  • Realized (today): {_format_realized_today(state)}\n\n"
        + "\n".join(port_lines)
        + "\n\n"
        + "\n".join(order_lines)
        + "\n\n"
        "Note: This report is informational only. Execution logic remains governed by your stop-loss + TP0 rules."
    )
    _send_status_email(cfg, subj, body)
    state["hourly_report_last"] = key
    state["activity_cutoff_ts"] = now.isoformat()

def _event_once(cfg: AutoExecConfig, lifecycle: TradeLifecycle, event_key: str, subject: str, body: str) -> None:
    """Dedup: send an event email once per lifecycle per event_key."""
    try:
        sent = (lifecycle.emailed_events or {}).get(event_key)
    except Exception:
        sent = None
    if sent:
        return
    _send_status_email(cfg, subject, body)
    try:
        lifecycle.emailed_events[event_key] = _now_et().isoformat()
    except Exception:
        pass



def _maybe_email_entry_skip(cfg: AutoExecConfig, lifecycle: TradeLifecycle, now: datetime, reason_code: str, detail: str = "") -> None:
    """Email once when a lifecycle is STAGED but an entry is not sent (observability)."""
    try:
        if not bool(getattr(cfg, "status_emails", False)):
            return
        if not bool(getattr(cfg, "email_on_entry_skip", True)):
            return
        if str(getattr(lifecycle, "stage", "") or "").upper() != "STAGED":
            return
        # Dedup by reason code
        code = ''.join(ch for ch in str(reason_code or "UNKNOWN").upper() if ch.isalnum() or ch == '_')[:40] or "UNKNOWN"
        event_key = f"ENTRY_NOT_SENT_{code}"
        # Compute timeout remaining (best-effort)
        rem = "—"
        try:
            created = datetime.fromisoformat(lifecycle.created_ts)
            age_min = (now - created).total_seconds() / 60.0
            timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
            rem = f"{max(0, timeout_m - int(age_min))}m remaining (timeout={timeout_m}m)"
        except Exception:
            pass

        subj = f"[AUTOEXEC] {lifecycle.symbol} ENTRY NOT SENT ({code})"
        body = (
            f"Time (ET): {now.isoformat()}\n"
            f"Symbol: {lifecycle.symbol}\n"
            f"Engine: {lifecycle.engine}\n\n"
            f"Stage: {lifecycle.stage}\n"
            f"Reason: {code}\n"
            + (f"Detail: {detail}\n" if detail else "")
            + f"\nWill keep evaluating this STAGED lifecycle on each refresh. {rem}.\n"
        )
        _event_once(cfg, lifecycle, event_key, subj, body)
    except Exception:
        pass


def _record_realized_trade_on_close(state: Dict[str, Any], lifecycle: TradeLifecycle, client: ETradeClient, account_id_key: str, reason: str) -> None:
    """Record best-effort realized P&L for a lifecycle close.

    This is for reporting only. Execution logic does not depend on it.
    """
    try:
        key = f"{lifecycle.symbol}:{lifecycle.created_ts}"
    except Exception:
        key = f"{getattr(lifecycle, 'symbol', 'UNK')}:{getattr(lifecycle, 'created_ts', '')}"

    ledger = state.setdefault("realized_trades", [])
    try:
        if any(isinstance(r, dict) and r.get("key") == key for r in ledger):
            return
    except Exception:
        pass

    entry_oid = _oid_int(lifecycle.entry_order_id)
    if entry_oid is None:
        # Can't compute without entry reference
        ledger.append(
            {
                "key": key,
                "symbol": lifecycle.symbol,
                "engine": lifecycle.engine,
                "closed_ts": _now_et().isoformat(),
                "reason": reason,
                "realized": None,
                "note": "missing_entry_order_id",
            }
        )
        return

    try:
        e_filled, e_avg = client.get_order_filled_and_avg_price(account_id_key, entry_oid)
    except Exception:
        e_filled, e_avg = (int(lifecycle.filled_qty or 0), None)

    # Gather exits (stop/tp0/market remainder)
    exit_components = []
    for label, oid_str in (
        ("STOP", lifecycle.stop_order_id),
        ("TP0", lifecycle.tp0_order_id),
        ("MKT", lifecycle.market_exit_order_id),
    ):
        oid = _oid_int(oid_str)
        if oid is None:
            continue
        try:
            fqty, avg = client.get_order_filled_and_avg_price(account_id_key, oid)
        except Exception:
            fqty, avg = (0, None)
        if int(fqty or 0) > 0:
            exit_components.append((label, int(fqty), avg))

    total_exit_qty = sum(q for _, q, _ in exit_components)
    realized = None
    note = ""
    if e_avg is None and lifecycle.entry_avg_price_cached is not None:
        try:
            e_avg = float(lifecycle.entry_avg_price_cached)
            note = "entry_avg_from_lifecycle_cache"
        except Exception:
            e_avg = None
    if e_avg is None:
        note = "missing_entry_avg_price"
    elif total_exit_qty <= 0:
        note = "no_exit_fills_detected"
    else:
        proceeds = 0.0
        missing = False
        for _, q, avg in exit_components:
            if avg is None:
                missing = True
                continue
            proceeds += float(q) * float(avg)
        if missing:
            note = "missing_exit_avg_price"
        else:
            realized = proceeds - float(total_exit_qty) * float(e_avg)

    ledger.append(
        {
            "key": key,
            "symbol": lifecycle.symbol,
            "engine": lifecycle.engine,
            "closed_ts": _now_et().isoformat(),
            "reason": reason,
            "qty": int(total_exit_qty or 0),
            "entry_avg": e_avg,
            "realized": realized,
            "note": note,
        }
    )
    try:
        if len(ledger) > 100:
            del ledger[:-100]
    except Exception:
        pass




def _has_active_lifecycle(state: dict, symbol: str) -> bool:
    """Return True if the symbol has any lifecycle that is not fully closed/canceled."""
    try:
        lifecycles = (state.get('lifecycles') or {}).get(symbol, [])
    except Exception:
        return False
    for lc in lifecycles:
        # lifecycles are persisted as plain dicts in st.session_state
        try:
            if isinstance(lc, dict):
                stage = str(lc.get('stage', '') or '').upper()
            else:
                stage = str(getattr(lc, 'stage', '') or '').upper()
        except Exception:
            stage = ''
        if stage and stage not in {'CLOSED', 'CANCELED'}:
            return True
    return False

def _ensure_brackets(client, account_id_key: str, symbol: str, lifecycle, cfg: AutoExecConfig, log_prefix: str = '') -> None:
    # Threshold-mode lifecycles must never place/repair broker stops.
    if _effective_exit_mode(lifecycle) == "THRESHOLD":
        return
    """Best-effort safety invariant for Streamlit reruns.

    If we are IN_POSITION, we must have a broker-recognized protective STOP SELL order. TP0 is handled as a trigger.
    Do *not* assume preview responses are authoritative; verify broker truth via OPEN orders.
    """
    try:
        now = _now_et()
        stage = (lifecycle.stage or '').upper()
        if stage != 'IN_POSITION':
            return
        try:
            filled_qty = int(lifecycle.filled_qty or 0)
        except Exception:
            filled_qty = 0
        if filled_qty <= 0:
            return

        # One snapshot per call (avoid hammering the broker).
        open_orders = _extract_open_orders_for_symbol(client, account_id_key, symbol)

        def _eps_for(price: float) -> float:
            # E*TRADE price increments are typically $0.01; use a tiny buffer for float noise.
            return 0.011

        def _attach_from_broker() -> None:
            """Attach broker OPEN orders to lifecycle if we are missing orderIds."""
            nonlocal open_orders
            if not open_orders:
                return

            def _raw_client_oid(order_row: dict) -> str:
                raw = (order_row or {}).get('raw') or {}
                direct = raw.get('clientOrderId') or raw.get('client_order_id')
                if direct not in (None, ''):
                    return str(direct).strip()
                od = raw.get('OrderDetail', raw.get('orderDetail', []))
                if isinstance(od, dict):
                    od = [od]
                if isinstance(od, list):
                    for detail in od:
                        if not isinstance(detail, dict):
                            continue
                        val = detail.get('clientOrderId') or detail.get('client_order_id')
                        if val not in (None, ''):
                            return str(val).strip()
                return ''

            # STOP attach
            if not lifecycle.stop_order_id and lifecycle.stop is not None:
                try:
                    tgt = float(getattr(lifecycle, "broker_stop", None) or lifecycle.stop)
                except Exception:
                    tgt = None
                stop_coid = str(getattr(lifecycle, 'stop_client_order_id', '') or '').strip()
                if tgt is not None:
                    for o in open_orders:
                        if (o.get('action') or '').upper() != 'SELL':
                            continue
                        ot = (o.get('order_type') or '').upper()
                        if 'STOP' not in ot:
                            continue
                        oq = o.get('qty')
                        if oq is not None and int(oq) != int(filled_qty):
                            continue
                        broker_coid = _raw_client_oid(o)
                        if stop_coid and broker_coid and broker_coid == stop_coid:
                            lifecycle.stop_order_id = str(o.get('order_id'))
                            _append_note(lifecycle, f"STOP_ATTACH_BROKER_COID oid={lifecycle.stop_order_id}")
                            _log(f"{log_prefix}Attached STOP from broker truth via clientOrderId oid={lifecycle.stop_order_id} sym={symbol}")
                            break
                        sp = o.get('stop_price')
                        if sp is None:
                            continue
                        if abs(float(sp) - tgt) <= _eps_for(tgt):
                            lifecycle.stop_order_id = str(o.get('order_id'))
                            _append_note(lifecycle, f"STOP_ATTACH_BROKER oid={lifecycle.stop_order_id}")
                            _log(f"{log_prefix}Attached STOP from broker truth oid={lifecycle.stop_order_id} sym={symbol}")
                            break

            # TP0 attach
            if not lifecycle.tp0_order_id and lifecycle.tp0 is not None:
                try:
                    tgt = float(lifecycle.tp0)
                except Exception:
                    tgt = None
                tp0_coid = str(getattr(lifecycle, 'tp0_client_order_id', '') or '').strip()
                if tgt is not None:
                    for o in open_orders:
                        if (o.get('action') or '').upper() != 'SELL':
                            continue
                        ot = (o.get('order_type') or '').upper()
                        if 'LIMIT' not in ot:
                            continue
                        oq = o.get('qty')
                        if oq is not None and int(oq) != int(filled_qty):
                            continue
                        broker_coid = _raw_client_oid(o)
                        if tp0_coid and broker_coid and broker_coid == tp0_coid:
                            lifecycle.tp0_order_id = str(o.get('order_id'))
                            _append_note(lifecycle, f"TP0_ATTACH_BROKER_COID oid={lifecycle.tp0_order_id}")
                            _log(f"{log_prefix}Attached TP0 from broker truth via clientOrderId oid={lifecycle.tp0_order_id} sym={symbol}")
                            break
                        lp = o.get('limit_price')
                        if lp is None:
                            continue
                        if abs(float(lp) - tgt) <= _eps_for(tgt):
                            lifecycle.tp0_order_id = str(o.get('order_id'))
                            _append_note(lifecycle, f"TP0_ATTACH_BROKER oid={lifecycle.tp0_order_id}")
                            _log(f"{log_prefix}Attached TP0 from broker truth oid={lifecycle.tp0_order_id} sym={symbol}")
                            break

        def _parse_iso(ts: str):
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                return None

        def _should_attempt(key: str, min_seconds: int = 90) -> bool:
            ts = getattr(lifecycle, key, None)
            if not ts:
                return True
            dt = _parse_iso(str(ts))
            if not dt:
                return True
            try:
                return (now - dt).total_seconds() >= float(min_seconds)
            except Exception:
                return True

        # 0) First, try to attach any already-open broker brackets (manual or previously succeeded).
        _attach_from_broker()
        # 1) TP0 leg
        # Synthetic bracket mode: TP0 is a trigger (not a resting order) to avoid E*TRADE share reservation (error 1037).
        # TP0 placement is handled in _reconcile_one() when price reaches the TP0 trigger.

        # 2) STOP leg
        if not lifecycle.stop_order_id and lifecycle.stop is not None and _should_attempt('stop_last_attempt_at'):
            lifecycle.stop_last_attempt_at = now.isoformat()
            stop_coid = str(lifecycle.stop_client_order_id or _mk_client_order_id(lifecycle.lifecycle_id, 'ST'))
            lifecycle.stop_client_order_id = stop_coid
            broker_stop_price, applied_stop_buffer = _compute_broker_stop_price(lifecycle, cfg)
            lifecycle.broker_stop = broker_stop_price
            lifecycle.stop_buffer_applied = applied_stop_buffer
            if _recent_submit(getattr(lifecycle, 'stop_submit_started_at', None), now, 90):
                open_orders = _extract_open_orders_for_symbol(client, account_id_key, symbol)
                _attach_from_broker()
                if not lifecycle.stop_order_id:
                    _append_note(lifecycle, 'STOP_SUBMIT_INFLIGHT<90s')
            else:
                lifecycle.stop_submit_started_at = now.isoformat()
                try:
                    stop_oid, stop_pid = client.place_equity_stop_order_ex(
                        account_id_key=account_id_key,
                        symbol=symbol,
                        action='SELL',
                        qty=int(filled_qty),
                        stop_price=float(broker_stop_price),
                        client_order_id=stop_coid,
                    )
                    lifecycle.stop_order_id = str(stop_oid) if stop_oid else None
                    if lifecycle.stop_order_id:
                        _append_note(lifecycle, f"STOP_PREVIEW_OK pid={stop_pid}")
                        _append_note(lifecycle, f"STOP_PLACE_OK oid={stop_oid}")
                        _log(f"{log_prefix}STOP placed oid={lifecycle.stop_order_id} sym={symbol}")
                except Exception as e:
                    emsg = str(e)
                    _append_note(lifecycle, f"STOP_PLACE_ERR {emsg[:200]}")
                    recovered = None
                    try:
                        recovered = client.find_order_by_client_order_id(
                            account_id_key=account_id_key,
                            client_order_id=stop_coid,
                            symbol=symbol,
                        )
                    except Exception:
                        recovered = None
                    if recovered and recovered.get('orderId'):
                        lifecycle.stop_order_id = str(recovered.get('orderId'))
                        _append_note(lifecycle, f"STOP_ATTACH_DUP_GUARD oid={lifecycle.stop_order_id}")
                    else:
                        open_orders = _extract_open_orders_for_symbol(client, account_id_key, symbol)
                        _attach_from_broker()

    except Exception as e:
        _log(f'{log_prefix}Bracket ensure failed for {symbol}: {e}')
def _compute_broker_stop_price(lifecycle: TradeLifecycle, cfg: AutoExecConfig) -> tuple[float, float]:
    """Return (broker_stop_price, applied_buffer).

    The signal/payload stop remains the thesis stop on the lifecycle. This helper only
    widens the actual broker-submitted protective stop when the operator enables the
    execution-layer stop buffer.
    """
    try:
        base_stop = float(lifecycle.stop or 0.0)
    except Exception:
        base_stop = 0.0
    if base_stop <= 0:
        return base_stop, 0.0

    use_buf = bool(getattr(cfg, "use_stop_buffer", False))
    try:
        raw_buf = float(getattr(cfg, "stop_buffer_amount", 0.01) or 0.0)
    except Exception:
        raw_buf = 0.0
    buf = min(max(raw_buf, 0.0), 0.05)
    if (not use_buf) or buf <= 0:
        return float(_tick_round(base_stop) or base_stop), 0.0

    try:
        risk = max(0.0, float(lifecycle.desired_entry or 0.0) - base_stop)
    except Exception:
        risk = 0.0
    if risk > 0:
        # Never let the widening exceed half of the original stop distance. This keeps
        # the execution-layer tweak bounded and prevents turning the thesis into a new trade.
        buf = min(buf, risk * 0.5)

    widened = max(0.01, base_stop - buf)
    widened = float(_tick_round(widened) or widened)
    try:
        desired_entry = float(lifecycle.desired_entry or 0.0)
    except Exception:
        desired_entry = 0.0
    if desired_entry > 0 and widened >= desired_entry:
        widened = float(_tick_round(base_stop) or base_stop)
        buf = 0.0
    return widened, max(0.0, round(base_stop - widened, 4))


def _extract_open_orders_for_symbol(client, account_id_key: str, symbol: str) -> list:
    """Best-effort: list OPEN orders for a symbol and return a normalized list of dicts.

    Used for orphan recovery + verifying whether broker-side brackets exist.
    """
    try:
        data = client.list_orders(account_id_key, status="OPEN", count=100, symbol=str(symbol).upper().strip())
    except Exception:
        return []
    orders = (
        data.get("OrdersResponse", {})
        .get("Orders", {})
        .get("Order", [])
        if isinstance(data, dict)
        else []
    )
    if isinstance(orders, dict):
        orders = [orders]
    out = []
    for o in orders or []:
        if not isinstance(o, dict):
            continue
        try:
            oid = int(o.get("orderId", o.get("orderID", 0)) or 0)
        except Exception:
            oid = 0
        order_type = str(o.get("orderType", o.get("ordType", "")) or "").upper()
        action = str(o.get("orderAction", o.get("action", "")) or "").upper()

        limit_price = None
        stop_price = None
        try:
            od = o.get("OrderDetail", o.get("orderDetail", []))
            if isinstance(od, dict):
                od = [od]
            if od and isinstance(od[0], dict):
                lp = od[0].get("limitPrice", od[0].get("price"))
                sp = od[0].get("stopPrice")
                limit_price = float(lp) if lp not in (None, "") else None
                stop_price = float(sp) if sp not in (None, "") else None
        except Exception:
            pass
        if limit_price is None:
            try:
                lp = o.get("limitPrice", o.get("price"))
                limit_price = float(lp) if lp not in (None, "") else None
            except Exception:
                limit_price = None
        if stop_price is None:
            try:
                sp = o.get("stopPrice")
                stop_price = float(sp) if sp not in (None, "") else None
            except Exception:
                stop_price = None

        qty = None
        try:
            qv = None
            # Try common E*TRADE fields first
            for k in ("orderedQuantity", "quantity", "orderQty", "qty"):
                if k in o and o.get(k) not in (None, ""):
                    qv = o.get(k)
                    break
            if qv is None:
                od = o.get("OrderDetail", o.get("orderDetail", []))
                if isinstance(od, dict):
                    od = [od]
                if od and isinstance(od[0], dict):
                    for k in ("orderedQuantity", "quantity", "orderQty", "qty"):
                        if k in od[0] and od[0].get(k) not in (None, ""):
                            qv = od[0].get(k)
                            break
            if qv is not None:
                qty = int(float(qv))
        except Exception:
            qty = None

        out.append(
            {
                "order_id": oid,
                "order_type": order_type,
                "action": action,
                "qty": qty,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "raw": o,
            }
        )
    return out




def _extract_client_order_id_from_open_order_row(order_row: dict) -> str:
    """Best-effort extraction of clientOrderId from a normalized OPEN order row."""
    try:
        raw = (order_row or {}).get('raw') or {}
        direct = raw.get('clientOrderId') or raw.get('client_order_id')
        if direct not in (None, ''):
            return str(direct).strip()
        od = raw.get('OrderDetail', raw.get('orderDetail', []))
        if isinstance(od, dict):
            od = [od]
        if isinstance(od, list):
            for detail in od:
                if not isinstance(detail, dict):
                    continue
                val = detail.get('clientOrderId') or detail.get('client_order_id')
                if val not in (None, ''):
                    return str(val).strip()
    except Exception:
        pass
    return ''


def _match_open_stop_order(open_orders: list, lifecycle, filled_qty: int | None = None) -> Optional[dict]:
    """Find the lifecycle's broker-recognized OPEN STOP order, if any."""
    try:
        target_qty = int(filled_qty if filled_qty is not None else (getattr(lifecycle, 'filled_qty', 0) or 0))
    except Exception:
        target_qty = 0
    stop_oid = str(getattr(lifecycle, 'stop_order_id', '') or '').strip()
    stop_coid = str(getattr(lifecycle, 'stop_client_order_id', '') or '').strip()
    try:
        target_stop = float(getattr(lifecycle, 'broker_stop', None) or getattr(lifecycle, 'stop', None) or 0.0)
    except Exception:
        target_stop = 0.0
    eps = 0.011
    for o in open_orders or []:
        if not isinstance(o, dict):
            continue
        if str(o.get('action') or '').upper() != 'SELL':
            continue
        if 'STOP' not in str(o.get('order_type') or '').upper():
            continue
        broker_oid = str(o.get('order_id') or '').strip()
        broker_coid = _extract_client_order_id_from_open_order_row(o)
        try:
            oq = o.get('qty')
            if target_qty > 0 and oq is not None and int(oq) != int(target_qty):
                continue
        except Exception:
            pass
        if stop_oid and broker_oid and broker_oid == stop_oid:
            return o
        if stop_coid and broker_coid and broker_coid == stop_coid:
            return o
        try:
            sp = o.get('stop_price')
            if target_stop > 0 and sp is not None and abs(float(sp) - float(target_stop)) <= eps:
                return o
        except Exception:
            pass
    return None


def _cancel_stop_and_confirm_absent(client, account_id_key: str, symbol: str, lifecycle, filled_qty: int | None = None, note_prefix: str = 'STOP_CANCEL') -> bool:
    """Best-effort cancel of the protective STOP, only clearing local ids when broker truth confirms final cancellation.

    Returns True only when the STOP is confirmed finalized/inactive (CANCELLED/REJECTED/EXPIRED),
    or there is no known stop and broker truth also does not show one.
    Returns False when the STOP still appears OPEN/CANCEL_REQUESTED/UNKNOWN or broker truth is ambiguous.
    """
    known_oid = str(getattr(lifecycle, 'stop_order_id', '') or '').strip()
    try:
        open_orders = _extract_open_orders_for_symbol(client, account_id_key, symbol)
    except Exception:
        open_orders = []
    existing = _match_open_stop_order(open_orders, lifecycle, filled_qty=filled_qty)

    broker_oid = existing.get('order_id') if isinstance(existing, dict) else None
    chosen_oid = known_oid or broker_oid or None
    if known_oid and broker_oid and str(known_oid) != str(broker_oid):
        _append_note(lifecycle, f"{note_prefix}_OID_MISMATCH known={known_oid} broker={broker_oid}")
    if chosen_oid not in (None, '', 0):
        lifecycle.stop_order_id = str(chosen_oid)

    try:
        oid_int = int(chosen_oid) if chosen_oid not in (None, '', 0) else None
    except Exception:
        oid_int = None

    if isinstance(existing, dict) and oid_int is not None:
        try:
            client.cancel_order(account_id_key, oid_int)
            _append_note(lifecycle, f"{note_prefix}_OK oid={oid_int}")
        except Exception as ce:
            _append_note(lifecycle, f"{note_prefix}_ERR {str(ce)[:180]}")
    elif isinstance(existing, dict):
        _append_note(lifecycle, f"{note_prefix}_NO_OID")

    try:
        refreshed = _extract_open_orders_for_symbol(client, account_id_key, symbol)
    except Exception:
        refreshed = []
    still_open = _match_open_stop_order(refreshed, lifecycle, filled_qty=filled_qty)
    if isinstance(still_open, dict):
        broker_oid2 = still_open.get('order_id')
        if broker_oid2 not in (None, '', 0):
            lifecycle.stop_order_id = str(broker_oid2)

    status_u = 'UNKNOWN'
    if oid_int is not None:
        try:
            status, _ = client.get_order_status_and_filled_qty(account_id_key, int(oid_int), symbol=symbol)
            status_u = str(status or 'UNKNOWN').upper().strip()
        except Exception as se:
            _append_note(lifecycle, f"{note_prefix}_STATUS_ERR {str(se)[:180]}")
            status_u = 'UNKNOWN'

    if status_u in {'CANCELLED', 'REJECTED', 'EXPIRED'}:
        lifecycle.stop_order_id = None
        _append_note(lifecycle, f"{note_prefix}_FINAL status={status_u}")
        return True

    # Only consider absence-without-id a success when we genuinely have no known stop id and
    # broker truth also cannot find a stop for this lifecycle. Otherwise keep retrying.
    if still_open is None and oid_int is None and not known_oid:
        lifecycle.stop_order_id = None
        _append_note(lifecycle, f"{note_prefix}_ABSENT_NO_OID")
        return True

    if still_open is None:
        _append_note(lifecycle, f"{note_prefix}_WAIT_STATUS status={status_u} oid={oid_int or known_oid or 'N/A'}")
    else:
        _append_note(lifecycle, f"{note_prefix}_PENDING status={status_u} oid={oid_int or known_oid or broker_oid or 'N/A'}")
    return False

def _orphan_recovery_and_protection_guard(cfg, state: dict, client, account_id_key: str, now) -> None:
    """Runs on every Streamlit rerun (when broker is ready).

    This is *not* a background daemon; it is a best-effort safety guard under
    Streamlit Community Cloud constraints.

    Policy A: Only use an existing bracket plan (stop/tp0) if it exists in
    our lifecycle state OR is discoverable at the broker (open STOP/LIMIT SELL).
    No synthetic stop/tp computation is done here.
    """
    lifecycles = state.get("lifecycles", {}) or {}

    # 1) Ensure brackets for known IN_POSITION lifecycles (state drift repair)
    for symbol, lst in list(lifecycles.items()):
        for idx, raw in enumerate(list(lst)):
            try:
                lifecycle = lifecycle_from_raw(raw)
            except Exception:
                continue
            if str(lifecycle.stage or "").upper() == "IN_POSITION":
                try:
                    _ensure_brackets(client, account_id_key, lifecycle.symbol, lifecycle, cfg, log_prefix="[GUARD] ")
                except Exception:
                    pass
                lst[idx] = asdict(lifecycle)

    # 2) Orphan recovery: broker has a position, but we have no lifecycle for the symbol
    try:
        positions_map = _get_positions_map_cached(state, client, account_id_key)
    except Exception:
        positions_map = {}

    for sym, pos in (positions_map or {}).items():
        symbol = str(sym).upper().strip()
        # Skip symbols we are already tracking
        if symbol in lifecycles and lifecycles.get(symbol):
            continue
        try:
            qty = float(pos.get("qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue

        open_orders = _extract_open_orders_for_symbol(client, account_id_key, symbol)
        stop_orders = [o for o in open_orders if ("STOP" in (o.get("order_type") or "")) and (o.get("action") == "SELL")]
        tp_orders = [o for o in open_orders if ("LIMIT" in (o.get("order_type") or "")) and (o.get("action") == "SELL")]

        recovered_stop = stop_orders[0] if stop_orders else None
        recovered_tp0 = tp_orders[0] if tp_orders else None

        if recovered_stop and recovered_tp0:
            # Minimal recovered lifecycle (uses broker-provided bracket plan)
            raw_lc = {
                "symbol": symbol,
                "engine": "RECOVERED",
                "created_ts": now.isoformat(),
                "stage": "IN_POSITION",
                "desired_entry": 0.0,
                "stop": float(recovered_stop.get("stop_price") or 0.0),
                "tp0": float(recovered_tp0.get("limit_price") or 0.0),
                "qty": int(qty),
                "reserved_dollars": 0.0,
                "filled_qty": int(qty),
                "stop_order_id": str(recovered_stop.get("order_id") or ""),
                "tp0_order_id": str(recovered_tp0.get("order_id") or ""),
                "notes": "recovered_from_broker_open_orders",
            }
            lifecycles.setdefault(symbol, []).append(raw_lc)
            try:
                tmp = lifecycle_from_raw(raw_lc)
                _event_once(
                    cfg,
                    tmp,
                    "ORPHAN_RECOVERED",
                    f"[AUTOEXEC] {symbol} ORPHAN RECOVERED (broker position + brackets found)",
                    f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nRecovered Qty: {qty}\n\nRecovered lifecycle from broker OPEN orders. Stop orderId={raw_lc.get('stop_order_id')} @ {raw_lc.get('stop')}. TP0 orderId={raw_lc.get('tp0_order_id')} @ {raw_lc.get('tp0')}.\n",
                )
            except Exception:
                pass
        else:
            # Orphan position with no bracket plan discoverable (policy A)
            tmp = TradeLifecycle(
                symbol=symbol,
                engine="RECOVERED",
                created_ts=now.isoformat(),
                stage="ORPHAN_POSITION",
                desired_entry=0.0,
                stop=0.0,
                tp0=0.0,
                qty=int(qty),
                reserved_dollars=0.0,
                filled_qty=int(qty),
            )
            _event_once(
                cfg,
                tmp,
                "ORPHAN_POSITION_DETECTED",
                f"[AUTOEXEC] {symbol} ORPHAN POSITION DETECTED (no bracket OPEN orders)",
                f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nPosition Qty: {qty}\n\nBroker reports an open position but auto-exec has no lifecycle and could not find existing STOP/TP0 OPEN orders to recover a plan (policy A).\n\nAction: Open E*TRADE and verify protection immediately.\n",
            )

    state["lifecycles"] = lifecycles

def _active_symbols(state: Dict[str, Any]) -> int:
    n = 0
    for sym, lst in state.get("lifecycles", {}).items():
        for l in lst:
            if l.get("stage") in {"STAGED", "ENTRY_SENT", "IN_POSITION", "EXIT_SENT"}:
                n += 1
                break
    return n


def _symbol_lifecycle_count_today(state: Dict[str, Any], symbol: str) -> int:
    return len(state.get("lifecycles", {}).get(symbol, []))


def _reserve_pool(state: Dict[str, Any], dollars: float, max_pool: float) -> bool:
    if state["pool_reserved"] + dollars > max_pool:
        return False
    state["pool_reserved"] += dollars
    _assert_pool_invariants(state)
    return True


def _release_pool(state: Dict[str, Any], dollars: float) -> None:
    state["pool_reserved"] = max(0.0, float(state.get("pool_reserved", 0.0)) - float(dollars))
    _assert_pool_invariants(state)



def _assert_pool_invariants(state: Dict[str, Any]) -> None:
    """Warn-only: pool_reserved should equal sum(reserved_dollars) of active lifecycles."""
    try:
        lifecycles = state.get('lifecycles', {}) or {}
        s = 0.0
        for _, lst in lifecycles.items():
            for raw in (lst or []):
                stg = str((raw or {}).get('stage', ''))
                if stg in {'STAGED', 'ENTRY_SENT', 'IN_POSITION'}:
                    s += float((raw or {}).get('reserved_dollars', 0.0) or 0.0)
        pool = float(state.get('pool_reserved', 0.0) or 0.0)
        if abs(pool - s) > 0.01:
            _log(f"[AUTOEXEC][BOOKKEEP] pool_reserved mismatch: state={pool:.2f} vs sum_active={s:.2f}")
    except Exception:
        pass
def _set_last_action(state: Dict[str, Any], summary: str) -> None:
    """Store a short human-readable summary of the bot's most recent meaningful action.

    Included in hourly P&L emails so you can quickly see what the bot last did
    without digging through all status emails.
    """
    try:
        state["last_action"] = str(summary or "")
    except Exception:
        pass


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace("$", "").strip()
        return float(x)
    except Exception:
        return None


def _pget(payload: Dict[str, Any], *keys: str, default=None):
    """Case/label-tolerant payload getter.

    Alert payloads can come from different engines and UI tables and may
    use different key casing or labels (e.g., 'Score' vs 'score',
    'Bias' vs 'bias', 'Pullback Band Low' vs 'pb_low').

    This helper makes auto-exec staging robust without touching any engine logic.
    """
    if not isinstance(payload, dict):
        return default
    for k in keys:
        if k in payload:
            return payload.get(k)
    # Fallback: case-insensitive lookup
    try:
        lower_map = {str(k).lower(): k for k in payload.keys()}
        for k in keys:
            kk = str(k).lower()
            if kk in lower_map:
                return payload.get(lower_map[kk])
    except Exception:
        pass
    return default


def build_desired_entry_for_ride(pbl: float, pbh: float, stage: str, score: Optional[float] = None) -> float:
    rng = max(0.0001, pbh - pbl)
    # For the strongest RIDE pullback alerts, bias entries closer to the top of the
    # pullback band regardless of PRE/CONF tier so we do not get overly patient on
    # the best continuation setups.
    if score is not None and score > 90.0:
        return pbl + 0.85 * rng
    if str(stage).upper().startswith("PRE"):
        return pbl + 0.33 * rng
    return pbl + 0.66 * rng


def compute_qty(max_dollars: float, entry: float) -> int:
    if entry <= 0:
        return 0
    return int(math.floor(max_dollars / entry))


def should_stage_lifecycle(cfg: AutoExecConfig, payload: Dict[str, Any]) -> bool:
    # LONG-only for v1
    if str(_pget(payload, "bias", "Bias", "BIAS", default="LONG")).upper() != "LONG":
        return False
    score = _parse_float(_pget(payload, "score", "Score", "SCORE"))
    if score is None or score < cfg.min_score:
        return False
    stage = str(_pget(payload, "stage", "Stage", "STAGE", "tier", "Tier", "TIER", default="")).upper()
    if cfg.confirm_only and "CONF" not in stage:
        return False
    return True


def stage_from_payload(cfg: AutoExecConfig, engine: str, payload: Dict[str, Any], stage: str = "STAGED") -> Optional[TradeLifecycle]:
    payload = normalize_alert_payload(payload)
    symbol = str(_pget(payload, "symbol", "Symbol", "SYMBOL", default="") or "").upper().strip()
    if not symbol:
        return None

    entry = _parse_float(_pget(payload, "entry", "Entry", "ENTRY"))
    stop = _parse_float(_pget(payload, "stop", "Stop", "STOP"))
    tp0 = _parse_float(_pget(payload, "tp0", "TP0", "Tp0"))
    if entry is None or stop is None or tp0 is None:
        return None

    # Adjust TP0 by deviation (sell a bit early). For longs: tp0 - dev
    tp0_adj = max(0.0, tp0 - float(cfg.tp0_deviation or 0.0))

    # Tick-size normalization (stored + used everywhere downstream)
    entry = _tick_round(entry) or entry
    stop = _tick_round(stop) or stop
    tp0_adj = _tick_round(tp0_adj) or tp0_adj

    desired_entry = entry
    ride_entry_mode_meta: Optional[str] = None
    pullback_band_low_meta: Optional[float] = None
    pullback_band_high_meta: Optional[float] = None
    if engine == "RIDE":
        ride_entry_mode = str(
            _pget(
                payload,
                "entry_mode", "EntryMode", "ENTRY_MODE",
                default=(payload.get("Extras") or {}).get("entry_mode") if isinstance(payload, dict) and isinstance(payload.get("Extras"), dict) else "",
            ) or ""
        ).upper().strip()
        ride_entry_mode_meta = ride_entry_mode or None
        if ride_entry_mode == "PULLBACK":
            # Pullback band can be provided with different labels and/or nested under Extras.
            # v72 contract note:
            #   - RIDE emits pb1/pb2 and pullback_entry inside Extras
            #   - normalize_alert_payload keeps Extras + __raw__ but does not lift pb1/pb2 to top-level
            # Only refine desired_entry from the pullback band when the engine itself selected
            # a PULLBACK execution plan. BREAKOUT entries should flow through untouched via Entry.
            def _extract_pb_band(src: Any) -> tuple[Optional[float], Optional[float]]:
                if not isinstance(src, dict):
                    return None, None
                # Direct low/high labels
                low = _parse_float(_pget(src, "pb_low", "PB_Low", "Pullback Band Low", "PB Low", "PullbackLow", "pullback_low"))
                high = _parse_float(_pget(src, "pb_high", "PB_High", "Pullback Band High", "PB High", "PullbackHigh", "pullback_high"))
                # Common v72 RIDE labels (band endpoints) — pb1/pb2
                p1 = _parse_float(_pget(src, "pb1", "PB1", "PB_1", "pullback_band_1"))
                p2 = _parse_float(_pget(src, "pb2", "PB2", "PB_2", "pullback_band_2"))
                # If low/high are missing but endpoints exist, derive low/high.
                if (low is None or high is None) and (p1 is not None and p2 is not None):
                    low = min(p1, p2)
                    high = max(p1, p2)
                return low, high

            pbl, pbh = _extract_pb_band(payload)
            if pbl is None or pbh is None:
                ex = payload.get("Extras") if isinstance(payload, dict) else None
                rbl, rbh = _extract_pb_band(ex)
                pbl = pbl if pbl is not None else rbl
                pbh = pbh if pbh is not None else rbh
            if pbl is None or pbh is None:
                raw = payload.get("__raw__") if isinstance(payload, dict) else None
                rbl, rbh = _extract_pb_band(raw)
                pbl = pbl if pbl is not None else rbl
                pbh = pbh if pbh is not None else rbh

            # Defensive: swap if reversed
            if pbl is not None and pbh is not None and pbl > pbh:
                pbl, pbh = pbh, pbl

            if pbl is not None and pbh is not None:
                pullback_band_low_meta = float(pbl)
                pullback_band_high_meta = float(pbh)
                desired_entry = build_desired_entry_for_ride(
                    pbl,
                    pbh,
                    str(_pget(payload, "stage", "Stage", "tier", "Tier", default="")),
                    _parse_float(_pget(payload, "score", "Score", "SCORE")),
                )

    
    desired_entry = _tick_round(desired_entry) or desired_entry
    qty = compute_qty(cfg.max_dollars_per_trade, desired_entry)
    if qty <= 0:
        return None

    reserved = qty * desired_entry
    if str(stage).upper() == "PRESTAGED":
        # PRESTAGED lifecycles are non-executable until OAuth + account binding exists.
        # Do not reserve capital yet; reserve on promotion to STAGED.
        reserved = 0.0
    return TradeLifecycle(
        symbol=symbol,
        engine=engine,
        created_ts=_now_et().isoformat(),
        stage=str(stage).upper(),
        desired_entry=float(desired_entry),
        stop=float(stop),
        tp0=float(tp0_adj),
        qty=qty,
        reserved_dollars=float(reserved),
        ride_entry_mode=ride_entry_mode_meta,
        pullback_band_low=pullback_band_low_meta,
        pullback_band_high=pullback_band_high_meta,
        notes="prestaged" if str(stage).upper() == "PRESTAGED" else "staged",
    )


def ensure_client(cfg: AutoExecConfig) -> Optional[ETradeClient]:
    state = _get_state()
    auth = state.get("auth", {})
    ck = auth.get("consumer_key")
    cs = auth.get("consumer_secret")
    at = auth.get("access_token")
    ats = auth.get("access_token_secret")
    if not (ck and cs and at and ats):
        return None
    try:
        return ETradeClient(
            consumer_key=ck,
            consumer_secret=cs,
            # Environment must match the tokens/consumer key that were used during auth.
            sandbox=bool(auth.get("sandbox", cfg.sandbox)),
            access_token=at,
            access_token_secret=ats,
        )
    except Exception:
        return None


def _broker_ping_cached(cfg: AutoExecConfig, state: dict, client: ETradeClient, account_id_key: str) -> tuple[bool, str]:
    """Return (ok, err). Uses a small cache in session_state to avoid spamming the API.

    The goal is to verify OAuth tokens are *working* (not only present).
    We ping an inexpensive endpoint (OPEN orders, count=1).
    """
    try:
        enabled = bool(getattr(cfg, 'broker_ping_enabled', True))
        interval = int(getattr(cfg, 'broker_ping_interval_sec', 60) or 60)
    except Exception:
        enabled, interval = True, 60
    if not enabled:
        return True, ''

    bp = state.get('broker_ping') if isinstance(state, dict) else None
    try:
        now_ts = _now_et().timestamp()
        last_ts = float((bp or {}).get('ts') or 0.0)
    except Exception:
        now_ts, last_ts = _now_et().timestamp(), 0.0

    # If we pinged recently, trust cached result (but still expose it in UI/state).
    if bp and (now_ts - last_ts) < interval:
        ok = bool(bp.get('ok', False))
        err = str(bp.get('err', '') or '')
        return ok, err

    ok = False
    err = ''
    try:
        # Lightweight token validity check (uses OAuth-signed request).
        client.list_orders(account_id_key, status='OPEN', count=1)
        ok = True
    except Exception as e:
        ok = False
        err = f'{type(e).__name__}: {e}'

    state['broker_ping'] = {'ok': bool(ok), 'ts': float(now_ts), 'err': err}
    return bool(ok), err


def _broker_ready(cfg: AutoExecConfig, state: Dict[str, Any]) -> Tuple[bool, Optional[ETradeClient], str, str]:
    """Return (ready, client, account_id_key, reason).

    "ready" means:
      - OAuth tokens present and a client can be constructed, AND
      - account_id_key is present (account selected/bound).

    This is the "armed" invariant for *any* broker action.
    """
    client = None
    try:
        client = ensure_client(cfg)
    except Exception:
        client = None
    account_id_key = str((state.get("auth", {}) or {}).get("account_id_key") or "")
    if client is None:
        return False, None, account_id_key, "missing_oauth"
    if not account_id_key:
        return False, None, "", "missing_account_id"
    # Optional health check: verify tokens are actually working via cached broker ping.
    # IMPORTANT: this must NOT hard-block execution/reconciliation.
    # E*TRADE read endpoints can intermittently time out even when place/cancel succeed.
    # We treat ping failure as a warning signal only and allow the caller to proceed.
    warn = ""
    try:
        if getattr(cfg, 'broker_ping_enabled', True):
            ok, perr = _broker_ping_cached(cfg, state, client, account_id_key)
            if not ok:
                warn = f'oauth_ping_failed:{perr}'
    except Exception:
        # Never crash readiness checks
        warn = ""
    return True, client, account_id_key, warn


def reconcile_and_execute(
    cfg: AutoExecConfig,
    allow_pre: bool,
    allow_opening: bool,
    allow_midday: bool,
    allow_power: bool,
    allow_after: bool,
    fetch_last_price_fn,
) -> None:
    """Runs every Streamlit rerun to reconcile order state and enforce EOD."""
    if not cfg.enabled:
        return

    state = _get_state()
    now = _now_et()


    # Hourly checkpoint email (P&L + simple analytics) is independent of the
    # execution windows. It should still fire even if we're outside the
    # opening/midday/power windows (as long as OAuth is active).
    try:
        if getattr(cfg, "hourly_pnl_emails", False):
            key = _should_send_hourly(now)
            if key and str(state.get("hourly_report_last", "") or "") != key:
                client_for_report = ensure_client(cfg)
                if client_for_report is not None:
                    _maybe_send_hourly_pnl(cfg, state, client_for_report)
                    # Persist any state mutations (dedupe key)
                    st.session_state["autoexec"] = state
    except Exception:
        # Never crash the app due to reporting.
        pass

    # Auto-exec digest email (every N minutes) for visibility — independent of broker readiness.
    try:
        if getattr(cfg, "digest_emails_enabled", False):
            _maybe_send_autoexec_digest(cfg, state, now)
            st.session_state["autoexec"] = state
    except Exception:
        pass

    broker_ready, client, account_id_key, broker_reason = _broker_ready(cfg, state)
    if not broker_ready:
        # Broker not armed: no broker calls can be made (reconcile/exits/liquidation).
        # Record explicit breadcrumbs so operators can see why lifecycles are not progressing.
        lifecycles = state.get("lifecycles", {})
        for symbol, lst in list(lifecycles.items()):
            for idx, raw in enumerate(list(lst)):
                try:
                    lifecycle = lifecycle_from_raw(raw)
                except Exception:
                    continue
                if lifecycle.stage in {"ENTRY_SENT", "IN_POSITION", "EXIT_SENT", "CANCEL_PENDING"}:
                    lifecycle.notes = (lifecycle.notes or "") + f" | broker_not_ready:{broker_reason}"
                    # One-time alert per lifecycle (critical only; avoids spam)
                    _event_once(
                        cfg,
                        lifecycle,
                        "BROKER_NOT_READY_RECONCILE",
                        f"[AUTOEXEC] {lifecycle.symbol} BROKER NOT READY (managing orders paused)",
                        f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\nStage: {lifecycle.stage}\n\nBroker is not armed ({broker_reason}). Order management (fills/stops/targets/liquidation) is paused until OAuth/account is restored.\n",
                    )
                lst[idx] = asdict(lifecycle)
        # Persist breadcrumbs
        st.session_state["autoexec"] = state
        return

    # Liquidation enforcement
    if _is_liquidation_time(now):
        _force_liquidate_all(client, account_id_key, cfg, state)
        return

    # IMPORTANT: do NOT stop reconciliation outside the entry windows.
    # Entries are gated inside try_send_entries() via cfg.enforce_entry_windows.
    in_window_now = _in_exec_window(now, cfg)

    # Guardrail: every rerun, attempt to (a) repair missing bracket orderIds for
    # known IN_POSITION lifecycles and (b) recover broker-side orphans using
    # existing broker brackets only (policy A).
    try:
        _orphan_recovery_and_protection_guard(cfg, state, client, account_id_key, now)
        st.session_state["autoexec"] = state
    except Exception:
        pass

    # Reconcile lifecycles
    lifecycles = state.get("lifecycles", {})
    for symbol, lst in list(lifecycles.items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = lifecycle_from_raw(raw)
            try:
                _reconcile_one(client, account_id_key, state, lifecycle, cfg, fetch_last_price_fn=fetch_last_price_fn)
            except Exception as e:
                lifecycle.notes = f"reconcile_error: {e}"
            lst[idx] = asdict(lifecycle)


def handle_alert_for_autoexec(
    cfg: AutoExecConfig,
    engine: str,
    payload: Dict[str, Any],
    allow_pre: bool,
    allow_opening: bool,
    allow_midday: bool,
    allow_power: bool,
    allow_after: bool,
) -> None:
    """Called only when the app has already decided to send an email alert."""
    if not cfg.enabled:
        return
    if engine not in set(cfg.engines):
        return

    now = _now_et()

    # Normalize payload keys for consistent terminology/casing across auto-exec.
    payload = normalize_alert_payload(payload)


    # Actual entry placement is gated inside try_send_entries() when cfg.enforce_entry_window
    # is enabled. Optionally, staging itself can also be restricted to the selected auto-exec
    # windows via cfg.stage_only_within_exec_windows.
    in_window = _in_exec_window(now, cfg)

    if not should_stage_lifecycle(cfg, payload):
        return
    if bool(getattr(cfg, "stage_only_within_exec_windows", False)) and not in_window:
        return

    state = _get_state()

    symbol = str(_pget(payload, "symbol", "Symbol", "SYMBOL", default="")).upper().strip()

    # Broker "armed" invariant: any executable lifecycle requires OAuth + account binding.
    broker_ready, _c, _acct, broker_reason = _broker_ready(cfg, state)

    def _skip_once(reason: str) -> None:
        key = f"{symbol}:{reason}"
        sent = state.get("skip_notices", {}).get(key)
        if sent:
            return
        state.setdefault("skip_notices", {})[key] = _now_et().isoformat()
        subj = f"[AUTOEXEC] {symbol} SKIP — {reason}"
        body = f"Time (ET): {_now_et().isoformat()}\nSymbol: {symbol}\nEngine: {engine}\nReason: {reason}\n\nThis is a one-time notice for today."
        _send_status_email(cfg, subj, body)
    def _finalize_stale_cancel_pending_for_symbol() -> None:
        """If the symbol is blocked only by a stale CANCEL_PENDING lifecycle, finalize it when broker is clean.

        This prevents 'active lifecycle exists' from blocking new tradable alerts after a confirmed cancel.
        """
        if not broker_ready:
            return
        try:
            lifecycles = (state.get("lifecycles") or {}).get(symbol, []) or []
        except Exception:
            return
        if not lifecycles:
            return

        CANCEL_PENDING_MAX_AGE_SEC = 180
        # Broker cleanliness check (open orders + positions) for the symbol
        try:
            today = datetime.now(ZoneInfo(ET_TZ)).date()
            from_s = (today - timedelta(days=7)).strftime("%m%d%Y")
            to_s = today.strftime("%m%d%Y")
            od = _c.list_orders(_acct, status="OPEN", count=50, symbol=symbol, from_date=from_s, to_date=to_s)
            orders = od.get("OrdersResponse", {}).get("Orders", {}).get("Order", []) if isinstance(od, dict) else []
            if isinstance(orders, dict):
                orders = [orders]
            has_open = False
            for o in (orders or []):
                sym = str(o.get("symbol", o.get("Symbol", "")) or "").upper().strip()
                if sym == symbol:
                    has_open = True
                    break
            pos_map = _c.get_positions_map(_acct) or {}
            has_pos = int(pos_map.get(symbol, 0) or 0) != 0
            broker_clean = (not has_open) and (not has_pos)
        except Exception:
            broker_clean = False

        if not broker_clean:
            return

        changed = False
        new_list = []
        for lc in lifecycles:
            if not isinstance(lc, dict):
                new_list.append(lc)
                continue
            stg = str(lc.get("stage", "") or "").upper().strip()
            if stg != "CANCEL_PENDING":
                new_list.append(lc)
                continue
            # Only finalize if stale
            cr = str(lc.get("cancel_requested_at", "") or "")
            try:
                cr_dt = datetime.fromisoformat(cr) if cr else None
            except Exception:
                cr_dt = None
            age_sec = (now - cr_dt).total_seconds() if cr_dt else 0.0
            if age_sec < CANCEL_PENDING_MAX_AGE_SEC:
                new_list.append(lc)
                continue

            # Safe finalize (broker clean + stale)
            try:
                if float(lc.get("reserved_dollars") or 0.0) > 0.0:
                    _release_pool(state, float(lc.get("reserved_dollars") or 0.0))
                    lc["reserved_dollars"] = 0.0
            except Exception:
                pass
            lc["entry_order_id"] = None
            lc["stop_order_id"] = None
            lc["tp0_order_id"] = None
            lc["market_exit_order_id"] = None
            notes = str(lc.get("notes", "") or "")
            if "pending_close" in notes:
                lc["stage"] = "CLOSED"
            else:
                lc["stage"] = "CANCELED"
            lc["notes"] = (notes + " | cancel_pending_timeout_finalize").strip()
            changed = True
            new_list.append(lc)

        if changed:
            state.setdefault("lifecycles", {})[symbol] = new_list



    # Session gating for STAGING (entries remain RTH-only in try_send_entries).
    # If the user disables a session in the main app settings, we do not stage
    # lifecycles during that session.
    try:
        ts = pd.Timestamp(now)
        actual_session = classify_session(
            ts,
            allow_opening=True,
            allow_midday=True,
            allow_power=True,
            allow_premarket=True,
            allow_afterhours=True,
        )
        allowed_session = classify_session(
            ts,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_pre,
            allow_afterhours=allow_after,
        )
        if allowed_session == "OFF":
            _skip_once(f"session_not_allowed ({actual_session})")
            return
    except Exception:
        # Never allow session gating logic to crash auto-exec staging
        pass

    # Prevent overlapping lifecycles for the same symbol.
    # You can have multiple attempts per day, but only one active at a time.
    if _has_active_lifecycle(state, symbol):
        try:
            _finalize_stale_cancel_pending_for_symbol()
        except Exception:
            pass
        if _has_active_lifecycle(state, symbol):
            _skip_once("active_lifecycle_exists")
            return

    # If we are not broker-armed yet, capture the signal as PRESTAGED so it's visible
    # (and can be promoted once OAuth/account is ready) but do NOT reserve capital.
    if not broker_ready:
        lifecycle = stage_from_payload(cfg, engine, payload, stage="PRESTAGED")
        if lifecycle is None:
            return
        lifecycle.notes = f"prestaged_broker_not_ready:{broker_reason}"
        state.setdefault("lifecycles", {}).setdefault(symbol, []).append(asdict(lifecycle))

        subj = f"[AUTOEXEC] {symbol} {engine} PRESTAGED (broker not armed)"
        body = (
            f"Time (ET): {now.isoformat()}\n"
            f"Symbol: {symbol}\nEngine: {engine}\n\n"
            f"PRESTAGED — OAuth/account not ready ({broker_reason}).\n"
            f"No order will be sent until authentication is complete and an account is bound.\n\n"
            f"Desired entry: {_fmt_price(lifecycle.desired_entry)}\n"
            f"Stop: {_fmt_price(lifecycle.stop)}\n"
            f"TP0 (exit limit): {_fmt_price(lifecycle.tp0)}\n"
            f"Qty (computed): {lifecycle.qty}\n"
        )
        _event_once(cfg, lifecycle, "PRESTAGED", subj, body)
        # Persist emailed_events mutation
        try:
            lifelist = state.get("lifecycles", {}).get(symbol)
            if isinstance(lifelist, list) and lifelist:
                lifelist[-1] = asdict(lifecycle)
        except Exception:
            pass
        return


    # Concurrency and lifecycle limits
    if _active_symbols(state) >= int(cfg.max_concurrent_symbols):
        _skip_once("max_concurrent_symbols")
        return

    if _symbol_lifecycle_count_today(state, symbol) >= int(cfg.lifecycles_per_symbol_per_day):
        _skip_once("lifecycle_cap")
        return

    lifecycle = stage_from_payload(cfg, engine, payload)
    if lifecycle is None:
        return

    if not in_window:
        lifecycle.notes = (lifecycle.notes or "") + " | staged outside entry window"

    if not _reserve_pool(state, lifecycle.reserved_dollars, float(cfg.max_pool_dollars)):
        _skip_once("pool_cap")
        return

    state.setdefault("lifecycles", {}).setdefault(symbol, []).append(asdict(lifecycle))

    # Status email: staged
    subj = f"[AUTOEXEC] {symbol} {engine} STAGED"
    score = payload.get("Score")
    tier = payload.get("Tier") or payload.get("Stage")
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Symbol: {symbol}\nEngine: {engine}\nTier: {tier}\nScore: {score}\n\n"
        f"STAGED — waiting for entry conditions.\n\n"
        f"Desired entry: {_fmt_price(lifecycle.desired_entry)}\n"
        f"Stop: {_fmt_price(lifecycle.stop)}\n"
        f"TP0 (exit limit): {_fmt_price(lifecycle.tp0)}\n"
        f"Qty: {lifecycle.qty}\n"
        f"Reserved: ${lifecycle.reserved_dollars:.2f}\n"
    )

    _event_once(cfg, lifecycle, "STAGED", subj, body)

    # _event_once mutates lifecycle.emailed_events for dedupe.
    # Persist that mutation back into session_state so reruns don't resend.
    try:
        lifelist = state.get("lifecycles", {}).get(symbol)
        if isinstance(lifelist, list) and lifelist:
            lifelist[-1] = asdict(lifecycle)
    except Exception:
        # If persistence fails, it's non-fatal; worst case is a duplicate status email.
        pass


def try_send_entries(cfg: AutoExecConfig, allow_opening: bool, allow_midday: bool, allow_power: bool, fetch_last_price_fn) -> None:
    """Places entry orders for STAGED lifecycles when price is in range.

    Safety:
      - entry is ONLY sent if last is ABOVE stop AND at/below desired entry.
      - staged/entry orders time out after cfg.timeout_minutes.
    """
    if not cfg.enabled:
        return
    state = _get_state()
    broker_ready, client, account_id_key, broker_reason = _broker_ready(cfg, state)

    now = _now_et()


    # Entry-window gating controls NEW entry submissions only.
    # Exits (stops/targets/EOD) are handled in reconcile.
    in_window_now = _in_exec_window(now, cfg)
    enforce_windows = bool(getattr(cfg, "enforce_entry_windows", True))
    grace_min = int(getattr(cfg, "entry_grace_minutes", 0) or 0)
    for symbol, lst in list(state.get("lifecycles", {}).items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = lifecycle_from_raw(raw)

            # If broker is not armed, we do not attempt any entries.
            # Record an explicit reason once per lifecycle to make wiring issues obvious.
            if not broker_ready:
                lifecycle.last_entry_eval = f"SKIP: broker_not_ready ({broker_reason})"
                if lifecycle.stage in {"STAGED", "PRESTAGED"}:
                    _event_once(
                        cfg,
                        lifecycle,
                        "BROKER_NOT_READY",
                        f"[AUTOEXEC] {lifecycle.symbol} BROKER NOT READY",
                        f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nAuto-exec is enabled but OAuth/account is not armed ({broker_reason}).\nNo orders can be placed until auth is completed.\n",
                    )
                lst[idx] = asdict(lifecycle)
                continue

            # Promote PRESTAGED -> STAGED once broker becomes armed.
            if lifecycle.stage == "PRESTAGED":
                # Reserve capital now (if possible) and begin normal entry evaluation.
                if not _reserve_pool(state, float(lifecycle.qty or 0) * float(lifecycle.desired_entry or 0.0), float(cfg.max_pool_dollars)):
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = "prestaged_promotion_failed:pool_cap"
                    lifecycle.last_entry_eval = "CANCELED: pool_cap on promotion"
                    _event_once(cfg, lifecycle, "PROMOTION_FAILED", f"[AUTOEXEC] {lifecycle.symbol} PROMOTION FAILED", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCould not reserve pool dollars when promoting PRESTAGED -> STAGED.\n")
                    lst[idx] = asdict(lifecycle)
                    continue
                lifecycle.reserved_dollars = float(lifecycle.qty or 0) * float(lifecycle.desired_entry or 0.0)
                lifecycle.stage = "STAGED"
                lifecycle.notes = (lifecycle.notes or "").strip() + " | promoted_from_prestaged"
                lifecycle.last_entry_eval = "PROMOTED: PRESTAGED->STAGED"
                _event_once(cfg, lifecycle, "PROMOTED", f"[AUTOEXEC] {lifecycle.symbol} PROMOTED", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nPromoted PRESTAGED -> STAGED now that OAuth/account is armed.\nReserved: ${float(lifecycle.reserved_dollars or 0.0):.2f}\n")
                lst[idx] = asdict(lifecycle)
                # Continue into standard STAGED checks this same rerun
                # (reload lifecycle from dict after mutation below)
                lifecycle = lifecycle_from_raw(lst[idx])

            # Timeout STAGED lifecycles that never got an entry window
            if lifecycle.stage == "STAGED":
                try:
                    created = datetime.fromisoformat(lifecycle.created_ts)
                    age_min = (now - created).total_seconds() / 60.0
                except Exception:
                    age_min = 0.0
                timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
                if age_min >= timeout_m:
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = f"staged_timeout_{timeout_m}m"
                    _event_once(cfg, lifecycle, "STAGED_TIMEOUT", f"[AUTOEXEC] {lifecycle.symbol} STAGED TIMEOUT", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCanceled: staged timeout ({timeout_m}m)\n")
                    _record_activity(state, "ENTRY_TIMEOUT", lifecycle, f"timeout={timeout_m}m")
                    # release unused reserved dollars back to the pool
                    _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                    lifecycle.reserved_dollars = 0.0
                    lst[idx] = asdict(lifecycle)
                    continue

            if lifecycle.stage != "STAGED":
                continue

            # Observability: count how many reruns have evaluated this STAGED lifecycle.
            try:
                lifecycle.entry_eval_count = int(getattr(lifecycle, "entry_eval_count", 0) or 0) + 1
            except Exception:
                lifecycle.entry_eval_count = 1

            # Entry window gating: prevents orders outside selected execution windows.
            # If enforce_entry_windows is ON, we only send entries during the window,
            # with an optional small grace period based on lifecycle.created_ts.
            if enforce_windows and not in_window_now:
                allow_via_grace = False
                if grace_min > 0:
                    try:
                        created_dt = datetime.fromisoformat(lifecycle.created_ts)
                        if _in_exec_window(created_dt, cfg):
                            age_min = (now - created_dt).total_seconds() / 60.0
                            if age_min <= float(grace_min):
                                allow_via_grace = True
                    except Exception:
                        allow_via_grace = False
                if not allow_via_grace:
                    lifecycle.last_entry_eval = f"SKIP: outside_exec_window ({_exec_window_label(now)})"
                    _maybe_email_entry_skip(cfg, lifecycle, now, "outside_exec_window", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue

            coid = str(lifecycle.entry_client_order_id or _mk_client_order_id(lifecycle.lifecycle_id, "EN"))
            lifecycle.entry_client_order_id = coid

            # Duplicate-send guard: if a submit was initiated recently, poll broker truth
            # first and avoid re-sending during the guard window. This protects against
            # Streamlit reruns / transient broker-response timing where the order was
            # accepted but our lifecycle has not yet advanced to ENTRY_SENT.
            try:
                recent_submit = False
                if lifecycle.entry_submit_started_at:
                    started_dt = datetime.fromisoformat(lifecycle.entry_submit_started_at)
                    recent_submit = (now - started_dt).total_seconds() < float(duplicate_guard_seconds)
                if recent_submit:
                    recovered = None
                    try:
                        recovered = client.find_order_by_client_order_id(
                            account_id_key=account_id_key,
                            client_order_id=coid,
                            symbol=symbol,
                            from_date=from_date,
                            to_date=to_date,
                        )
                    except Exception:
                        recovered = None
                    if recovered and recovered.get("orderId"):
                        oid = int(recovered.get("orderId"))
                        lifecycle.entry_order_id = oid
                        lifecycle.entry_sent_ts = lifecycle.entry_sent_ts or lifecycle.entry_submit_started_at or now.isoformat()
                        lifecycle.entry_submit_started_at = None
                        lifecycle.stage = "ENTRY_SENT"
                        _append_note(lifecycle, f"ENTRY_ATTACH_DUP_GUARD oid={oid}")
                        lifecycle.last_entry_eval = f"SENT: attached_recent_submit oid={oid}"
                        lst[idx] = asdict(lifecycle)
                        continue
                    else:
                        lifecycle.last_entry_eval = f"SKIP: entry_submit_inflight<{duplicate_guard_seconds}s"
                        lst[idx] = asdict(lifecycle)
                        continue
            except Exception:
                pass

            try:
                last = _parse_float(fetch_last_price_fn(symbol))
            except Exception:
                last = None
            if last is None:
                lifecycle.last_entry_eval = "SKIP: last_price_unavailable"
                _maybe_email_entry_skip(cfg, lifecycle, now, "last_price_unavailable", lifecycle.last_entry_eval)
                lst[idx] = asdict(lifecycle)
                continue


            # Entry gating / entry mode:
            # - touch_required: only place entry once we observe last <= desired_entry (and last > stop)
            # - early_band: place a resting limit order slightly ABOVE desired_entry to avoid missing fills due to refresh cadence,
            #               still requires last > stop and last within a configurable band (bps) above desired_entry.
            # - immediate_on_stage: as soon as we are STAGED (and within allowed entry window), place a resting limit order at desired_entry
            #                       as long as last > stop. This maximizes fill capture on pullbacks.
            entry_mode = str(getattr(cfg, "entry_mode", "") or "").strip().lower()
            if not entry_mode:
                # Backward-compat: if old toggle is on, treat as early_band, else touch_required
                entry_mode = "early_band" if bool(getattr(cfg, "early_entry_limit_orders", False)) else "touch_required"

            guard_bps = float(getattr(cfg, "entry_distance_guard_bps", 25.0) or 0.0)

            if entry_mode == "immediate_on_stage":
                if not (last > lifecycle.stop):
                    lifecycle.last_entry_eval = f"SKIP: last<=stop (mode=immediate) last={last} stop={lifecycle.stop}"
                    _maybe_email_entry_skip(cfg, lifecycle, now, "last_le_stop", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue
            elif entry_mode == "early_band":
                band_mult = 1.0 + max(0.0, guard_bps) / 10000.0
                if not (last > lifecycle.stop and last <= float(lifecycle.desired_entry) * band_mult):
                    lifecycle.last_entry_eval = (
                        f"SKIP: early_band_not_met last={last} stop={lifecycle.stop} desired={lifecycle.desired_entry} "
                        f"guard_bps={guard_bps} band_max={float(lifecycle.desired_entry)*band_mult}"
                    )
                    _maybe_email_entry_skip(cfg, lifecycle, now, "early_band_not_met", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue
            else:
                # touch_required
                if not (last <= lifecycle.desired_entry and last > lifecycle.stop):
                    lifecycle.last_entry_eval = (
                        f"SKIP: touch_required_not_met last={last} stop={lifecycle.stop} desired={lifecycle.desired_entry}"
                    )
                    _maybe_email_entry_skip(cfg, lifecycle, now, "touch_required_not_met", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue
            # Place entry order (limit at desired_entry)
            try:
                # Determine entry limit price. In immediate_on_stage mode we can optionally apply
                # a small "marketable-limit" buffer to improve fill probability, while enforcing
                # strict safety bounds (tick rounding + stop invariant).
                entry_limit = float(_tick_round(lifecycle.desired_entry) or lifecycle.desired_entry)
                if entry_mode == "immediate_on_stage" and bool(getattr(cfg, "use_entry_buffer", False)):
                    raw_buf = float(getattr(cfg, "entry_buffer_max", 0.01) or 0.0)
                    buf = min(max(raw_buf, 0.0), 0.03)  # hard cap
                    entry_limit = float(_tick_round(lifecycle.desired_entry + buf) or (lifecycle.desired_entry + buf))
                    if entry_limit <= float(lifecycle.stop):
                        lifecycle.last_entry_eval = (
                            f"SKIP: entry_buffer_violates_stop entry_limit={entry_limit} stop={lifecycle.stop}"
                        )
                        lst[idx] = asdict(lifecycle)
                        continue
                lifecycle.entry_submit_started_at = now.isoformat()
                lifecycle.entry_client_order_id = coid
                lst[idx] = asdict(lifecycle)
                oid, pid = client.place_equity_limit_order_ex(
                    account_id_key=account_id_key,
                    symbol=symbol,
                    qty=lifecycle.qty,
                    limit_price=float(entry_limit),
                    action="BUY",
                    market_session="REGULAR",
                    client_order_id=coid,
                )
                lifecycle.entry_order_id = oid
                lifecycle.entry_sent_ts = now.isoformat()
                lifecycle.entry_submit_started_at = None
                lifecycle.stage = "ENTRY_SENT"
                lifecycle.notes = f"entry_sent@{lifecycle.desired_entry}"
                _append_note(lifecycle, f"ENTRY_PREVIEW_OK pid={pid}")
                _append_note(lifecycle, f"ENTRY_PLACE_OK oid={oid}")
                lifecycle.last_entry_eval = f"SENT: entry_limit oid={oid} pid={pid}"
                _log(f"[AUTOEXEC][BROKER] place_ok sym={symbol} lc={lifecycle.lifecycle_id} coid={coid} pid={pid} oid={oid}")
                _event_once(cfg, lifecycle, "ENTRY_SENT", f"[AUTOEXEC] {symbol} ENTRY SENT", f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nEntry limit placed.\nQty: {lifecycle.qty}\nLimit: {lifecycle.desired_entry}\nStop (planned): {lifecycle.stop}\nTP0 (planned exit): {lifecycle.tp0}\nclientOrderId: {coid}\npreviewId: {pid}\norderId: {oid}\n")
                _record_activity(state, "ENTRY_PLACED", lifecycle, f"oid={oid} pid={pid}")
            except Exception as e:
                emsg = str(e)
                # Special-case: E*TRADE may reply "This is a duplicate order" (code 1028)
                # even when the original request was accepted and is in flight. If we keep
                # retrying from STAGED we will spam duplicates and never reconcile.
                if ("1028" in emsg) or ("duplicate order" in emsg.lower()):
                    recovered = None
                    try:
                        recovered = client.find_order_by_client_order_id(
                            account_id_key=account_id_key,
                            client_order_id=coid,
                            symbol=symbol,
                            from_date=from_date,
                            to_date=to_date,
                        )
                    except Exception:
                        recovered = None

                    if recovered and recovered.get("orderId"):
                        oid = int(recovered.get("orderId"))
                        lifecycle.entry_order_id = oid
                        lifecycle.entry_sent_ts = lifecycle.entry_sent_ts or lifecycle.entry_submit_started_at or now.isoformat()
                        lifecycle.entry_submit_started_at = None
                        lifecycle.stage = "ENTRY_SENT"
                        lifecycle.notes = "entry_send_duplicate_recovered"
                        lifecycle.last_entry_eval = f"SENT: recovered_duplicate oid={oid}"
                        _log(f"[AUTOEXEC][BROKER] duplicate_recovered sym={symbol} lc={lifecycle.lifecycle_id} coid={coid} oid={oid}")
                        _event_once(
                            cfg,
                            lifecycle,
                            "ENTRY_PLACED_RECOVERED",
                            f"[AUTOEXEC] {symbol} ENTRY PLACED (recovered)",
                            f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nRecovered orderId after duplicate-order response.\nclientOrderId: {coid}\norderId: {oid}\n",
                        )
                        _record_activity(state, "ENTRY_PLACED", lifecycle, f"oid={oid} (recovered)")
                    else:
                        lifecycle.notes = f"entry_send_pending_duplicate: {e}"
                        lifecycle.last_entry_eval = f"PENDING: duplicate_order_no_recovery ({emsg})"
                else:
                    lifecycle.entry_submit_started_at = None
                    lifecycle.notes = f"entry_send_failed: {e}"
                    lifecycle.last_entry_eval = f"FAILED: entry_send_failed ({e})"
                    _event_once(cfg, lifecycle, "ENTRY_SEND_FAILED", f"[AUTOEXEC] {symbol} ENTRY SEND FAILED", f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nEntry placement failed: {e}\n")

            lst[idx] = asdict(lifecycle)


def _oid_int(order_id: Any) -> Optional[int]:
    """Convert stored order id (often a str) into an int for E*TRADE calls."""
    if order_id is None:
        return None
    if isinstance(order_id, int):
        return order_id
    try:
        s = str(order_id).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _reconcile_one(client: ETradeClient, account_id_key: str, state: Dict[str, Any], lifecycle: TradeLifecycle, cfg: AutoExecConfig, fetch_last_price_fn=None) -> None:
    """Update lifecycle state based on order statuses.

    Contract goals:
      - All broker calls use the ETradeClient wrapper signatures (keyword args, correct types)
      - Lifecycle does NOT transition to CLOSED unless we have evidence of an exit
      - Bookkeeping reserve is released exactly once on close/cancel
    """
    now = _now_et()
    # Increment reconcile evaluation counter early (before broker calls) so digests reflect live reconciliation cadence.
    try:
        lifecycle.reconcile_eval_count = int(getattr(lifecycle, 'reconcile_eval_count', 0) or 0) + 1
    except Exception:
        lifecycle.reconcile_eval_count = 1
    def _oid_int_safe(x) -> Optional[int]:
        try:
            return int(x) if x not in (None, "", 0) else None
        except Exception:
            return None

    FINAL_STATUSES = {"EXECUTED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"}

    # --- Broker snapshot caches (in-memory state dict, NOT st.session_state) ---
    # These reduce API chatter during rapid reruns without affecting correctness.
    _bc = state.setdefault("_broker_cache", {})

    def _get_positions_map_cached(ttl_sec: int = 15, max_stale_sec: int = 300) -> dict:
        """Return positions map with a short TTL cache.

        If a refresh call fails, fall back to the last known-good snapshot for up to
        max_stale_sec. This prevents 'missed fills' during temporary broker/API hiccups.
        """
        now_ts = pytime.time()
        ent = _bc.get("positions_map") if isinstance(_bc, dict) else None
        if isinstance(ent, dict):
            ts = float(ent.get("ts", 0) or 0)
            if (now_ts - ts) <= float(ttl_sec):
                data = ent.get("data")
                if isinstance(data, dict):
                    return data
        # IMPORTANT: cache wrapper must call the broker client, not recurse.
        try:
            data = client.get_positions_map(account_id_key)
        except Exception:
            data = {}

        # If refresh failed (or returned empty), use last_good if it's not too stale.
        if not data and isinstance(_bc, dict):
            last_good = _bc.get("positions_map_last_good")
            if isinstance(last_good, dict):
                ts_good = float(last_good.get("ts", 0) or 0)
                if ts_good and (now_ts - ts_good) <= float(max_stale_sec):
                    data = last_good.get("data") if isinstance(last_good.get("data"), dict) else data

        # Store current snapshot
        if isinstance(_bc, dict):
            _bc["positions_map"] = {"ts": now_ts, "data": data}
            if isinstance(data, dict) and data:
                _bc["positions_map_last_good"] = {"ts": now_ts, "data": data}
        return data

    def _get_portfolio_positions_cached(ttl_sec: int = 15, max_stale_sec: int = 300) -> list[dict]:
        """Return raw portfolio position rows with short TTL caching.

        This is heavier than the positions map but lets reconcile use broker portfolio
        average-cost fields as a secondary truth source for stabilizing entry_avg_price_cached.
        """
        now_ts = pytime.time()
        ent = _bc.get("portfolio_positions") if isinstance(_bc, dict) else None
        if isinstance(ent, dict):
            ts = float(ent.get("ts", 0) or 0)
            if (now_ts - ts) <= float(ttl_sec):
                data = ent.get("data")
                if isinstance(data, list):
                    return data
        try:
            pj = client.get_portfolio(account_id_key)
            data = _extract_positions(pj) if isinstance(pj, dict) else []
        except Exception:
            data = []

        if not data and isinstance(_bc, dict):
            last_good = _bc.get("portfolio_positions_last_good")
            if isinstance(last_good, dict):
                ts_good = float(last_good.get("ts", 0) or 0)
                if ts_good and (now_ts - ts_good) <= float(max_stale_sec):
                    cand = last_good.get("data")
                    if isinstance(cand, list):
                        data = cand

        if isinstance(_bc, dict):
            _bc["portfolio_positions"] = {"ts": now_ts, "data": data}
            if isinstance(data, list) and data:
                _bc["portfolio_positions_last_good"] = {"ts": now_ts, "data": data}
        return data

    def _get_portfolio_position_snapshot(symbol: str, *, force_refresh: bool = False) -> dict:
        sym_u = str(symbol or "").upper().strip()
        if not sym_u:
            return {}
        ttl = 0 if force_refresh else 15
        for pos in _get_portfolio_positions_cached(ttl_sec=ttl) or []:
            try:
                if _pos_symbol(pos) == sym_u:
                    qty = _extract_position_qty(pos)
                    avg = _extract_position_entry_avg(pos)
                    return {"raw": pos, "qty": int(qty or 0), "avg_entry": avg}
            except Exception:
                continue
        return {}

    def _extract_qty_from_position_like(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                return int(float(value))
            except Exception:
                return None
        if isinstance(value, dict):
            for k in ("qty", "quantity", "positionQuantity", "position_qty", "shares"):
                if k in value:
                    try:
                        return int(float(value.get(k) or 0))
                    except Exception:
                        pass
        return None

    def _get_broker_position_qty(symbol: str, *, force_refresh: bool = False) -> Optional[int]:
        """Return broker position quantity for symbol using the simple positions map first,
        then portfolio snapshot truth as a fallback. Handles both {symbol: qty} and
        accidental {symbol: {qty: ...}} shapes so TP0/close logic stays cohesive.
        """
        sym_u = str(symbol or "").upper().strip()
        if not sym_u:
            return None

        pos_qty = None
        try:
            pos_map = _get_positions_map_cached(
                ttl_sec=(0 if force_refresh else 15),
                max_stale_sec=(0 if force_refresh else 300),
            ) or {}
        except Exception:
            pos_map = {}

        if isinstance(pos_map, dict):
            candidates = [sym_u, str(symbol or "").strip()]
            normalized = ''.join(ch for ch in sym_u if ch.isalnum())
            if normalized and normalized not in candidates:
                candidates.append(normalized)
            for key in candidates:
                if key in pos_map:
                    pos_qty = _extract_qty_from_position_like(pos_map.get(key))
                    if pos_qty is not None:
                        break
            if pos_qty is None:
                for k, v in pos_map.items():
                    try:
                        if str(k or '').upper().strip() == sym_u:
                            pos_qty = _extract_qty_from_position_like(v)
                            if pos_qty is not None:
                                break
                    except Exception:
                        continue

        if pos_qty is None:
            try:
                snap = _get_portfolio_position_snapshot(sym_u, force_refresh=force_refresh)
                if isinstance(snap, dict):
                    pos_qty = _extract_qty_from_position_like(snap.get("qty"))
            except Exception:
                pos_qty = None

        # For exit/flat detection, a forced refresh should treat complete symbol absence
        # from both broker position sources as flat rather than "unknown".
        if force_refresh and pos_qty is None:
            return 0

        return pos_qty

    def _poll_order_status_cached(order_id: int, symbol: str, ttl_sec: int = 2) -> tuple[str, int]:
        """Poll broker order status/filled qty by orderId with short TTL cache."""
        now_ts = pytime.time()
        key = f"order:{int(order_id)}"
        ent = _bc.get(key) if isinstance(_bc, dict) else None
        if isinstance(ent, dict):
            ts = float(ent.get("ts", 0) or 0)
            if (now_ts - ts) <= float(ttl_sec):
                st = ent.get("status")
                fq = ent.get("filled_qty")
                if isinstance(st, str) and fq is not None:
                    try:
                        return (st, int(fq))
                    except Exception:
                        pass
        st, fq = client.get_order_status_and_filled_qty(account_id_key, int(order_id), symbol=symbol)
        if isinstance(_bc, dict):
            _bc[key] = {"ts": now_ts, "status": st, "filled_qty": int(fq or 0)}
        return (st, int(fq or 0))

    def _stabilize_entry_avg_cache(*, observed_filled_qty: int, observed_status: Optional[str], force_refresh: bool = False) -> None:
        """Stabilize lifecycle.entry_avg_price_cached using broker order truth first,
        then portfolio avg/cost-basis truth while fills are still evolving. Once the
        qty/avg settle for consecutive passes, lock the cache against noisy rewrites.
        """
        try:
            oid_local = _oid_int_safe(lifecycle.entry_order_id)
            current_cached_avg = _parse_float(getattr(lifecycle, "entry_avg_price_cached", None))
            cache_locked = bool(getattr(lifecycle, "entry_avg_price_locked", False))
            stable_passes = int(getattr(lifecycle, "entry_avg_stable_passes", 0) or 0)
            last_qty = int(getattr(lifecycle, "entry_avg_last_qty", 0) or 0)

            order_avg = None
            if oid_local is not None:
                try:
                    _filled_tmp, _entry_avg = client.get_order_filled_and_avg_price(account_id_key, oid_local)
                    if _entry_avg is not None:
                        order_avg = float(_entry_avg)
                except Exception as _e:
                    _append_note(lifecycle, f"ENTRY_AVG_ORDER_FETCH_ERR {type(_e).__name__}:{str(_e)[:80]}")

            force_portfolio_refresh = bool(force_refresh) or (order_avg is None) or (int(observed_filled_qty or 0) > int(last_qty or 0))
            pos_snap = _get_portfolio_position_snapshot(lifecycle.symbol, force_refresh=bool(force_portfolio_refresh))
            pos_qty = int(_coerce_int((pos_snap or {}).get("qty", 0), 0))
            portfolio_avg = _parse_float((pos_snap or {}).get("avg_entry"))

            expected_qty = max(int(observed_filled_qty or 0), int(pos_qty or 0), int(lifecycle.filled_qty or 0))
            order_final = str(observed_status or lifecycle.entry_status_cached or "").upper() in {"EXECUTED", "FILLED"}
            fill_still_evolving = (
                (int(observed_filled_qty or 0) < int(lifecycle.qty or 0))
                or (int(pos_qty or 0) < int(lifecycle.qty or 0) and int(pos_qty or 0) > 0)
                or (int(observed_filled_qty or 0) > int(last_qty or 0))
                or (int(pos_qty or 0) > int(last_qty or 0))
                or not order_final
            )

            chosen_avg = None
            chosen_source = None
            if order_avg is not None and order_avg > 0:
                chosen_avg = float(order_avg)
                chosen_source = "order"
            elif portfolio_avg is not None and portfolio_avg > 0:
                chosen_avg = float(portfolio_avg)
                chosen_source = "portfolio"
            elif current_cached_avg is not None and current_cached_avg > 0:
                chosen_avg = float(current_cached_avg)
                chosen_source = str(getattr(lifecycle, "entry_avg_last_source", None) or "cache")

            if (not cache_locked) and chosen_avg is not None and chosen_avg > 0:
                changed = (
                    current_cached_avg is None
                    or abs(float(current_cached_avg) - float(chosen_avg)) > 1e-9
                    or str(getattr(lifecycle, "entry_avg_last_source", None) or "") != str(chosen_source or "")
                )
                lifecycle.entry_avg_price_cached = float(chosen_avg)
                lifecycle.entry_avg_last_source = str(chosen_source or "cache")
                lifecycle.entry_avg_last_qty = int(expected_qty or observed_filled_qty or 0)
                if changed:
                    _append_note(lifecycle, f"ENTRY_AVG_CACHED {lifecycle.entry_avg_price_cached} src={lifecycle.entry_avg_last_source} qty={lifecycle.entry_avg_last_qty}")

            if cache_locked:
                lifecycle.entry_avg_last_qty = max(int(lifecycle.entry_avg_last_qty or 0), int(expected_qty or 0))
                return

            qty_stable = int(expected_qty or 0) > 0 and int(expected_qty or 0) == int(last_qty or 0)
            avg_present = _parse_float(getattr(lifecycle, "entry_avg_price_cached", None))
            if avg_present is not None and qty_stable and order_final and not fill_still_evolving:
                stable_passes += 1
            else:
                stable_passes = 0
            lifecycle.entry_avg_stable_passes = int(stable_passes)
            lifecycle.entry_avg_last_qty = int(expected_qty or 0)
            if int(stable_passes) >= 2 and avg_present is not None:
                lifecycle.entry_avg_price_locked = True
                _append_note(lifecycle, f"ENTRY_AVG_LOCKED {float(avg_present):.6f} qty={lifecycle.entry_avg_last_qty} src={getattr(lifecycle, 'entry_avg_last_source', None) or 'cache'}")
        except Exception as _e:
            lifecycle.notes = (lifecycle.notes or "") + f" | entry_avg_cache_failed:{type(_e).__name__}:{str(_e)[:120]}"


    def _order_is_inactive(order_id: int) -> bool:
        """Return True only when we have *positive* evidence an order is no longer active.

        IMPORTANT: We do **not** treat UNKNOWN as inactive.
        In E*TRADE (especially sandbox), ListOrders can intermittently omit an order (pagination,
        backend delays, or status buckets). If we treat UNKNOWN as inactive we can:
          - prematurely release pool reserve
          - stop monitoring a live order
          - mark a lifecycle CANCELED while the broker still has the order

        Therefore:
          - UNKNOWN => assume ACTIVE (return False)
          - Only final broker statuses => inactive
        """
        try:
            st, _fq = client.get_order_status_and_filled_qty(account_id_key, int(order_id), symbol=lifecycle.symbol)
        except Exception:
            return False
        st = str(st or "UNKNOWN").upper().strip()
        if st == "UNKNOWN":
            return False
        return st in FINAL_STATUSES

    def _cancel_with_verify(order_id_str: Optional[str], label: str) -> bool:
        """Best-effort cancel with verification via list_orders search.

        Returns True if order is confirmed inactive (or missing), else False.
        """
        oid = _oid_int_safe(order_id_str)
        if oid is None:
            return True
        try:
            client.cancel_order(account_id_key, oid)
        except Exception:
            # ignore; verification below will decide
            pass
        return _order_is_inactive(oid)

    # --- CANCEL_PENDING recovery ---
    if lifecycle.stage == "CANCEL_PENDING":
        # Attempt to cancel any remaining known orders and only finalize once broker confirms inactivity.
        pending_labels = [
            ("ENTRY", lifecycle.entry_order_id),
            ("STOP", lifecycle.stop_order_id),
            ("TP0", lifecycle.tp0_order_id),
            ("MKT_EXIT", lifecycle.market_exit_order_id),
        ]
        all_inactive = True
        for lbl, oid_str in pending_labels:
            if oid_str:
                ok = _cancel_with_verify(oid_str, lbl)
                if not ok:
                    all_inactive = False


        # If broker verification is flaky or delayed, do not allow CANCEL_PENDING to block the symbol forever.
        # After a short max-age, if the broker confirms there are NO open orders and NO position for the symbol,
        # we can safely finalize the lifecycle as CANCELED/CLOSED to unblock future alerts.
        CANCEL_PENDING_MAX_AGE_SEC = 180
        if not all_inactive:
            try:
                cr = getattr(lifecycle, "cancel_requested_at", None) or ""
                cr_dt = datetime.fromisoformat(cr) if cr else None
            except Exception:
                cr_dt = None
            age_sec = (now - cr_dt).total_seconds() if cr_dt else 0.0
            if age_sec >= CANCEL_PENDING_MAX_AGE_SEC:
                broker_clean = False
                try:
                    today = datetime.now(ZoneInfo("America/New_York")).date()
                    from_dt = today - timedelta(days=7)
                    from_s = from_dt.strftime("%m%d%Y")
                    to_s = today.strftime("%m%d%Y")
                    od = client.list_orders(account_id_key, status="OPEN", count=50, symbol=lifecycle.symbol, from_date=from_s, to_date=to_s)
                    orders = od.get("OrdersResponse", {}).get("Orders", {}).get("Order", []) if isinstance(od, dict) else []
                    if isinstance(orders, dict):
                        orders = [orders]
                    has_open = False
                    for o in (orders or []):
                        sym = str(o.get("symbol", o.get("Symbol", "")) or "").upper().strip()
                        if sym == str(lifecycle.symbol).upper().strip():
                            has_open = True
                            break
                    pos_map = _get_positions_map_cached() or {}
                    has_pos = int(pos_map.get(str(lifecycle.symbol).upper().strip(), 0) or 0) != 0
                    broker_clean = (not has_open) and (not has_pos) and int(lifecycle.filled_qty or 0) == 0
                except Exception:
                    broker_clean = False

                if broker_clean:
                    # Release any reserved dollars exactly once
                    if float(lifecycle.reserved_dollars or 0.0) > 0.0:
                        _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                        lifecycle.reserved_dollars = 0.0

                    lifecycle.entry_order_id = None
                    lifecycle.stop_order_id = None
                    lifecycle.tp0_order_id = None
                    lifecycle.market_exit_order_id = None
                    lifecycle.bracket_qty = int(lifecycle.filled_qty or 0)

                    if "pending_close" in (lifecycle.notes or ""):
                        lifecycle.stage = "CLOSED"
                    else:
                        lifecycle.stage = "CANCELED"
                    lifecycle.notes = (lifecycle.notes or "") + " | cancel_pending_timeout_finalize"

                    _event_once(
                        cfg,
                        lifecycle,
                        "CANCEL_PENDING_TIMEOUT_FINALIZE",
                        f"[AUTOEXEC] {lifecycle.symbol} CANCEL PENDING TIMEOUT FINALIZE",
                        f"""Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nFinalized stale CANCEL_PENDING after {int(age_sec)}s (broker clean). Stage now: {lifecycle.stage}\n""",
                    )
                    return

        if all_inactive:
            # If this was an entry-side cancel, reserved dollars were still held; release now.
            if float(lifecycle.reserved_dollars or 0.0) > 0.0:
                _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                lifecycle.reserved_dollars = 0.0

            lifecycle.entry_order_id = None
            lifecycle.stop_order_id = None
            lifecycle.tp0_order_id = None
            lifecycle.market_exit_order_id = None
            lifecycle.bracket_qty = int(lifecycle.filled_qty or 0)

            # Preserve prior terminal intent in notes
            if "pending_close" in (lifecycle.notes or ""):
                lifecycle.stage = "CLOSED"
                # Best-effort realized P&L capture for pending-close lifecycles
                try:
                    m = re.search(r"pending_close_reason:([^|]+)", str(lifecycle.notes or ""))
                    close_reason = m.group(1).strip() if m else "PENDING_CLOSE"
                except Exception:
                    close_reason = "PENDING_CLOSE"
                try:
                    _record_realized_trade_on_close(state, lifecycle, client, account_id_key, close_reason)
                except Exception:
                    pass
            else:
                lifecycle.stage = "CANCELED"

            _event_once(
                cfg,
                lifecycle,
                "CANCEL_PENDING_RESOLVED",
                f"[AUTOEXEC] {lifecycle.symbol} CANCEL PENDING RESOLVED",
                f"""Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nResolved cancel-pending state. Stage now: {lifecycle.stage}\n""",
            )
        # Persist and return either way
        return

    # --- Bracket invariants (safe scope: we have client/account/lifecycle here) ---
    # If we're IN_POSITION and missing broker bracket orderIds, recover/attach/place best-effort.
    if lifecycle.stage == "IN_POSITION" and int(lifecycle.filled_qty or 0) > 0:
        try:
            _stabilize_entry_avg_cache(observed_filled_qty=int(lifecycle.filled_qty or 0), observed_status=(lifecycle.entry_status_cached or "EXECUTED"), force_refresh=False)
        except Exception as e:
            lifecycle.notes = (lifecycle.notes or "") + f" | entry_avg_invar_error:{e}"
        try:
            _ensure_brackets(client, account_id_key, lifecycle.symbol, lifecycle, cfg, log_prefix="[INVAR] ")
            # If both legs are now known, record bracket_qty for idempotency.
            try:
                fq = int(lifecycle.filled_qty or 0)
            except Exception:
                fq = 0
            if fq > 0 and lifecycle.stop_order_id:
                lifecycle.bracket_qty = fq
        except Exception as e:
            lifecycle.notes = (lifecycle.notes or "") + f" | bracket_invariant_error:{e}"

    # ---- ENTRY SENT: monitor fill / timeout ----
    if lifecycle.stage == "ENTRY_SENT" and lifecycle.entry_order_id:
        # Broker-truth safety: if E*TRADE already shows a position for this symbol,
        # treat the entry as filled even if order-status lookups are laggy.
        # This prevents (a) cancel-before-fill invalidations and (b) missing bracket placement.
        force_filled_qty = 0
        if int(lifecycle.filled_qty or 0) == 0:
            try:
                force_filled_qty = int(_get_broker_position_qty(lifecycle.symbol) or 0)
            except Exception:
                force_filled_qty = 0

        # Stop-breach safety: if price breaks the stop BEFORE entry fills, cancel the resting entry.
        # This prevents late fills on invalidated setups.
        if force_filled_qty == 0 and fetch_last_price_fn is not None and int(lifecycle.filled_qty or 0) == 0:
            try:
                last_px = _parse_float(fetch_last_price_fn(lifecycle.symbol))
            except Exception:
                last_px = None
            if last_px is not None and lifecycle.stop is not None and float(last_px) <= float(lifecycle.stop):
                ok_cancel = _cancel_with_verify(lifecycle.entry_order_id, "ENTRY_STOP_BREACH")
                if ok_cancel:
                    lifecycle.entry_order_id = None
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = "entry_canceled_stop_breach"
                    _event_once(
                        cfg,
                        lifecycle,
                        "ENTRY_STOP_BREACH_CANCEL",
                        f"[AUTOEXEC] {lifecycle.symbol} ENTRY CANCELED (STOP BREACH)",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Canceled: entry invalidated because last <= stop before fill.
Last: {last_px}
Stop: {lifecycle.stop}
""",
                    )
                    _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                    lifecycle.reserved_dollars = 0.0
                    return
                else:
                    lifecycle.stage = "CANCEL_PENDING"
                    lifecycle.cancel_requested_at = now.isoformat()
                    lifecycle.cancel_attempts = int(getattr(lifecycle, "cancel_attempts", 0) or 0)
                    lifecycle.notes = "pending_entry_cancel | stop_breach"
                    _event_once(
                        cfg,
                        lifecycle,
                        "ENTRY_STOP_BREACH_CANCEL_PENDING",
                        f"[AUTOEXEC] {lifecycle.symbol} ENTRY CANCEL PENDING (STOP BREACH)",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Cancel requested: entry invalidated because last <= stop before fill.
Last: {last_px}
Stop: {lifecycle.stop}
""",
                    )
                    return

        # Timeout (skip if we already detected a position via portfolio fallback)
        try:
            sent = datetime.fromisoformat(lifecycle.entry_sent_ts) if lifecycle.entry_sent_ts else datetime.fromisoformat(lifecycle.created_ts)
            age_min = (now - sent).total_seconds() / 60.0
        except Exception:
            age_min = 0.0

        timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
        if force_filled_qty == 0 and age_min >= timeout_m:
            ok_cancel = _cancel_with_verify(lifecycle.entry_order_id, "ENTRY")
            if ok_cancel:
                lifecycle.entry_order_id = None
                lifecycle.stage = "CANCELED"
                lifecycle.notes = f"entry_timeout_{timeout_m}m"
            else:
                lifecycle.stage = "CANCEL_PENDING"
                lifecycle.cancel_requested_at = now.isoformat()
                lifecycle.cancel_attempts = int(getattr(lifecycle, "cancel_attempts", 0) or 0)
                lifecycle.notes = f"pending_entry_cancel | entry_timeout_{timeout_m}m"
            _event_once(
                cfg,
                lifecycle,
                "ENTRY_TIMEOUT",
                f"[AUTOEXEC] {lifecycle.symbol} ENTRY TIMEOUT",
                f"""Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCanceled: entry order timeout ({timeout_m}m).\n""",
            )
            if lifecycle.stage == "CANCELED":
                _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                lifecycle.reserved_dollars = 0.0
            return

        oid_int = _oid_int(lifecycle.entry_order_id)
        if oid_int is None:
            lifecycle.notes = "entry_order_id_invalid"
            return

        prev_filled = int(lifecycle.filled_qty or 0)
        # Default to sticky cached values so we don't regress to UNKNOWN on intermittent API issues.
        status = lifecycle.entry_status_cached or "UNKNOWN"
        filled_qty = int(lifecycle.entry_filled_qty_cached or 0)

        if force_filled_qty > 0:
            status, filled_qty = ("EXECUTED", int(force_filled_qty))
            lifecycle.entry_status_cached = status
            lifecycle.entry_filled_qty_cached = int(filled_qty)
            lifecycle.entry_exec_detected_at = lifecycle.entry_exec_detected_at or now.isoformat()
            lifecycle.entry_last_checked_at = now.isoformat()
            lifecycle.entry_poll_failures = 0
            lifecycle.notes = f"entry_status={status} | pos_fill_fallback"
        else:
            try:
                status, filled_qty = _poll_order_status_cached(oid_int, symbol=str(lifecycle.symbol or "").upper())
                lifecycle.entry_status_cached = status or lifecycle.entry_status_cached
                lifecycle.entry_filled_qty_cached = int(filled_qty or lifecycle.entry_filled_qty_cached or 0)
                lifecycle.entry_last_checked_at = now.isoformat()
                lifecycle.entry_poll_failures = 0
                if str(status or "").upper() == "EXECUTED" and int(filled_qty or 0) > 0:
                    lifecycle.entry_exec_detected_at = lifecycle.entry_exec_detected_at or now.isoformat()
                lifecycle.notes = f"entry_status={status}"
            except Exception as e:
                lifecycle.entry_poll_failures = int(getattr(lifecycle, "entry_poll_failures", 0) or 0) + 1
                lifecycle.entry_last_checked_at = now.isoformat()
                # Keep cached status/qty; just annotate.
                lifecycle.notes = f"entry_status={status} | poll_failed:{type(e).__name__}:{str(e)[:120]}"

        # If we got any fills, manage position (even if partial).
        if filled_qty and int(filled_qty) > 0:
            filled_qty = int(filled_qty)
            used = float(filled_qty) * float(lifecycle.desired_entry)
            unused = max(0.0, float(lifecycle.reserved_dollars or 0.0) - used)
            if unused > 0:
                _release_pool(state, unused)
                lifecycle.reserved_dollars = used

            lifecycle.filled_qty = filled_qty
            # Sticky: once we have a fill, record it so we never regress to UNKNOWN later.
            lifecycle.entry_status_cached = "EXECUTED"
            lifecycle.entry_filled_qty_cached = int(filled_qty)
            lifecycle.entry_exec_detected_at = lifecycle.entry_exec_detected_at or now.isoformat()
            _stabilize_entry_avg_cache(observed_filled_qty=int(filled_qty or 0), observed_status=status, force_refresh=bool(int(filled_qty or 0) > int(prev_filled or 0)))
            if filled_qty > prev_filled:
                _event_once(
                    cfg,
                    lifecycle,
                    f"FILL_{filled_qty}",
                    f"[AUTOEXEC] {lifecycle.symbol} FILL UPDATE",
                    f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Filled qty: {filled_qty} (prev {prev_filled})
Entry: {lifecycle.desired_entry}
OrderId: {lifecycle.entry_order_id}
""",
                )

            # First-time execution email (separate from incremental fill updates)
            if prev_filled == 0:
                _record_activity(state, "ENTRY_EXECUTED", lifecycle, f"filled={filled_qty}")
                _event_once(
                    cfg,
                    lifecycle,
                    "ENTRY_EXECUTED",
                    f"[AUTOEXEC] {lifecycle.symbol} ENTRY EXECUTED",
                    f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Entry filled.
OrderId: {lifecycle.entry_order_id}
Filled qty: {filled_qty}
""",
                )


            # AUTOPILOT SAFETY: If the entry order partially fills, cancel the remaining
            # entry quantity immediately. Otherwise, the remaining open limit could fill
            # later while we are already IN_POSITION, and we would not resize brackets
            # because we stop monitoring the entry order after transitioning stages.
            try:
                desired_qty = int(lifecycle.qty or 0)
            except Exception:
                desired_qty = 0
            if desired_qty > 0 and filled_qty < desired_qty:
                ok_rem = _cancel_with_verify(lifecycle.entry_order_id, "ENTRY")
                if ok_rem:
                    lifecycle.entry_order_id = None
                    lifecycle.notes = (lifecycle.notes or "") + " | entry_remainder_canceled"
                else:
                    lifecycle.notes = (lifecycle.notes or "") + " | entry_remainder_cancel_pending"
                    # Keep monitoring entry order until broker confirms remainder is inactive.
                    return
                _event_once(
                    cfg,
                    lifecycle,
                    f"ENTRY_PARTIAL_CANCEL_{filled_qty}",
                    f"[AUTOEXEC] {lifecycle.symbol} ENTRY PARTIAL — REMAINDER CANCELED",
                    f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Entry partially filled; canceling remainder for safety.
Desired qty: {desired_qty}
Filled qty: {filled_qty}
""",
                )
                _record_activity(state, "ENTRY_CANCELED", lifecycle, f"partial_remainder_cancel filled={filled_qty}")

            # Place / ensure brackets (STOP + TP0) for the executed quantity.
            # IMPORTANT: Do not cancel an existing STOP just because TP0 is missing.
            need_resize = False
            try:
                prev_bq = int(lifecycle.bracket_qty or 0)
            except Exception:
                prev_bq = 0
            if prev_bq > 0 and prev_bq != int(filled_qty):
                need_resize = True

            if need_resize:
                # Cancel old brackets before resizing to the new filled qty.
                for oid in [lifecycle.stop_order_id, lifecycle.tp0_order_id]:
                    oid_i = _oid_int(oid)
                    if oid_i is not None:
                        try:
                            client.cancel_order(account_id_key, oid_i)
                        except Exception:
                            pass
                lifecycle.stop_order_id = None
                lifecycle.tp0_order_id = None
                lifecycle.bracket_qty = 0

            # Transition first so same-pass protection placement/attach is allowed on the
            # exact rerun where execution is first detected.
            lifecycle.stage = "IN_POSITION"

            if bool(getattr(cfg, "threshold_exit_enabled", False)):
                lifecycle.exit_mode = "THRESHOLD"
                lifecycle.bracket_qty = int(filled_qty)
                lifecycle.threshold_exit_activated_at = now.isoformat()
                _append_note(lifecycle, f"EXIT_MODE_LOCKED=THRESHOLD qty={filled_qty}")
                entry_px_dbg = None
                entry_src_dbg = "desired_entry"
                try:
                    if lifecycle.entry_avg_price_cached is not None:
                        entry_px_dbg = float(lifecycle.entry_avg_price_cached)
                        entry_src_dbg = "entry_avg_price_cached"
                except Exception:
                    entry_px_dbg = None
                if entry_px_dbg is None:
                    try:
                        entry_px_dbg = float(lifecycle.desired_entry or 0.0)
                    except Exception:
                        entry_px_dbg = None
                _thr_gain, _thr_loss, _thr_src = _threshold_engine_trigger_pcts(cfg, lifecycle)
                body = _threshold_exit_state_block(
                    cfg, lifecycle, now,
                    shared_last_px=None,
                    entry_px=entry_px_dbg,
                    profit_basis_source=entry_src_dbg,
                    gain_pct=None,
                    loss_pct_now=None,
                    gain_trigger_pct=float(_thr_gain),
                    loss_trigger_pct=float(_thr_loss),
                    broker_qty=int(filled_qty),
                    rem_qty=int(filled_qty),
                )
                body += f"Threshold source: {_thr_src}\n"
                _event_once(
                    cfg,
                    lifecycle,
                    f"THRESHOLD_EXIT_ACTIVE_{filled_qty}",
                    f"[AUTOEXEC] {lifecycle.symbol} THRESHOLD TRIGGER SELL IS ACTIVE",
                    body + "\nThreshold-trigger sell mode is active. No broker stop was placed. The bot will monitor gain/loss price fences on each reconcile cycle and market-sell if either threshold is crossed.\n",
                )
                _record_activity(state, "THRESHOLD_EXIT_ACTIVE", lifecycle, f"qty={filled_qty}")
            else:
                lifecycle.exit_mode = "TIME_PROFIT" if bool(getattr(cfg, "enable_time_profit_capture", False)) else "STOP"
                _append_note(lifecycle, f"EXIT_MODE_LOCKED={lifecycle.exit_mode} qty={filled_qty}")
                # Ensure brackets exist at broker (attach broker truth first, then place missing legs best-effort).
                _ensure_brackets(client, account_id_key, lifecycle.symbol, lifecycle, cfg, log_prefix="[AUTOEXEC] ")

                if lifecycle.stop_order_id:
                    lifecycle.bracket_qty = int(filled_qty)
                    lifecycle.notes = f"stop_protection_sent stop={lifecycle.stop} qty={filled_qty}"
                    _event_once(
                        cfg,
                        lifecycle,
                        f"STOP_PROTECTION_{filled_qty}",
                        f"[AUTOEXEC] {lifecycle.symbol} STOP PROTECTION PLACED",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Protective STOP placed for filled qty {filled_qty}.
STOP: {lifecycle.stop} (order {lifecycle.stop_order_id})
TP0: synthetic trigger only (no resting broker order)
""",
                    )
                    _record_activity(state, "BRACKETS_SENT", lifecycle, f"stop_oid={lifecycle.stop_order_id} qty={filled_qty}")
                else:
                    # Protective STOP incomplete — keep retrying on future reruns (with backoff inside _ensure_brackets).
                    lifecycle.brackets_attempts = int(getattr(lifecycle, "brackets_attempts", 0) or 0) + 1
                    lifecycle.brackets_last_attempt_at = now.isoformat()
                    lifecycle.brackets_last_error = "missing=STOP"
                    _record_activity(state, "BRACKETS_INCOMPLETE", lifecycle, lifecycle.brackets_last_error)
                    subj = f"[AUTOEXEC] {lifecycle.symbol} STOP PROTECTION INCOMPLETE"
                    _event_once(
                        cfg,
                        lifecycle,
                        f"STOP_PROTECTION_INCOMPLETE_{filled_qty}_{lifecycle.brackets_attempts}",
                        subj,
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Entry is filled but protective STOP is incomplete.
OrderId: {lifecycle.entry_order_id}
Filled qty: {filled_qty}

Missing: STOP
Known STOP oid: {lifecycle.stop_order_id}
TP0: synthetic trigger only (no resting broker order)

Last notes: {lifecycle.notes}
""",
                    )
            return

        # No fills and order ended
        if str(status).upper() in {"CANCELLED", "REJECTED", "EXPIRED"}:
            lifecycle.stage = "CANCELED"
            lifecycle.notes = f"entry_{str(status).lower()}"
            _event_once(
                cfg,
                lifecycle,
                f"ENTRY_{status}",
                f"[AUTOEXEC] {lifecycle.symbol} ENTRY {status}",
                f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Entry order ended with status: {status}.
""",
            )
            _record_activity(state, "ENTRY_CANCELED", lifecycle, f"status={status}")
            _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
            lifecycle.reserved_dollars = 0.0
            return

    # ---- IN POSITION / EXIT SENT: monitor exits ----
    if lifecycle.stage in {"IN_POSITION", "EXIT_SENT"}:
        bracket_qty = int(lifecycle.bracket_qty or lifecycle.filled_qty or 0)

        # Helper to close + release
        def _close(reason: str) -> None:
            # --- OCO / bracket safety (best-effort) ---
            # If TP executes, cancel the STOP sibling. If STOP executes, cancel the TP sibling.
            def _cancel_order_best_effort(order_id_str: Optional[str], label: str) -> bool:
                oid = _oid_int(order_id_str)
                if oid is None:
                    return True
                try:
                    client.cancel_order(account_id_key, oid)
                except Exception as e:
                    # log but still verify below
                    _event_once(
                        cfg,
                        lifecycle,
                        f"OCO_CANCEL_ERR_{label}_{oid}",
                        f"[AUTOEXEC] {lifecycle.symbol} OCO cancel error ({label})",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

OCO cleanup cancel attempt had error for {label} order_id={oid}
Close reason: {reason}
Error: {repr(e)}
""",
                    )

                inactive = _order_is_inactive(int(oid))
                if inactive:
                    _event_once(
                        cfg,
                        lifecycle,
                        f"OCO_CANCEL_{label}_{oid}",
                        f"[AUTOEXEC] {lifecycle.symbol} OCO cancel {label}",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

OCO cleanup: confirmed inactive {label} order_id={oid}
Close reason: {reason}
""",
                    )
                else:
                    _event_once(
                        cfg,
                        lifecycle,
                        f"OCO_CANCEL_PENDING_{label}_{oid}",
                        f"[AUTOEXEC] {lifecycle.symbol} OCO cancel pending ({label})",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

OCO cleanup: {label} order_id={oid} still appears ACTIVE after cancel attempt.
Close reason: {reason}
""",
                    )
                return inactive

            pending_cleanup = False
            if reason == "TP0_EXECUTED":
                ok = _cancel_order_best_effort(lifecycle.stop_order_id, "STOP")
                if not ok:
                    pending_cleanup = True
            elif reason == "STOP_EXECUTED":
                ok = _cancel_order_best_effort(lifecycle.tp0_order_id, "TP0")
                if not ok:
                    pending_cleanup = True

            # --- Bookkeeping: release any remaining reserved capital exactly once ---
            if float(lifecycle.reserved_dollars or 0.0) > 0.0:
                _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                lifecycle.reserved_dollars = 0.0

            if pending_cleanup:
                lifecycle.stage = "CANCEL_PENDING"
                lifecycle.cancel_requested_at = now.isoformat()
                lifecycle.cancel_attempts = int(getattr(lifecycle, "cancel_attempts", 0) or 0)
                lifecycle.notes = (lifecycle.notes or "") + f" | pending_close_reason:{reason} | pending_close"
                return

            # Record realized P&L for reporting (best-effort)
            try:
                _record_realized_trade_on_close(state, lifecycle, client, account_id_key, reason)
            except Exception:
                pass

            # Clear broker order ids once closed (prevents orphan state)
            lifecycle.entry_order_id = None
            lifecycle.stop_order_id = None
            lifecycle.tp0_order_id = None
            lifecycle.market_exit_order_id = None

            lifecycle.stage = "CLOSED"
            lifecycle.notes = reason
            _event_once(
                cfg,
                lifecycle,
                f"CLOSE_{reason}",
                f"[AUTOEXEC] {lifecycle.symbol} CLOSED",
                f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Closed: {reason}
""",
            )
            _record_activity(state, "EXIT_EXECUTED", lifecycle, str(reason))

        def _pending_exit_reason(default: str = "MARKET_EXIT_EXECUTED") -> str:
            try:
                m = re.search(r"EXIT_SENT reason=([^|]+)", str(lifecycle.notes or ""))
                if m:
                    return str(m.group(1)).strip()
            except Exception:
                pass
            return default

        # --- Exit truth via broker position (safety backstop) ---
        # E*TRADE order-status endpoints can be laggy/inconsistent. The broker position is the most reliable truth.
        # If we were in-position and the broker now shows 0 shares, we must be flat (some exit occurred).
        # If the position is reduced (partial exit), we enforce "close fully" by flattening remainder at market.
        pos_qty = None
        try:
            # Use fresh broker truth here; stale last-good caches are helpful for fill healing,
            # but can prevent exit/lifecycle-close detection after a real stop/TP0 execution.
            pos_qty = _get_broker_position_qty(lifecycle.symbol, force_refresh=True)
        except Exception:
            pos_qty = None

        # Broker-flat debounce: if E*TRADE briefly reports 0 shares right after a fill or during
        # a transient snapshot hiccup, do not close immediately. Start a timer and only auto-close
        # the lifecycle if broker position remains flat for a sustained period.
        FLAT_CLOSE_DEBOUNCE_SEC = 120
        if pos_qty is not None:
            if int(pos_qty) > 0:
                lifecycle.flat_detected_at = None
            elif int(pos_qty) == 0:
                flat_ts = getattr(lifecycle, "flat_detected_at", None)
                if not flat_ts:
                    lifecycle.flat_detected_at = now.isoformat()
                else:
                    try:
                        flat_dt = datetime.fromisoformat(str(flat_ts))
                        flat_age_sec = max(0.0, (now - flat_dt).total_seconds())
                    except Exception:
                        lifecycle.flat_detected_at = now.isoformat()
                        flat_age_sec = 0.0

                    if flat_age_sec >= FLAT_CLOSE_DEBOUNCE_SEC:
                        inferred = None
                        # Try to infer which leg executed for reporting only.
                        try:
                            if lifecycle.tp0_order_id:
                                _toid = _oid_int(lifecycle.tp0_order_id)
                                if _toid is not None:
                                    try:
                                        _ts, _tf = client.get_order_status_and_filled_qty(account_id_key, _toid, symbol=lifecycle.symbol)
                                    except Exception:
                                        _ts, _tf = ("UNKNOWN", 0)
                                    if str(_ts).upper() in {"EXECUTED", "FILLED"} or (_tf and int(_tf) >= bracket_qty):
                                        inferred = "TP0_EXECUTED"
                            if inferred is None and lifecycle.stop_order_id:
                                _soid = _oid_int(lifecycle.stop_order_id)
                                if _soid is not None:
                                    try:
                                        _ss, _sf = client.get_order_status_and_filled_qty(account_id_key, _soid, symbol=lifecycle.symbol)
                                    except Exception:
                                        _ss, _sf = ("UNKNOWN", 0)
                                    if str(_ss).upper() in {"EXECUTED", "FILLED"} or (_sf and int(_sf) >= bracket_qty):
                                        inferred = "STOP_EXECUTED"
                        except Exception:
                            inferred = None

                        if inferred is None:
                            try:
                                if getattr(lifecycle, "tp0_triggered_at", None) or lifecycle.tp0_order_id or getattr(lifecycle, "tp0_submit_started_at", None):
                                    inferred = "TP0_EXECUTED"
                                elif lifecycle.stop_order_id:
                                    inferred = "STOP_EXECUTED"
                            except Exception:
                                pass

                        _close(inferred or _pending_exit_reason("POSITION_ZERO_CLOSED"))
                        return

        if bracket_qty > 0 and pos_qty is not None:
            # Partial exit at broker -> enforce "close fully" by flattening remainder.
            if 0 < pos_qty < bracket_qty:
                # cancel remaining STOP/TP0 and market-sell remaining position
                try:
                    if lifecycle.stop_order_id:
                        _soid = _oid_int(lifecycle.stop_order_id)
                        if _soid is not None:
                            try:
                                client.cancel_order(account_id_key, _soid)
                            except Exception:
                                pass
                        lifecycle.stop_order_id = None
                    if lifecycle.tp0_order_id:
                        _toid = _oid_int(lifecycle.tp0_order_id)
                        if _toid is not None:
                            try:
                                client.cancel_order(account_id_key, _toid)
                            except Exception:
                                pass
                        lifecycle.tp0_order_id = None
                except Exception:
                    pass

                try:
                    mcoid = _mk_client_order_id(lifecycle.lifecycle_id, 'MK')
                    mode, mid, mpid = _place_sell_close_best_effort(
                        client=client,
                        account_id_key=account_id_key,
                        symbol=lifecycle.symbol,
                        qty=int(pos_qty),
                        client_order_id=mcoid,
                        last_px=None,
                        lifecycle=lifecycle,
                        note_prefix="MK",
                    )
                    lifecycle.market_exit_order_id = str(mid) if mid is not None else None

                    _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={mcoid} pid={mpid} oid={mid} leg=MK_PARTIAL")
                except Exception:
                    pass

                _mark_exit_sent(lifecycle, now, "POSITION_PARTIAL_FLATTEN", order_id=lifecycle.market_exit_order_id, mode="broker_backstop")
                return

        



        # If an explicit exit order has been submitted, keep reconciling until broker confirms flat/executed.
        if lifecycle.stage == "EXIT_SENT":
            try:
                if not lifecycle.market_exit_order_id and getattr(lifecycle, "close_client_order_id", None):
                    recovered = client.find_order_by_client_order_id(
                        account_id_key=account_id_key,
                        client_order_id=str(lifecycle.close_client_order_id),
                        symbol=lifecycle.symbol,
                    )
                    if recovered and recovered.get("orderId"):
                        lifecycle.market_exit_order_id = str(recovered.get("orderId"))
                        _append_note(lifecycle, f"EXIT_ATTACH_DUP_GUARD oid={lifecycle.market_exit_order_id}")
            except Exception:
                pass

            if lifecycle.market_exit_order_id:
                moid = _oid_int(lifecycle.market_exit_order_id)
                if moid is not None:
                    try:
                        m_status, m_filled = _poll_order_status_cached(moid, lifecycle.symbol)
                    except Exception:
                        m_status, m_filled = ("UNKNOWN", 0)
                    m_status_u = str(m_status or "UNKNOWN").upper()
                    if m_status_u in {"EXECUTED", "FILLED"} or (bracket_qty > 0 and int(m_filled or 0) >= bracket_qty):
                        _close(_pending_exit_reason())
                        return
                    if m_status_u in {"CANCELLED", "REJECTED", "EXPIRED"}:
                        _append_note(lifecycle, f"EXIT_ORDER_INACTIVE status={m_status_u}")
                        lifecycle.market_exit_order_id = None
                        lifecycle.close_submit_started_at = None
            # Keep waiting for broker position/order truth on future reruns.
            return

        # --- Shared last price / TP0 context for exit logic ---
        tp_symbol = str(lifecycle.symbol or "").upper()
        tp0_px = None
        if lifecycle.tp0 is not None:
            try:
                tp0_px = float(lifecycle.tp0)
            except Exception:
                tp0_px = None

        shared_last_px = None
        if fetch_last_price_fn is not None:
            try:
                shared_last_px = _parse_float(fetch_last_price_fn(tp_symbol or lifecycle.symbol))
            except Exception:
                shared_last_px = None

        # --- Threshold-trigger sell mode (no broker stop) ---
        try:
            if lifecycle.stage == "IN_POSITION" and bool(getattr(cfg, "threshold_exit_enabled", False)) and _effective_exit_mode(lifecycle) == "THRESHOLD":
                entry_px = None
                profit_basis_source = "desired_entry"
                if lifecycle.entry_avg_price_cached is not None:
                    try:
                        entry_px = float(lifecycle.entry_avg_price_cached)
                        profit_basis_source = "entry_avg_price_cached"
                    except Exception:
                        entry_px = None
                if entry_px is None:
                    try:
                        entry_px = float(lifecycle.desired_entry or 0.0)
                    except Exception:
                        entry_px = None
                gain_trigger_pct, loss_trigger_pct, threshold_source = _threshold_engine_trigger_pcts(cfg, lifecycle)
                try:
                    adaptive_pcts = _adaptive_threshold_engine_trigger_pcts(cfg, lifecycle, entry_px)
                    if adaptive_pcts is not None:
                        gain_trigger_pct, loss_trigger_pct, threshold_source = adaptive_pcts
                except Exception:
                    pass
                if (shared_last_px is None) or (not entry_px) or (entry_px <= 0):
                    body_base = _threshold_exit_state_block(cfg, lifecycle, now, shared_last_px=shared_last_px, entry_px=entry_px, profit_basis_source=profit_basis_source, gain_pct=None, loss_pct_now=None, gain_trigger_pct=gain_trigger_pct, loss_trigger_pct=loss_trigger_pct)
                    body_base += f"Threshold source: {threshold_source}\n"
                    _append_note(lifecycle, f"THRESHOLD_EXIT_WAITING_FOR_PRICE last={shared_last_px} basis={entry_px} thresh_src={threshold_source}")
                    _threshold_step_email(cfg, lifecycle, "THRESHOLD_EXIT_WAITING_FOR_PRICE", "WAITING FOR PRICE", body_base + "\nThreshold-trigger sell mode is active but a usable last price / profit basis was not available yet. Will retry on the next reconcile cycle.\n")
                else:
                    gain_pct = (float(shared_last_px) - float(entry_px)) / float(entry_px) * 100.0
                    loss_basis_px = float(entry_px)
                    loss_basis_source = profit_basis_source
                    try:
                        use_engine_specific = bool(getattr(cfg, "threshold_exit_use_engine_specific", False))
                        if use_engine_specific and str(getattr(lifecycle, "engine", "") or "").upper() == "RIDE" and str(getattr(lifecycle, "ride_entry_mode", "") or "").upper() == "PULLBACK":
                            pbl = _coerce_float(getattr(lifecycle, "pullback_band_low", None), None)
                            pbh = _coerce_float(getattr(lifecycle, "pullback_band_high", None), None)
                            if pbl is not None and pbh is not None:
                                bias = str(getattr(lifecycle, "notes", "") or "").upper()
                                # Lifecycle staging is currently LONG-only in this code path, but keep
                                # mirror-safe semantics for future short support.
                                if "BIAS=SHORT" in bias or "SHORT" in bias:
                                    loss_basis_px = float(max(pbl, pbh))
                                    loss_pct_now = (float(shared_last_px) - float(loss_basis_px)) / float(loss_basis_px) * 100.0
                                    loss_basis_source = "pullback_band_high"
                                else:
                                    loss_basis_px = float(min(pbl, pbh))
                                    loss_pct_now = (float(loss_basis_px) - float(shared_last_px)) / float(loss_basis_px) * 100.0
                                    loss_basis_source = "pullback_band_low"
                            else:
                                loss_pct_now = (float(entry_px) - float(shared_last_px)) / float(entry_px) * 100.0
                        else:
                            loss_pct_now = (float(entry_px) - float(shared_last_px)) / float(entry_px) * 100.0
                    except Exception:
                        loss_pct_now = (float(entry_px) - float(shared_last_px)) / float(entry_px) * 100.0
                        loss_basis_px = float(entry_px)
                        loss_basis_source = profit_basis_source
                    body_base = _threshold_exit_state_block(cfg, lifecycle, now, shared_last_px=shared_last_px, entry_px=entry_px, profit_basis_source=profit_basis_source, gain_pct=gain_pct, loss_pct_now=loss_pct_now, gain_trigger_pct=gain_trigger_pct, loss_trigger_pct=loss_trigger_pct)
                    body_base += f"Threshold source: {threshold_source}\n"
                    _append_note(lifecycle, f"THRESHOLD_EXIT_EVAL last={shared_last_px} basis={entry_px} source={profit_basis_source} thresh_src={threshold_source} gain={gain_pct:.2f}% loss={loss_pct_now:.2f}% gain_trig={gain_trigger_pct:.2f}% loss_trig={loss_trigger_pct:.2f}%")
                    exit_reason = None
                    if gain_pct >= gain_trigger_pct:
                        exit_reason = "THRESHOLD_GAIN"
                        _threshold_step_email(cfg, lifecycle, "THRESHOLD_EXIT_GAIN_TRIGGER", "GAIN TRIGGER FIRED", body_base + "\nGain threshold crossed. Beginning explicit SELL close preview/place workflow.\n")
                    elif loss_pct_now >= loss_trigger_pct:
                        exit_reason = "THRESHOLD_LOSS"
                        _threshold_step_email(cfg, lifecycle, "THRESHOLD_EXIT_LOSS_TRIGGER", "LOSS TRIGGER FIRED", body_base + "\nLoss threshold crossed. Beginning explicit SELL close preview/place workflow.\n")
                    else:
                        _threshold_step_email(cfg, lifecycle, "THRESHOLD_EXIT_MONITORING", "MONITORING", body_base + "\nThreshold-trigger sell mode remains active. Neither gain nor loss threshold has been crossed yet. Will check again on the next reconcile cycle.\n")
                    if exit_reason is not None:
                        try:
                            rem_qty = int(_get_broker_position_qty(lifecycle.symbol, force_refresh=True) or 0)
                        except Exception:
                            rem_qty = int(lifecycle.filled_qty or 0)
                        if rem_qty <= 0:
                            _append_note(lifecycle, "THRESHOLD_EXIT_NO_REMAINING_QTY")
                            body_qty = _threshold_exit_state_block(cfg, lifecycle, now, shared_last_px=shared_last_px, entry_px=entry_px, profit_basis_source=profit_basis_source, gain_pct=gain_pct, loss_pct_now=loss_pct_now, gain_trigger_pct=gain_trigger_pct, loss_trigger_pct=loss_trigger_pct, rem_qty=rem_qty)
                            _threshold_step_email(cfg, lifecycle, f"THRESHOLD_EXIT_NO_REMAINING_QTY_{exit_reason}", "NO REMAINING QTY", body_qty + "\nBroker position appears flat already; no SELL close sent.\n")
                            return
                        close_coid = _mk_client_order_id(lifecycle.lifecycle_id, 'MKTH')
                        if _close_inflight(lifecycle, now, close_coid, 90):
                            _append_note(lifecycle, 'THRESHOLD_EXIT_SKIP_DUP')
                            body_qty = _threshold_exit_state_block(cfg, lifecycle, now, shared_last_px=shared_last_px, entry_px=entry_px, profit_basis_source=profit_basis_source, gain_pct=gain_pct, loss_pct_now=loss_pct_now, gain_trigger_pct=gain_trigger_pct, loss_trigger_pct=loss_trigger_pct, rem_qty=rem_qty)
                            _threshold_step_email(cfg, lifecycle, f"THRESHOLD_EXIT_CLOSE_INFLIGHT_{exit_reason}", "CLOSE ALREADY INFLIGHT", body_qty + "\nA close order is already inflight; skipping duplicate submit.\n")
                            return
                        _mark_close_submit_started(lifecycle, now, close_coid)
                        try:
                            close_oid, close_pid, close_session = _place_timeout_profit_market_close_explicit(client=client, account_id_key=account_id_key, symbol=lifecycle.symbol, qty=int(rem_qty), client_order_id=close_coid, lifecycle=lifecycle, now=now)
                            lifecycle.market_exit_order_id = str(close_oid)
                            body_qty = _threshold_exit_state_block(cfg, lifecycle, now, shared_last_px=shared_last_px, entry_px=entry_px, profit_basis_source=profit_basis_source, gain_pct=gain_pct, loss_pct_now=loss_pct_now, gain_trigger_pct=gain_trigger_pct, loss_trigger_pct=loss_trigger_pct, rem_qty=rem_qty)
                            _threshold_step_email(cfg, lifecycle, f"THRESHOLD_EXIT_PREVIEW_OK_{exit_reason}", "EXIT PREVIEW OK", body_qty + f"\nPreview succeeded. preview_id={close_pid} session={close_session} qty={rem_qty}.\n")
                            _threshold_step_email(cfg, lifecycle, f"THRESHOLD_EXIT_SENT_{exit_reason}", "EXIT ORDER SENT", body_qty + f"\nSELL close placed successfully. order_id={close_oid} preview_id={close_pid} qty={rem_qty} session={close_session}.\n")
                        except Exception as place_exc:
                            lifecycle.close_submit_started_at = None
                            lifecycle.close_client_order_id = None
                            body_qty = _threshold_exit_state_block(cfg, lifecycle, now, shared_last_px=shared_last_px, entry_px=entry_px, profit_basis_source=profit_basis_source, gain_pct=gain_pct, loss_pct_now=loss_pct_now, gain_trigger_pct=gain_trigger_pct, loss_trigger_pct=loss_trigger_pct, rem_qty=rem_qty)
                            _append_note(lifecycle, f"THRESHOLD_EXIT_ERR {str(place_exc)[:240]}")
                            _threshold_step_email(cfg, lifecycle, f"THRESHOLD_EXIT_ERROR_{exit_reason}", "EXIT ORDER ERROR", body_qty + f"\nSELL close preview/place failed: {place_exc}\n")
                            return
                        _append_note(lifecycle, f"THRESHOLD_EXIT_{exit_reason} gain={gain_pct:.2f}% loss={loss_pct_now:.2f}%")
                        _mark_exit_sent(lifecycle, now, exit_reason, order_id=lifecycle.market_exit_order_id, mode="MARKET")
                        return
        except Exception as e:
            lifecycle.notes = (lifecycle.notes or "") + f" | threshold_exit_error:{e}"
            _threshold_step_email(cfg, lifecycle, "THRESHOLD_EXIT_FLOW_ERROR", "THRESHOLD FLOW ERROR", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nUnhandled threshold-exit flow error: {e}\n")

        # --- Simplified timeout-based profit capture (TP0 kept informational only) ---
        try:
            if lifecycle.stage == "IN_POSITION" and bool(getattr(cfg, "enable_time_profit_capture", False)) and not bool(getattr(cfg, "threshold_exit_enabled", False)) and _effective_exit_mode(lifecycle) != "THRESHOLD":
                entry_ts = lifecycle.entry_exec_detected_at or lifecycle.entry_sent_ts or lifecycle.created_ts
                try:
                    entry_dt = datetime.fromisoformat(str(entry_ts)) if entry_ts else None
                except Exception:
                    entry_dt = None

                mins_cfg = float(getattr(cfg, "time_profit_capture_minutes", 12) or 0)
                thresh_cfg = float(getattr(cfg, "time_profit_capture_profit_pct", 0.5) or 0.0)

                if entry_dt is not None and mins_cfg > 0 and (now - entry_dt).total_seconds() >= mins_cfg * 60.0:
                    entry_px = None
                    profit_basis_source = "desired_entry"
                    if lifecycle.entry_avg_price_cached is not None:
                        try:
                            entry_px = float(lifecycle.entry_avg_price_cached)
                            profit_basis_source = "entry_avg_price_cached"
                        except Exception:
                            entry_px = None
                    if entry_px is None:
                        try:
                            entry_px = float(lifecycle.desired_entry or 0.0)
                            profit_basis_source = "desired_entry"
                        except Exception:
                            entry_px = None

                    if (shared_last_px is None) or (not entry_px) or (entry_px <= 0):
                        body_base = _timeout_profit_state_block(
                            cfg, lifecycle, now,
                            shared_last_px=shared_last_px,
                            entry_px=entry_px,
                            profit_basis_source=profit_basis_source,
                            gain_pct=None,
                            mins_cfg=mins_cfg,
                            thresh_cfg=thresh_cfg,
                        )
                        _append_note(lifecycle, f"TIMEOUT_PROFIT_WAITING_FOR_PRICE last={shared_last_px} basis={entry_px}")
                        _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_WAITING_FOR_PRICE", "WAITING FOR PRICE", body_base + "\nTimeout elapsed but a usable last price / profit basis was not available yet. Will retry on the next reconcile cycle.\n")
                    else:
                        gain_pct = (float(shared_last_px) - float(entry_px)) / float(entry_px) * 100.0
                        body_base = _timeout_profit_state_block(
                            cfg, lifecycle, now,
                            shared_last_px=shared_last_px,
                            entry_px=entry_px,
                            profit_basis_source=profit_basis_source,
                            gain_pct=gain_pct,
                            mins_cfg=mins_cfg,
                            thresh_cfg=thresh_cfg,
                        )
                        _append_note(lifecycle, f"TIMEOUT_PROFIT_EVAL last={shared_last_px} basis={entry_px} source={profit_basis_source} gain={gain_pct:.2f}% mins={mins_cfg:.1f} thresh={thresh_cfg:.2f}%")
                        if gain_pct < thresh_cfg:
                            _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_BELOW_THRESHOLD", "BELOW THRESHOLD", body_base + "\nTimeout elapsed but gain is still below the configured threshold. Leaving the stop in place and checking again on the next reconcile cycle.\n")
                        if gain_pct >= thresh_cfg:
                            _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_TRIGGER_FIRED", "TIMEOUT TRIGGER FIRED", body_base + "\nTimeout-profit criteria met. Beginning stop cancel workflow.\n")

                            stop_absent = False
                            try:
                                if lifecycle.stop_order_id:
                                    _tp0_step_email(
                                        cfg, lifecycle, "TP0_TIMEOUT_STOP_CANCEL_REQUESTED", "STOP CANCEL REQUESTED",
                                        body_base + f"\nAttempting cancel for stop_order_id={lifecycle.stop_order_id}.\n"
                                    )
                                stop_absent = _cancel_stop_and_confirm_absent(
                                    client,
                                    account_id_key,
                                    lifecycle.symbol,
                                    lifecycle,
                                    filled_qty=bracket_qty,
                                    note_prefix="TIMEOUT_PROFIT_STOP_CANCEL",
                                )
                            except Exception as cancel_exc:
                                _append_note(lifecycle, f"TIMEOUT_PROFIT_STOP_CANCEL_ERR {str(cancel_exc)[:240]}")
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_STOP_CANCEL_ERROR", "STOP CANCEL ERROR", body_base + f"\nError while canceling stop: {cancel_exc}\n")
                                return

                            if not stop_absent:
                                _append_note(lifecycle, "TIMEOUT_PROFIT_WAITING_STOP_CANCEL_CONFIRM")
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_STOP_CANCEL_PENDING", "STOP CANCEL PENDING", body_base + "\nStop cancel is not fully confirmed yet (OPEN/CANCEL_REQUESTED/UNKNOWN broker state). Will retry on the next reconcile cycle before attempting the SELL close.\n")
                                return

                            _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_STOP_CANCEL_CONFIRMED", "STOP CANCEL CONFIRMED", body_base + "\nStop no longer appears in OPEN orders. Proceeding to explicit SELL close preview/place.\n")

                            rem_qty = 0
                            try:
                                rem_qty = int(_get_broker_position_qty(lifecycle.symbol, force_refresh=True) or 0)
                            except Exception:
                                rem_qty = int(lifecycle.filled_qty or 0)

                            if rem_qty <= 0:
                                _append_note(lifecycle, "TIMEOUT_PROFIT_NO_REMAINING_QTY")
                                body_with_qty = _timeout_profit_state_block(
                                    cfg, lifecycle, now,
                                    shared_last_px=shared_last_px,
                                    entry_px=entry_px,
                                    profit_basis_source=profit_basis_source,
                                    gain_pct=gain_pct,
                                    mins_cfg=mins_cfg,
                                    thresh_cfg=thresh_cfg,
                                    rem_qty=rem_qty,
                                )
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_NO_REMAINING_QTY", "NO REMAINING QTY", body_with_qty + "\nBroker position appears flat already; no SELL close sent.\n")
                                return

                            close_coid = _mk_client_order_id(lifecycle.lifecycle_id, 'MKPC')
                            if _close_inflight(lifecycle, now, close_coid, 90):
                                _append_note(lifecycle, 'TIMEOUT_PROFIT_SKIP_DUP')
                                body_with_qty = _timeout_profit_state_block(
                                    cfg, lifecycle, now,
                                    shared_last_px=shared_last_px,
                                    entry_px=entry_px,
                                    profit_basis_source=profit_basis_source,
                                    gain_pct=gain_pct,
                                    mins_cfg=mins_cfg,
                                    thresh_cfg=thresh_cfg,
                                    rem_qty=rem_qty,
                                )
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_CLOSE_INFLIGHT", "CLOSE ALREADY INFLIGHT", body_with_qty + "\nA close order is already inflight; skipping duplicate submit.\n")
                                return

                            _mark_close_submit_started(lifecycle, now, close_coid)
                            try:
                                close_oid, close_pid, close_session = _place_timeout_profit_market_close_explicit(
                                    client=client,
                                    account_id_key=account_id_key,
                                    symbol=lifecycle.symbol,
                                    qty=int(rem_qty),
                                    client_order_id=close_coid,
                                    lifecycle=lifecycle,
                                    now=now,
                                )
                                lifecycle.market_exit_order_id = str(close_oid)
                                body_with_qty = _timeout_profit_state_block(
                                    cfg, lifecycle, now,
                                    shared_last_px=shared_last_px,
                                    entry_px=entry_px,
                                    profit_basis_source=profit_basis_source,
                                    gain_pct=gain_pct,
                                    mins_cfg=mins_cfg,
                                    thresh_cfg=thresh_cfg,
                                    rem_qty=rem_qty,
                                )
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_EXIT_PREVIEW_OK", "EXIT PREVIEW OK", body_with_qty + f"\nPreview succeeded. preview_id={close_pid} session={close_session} qty={rem_qty}.\n")
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_EXIT_SENT", "EXIT ORDER SENT", body_with_qty + f"\nSELL close placed successfully. order_id={close_oid} preview_id={close_pid} qty={rem_qty} session={close_session}.\n")
                            except Exception as place_exc:
                                lifecycle.close_submit_started_at = None
                                lifecycle.close_client_order_id = None
                                emsg = str(place_exc)
                                _append_note(lifecycle, f"TIMEOUT_PROFIT_EXIT_ERR {emsg[:240]}")
                                body_with_qty = _timeout_profit_state_block(
                                    cfg, lifecycle, now,
                                    shared_last_px=shared_last_px,
                                    entry_px=entry_px,
                                    profit_basis_source=profit_basis_source,
                                    gain_pct=gain_pct,
                                    mins_cfg=mins_cfg,
                                    thresh_cfg=thresh_cfg,
                                    rem_qty=rem_qty,
                                )
                                emsg_u = emsg.upper()
                                if 'NOT ENOUGH AVAILABLE SHARES' in emsg_u or 'NOT FIND ENOUGH AVAILABLE SHARES' in emsg_u:
                                    _append_note(lifecycle, 'TIMEOUT_PROFIT_STOP_CANCEL_NOT_FINALIZED')
                                    _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_STOP_CANCEL_NOT_FINALIZED", "STOP CANCEL NOT FINALIZED", body_with_qty + f"\nSELL close preview failed because E*TRADE still reports shares unavailable. Treating stop cancellation as not finalized yet and will retry on the next reconcile cycle. Original broker error: {place_exc}\n")
                                    return
                                _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_EXIT_ERROR", "EXIT ORDER ERROR", body_with_qty + f"\nSELL close preview/place failed: {place_exc}\n")
                                return

                            _append_note(lifecycle, f"TIMEOUT_PROFIT_CAPTURE gain_pct={gain_pct:.2f}% mins={mins_cfg:.1f}")
                            _mark_exit_sent(lifecycle, now, "TIMEOUT_PROFIT_CAPTURE", order_id=lifecycle.market_exit_order_id, mode="MARKET")
                            return
        except Exception as e:
            lifecycle.notes = (lifecycle.notes or "") + f" | time_profit_capture_error:{e}"
            _tp0_step_email(cfg, lifecycle, "TP0_TIMEOUT_FLOW_ERROR", "TIMEOUT FLOW ERROR", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nUnhandled timeout-profit flow error: {e}\n")

        # Check STOP (protective)
        if lifecycle.stop_order_id:
            soid = _oid_int(lifecycle.stop_order_id)
            if soid is not None:
                try:
                    s_status, s_filled = client.get_order_status_and_filled_qty(account_id_key, soid, symbol=lifecycle.symbol)
                except Exception:
                    s_status, s_filled = ("UNKNOWN", 0)

                # Partial STOP fill: enforce full exit (mirror TP0 partial-flatten behavior)
                # If the STOP order has filled some but not all shares, we cancel remaining STOP + TP0 and
                # market-sell any remaining position to eliminate orphan-risk.
                if (
                    bracket_qty > 0
                    and s_filled
                    and 0 < int(s_filled) < bracket_qty
                    and str(s_status).upper() not in {"CANCELLED", "REJECTED", "EXPIRED"}
                ):
                    # cancel remaining STOP and TP0, market-sell remainder
                    try:
                        client.cancel_order(account_id_key, soid)
                    except Exception:
                        pass
                    lifecycle.stop_order_id = None

                    if lifecycle.tp0_order_id:
                        toid = _oid_int(lifecycle.tp0_order_id)
                        if toid is not None:
                            try:
                                client.cancel_order(account_id_key, toid)
                            except Exception:
                                pass
                        lifecycle.tp0_order_id = None

                    try:
                        rem = int(_get_broker_position_qty(lifecycle.symbol, force_refresh=True) or 0)
                    except Exception:
                        rem = 0

                    if rem > 0:
                        try:
                            mcoid = _mk_client_order_id(lifecycle.lifecycle_id, 'MK')
                            if _close_inflight(lifecycle, now, mcoid, 90):
                                _append_note(lifecycle, 'MK_SKIP_DUP')
                            else:
                                _mark_close_submit_started(lifecycle, now, mcoid)
                                mode, mid, mpid = _place_sell_close_best_effort(
                                    client=client,
                                    account_id_key=account_id_key,
                                    symbol=lifecycle.symbol,
                                    qty=int(rem),
                                    client_order_id=mcoid,
                                    last_px=None,
                                    lifecycle=lifecycle,
                                    note_prefix="MK",
                                )
                                lifecycle.market_exit_order_id = str(mid) if mid is not None else None

                            _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={mcoid} pid={mpid} oid={mid} leg=MK")
                        except Exception:
                            pass

                    _mark_exit_sent(lifecycle, now, "STOP_PARTIAL_FLATTEN", order_id=lifecycle.market_exit_order_id, mode="market_close")
                    return
                if str(s_status).upper() in {"EXECUTED", "FILLED"} or (s_filled and int(s_filled) >= bracket_qty and bracket_qty > 0):
                    _close("STOP_EXECUTED")
                    return

        # Check TP0
        if lifecycle.tp0_order_id:
            toid = _oid_int(lifecycle.tp0_order_id)
            if toid is not None:
                try:
                    t_status, t_filled = client.get_order_status_and_filled_qty(account_id_key, toid, symbol=lifecycle.symbol)
                except Exception:
                    t_status, t_filled = ("UNKNOWN", 0)

                # Partial TP fill: enforce full exit
                if t_filled and 0 < int(t_filled) < bracket_qty and str(t_status).upper() not in {"EXECUTED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                    # cancel remaining TP and stop, market-sell remainder
                    try:
                        client.cancel_order(account_id_key, toid)
                    except Exception:
                        pass
                    lifecycle.tp0_order_id = None
                    if lifecycle.stop_order_id:
                        soid = _oid_int(lifecycle.stop_order_id)
                        if soid is not None:
                            try:
                                client.cancel_order(account_id_key, soid)
                            except Exception:
                                pass
                        lifecycle.stop_order_id = None
                    try:
                        rem = int(_get_broker_position_qty(lifecycle.symbol, force_refresh=True) or 0)
                    except Exception:
                        rem = 0
                    if rem > 0:
                        try:
                            mcoid = _mk_client_order_id(lifecycle.lifecycle_id, 'MK')
                            if _close_inflight(lifecycle, now, mcoid, 90):
                                _append_note(lifecycle, 'MK_SKIP_DUP')
                            else:
                                _mark_close_submit_started(lifecycle, now, mcoid)
                                mode, mid, mpid = _place_sell_close_best_effort(
                                    client=client,
                                    account_id_key=account_id_key,
                                    symbol=lifecycle.symbol,
                                    qty=int(rem),
                                    client_order_id=mcoid,
                                    last_px=None,
                                    lifecycle=lifecycle,
                                    note_prefix="MK",
                                )
                                lifecycle.market_exit_order_id = str(mid) if mid is not None else None

                            _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={mcoid} pid={mpid} oid={mid} leg=MK")
                        except Exception:
                            pass
                    _mark_exit_sent(lifecycle, now, "TP0_PARTIAL_FLATTEN", order_id=lifecycle.market_exit_order_id, mode="market_close")
                    return

                if str(t_status).upper() in {"EXECUTED", "FILLED"} or (t_filled and int(t_filled) >= bracket_qty and bracket_qty > 0):
                    _close("TP0_EXECUTED")
                    return

        # Otherwise remain in position
        return


def _force_liquidate_all(client: ETradeClient, account_id_key: str, cfg: AutoExecConfig, state: Dict[str, Any]) -> None:
    """Hard liquidation by 15:55 ET.

    Actions:
      - Cancel any open entry/stop/tp0 orders for managed lifecycles
      - Market-sell any remaining positions for managed symbols
      - Mark lifecycles CLOSED and release reserved pool dollars
    """
    now = _now_et()

    managed_syms = set(state.get("lifecycles", {}).keys())

    # Cancel orders + close lifecycles
    for symbol, lst in list(state.get("lifecycles", {}).items()):
        new_lst = []
        for raw in list(lst):
            lc = lifecycle_from_raw(raw)
            if lc.stage in {"CLOSED", "CANCELED"}:
                new_lst.append(asdict(lc))
                continue

            for oid in [lc.entry_order_id, lc.stop_order_id, lc.tp0_order_id]:
                oid_i = _oid_int(oid)
                if oid_i is not None:
                    try:
                        client.cancel_order(account_id_key, oid_i)
                    except Exception:
                        pass

            # Release any reserved capital
            if float(lc.reserved_dollars or 0.0) > 0.0:
                _release_pool(state, float(lc.reserved_dollars or 0.0))
                lc.reserved_dollars = 0.0

            lc.stage = "CLOSED"
            lc.notes = "EOD_LIQUIDATION"
            _event_once(
                cfg,
                lc,
                "EOD_LIQUIDATION",
                f"[AUTOEXEC] {lc.symbol} EOD LIQUIDATION",
                f"""Time (ET): {now.isoformat()}\nSymbol: {lc.symbol}\nEngine: {lc.engine}\n\nEOD liquidation: canceled open orders; flattening remaining position if any.\n""",
            )
            new_lst.append(asdict(lc))

        state.setdefault("lifecycles", {})[symbol] = new_lst

    # Market sell remaining positions for managed symbols
    try:
        positions = _get_positions_map_cached()
    except Exception:
        positions = {}

    for sym, qty in (positions or {}).items():
        if sym not in managed_syms:
            continue
        try:
            q = int(qty or 0)
        except Exception:
            q = 0
        if q <= 0:
            continue
        try:
            mcoid = _mk_client_order_id(f'FLAT{sym}{now.strftime("%H%M")}', 'MK')
            if _close_inflight(lifecycle, now, mcoid, 90):
                _append_note(lifecycle, 'FLAT_SKIP_DUP')
            else:
                _mark_close_submit_started(lifecycle, now, mcoid)
                mode, mid, mpid = _place_sell_close_best_effort(
                    client=client,
                    account_id_key=account_id_key,
                    symbol=sym,
                    qty=int(q),
                    client_order_id=mcoid,
                    last_px=None,
                    lifecycle=lifecycle,
                    note_prefix="FLAT_MK",
                )

            _log(f"[AUTOEXEC][BROKER] force_flat_ok sym={sym} coid={mcoid} pid={mpid} oid={mid}")
        except Exception:
            pass

    # Persist state
    st.session_state["autoexec"] = state