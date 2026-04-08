from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import math

from indicators import (
    vwap as calc_vwap,
    session_vwap as calc_session_vwap,
    atr as calc_atr,
    ema as calc_ema,
    adx as calc_adx,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    find_breaker_block,
    in_zone,
)
from sessions import classify_session, classify_liquidity_phase


def _cap_score(x: float | int | None) -> int:
    """Scores are treated as 0..100 for UI + alerting.

    The internal point system can temporarily exceed 100 when multiple features
    stack or when ATR normalization scales up. We cap here so the UI never
    shows impossible percentages (e.g., 113%).
    """
    try:
        if x is None:
            return 0
        return int(np.clip(float(x), 0.0, 100.0))
    except Exception:
        return 0


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100 (calibrated)
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/PREMARKET/AFTERHOURS/OFF
    extras: Dict[str, Any]


# ---------------------------
# SWING / Intraday-Swing (structure-first) signal family
# ---------------------------

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample intraday OHLCV to a higher timeframe without additional API calls.

    We use this for Swing alerts so we don't add extra Alpha Vantage calls.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    out = (
        df[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
    )
    return out


def compute_swing_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi5: pd.Series,
    rsi14: pd.Series,
    macd_hist: pd.Series,
    *,
    interval: str = "1min",
    pro_mode: bool = False,
    # Time filters
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    # Shared options
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    fib_lookback_bars: int = 240,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    target_atr_pct: float | None = None,
) -> SignalResult:
    """SWING v2 — HTF dip-buy positioning with 5m confirmation (CONFIRM-only emails).

    Design principles:
      - SWING is a *positioning* engine (define cheap-risk buy zones) not an execution engine.
      - Use HTF structure (30m derived from intraday bars) to define the impulse leg and retrace band.
      - Stages:
          STALK   : candidate is valid; waiting for price to enter the buy zone
          BUY ZONE: price has entered the retrace band; waiting for confirmation
          CONFIRM : 5m confirmation (2-of-3); actionable (email is sent on CONFIRM only)
      - Streamlit safety: extras contain primitives only (no dict/list objects).
    """
    # -------------------------
    # Basic guards
    # -------------------------
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
        return SignalResult(
            symbol, "CHOP", 0, "Not enough data",
            None, None, None, None,
            None, None, "OFF",
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF"}
        )

    df = ohlcv.copy()
    if use_last_closed_only and len(df) >= 2:
        df = df.iloc[:-1].copy()

    last_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
    last_price = float(df["close"].iloc[-1])

    # -------------------------
    # Session gating (respects allowed_sessions toggles)
    # -------------------------
    try:
        sess = classify_session(
            last_ts,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
        )
    except Exception:
        sess = "OFF"

    if sess == "OFF":
        return SignalResult(
            symbol, "CHOP", 0, "Outside allowed session",
            None, None, None, None,
            last_price, last_ts, "OFF",
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF"}
        )

    # -------------------------
    # HTF (30m) context — derived via resample (no extra API call)
    #   - When running 5m, resampling preserves your interval choice.
    # -------------------------
    htf = _resample_ohlcv(df, "30T").tail(5 * 13 * 10).copy()  # ~10 trading days of 30m bars
    if len(htf) < 60:
        return SignalResult(
            symbol, "CHOP", 0, "Not enough HTF bars",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF"}
        )

    # Core HTF measures
    htf["ema20"] = calc_ema(htf["close"], 20)
    htf["ema50"] = calc_ema(htf["close"], 50)
    htf["atr"] = calc_atr(htf, 14)

    atr30 = float(htf["atr"].iloc[-1]) if np.isfinite(htf["atr"].iloc[-1]) else float(calc_atr(df, 14).iloc[-1])
    atr30 = atr30 if np.isfinite(atr30) and atr30 > 0 else max(1e-9, float(df["high"].tail(30).max() - df["low"].tail(30).min()) / 30.0)

    # HTF VWAP proxy (cumulative on HTF bars)
    try:
        htf_vwap = calc_vwap(htf).iloc[-1]
        htf_vwap = float(htf_vwap) if np.isfinite(htf_vwap) else None
    except Exception:
        htf_vwap = None

    # -------------------------
    # HTF pivot structure
    # -------------------------
    is_low = rolling_swing_lows(htf["low"], left=3, right=3)
    is_high = rolling_swing_highs(htf["high"], left=3, right=3)

    pivot_lows = htf.loc[is_low]
    pivot_highs = htf.loc[is_high]

    if len(pivot_lows) < 2 or len(pivot_highs) < 2:
        return SignalResult(
            symbol, "CHOP", 0, "Awaiting structure",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK"}
        )

    last_low = float(pivot_lows["low"].iloc[-1])
    prev_low = float(pivot_lows["low"].iloc[-2])
    last_high = float(pivot_highs["high"].iloc[-1])
    prev_high = float(pivot_highs["high"].iloc[-2])

    uptrend = (last_high > prev_high) and (last_low >= prev_low)
    downtrend = (last_low < prev_low) and (last_high <= prev_high)

    if not uptrend and not downtrend:
        return SignalResult(
            symbol, "CHOP", 0, "CHOP (HTF trend unclear)",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK"}
        )

    # IMPORTANT WIRING NOTE:
    # Downstream (tables + email gating) expects SWING bias labels:
    #   SWING_LONG / SWING_SHORT / CHOP
    # We keep a separate direction helper for internal comparisons.
    bias_dir = "LONG" if uptrend else "SHORT"
    bias = "SWING_LONG" if bias_dir == "LONG" else "SWING_SHORT"

    # -------------------------
    # Trend lock score (0..5)
    # -------------------------
    tl = 0
    # (1) EMA stack
    ema20 = float(htf["ema20"].iloc[-1]) if np.isfinite(htf["ema20"].iloc[-1]) else None
    ema50 = float(htf["ema50"].iloc[-1]) if np.isfinite(htf["ema50"].iloc[-1]) else None
    if ema20 is not None and ema50 is not None:
        if bias_dir == "LONG" and ema20 >= ema50:
            tl += 1
        if bias_dir == "SHORT" and ema20 <= ema50:
            tl += 1
    # (2) Price vs EMA20
    if ema20 is not None:
        if bias_dir == "LONG" and float(htf["close"].iloc[-1]) >= ema20:
            tl += 1
        if bias_dir == "SHORT" and float(htf["close"].iloc[-1]) <= ema20:
            tl += 1
    # (3) HTF VWAP alignment
    if htf_vwap is not None:
        if bias_dir == "LONG" and float(htf["close"].iloc[-1]) >= htf_vwap:
            tl += 1
        if bias_dir == "SHORT" and float(htf["close"].iloc[-1]) <= htf_vwap:
            tl += 1
    # (4) Structure confirmation
    tl += 1  # uptrend/downtrend already established
    # (5) DI spread proxy via candle range expansion in trend direction (robust, no extra indicators)
    # A small boost if the last HTF candle expanded in the trend direction.
    last_htf = htf.iloc[-1]
    if bias_dir == "LONG" and float(last_htf["close"]) >= float(last_htf["open"]):
        tl += 1
    if bias_dir == "SHORT" and float(last_htf["close"]) <= float(last_htf["open"]):
        tl += 1
    trend_lock_score = int(max(0, min(5, tl)))

    if trend_lock_score < 2:
        return SignalResult(
            symbol, bias, 0, "WATCH - Trend not locked",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", "trend_lock_score": trend_lock_score}
        )

    # -------------------------
    # Define impulse leg from HTF pivots
    # -------------------------
    # Build ordered pivot list
    piv = []
    for ts, row in pivot_lows.tail(12).iterrows():
        piv.append((ts, "L", float(row["low"])))
    for ts, row in pivot_highs.tail(12).iterrows():
        piv.append((ts, "H", float(row["high"])))
    piv.sort(key=lambda x: x[0])

    impulse_start = None
    impulse_end = None
    # Scan from the end to find the latest complete impulse in the trend direction.
    if bias_dir == "LONG":
        # need L then later H
        for i in range(len(piv) - 1, 0, -1):
            if piv[i][1] == "H":
                # find prior L
                for j in range(i - 1, -1, -1):
                    if piv[j][1] == "L":
                        impulse_start = piv[j][2]
                        impulse_end = piv[i][2]
                        break
            if impulse_start is not None:
                break
    else:
        # need H then later L
        for i in range(len(piv) - 1, 0, -1):
            if piv[i][1] == "L":
                for j in range(i - 1, -1, -1):
                    if piv[j][1] == "H":
                        impulse_start = piv[j][2]
                        impulse_end = piv[i][2]
                        break
            if impulse_start is not None:
                break

    if impulse_start is None or impulse_end is None:
        return SignalResult(
            symbol, bias, 0, "WATCH - Awaiting impulse leg",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", "trend_lock_score": trend_lock_score}
        )

    impulse_range = abs(impulse_end - impulse_start)
    if not np.isfinite(impulse_range) or impulse_range < 1.2 * atr30:
        return SignalResult(
            symbol, bias, 0, "WATCH - Impulse too small",
            None, None, None, None,
            last_price, last_ts, sess,
            {
                "family": "SWING",
                "stage": "PRE",
                "swing_stage": "STALK",
                "trend_lock_score": trend_lock_score,
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_mode": "pivot-leg",
            }
        )

    # -------------------------
    # Retrace band (38.2%..61.8%) + sweet spot 50%
    # -------------------------
    if bias_dir == "LONG":
        # Retrace from impulse_end downwards
        pb1 = float(impulse_end - 0.382 * impulse_range)
        pb2 = float(impulse_end - 0.618 * impulse_range)
        band_high = max(pb1, pb2)
        band_low = min(pb1, pb2)
        retrace_pct = 100.0 * max(0.0, min(1.0, (impulse_end - last_price) / impulse_range))
        invalidated = last_price < (impulse_start - 0.10 * atr30)
    else:
        pb1 = float(impulse_end + 0.382 * impulse_range)
        pb2 = float(impulse_end + 0.618 * impulse_range)
        band_high = max(pb1, pb2)
        band_low = min(pb1, pb2)
        retrace_pct = 100.0 * max(0.0, min(1.0, (last_price - impulse_end) / impulse_range))
        invalidated = last_price > (impulse_start + 0.10 * atr30)

    if invalidated:
        return SignalResult(
            symbol, bias, 0, "Invalidated (broke impulse start)",
            None, None, None, None,
            last_price, last_ts, sess,
            {
                "family": "SWING",
                "stage": "PRE",
                "swing_stage": "FAIL",
                "trend_lock_score": trend_lock_score,
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_pct": float(retrace_pct),
                "pb1": float(band_low),
                "pb2": float(band_high),
                "retrace_mode": "pivot-leg",
            }
        )

    # Pullback quality (0..6)
    # - 6 if in band & near 50% retrace; lower if shallow/deep or outside.
    target_retrace = 0.50
    if bias_dir == "LONG":
        retr = (impulse_end - last_price) / impulse_range
    else:
        retr = (last_price - impulse_end) / impulse_range
    retr = float(retr) if np.isfinite(retr) else 0.0
    pb_in_band = (last_price >= band_low) and (last_price <= band_high)
    dist_to_mid = abs(retr - target_retrace)
    pbq = 0
    pbq_reasons = []
    if pb_in_band:
        pbq = 4
        pbq_reasons.append("In band")
        if dist_to_mid <= 0.08:
            pbq += 2
            pbq_reasons.append("Near 50%")
        elif dist_to_mid <= 0.14:
            pbq += 1
            pbq_reasons.append("Near mid")
    else:
        # shallow pullback
        if retr < 0.30:
            pbq = 1
            pbq_reasons.append("Too shallow")
        # deep pullback
        elif retr > 0.75:
            pbq = 1
            pbq_reasons.append("Too deep")
        else:
            pbq = 2
            pbq_reasons.append("Approaching band")

    pullback_quality = int(max(0, min(6, pbq)))

    # -------------------------
    # Confluence (capped)
    # -------------------------
    confluences = []
    conf_n = 0
    tol = 0.25 * atr30

    # (1) HTF VWAP overlap
    if htf_vwap is not None and (band_low - tol) <= htf_vwap <= (band_high + tol):
        conf_n += 1
        confluences.append("HTF VWAP")

    # (2) Prior day levels from intraday df
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            d = df.copy()
            # Use last completed date
            last_date = d.index[-1].date()
            prev_days = sorted({x.date() for x in d.index if x.date() < last_date})
            if prev_days:
                pd_date = prev_days[-1]
                d_prev = d[d.index.date == pd_date]
                if len(d_prev) >= 5:
                    pdh = float(d_prev["high"].max())
                    pdl = float(d_prev["low"].min())
                    pdo = float(d_prev["open"].iloc[0])
                    for name, lvl in [("PDH", pdh), ("PDL", pdl), ("PDO", pdo)]:
                        if (band_low - tol) <= lvl <= (band_high + tol):
                            conf_n += 1
                            confluences.append(name)
    except Exception:
        pass

    # (3) Pivot cluster overlap (nearest prior pivot opposite side)
    try:
        # nearest HTF pivot level near band center
        band_mid = (band_low + band_high) / 2.0
        # use recent pivot highs/lows levels
        levels = []
        levels += [float(x) for x in pivot_highs["high"].tail(8).values]
        levels += [float(x) for x in pivot_lows["low"].tail(8).values]
        if levels:
            closest = min(levels, key=lambda x: abs(x - band_mid))
            if abs(closest - band_mid) <= tol:
                conf_n += 1
                confluences.append("HTF Pivot")
    except Exception:
        pass

    confluence_count = int(min(3, conf_n))
    confluences_str = ", ".join(confluences[:6]) if confluences else ""

    # -------------------------
    # Stage determination: STALK / BUY ZONE / CONFIRM
    # -------------------------
    stage = "STALK"
    entry_trigger_reason = ""
    # Zone entry check (within band OR within proximity)
    prox = 0.20 * atr30
    in_zone_now = (last_price >= (band_low - prox)) and (last_price <= (band_high + prox))
    if in_zone_now:
        stage = "BUY ZONE"

    # Confirmation checks on execution timeframe (df assumed to match selected interval, usually 5m)
    # We only allow confirm if we have a recent zone touch.
    recent = df.tail(20).copy()
    touched = False
    if bias_dir == "LONG":
        touched = float(recent["low"].min()) <= (band_high + prox)
    else:
        touched = float(recent["high"].max()) >= (band_low - prox)

    # Compute 5m ATR for stop sizing
    atr5 = float(calc_atr(df, 14).iloc[-1]) if len(df) >= 20 else atr30
    atr5 = atr5 if np.isfinite(atr5) and atr5 > 0 else atr30

    # Confirm rules (2-of-3)
    confirm_hits = 0
    confirm_reasons = []

    # (1) Reclaim VWAP after zone touch
    vwap_s = None
    try:
        vwap_s = calc_session_vwap(df).iloc[-1]
        vwap_s = float(vwap_s) if np.isfinite(vwap_s) else None
    except Exception:
        vwap_s = None

    if touched and vwap_s is not None and len(df) >= 2:
        prev_close = float(df["close"].iloc[-2])
        cur_close = float(df["close"].iloc[-1])
        if bias_dir == "LONG" and prev_close < vwap_s <= cur_close:
            confirm_hits += 1
            confirm_reasons.append("VWAP reclaim")
        if bias_dir == "SHORT" and prev_close > vwap_s >= cur_close:
            confirm_hits += 1
            confirm_reasons.append("VWAP reject")

    # (2) Two-candle reversal (robust)
    if touched and len(df) >= 3:
        c1 = df.iloc[-2]
        c2 = df.iloc[-1]
        if bias_dir == "LONG":
            if float(c1["close"]) < float(c1["open"]) and float(c2["close"]) > float(c2["open"]) and float(c2["close"]) > float(c1["high"]):
                confirm_hits += 1
                confirm_reasons.append("2-bar reversal")
        else:
            if float(c1["close"]) > float(c1["open"]) and float(c2["close"]) < float(c2["open"]) and float(c2["close"]) < float(c1["low"]):
                confirm_hits += 1
                confirm_reasons.append("2-bar reversal")

    # (3) RSI regime cross (compute from provided rsi14 if aligned; else compute on df closes)
    rsi_now = None
    rsi_prev = None
    try:
        if rsi14 is not None and len(rsi14) >= len(df):
            rsi_now = float(rsi14.iloc[-1])
            rsi_prev = float(rsi14.iloc[-2]) if len(rsi14) >= 2 else None
        else:
            # lightweight recompute using calc_rsi provided by engine (if available) is outside this module
            # so we compute locally here
            closes = df["close"].astype(float)
            delta = closes.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rs = up / down.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_now = float(rsi_series.iloc[-1])
            rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else None
    except Exception:
        rsi_now = None
        rsi_prev = None

    if touched and rsi_now is not None and rsi_prev is not None:
        if bias_dir == "LONG" and (rsi_prev < 50 <= rsi_now):
            confirm_hits += 1
            confirm_reasons.append("RSI reclaim")
        if bias_dir == "SHORT" and (rsi_prev > 50 >= rsi_now):
            confirm_hits += 1
            confirm_reasons.append("RSI roll")

    confirmed = touched and (confirm_hits >= 2)

    # -------------------------
    # Score (0..100) — optimized for entry quality
    # -------------------------
    score = 0
    score += 10 * trend_lock_score  # 0..50
    if impulse_range >= 2.0 * atr30:
        score += 8
    if impulse_range >= 3.0 * atr30:
        score += 6  # stack a bit more for bigger displacement
    # Pullback quality boost
    if pullback_quality > 2:
        score += 12 * (pullback_quality - 2)  # max +48
    # Confluence
    score += 8 * confluence_count  # max +24
    # Stage boosts
    if stage == "BUY ZONE":
        score += 10
    if confirmed:
        score += 18
    # Penalties
    # "Actionable" here is a scoring concept (do we have enough quality to consider the setup real?).
    # Email/alert gating is handled separately via extras["actionable"].
    actionable_ok = (confluence_count >= 1) and (pullback_quality >= 2)
    if not actionable_ok:
        score -= 8
    # Character ok: avoid over-extended RSI > 80 longs / <20 shorts
    character_ok = True
    if rsi_now is not None:
        if bias_dir == "LONG" and rsi_now > 80:
            character_ok = False
        if bias_dir == "SHORT" and rsi_now < 20:
            character_ok = False
    if not character_ok:
        score -= 20

    score = int(_cap_score(score))

    # -------------------------
    # Entry / stop / targets (only actionable on CONFIRM)
    # -------------------------
    entry = None
    stop = None
    tp0 = None
    tp1 = None

    # TP0 preference: mean-reversion to session VWAP if available; else impulse end
    if vwap_s is not None and np.isfinite(vwap_s):
        tp0 = float(vwap_s)
    else:
        tp0 = float(impulse_end)

    tp1 = float(impulse_end)

    if confirmed:
        stage = "CONFIRM"
        entry_trigger_reason = ", ".join(confirm_reasons)
        entry = last_price
        if bias_dir == "LONG":
            stop = float(band_low - 0.25 * atr5)
        else:
            stop = float(band_high + 0.25 * atr5)

        # Ensure targets are in the correct direction for R-multiples
        if bias_dir == "LONG":
            tp0 = max(tp0, entry + 0.5 * atr5)
            tp1 = max(tp1, entry + 1.0 * atr5)
        else:
            tp0 = min(tp0, entry - 0.5 * atr5)
            tp1 = min(tp1, entry - 1.0 * atr5)

        return SignalResult(
            symbol,
            bias,
            score,
            "CONFIRM - Dip-buy confirmed",
            float(entry),
            float(stop),
            float(tp0),
            float(tp1),
            last_price,
            last_ts,
            sess,
            {
                "family": "SWING",
                "stage": "CONFIRMED",
                # Email gating expects an explicit boolean.
                # For SWING we only want emails on CONFIRMED signals.
                "actionable": True,
                "swing_stage": "CONFIRM",
                "trend_lock_score": trend_lock_score,
                "retrace_pct": float(retrace_pct),
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_mode": "pivot-leg",
                "pullback_quality": pullback_quality,
                "pullback_quality_reasons": "; ".join(pbq_reasons),
                "confluence_count": confluence_count,
                "confluences": confluences_str,
                "pb1": float(band_low),
                "pb2": float(band_high),
                "pullback_band": [float(band_low), float(band_high)],
                "entry_zone": f"{band_low:.4f}–{band_high:.4f}",
                "entry_trigger_reason": entry_trigger_reason,
            }
        )

    # Not confirmed yet (no email)
    why = "WATCH - Trend locked, awaiting zone entry"
    if stage == "BUY ZONE":
        why = "SETUP - In buy zone, awaiting confirmation"
    return SignalResult(
        symbol,
        bias,
        score,
        why,
        None,
        None,
        None,
        None,
        last_price,
        last_ts,
        sess,
        {
            "family": "SWING",
            "stage": "PRE",
            "actionable": False,
            "swing_stage": stage,
            "trend_lock_score": trend_lock_score,
            "retrace_pct": float(retrace_pct),
            "impulse_start": float(impulse_start),
            "impulse_end": float(impulse_end),
            "retrace_mode": "pivot-leg",
            "pullback_quality": pullback_quality,
            "pullback_quality_reasons": "; ".join(pbq_reasons),
            "confluence_count": confluence_count,
            "confluences": confluences_str,
            "pb1": float(band_low),
            "pb2": float(band_high),
            "pullback_band": [float(band_low), float(band_high)],
            "entry_zone": f"{band_low:.4f}–{band_high:.4f}",
        }
    )


def _mfe_percentile_from_history(
    df: pd.DataFrame,
    *,
    direction: str,
    occur_mask: pd.Series,
    horizon_bars: int,
    pct: float,
) -> tuple[float | None, int]:
    """Compute a percentile of forward MFE for occurrences marked by occur_mask.

    LONG MFE is max(high fwd) - close at signal bar.
    SHORT MFE is close - min(low fwd).
    Returns (mfe_pct, n_samples).
    """
    try:
        h = int(horizon_bars)
        if h <= 0:
            return None, 0
    except Exception:
        return None, 0

    if occur_mask is None or df is None or len(df) == 0:
        return None, 0

    try:
        close = df["close"].astype(float)
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
    except Exception:
        return None, 0

    idxs = [i for i, ok in enumerate(occur_mask.values.tolist()) if bool(ok)]
    idxs = [i for i in idxs if i + h < len(df)]
    if len(idxs) < 10:
        return None, len(idxs)

    mfes: list[float] = []
    for i in idxs:
        ref = float(close.iloc[i])
        if direction.upper() == "LONG":
            fwd_max = float(hi.iloc[i + 1 : i + h + 1].max())
            mfes.append(max(0.0, fwd_max - ref))
        else:
            fwd_min = float(lo.iloc[i + 1 : i + h + 1].min())
            mfes.append(max(0.0, ref - fwd_min))

    if not mfes:
        return None, 0

    mfes.sort()
    k = int(round((pct / 100.0) * (len(mfes) - 1)))
    k = max(0, min(len(mfes) - 1, k))
    return float(mfes[k]), len(mfes)


def _tp3_from_expected_excursion(
    df: pd.DataFrame,
    *,
    direction: str,
    signature: dict,
    entry_px: float,
    interval_mins: int,
    lookback_bars: int = 600,
    horizon_bars: int | None = None,
) -> tuple[float | None, dict]:
    """Compute TP3 using expected excursion (rolling MFE) for similar historical signatures.

    Lightweight rolling backtest per symbol+interval:
    - Find prior bars where the same boolean signature fired
    - Compute forward Max Favorable Excursion (MFE) over horizon
    - Use a high percentile (95th) as TP3 (runner/lottery)

    Returns (tp3, diagnostics).
    """
    diag = {
        "tp3_mode": "mfe_p95",
        "samples": 0,
        "horizon_bars": None,
        "signature": signature,
    }
    if df is None or len(df) < 60:
        return None, diag

    try:
        n = int(lookback_bars)
    except Exception:
        n = 600
    n = max(120, min(len(df), n))
    d = df.iloc[-n:].copy()

    # Default horizon: 1m -> 15 bars (15m); 5m -> 6 bars (~30m)
    if horizon_bars is None:
        hb = 15 if int(interval_mins) <= 1 else 6
    else:
        hb = int(horizon_bars)
    hb = max(3, hb)
    diag["horizon_bars"] = hb

    # vwap series for signature matching (prefer a precomputed 'vwap_use')
    if "vwap_use" in d.columns:
        vwap_use = d["vwap_use"].astype(float)
    elif "vwap_sess" in d.columns:
        vwap_use = d["vwap_sess"].astype(float)
    elif "vwap_cum" in d.columns:
        vwap_use = d["vwap_cum"].astype(float)
    else:
        return None, diag

    close = d["close"].astype(float)

    # Recompute simple boolean events in-window to find prior occurrences.
    was_below = (close.shift(3) < vwap_use.shift(3)) | (close.shift(5) < vwap_use.shift(5))
    reclaim = (close > vwap_use) & (close.shift(1) <= vwap_use.shift(1))
    was_above = (close.shift(3) > vwap_use.shift(3)) | (close.shift(5) > vwap_use.shift(5))
    reject = (close < vwap_use) & (close.shift(1) >= vwap_use.shift(1))

    rsi5 = d.get("rsi5")
    rsi14 = d.get("rsi14")
    macd_hist = d.get("macd_hist")
    vol = d.get("volume")

    if rsi5 is not None:
        rsi5 = rsi5.astype(float)
    if rsi14 is not None:
        rsi14 = rsi14.astype(float)
    if macd_hist is not None:
        macd_hist = macd_hist.astype(float)

    # RSI events (match current engine semantics approximately)
    rsi_snap = None
    rsi_down = None
    if rsi5 is not None:
        rsi_snap = ((rsi5 >= 30) & (rsi5.shift(1) < 30)) | ((rsi5 >= 25) & (rsi5.shift(1) < 25))
        rsi_down = ((rsi5 <= 70) & (rsi5.shift(1) > 70)) | ((rsi5 <= 75) & (rsi5.shift(1) > 75))

    # MACD turns
    macd_up = None
    macd_dn = None
    if macd_hist is not None:
        macd_up = (macd_hist > macd_hist.shift(1)) & (macd_hist.shift(1) > macd_hist.shift(2))
        macd_dn = (macd_hist < macd_hist.shift(1)) & (macd_hist.shift(1) < macd_hist.shift(2))

    # Volume confirm: last bar volume >= multiplier * rolling median(30)
    vol_ok = None
    if vol is not None:
        v = vol.astype(float)
        med = v.rolling(30, min_periods=10).median()
        mult = float(signature.get("vol_mult") or 1.25)
        vol_ok = v >= (mult * med)

    # Micro-structure: higher-low / lower-high
    hl_ok = None
    lh_ok = None
    try:
        lows = d["low"].astype(float)
        highs = d["high"].astype(float)
        hl_ok = lows.iloc[-1] > lows.rolling(10, min_periods=5).min()
        lh_ok = highs.iloc[-1] < highs.rolling(10, min_periods=5).max()
    except Exception:
        pass

    # Build occurrence mask to match the CURRENT signature
    diru = direction.upper()
    if diru == "LONG":
        m = (was_below & reclaim)
        if signature.get("rsi_event") and rsi_snap is not None:
            m = m & rsi_snap
        if signature.get("macd_event") and macd_up is not None:
            m = m & macd_up
        if signature.get("vol_event") and vol_ok is not None:
            m = m & vol_ok
        if signature.get("struct_event") and hl_ok is not None:
            m = m & hl_ok
    else:
        m = (was_above & reject)
        if signature.get("rsi_event") and rsi_down is not None:
            m = m & rsi_down
        if signature.get("macd_event") and macd_dn is not None:
            m = m & macd_dn
        if signature.get("vol_event") and vol_ok is not None:
            m = m & vol_ok
        if signature.get("struct_event") and lh_ok is not None:
            m = m & lh_ok

    mfe95, n_samples = _mfe_percentile_from_history(d, direction=diru, occur_mask=m.fillna(False), horizon_bars=hb, pct=95.0)
    diag["samples"] = int(n_samples)
    if mfe95 is None or not np.isfinite(mfe95):
        return None, diag

    try:
        mfe95 = float(mfe95)
        if diru == "LONG":
            return float(entry_px) + mfe95, diag
        return float(entry_px) - mfe95, diag
    except Exception:
        return None, diag

def _candidate_levels_from_context(
    *,
    levels: Dict[str, Any],
    recent_swing_high: float,
    recent_swing_low: float,
    hi: float,
    lo: float,
) -> Dict[str, float]:
    """Collect common structure/liquidity levels into a flat dict of floats.

    We use these as *potential* scalp targets (TP0). We intentionally favor
    levels that are meaningful to traders (prior day hi/lo, ORB, swing pivots),
    but fall back gracefully when some session levels aren't available.
    """
    out: Dict[str, float] = {}

    def _add(name: str, v: Any):
        try:
            if v is None:
                return
            fv = float(v)
            if np.isfinite(fv):
                out[name] = fv
        except Exception:
            return

    # Session liquidity levels (may be None)
    _add("orb_high", levels.get("orb_high"))
    _add("orb_low", levels.get("orb_low"))
    _add("prior_high", levels.get("prior_high"))
    _add("prior_low", levels.get("prior_low"))
    _add("premarket_high", levels.get("premarket_high"))
    _add("premarket_low", levels.get("premarket_low"))

    # Swing + range context
    _add("recent_swing_high", recent_swing_high)
    _add("recent_swing_low", recent_swing_low)
    _add("range_high", hi)
    _add("range_low", lo)
    return out


def _pick_tp0(
    direction: str,
    *,
    entry_px: float,
    last_px: float,
    atr_last: float,
    levels: Dict[str, float],
) -> Optional[float]:
    """Pick TP0 as the nearest meaningful level beyond entry.

    For scalping, TP0 should usually be *closer* than 1R/2R and should map to
    real structure. If no structure exists in-range, we fall back to an ATR-based
    objective.
    """
    try:
        entry_px = float(entry_px)
        last_px = float(last_px)
    except Exception:
        return None

    max_dist = None
    if atr_last and atr_last > 0:
        # Don't pick a target 10 ATR away for a scalp; keep it sane.
        max_dist = 3.0 * float(atr_last)

    cands: List[float] = []
    if direction == "LONG":
        for _, lvl in levels.items():
            if lvl > entry_px:
                cands.append(float(lvl))
        if cands:
            tp0 = min(cands, key=lambda x: abs(x - entry_px))
            if max_dist is None or abs(tp0 - entry_px) <= max_dist:
                return float(tp0)
        # Fallback: small objective beyond last/entry
        bump = 0.8 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
        return float(max(entry_px, last_px) + bump)

    # SHORT
    for _, lvl in levels.items():
        if lvl < entry_px:
            cands.append(float(lvl))
    if cands:
        tp0 = min(cands, key=lambda x: abs(x - entry_px))
        if max_dist is None or abs(tp0 - entry_px) <= max_dist:
            return float(tp0)
    bump = 0.8 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
    return float(min(entry_px, last_px) - bump)


def _eta_minutes_to_tp0(
    *,
    last_px: float,
    tp0: Optional[float],
    atr_last: float,
    interval_mins: int,
    liquidity_mult: float,
) -> Optional[float]:
    """Rough expected minutes to TP0 using ATR as a speed proxy.

    This is not meant to be precise. It's a UI helper to detect *slow* setups
    (common midday / low-liquidity conditions).
    """
    try:
        if tp0 is None:
            return None
        if not atr_last or atr_last <= 0:
            return None
        dist = abs(float(tp0) - float(last_px))
        bars = dist / float(atr_last)
        # liquidity_mult >1 means faster; <1 slower.
        speed = max(0.5, float(liquidity_mult))
        mins = bars * float(interval_mins) / speed
        return float(min(max(mins, 0.0), 999.0))
    except Exception:
        return None


def _entry_limit_and_chase(
    direction: str,
    *,
    entry_px: float,
    last_px: float,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> Tuple[float, float]:
    """Return (limit_entry, chase_line).

    - limit_entry: your planned limit.
    - chase_line: a "max pain" price where, if crossed, you're late and should
      reassess or switch to a different execution model.
    """
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=float(atr_last or 0.0),
        atr_fraction_slippage=float(atr_fraction_slippage or 0.0),
    )
    try:
        entry_px = float(entry_px)
        last_px = float(last_px)
    except Exception:
        return entry_px, entry_px

    # "Chase" is intentionally tight for scalps.
    chase_pad = 0.25 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
    if direction == "LONG":
        chase = max(entry_px, last_px) + chase_pad + slip
        return float(entry_px), float(chase)
    chase = min(entry_px, last_px) - chase_pad - slip
    return float(entry_px), float(chase)


def _is_rising(series: pd.Series, bars: int = 3) -> bool:
    """Simple monotonic rise check over the last N bars."""
    try:
        s = series.dropna().tail(int(bars))
        if len(s) < int(bars):
            return False
        return bool(all(float(s.iloc[i]) > float(s.iloc[i - 1]) for i in range(1, len(s))))
    except Exception:
        return False


def _is_falling(series: pd.Series, bars: int = 3) -> bool:
    """Simple monotonic fall check over the last N bars."""
    try:
        s = series.dropna().tail(int(bars))
        if len(s) < int(bars):
            return False
        return bool(all(float(s.iloc[i]) < float(s.iloc[i - 1]) for i in range(1, len(s))))
    except Exception:
        return False


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def _fib_retracement_levels(hi: float, lo: float) -> List[Tuple[str, float]]:
    ratios = [0.382, 0.5, 0.618, 0.786]
    rng = hi - lo
    if rng <= 0:
        return []
    # "pullback" levels for an up-move: hi - r*(hi-lo)
    return [(f"Fib {r:g}", hi - r * rng) for r in ratios]


def _fib_extensions(hi: float, lo: float) -> List[Tuple[str, float]]:
    # extensions above hi for longs, below lo for shorts (we'll mirror in logic)
    ratios = [1.0, 1.272, 1.618]
    rng = hi - lo
    if rng <= 0:
        return []
    return [(f"Ext {r:g}", hi + (r - 1.0) * rng) for r in ratios]


def _closest_level(price: float, levels: List[Tuple[str, float]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not levels:
        return None, None, None
    name, lvl = min(levels, key=lambda x: abs(price - x[1]))
    return name, float(lvl), float(abs(price - lvl))


def _session_liquidity_levels(df: pd.DataFrame, interval_mins: int, orb_minutes: int):
    """Compute simple liquidity levels: prior session high/low, today's premarket high/low, and ORB high/low."""
    if df is None or len(df) < 5:
        return {}
    # normalize timestamps to ET
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    else:
        ts = pd.to_datetime(df.index)

    try:
        ts = ts.dt.tz_localize("America/New_York") if getattr(ts.dt, "tz", None) is None else ts.dt.tz_convert("America/New_York")
    except Exception:
        try:
            ts = ts.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            # if tz ops fail, fall back to naive dates
            pass

    d = df.copy()
    d["_ts"] = ts
    # derive dates
    try:
        cur_date = d["_ts"].iloc[-1].date()
        dates = sorted({x.date() for x in d["_ts"] if pd.notna(x)})
    except Exception:
        cur_date = pd.to_datetime(df.index[-1]).date()
        dates = sorted({pd.to_datetime(x).date() for x in df.index})

    prev_date = dates[-2] if len(dates) >= 2 else cur_date

    def _t(x):
        try:
            return x.time()
        except Exception:
            return None

    def _is_pre(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("04:00").time()) and (t < pd.Timestamp("09:30").time())

    def _is_rth(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("09:30").time()) and (t <= pd.Timestamp("16:00").time())

    prev = d[d["_ts"].dt.date == prev_date] if "_ts" in d else df.iloc[:0]
    prev_rth = prev[prev["_ts"].apply(_is_rth)] if len(prev) else prev
    prior_high = float(prev_rth["high"].max()) if len(prev_rth) else (float(prev["high"].max()) if len(prev) else None)
    prior_low = float(prev_rth["low"].min()) if len(prev_rth) else (float(prev["low"].min()) if len(prev) else None)

    cur = d[d["_ts"].dt.date == cur_date] if "_ts" in d else df
    cur_pre = cur[cur["_ts"].apply(_is_pre)] if len(cur) else cur
    pre_hi = float(cur_pre["high"].max()) if len(cur_pre) else None
    pre_lo = float(cur_pre["low"].min()) if len(cur_pre) else None

    cur_rth = cur[cur["_ts"].apply(_is_rth)] if len(cur) else cur
    orb_bars = max(1, int(math.ceil(float(orb_minutes) / max(float(interval_mins), 1.0))))
    orb_slice = cur_rth.head(orb_bars)
    orb_hi = float(orb_slice["high"].max()) if len(orb_slice) else None
    orb_lo = float(orb_slice["low"].min()) if len(orb_slice) else None

    return {
        "prior_high": prior_high, "prior_low": prior_low,
        "premarket_high": pre_hi, "premarket_low": pre_lo,
        "orb_high": orb_hi, "orb_low": orb_lo,
    }

def _asof_slice(df: pd.DataFrame, interval_mins: int, use_last_closed_only: bool, bar_closed_guard: bool) -> pd.DataFrame:
    """Return df truncated so the last row represents the 'as-of' bar we can legally use."""
    if df is None or len(df) < 3:
        return df
    asof_idx = len(df) - 1

    # Always allow "snapshot mode" to use last fully completed bar
    if use_last_closed_only:
        asof_idx = max(0, len(df) - 2)

    if bar_closed_guard and len(df) >= 2:
        try:
            # Determine timestamp of latest bar
            if "time" in df.columns:
                last_ts = pd.to_datetime(df["time"].iloc[-1], utc=False)
            else:
                last_ts = pd.to_datetime(df.index[-1], utc=False)

            # Normalize to ET if timezone-naive
            now = pd.Timestamp.now(tz="America/New_York")
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("America/New_York")
            else:
                last_ts = last_ts.tz_convert("America/New_York")

            bar_end = last_ts + pd.Timedelta(minutes=int(interval_mins))
            # If bar hasn't ended yet, step back one candle (avoid partial)
            if now < bar_end:
                asof_idx = min(asof_idx, len(df) - 2)
        except Exception:
            # If anything goes sideways, be conservative
            asof_idx = min(asof_idx, len(df) - 2)

    asof_idx = max(0, int(asof_idx))
    return df.iloc[: asof_idx + 1].copy()


def _detect_liquidity_sweep(df: pd.DataFrame, levels: dict, *, atr_last: float | None = None, buffer: float = 0.0):
    """Liquidity sweep with confirmation (reclaim + displacement).

    We only count a sweep when ALL are true on the latest bar:
      1) Liquidity grab (wick through a key level)
      2) Reclaim (close back on the 'correct' side of the level)
      3) Displacement (range >= ~1.2x ATR) to filter chop/fakes

    Returns:
      {"type": "...", "level": float(level), "confirmed": bool}
    or None.
    """
    if df is None or len(df) < 2 or not levels:
        return None

    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    # Displacement filter (keep it mild; still allow if ATR isn't available)
    disp_ok = True
    if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
        disp_ok = float(h - l) >= 1.2 * float(atr_last)

    def _bull(level: float) -> Optional[dict]:
        # wick below, reclaim above
        if l < level - buffer and c > level + buffer and disp_ok:
            return {"type": "bull_sweep", "level": float(level), "confirmed": True}
        return None

    def _bear(level: float) -> Optional[dict]:
        # wick above, reclaim below
        if h > level + buffer and c < level - buffer and disp_ok:
            return {"type": "bear_sweep", "level": float(level), "confirmed": True}
        return None

    # Priority: prior day hi/lo, then premarket hi/lo
    ph = levels.get("prior_high")
    pl = levels.get("prior_low")
    if ph is not None:
        out = _bear(float(ph))
        if out:
            out["type"] = "bear_sweep_prior_high"
            return out
    if pl is not None:
        out = _bull(float(pl))
        if out:
            out["type"] = "bull_sweep_prior_low"
            return out

    pmah = levels.get("premarket_high")
    pmal = levels.get("premarket_low")
    if pmah is not None:
        out = _bear(float(pmah))
        if out:
            out["type"] = "bear_sweep_premarket_high"
            return out
    if pmal is not None:
        out = _bull(float(pmal))
        if out:
            out["type"] = "bull_sweep_premarket_low"
            return out

    return None


def _orb_three_stage(
    df: pd.DataFrame,
    *,
    orb_high: float | None,
    orb_low: float | None,
    buffer: float,
    lookback_bars: int = 30,
    accept_bars: int = 2,
) -> Dict[str, bool]:
    """ORB as a 3-stage sequence: break -> accept -> retest.

    Bull:
      - break: close crosses above orb_high
      - accept: next `accept_bars` closes stay above orb_high
      - retest: subsequent bar(s) touch orb_high (within buffer) and close back above

    Bear mirrors below orb_low.

    Returns dict with:
      {"bull_orb_seq": bool, "bear_orb_seq": bool, "bull_break": bool, "bear_break": bool}
    """
    out = {"bull_orb_seq": False, "bear_orb_seq": False, "bull_break": False, "bear_break": False}
    if df is None or len(df) < 8:
        return out

    d = df.tail(int(min(max(10, lookback_bars), len(df)))).copy()
    c = d["close"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)

    # --- Bull sequence ---
    if orb_high is not None and np.isfinite(float(orb_high)):
        level = float(orb_high)
        broke_idx = None
        for i in range(1, len(d)):
            if c.iloc[i] > level + buffer and c.iloc[i - 1] <= level + buffer:
                broke_idx = i
        if broke_idx is not None:
            out["bull_break"] = True
            # accept: next N closes remain above
            end_acc = min(len(d), broke_idx + 1 + int(accept_bars))
            acc_ok = True
            for j in range(broke_idx + 1, end_acc):
                if c.iloc[j] <= level + buffer:
                    acc_ok = False
                    break
            if acc_ok and end_acc <= len(d) - 1:
                # retest: any later bar tags level (low <= level+buffer) and closes back above
                for k in range(end_acc, len(d)):
                    if l.iloc[k] <= level + buffer and c.iloc[k] > level + buffer:
                        out["bull_orb_seq"] = True
                        break

    # --- Bear sequence ---
    if orb_low is not None and np.isfinite(float(orb_low)):
        level = float(orb_low)
        broke_idx = None
        for i in range(1, len(d)):
            if c.iloc[i] < level - buffer and c.iloc[i - 1] >= level - buffer:
                broke_idx = i
        if broke_idx is not None:
            out["bear_break"] = True
            end_acc = min(len(d), broke_idx + 1 + int(accept_bars))
            acc_ok = True
            for j in range(broke_idx + 1, end_acc):
                if c.iloc[j] >= level - buffer:
                    acc_ok = False
                    break
            if acc_ok and end_acc <= len(d) - 1:
                for k in range(end_acc, len(d)):
                    if h.iloc[k] >= level - buffer and c.iloc[k] < level - buffer:
                        out["bear_orb_seq"] = True
                        break

    return out



def _detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series | None = None,
    *,
    lookback: int = 160,
    pivot_lr: int = 3,
    min_price_delta_atr: float = 0.20,
    min_rsi_delta: float = 3.0,
) -> Optional[Dict[str, float | str]]:
    """Pivot-based RSI divergence with RSI-5 timing + RSI-14 validation.

    We use PRICE pivots (swing highs/lows) and compare RSI values at those pivots.
    - RSI-5 provides the timing (fast divergence signal)
    - RSI-14 acts as a validator (should not *contradict* the divergence)

    Bullish divergence:
      price pivot low2 < low1 by >= min_price_delta_atr * ATR
      AND RSI-5 at low2 > RSI-5 at low1 by >= min_rsi_delta
      AND RSI-14 at low2 >= RSI-14 at low1 - 1 (soft validation)

    Bearish divergence:
      price pivot high2 > high1 by >= min_price_delta_atr * ATR
      AND RSI-5 at high2 < RSI-5 at high1 by >= min_rsi_delta
      AND RSI-14 at high2 <= RSI-14 at high1 + 1 (soft validation)

    Returns dict like:
      {"type": "bull"|"bear", "strength": float, ...}
    """
    if df is None or len(df) < 25 or rsi_fast is None or len(rsi_fast) < 25:
        return None

    d = df.tail(int(min(max(60, lookback), len(df)))).copy()
    r5 = rsi_fast.reindex(d.index).ffill()
    if r5.isna().all():
        return None
    r14 = None
    if rsi_slow is not None:
        r14 = rsi_slow.reindex(d.index).ffill()

    # ATR for scaling (fallback to price*0.002 if missing)
    atr_last = None
    try:
        if "atr14" in d.columns and np.isfinite(float(d["atr14"].iloc[-1])):
            atr_last = float(d["atr14"].iloc[-1])
    except Exception:
        atr_last = None
    atr_scale = atr_last if (atr_last is not None and atr_last > 0) else float(d["close"].iloc[-1]) * 0.002

    # Price pivots
    lows_mask = rolling_swing_lows(d["low"], left=int(pivot_lr), right=int(pivot_lr))
    highs_mask = rolling_swing_highs(d["high"], left=int(pivot_lr), right=int(pivot_lr))
    piv_lows = d.loc[lows_mask, ["low"]].tail(6)
    piv_highs = d.loc[highs_mask, ["high"]].tail(6)

    # --- Bull divergence on the last two pivot lows ---
    if len(piv_lows) >= 2:
        a_idx = piv_lows.index[-2]
        b_idx = piv_lows.index[-1]
        p_a = float(d.loc[a_idx, "low"])
        p_b = float(d.loc[b_idx, "low"])
        r_a = float(r5.loc[a_idx])
        r_b = float(r5.loc[b_idx])

        price_ok = (p_b < p_a) and ((p_a - p_b) >= float(min_price_delta_atr) * atr_scale)
        rsi_ok = (r_b > r_a) and ((r_b - r_a) >= float(min_rsi_delta))
        slow_ok = True
        if r14 is not None and not r14.isna().all():
            try:
                s_a = float(r14.loc[a_idx])
                s_b = float(r14.loc[b_idx])
                slow_ok = (s_b >= s_a - 1.0)  # don't contradict
            except Exception:
                slow_ok = True

        if price_ok and rsi_ok and slow_ok:
            strength = float((r_b - r_a) / max(1.0, min_rsi_delta)) + float((p_a - p_b) / max(1e-9, atr_scale))
            return {"type": "bull", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    # --- Bear divergence on the last two pivot highs ---
    if len(piv_highs) >= 2:
        a_idx = piv_highs.index[-2]
        b_idx = piv_highs.index[-1]
        p_a = float(d.loc[a_idx, "high"])
        p_b = float(d.loc[b_idx, "high"])
        r_a = float(r5.loc[a_idx])
        r_b = float(r5.loc[b_idx])

        price_ok = (p_b > p_a) and ((p_b - p_a) >= float(min_price_delta_atr) * atr_scale)
        rsi_ok = (r_b < r_a) and ((r_a - r_b) >= float(min_rsi_delta))
        slow_ok = True
        if r14 is not None and not r14.isna().all():
            try:
                s_a = float(r14.loc[a_idx])
                s_b = float(r14.loc[b_idx])
                slow_ok = (s_b <= s_a + 1.0)
            except Exception:
                slow_ok = True

        if price_ok and rsi_ok and slow_ok:
            strength = float((r_a - r_b) / max(1.0, min_rsi_delta)) + float((p_b - p_a) / max(1e-9, atr_scale))
            return {"type": "bear", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    return None





def _is_deep_zone_touch(
    zone_low: float,
    zone_high: float,
    candle_low: float,
    candle_high: float,
    atr_v: float,
    zone_type: str,
) -> bool:
    """Count a later touch only if price penetrates meaningfully into the zone."""
    try:
        zl = float(zone_low); zh = float(zone_high); lo = float(candle_low); hi = float(candle_high); atr = float(atr_v)
    except Exception:
        return False
    zone_width = max(0.0, zh - zl)
    deep_thresh = max(0.30 * zone_width, 0.08 * max(atr, 1e-9))
    zt = str(zone_type).upper()
    if zt == "BULLISH_DEMAND":
        return bool(lo <= (zh - deep_thresh))
    if zt == "BEARISH_SUPPLY":
        return bool(hi >= (zl + deep_thresh))
    return False

def _evaluate_entry_zone_context(
    df: pd.DataFrame,
    *,
    entry_price: float | None,
    direction: str,
    atr_last: float | None,
    lookback: int = 10,
) -> dict:
    """Assess simple recent demand/supply zone context around a proposed entry.

    Uses only existing OHLCV data already fetched by the engine. A candle must show
    meaningful rejection *and* enough displacement/volume to qualify as a zone.
    Returns a small, local context signal that can be used as a score tilt without
    introducing new data dependencies or engine gridlock.
    """
    out = {
        "favorable": False,
        "hostile": False,
        "favorable_type": None,
        "hostile_type": None,
        "favorable_dist": None,
        "hostile_dist": None,
        "favorable_inside": False,
        "hostile_inside": False,
        "zone_score_adj": 0,
        "zone_ref_price": None,
        "zone_quality": None,
        "favorable_quality": None,
        "hostile_quality": None,
    }
    try:
        if entry_price is None or not np.isfinite(float(entry_price)):
            return out
        if df is None or len(df) < 6:
            return out
        ep = float(entry_price)
        atr_v = float(atr_last) if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0 else max(1e-6, 0.005 * max(ep, 1.0))
        prox = max(0.18 * atr_v, 0.001 * max(ep, 1.0))

        sub = df.tail(int(max(6, lookback)) + 2).copy()
        vol = sub["volume"].astype(float) if "volume" in sub.columns else None
        vol_sma = vol.rolling(5, min_periods=3).mean() if vol is not None else None
        start = 0
        end = len(sub) - 1
        bull_zones = []
        bear_zones = []
        for i in range(start, end):
            o = float(sub["open"].iloc[i]); h = float(sub["high"].iloc[i]); l = float(sub["low"].iloc[i]); c = float(sub["close"].iloc[i])
            rng = max(1e-9, h - l)
            body = abs(c - o)
            upper = h - max(o, c)
            lower = min(o, c) - l
            close_pos = (c - l) / rng
            next_df = sub.iloc[i + 1 : min(len(sub), i + 3)]
            next_closes = next_df["close"].astype(float) if len(next_df) else pd.Series(dtype=float)
            next_lows = next_df["low"].astype(float) if len(next_df) else pd.Series(dtype=float)
            next_highs = next_df["high"].astype(float) if len(next_df) else pd.Series(dtype=float)

            disp_ok = bool(rng >= 0.60 * atr_v)
            vol_ok = True
            if vol is not None and vol_sma is not None and i < len(vol_sma):
                v = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else np.nan
                vs = float(vol_sma.iloc[i]) if np.isfinite(vol_sma.iloc[i]) else np.nan
                if np.isfinite(v) and np.isfinite(vs) and vs > 0:
                    vol_ok = bool(v >= 1.20 * vs)

            bull_shape = bool(
                lower >= max(1.5 * body, 0.35 * rng)
                and close_pos >= 0.45
            )
            bear_shape = bool(
                upper >= max(1.5 * body, 0.35 * rng)
                and close_pos <= 0.55
            )

            bull_cand = bool(
                bull_shape
                and disp_ok
                and vol_ok
                and len(next_closes) > 0
                and float(next_closes.max()) >= (max(o, c) - 0.05 * rng)
                and float(next_lows.min()) >= (l - 0.08 * atr_v)
            )
            bear_cand = bool(
                bear_shape
                and disp_ok
                and vol_ok
                and len(next_closes) > 0
                and float(next_closes.min()) <= (min(o, c) + 0.05 * rng)
                and float(next_highs.max()) <= (h + 0.08 * atr_v)
            )

            if bull_cand:
                zone_low = l
                zone_high = max(o, c)
                dist = 0.0 if zone_low - prox <= ep <= zone_high + prox else min(abs(ep - zone_low), abs(ep - zone_high))
                disp_move = float(next_closes.max() - c) if len(next_closes) > 0 else 0.0
                disp_score = float(min(1.0, max(0.0, disp_move) / max(1e-9, 1.5 * atr_v)))
                v = float(vol.iloc[i]) if (vol is not None and i < len(vol) and np.isfinite(vol.iloc[i])) else np.nan
                vs = float(vol_sma.iloc[i]) if (vol_sma is not None and i < len(vol_sma) and np.isfinite(vol_sma.iloc[i])) else np.nan
                vol_score = float(min(1.0, (v / vs) / 1.5)) if np.isfinite(v) and np.isfinite(vs) and vs > 0 else 0.5
                later_df = sub.iloc[i + 1 :]
                touch_count = int(sum(
                    1 for _, rr in later_df.iterrows()
                    if _is_deep_zone_touch(
                        zone_low,
                        zone_high,
                        float(rr["low"]),
                        float(rr["high"]),
                        atr_v,
                        "BULLISH_DEMAND",
                    )
                )) if len(later_df) else 0
                fresh_score = float(max(0.0, 1.0 - 0.25 * max(0, touch_count - 1)))
                zone_width = max(1e-9, zone_high - zone_low)
                precision_score = float(max(0.0, min(1.0, 1.0 - (zone_width / max(1e-9, 0.80 * atr_v)))))
                hold_score = float(max(0.0, min(1.0, (float(next_closes.iloc[-1]) - zone_high) / max(1e-9, 0.90 * atr_v)))) if len(next_closes) > 0 else 0.0
                reaction_score = float(max(0.0, min(1.0, 0.55 * disp_score + 0.45 * hold_score)))
                zone_quality = float(max(0.0, min(1.0, 0.30 * disp_score + 0.20 * vol_score + 0.20 * fresh_score + 0.20 * reaction_score + 0.10 * precision_score)))
                inside = bool(zone_low <= ep <= zone_high)
                bull_zones.append({"type": "BULLISH_DEMAND", "dist": float(dist), "ref": float(zone_high), "i": i, "quality": zone_quality, "inside": inside})
            if bear_cand:
                zone_low = min(o, c)
                zone_high = h
                dist = 0.0 if zone_low - prox <= ep <= zone_high + prox else min(abs(ep - zone_low), abs(ep - zone_high))
                disp_move = float(c - next_closes.min()) if len(next_closes) > 0 else 0.0
                disp_score = float(min(1.0, max(0.0, disp_move) / max(1e-9, 1.5 * atr_v)))
                v = float(vol.iloc[i]) if (vol is not None and i < len(vol) and np.isfinite(vol.iloc[i])) else np.nan
                vs = float(vol_sma.iloc[i]) if (vol_sma is not None and i < len(vol_sma) and np.isfinite(vol_sma.iloc[i])) else np.nan
                vol_score = float(min(1.0, (v / vs) / 1.5)) if np.isfinite(v) and np.isfinite(vs) and vs > 0 else 0.5
                later_df = sub.iloc[i + 1 :]
                touch_count = int(sum(
                    1 for _, rr in later_df.iterrows()
                    if _is_deep_zone_touch(
                        zone_low,
                        zone_high,
                        float(rr["low"]),
                        float(rr["high"]),
                        atr_v,
                        "BEARISH_SUPPLY",
                    )
                )) if len(later_df) else 0
                fresh_score = float(max(0.0, 1.0 - 0.25 * max(0, touch_count - 1)))
                zone_width = max(1e-9, zone_high - zone_low)
                precision_score = float(max(0.0, min(1.0, 1.0 - (zone_width / max(1e-9, 0.80 * atr_v)))))
                hold_score = float(max(0.0, min(1.0, (zone_low - float(next_closes.iloc[-1])) / max(1e-9, 0.90 * atr_v)))) if len(next_closes) > 0 else 0.0
                reaction_score = float(max(0.0, min(1.0, 0.55 * disp_score + 0.45 * hold_score)))
                zone_quality = float(max(0.0, min(1.0, 0.30 * disp_score + 0.20 * vol_score + 0.20 * fresh_score + 0.20 * reaction_score + 0.10 * precision_score)))
                inside = bool(zone_low <= ep <= zone_high)
                bear_zones.append({"type": "BEARISH_SUPPLY", "dist": float(dist), "ref": float(zone_low), "i": i, "quality": zone_quality, "inside": inside})

        best_bull = min(bull_zones, key=lambda z: (z["dist"], -z["i"])) if bull_zones else None
        best_bear = min(bear_zones, key=lambda z: (z["dist"], -z["i"])) if bear_zones else None

        if str(direction).upper() == "LONG":
            if best_bull and best_bull["dist"] <= prox:
                out["favorable"] = True
                out["favorable_type"] = best_bull["type"]
                out["favorable_dist"] = float(best_bull["dist"])
                out["favorable_quality"] = float(best_bull.get("quality", 0.0))
                out["favorable_inside"] = bool(best_bull.get("inside", False))
                out["zone_ref_price"] = float(best_bull["ref"])
                out["zone_quality"] = float(best_bull.get("quality", 0.0))
            if best_bear and best_bear["dist"] <= prox:
                out["hostile"] = True
                out["hostile_type"] = best_bear["type"]
                out["hostile_dist"] = float(best_bear["dist"])
                out["hostile_quality"] = float(best_bear.get("quality", 0.0))
                out["hostile_inside"] = bool(best_bear.get("inside", False))
                out["zone_ref_price"] = float(best_bear["ref"])
                out["zone_quality"] = float(best_bear.get("quality", 0.0))
        else:
            if best_bear and best_bear["dist"] <= prox:
                out["favorable"] = True
                out["favorable_type"] = best_bear["type"]
                out["favorable_dist"] = float(best_bear["dist"])
                out["favorable_quality"] = float(best_bear.get("quality", 0.0))
                out["favorable_inside"] = bool(best_bear.get("inside", False))
                out["zone_ref_price"] = float(best_bear["ref"])
                out["zone_quality"] = float(best_bear.get("quality", 0.0))
            if best_bull and best_bull["dist"] <= prox:
                out["hostile"] = True
                out["hostile_type"] = best_bull["type"]
                out["hostile_dist"] = float(best_bull["dist"])
                out["hostile_quality"] = float(best_bull.get("quality", 0.0))
                out["hostile_inside"] = bool(best_bull.get("inside", False))
                out["zone_ref_price"] = float(best_bull["ref"])
                out["zone_quality"] = float(best_bull.get("quality", 0.0))
    except Exception:
        return out
    return out

def _compute_atr_pct_series(df: pd.DataFrame, period: int = 14):
    if df is None or len(df) < period + 2:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / close.replace(0, np.nan)


def _apply_atr_score_normalization(score: float, df: pd.DataFrame, lookback: int = 200, period: int = 14):
    atr_pct = _compute_atr_pct_series(df, period=period)
    if atr_pct is None:
        return score, None, None, 1.0
    cur = atr_pct.iloc[-1]
    if pd.isna(cur) or float(cur) <= 0:
        return score, (None if pd.isna(cur) else float(cur)), None, 1.0
    tail = atr_pct.dropna().tail(int(lookback))
    baseline = float(tail.median()) if len(tail) else None
    if baseline is None or baseline <= 0:
        return score, float(cur), baseline, 1.0
    scale = float(baseline / float(cur))
    scale = max(0.75, min(1.35, scale))
    return max(0.0, min(100.0, float(score) * scale)), float(cur), baseline, scale


def _tape_bonus_from_readiness(
    readiness: float,
    *,
    cap: int = 4,
    thresholds: tuple[float, float, float, float] = (4.0, 5.5, 7.0, 8.0),
) -> int:
    try:
        r = float(readiness)
    except Exception:
        return 0
    t1, t2, t3, t4 = [float(x) for x in thresholds]
    if r >= t4:
        return int(min(cap, 4))
    if r >= t3:
        return int(min(cap, 3))
    if r >= t2:
        return int(min(cap, 2))
    if r >= t1:
        return int(min(cap, 1))
    return 0


def _compute_tape_readiness(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    release_level: float | None,
    structural_level: float | None = None,
    trigger_near: bool = False,
    baseline_ok: bool = False,
) -> Dict[str, float | bool | None]:
    """Small, behavior-first tape diagnostic for chaotic $1-$5 tape.

    Uses only already-fetched OHLCV + indicator context. It intentionally avoids
    explicit pattern labels and instead scores the behavior behind useful coils:
      - tightening
      - structural hold
      - directional pressure
      - proximity to a meaningful release area
    """
    out: Dict[str, float | bool | None] = {
        "eligible": False,
        "tightening": 0.0,
        "structural_hold": 0.0,
        "pressure": 0.0,
        "release_proximity": 0.0,
        "readiness": 0.0,
        "recent_range_ratio": None,
        "recent_body_ratio": None,
        "release_dist_atr": None,
        "prior_impulse_span_atr": None,
        "prior_impulse_push_atr": None,
        "impulse_floor_ok": False,
        "macd_directional_build": False,
    }
    try:
        if df is None or len(df) < 12 or (not baseline_ok):
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0:
            return out
        release = float(release_level) if release_level is not None and np.isfinite(release_level) else float("nan")
        if not np.isfinite(release):
            return out
        structural = float(structural_level) if structural_level is not None and np.isfinite(structural_level) else release
        direction = str(direction or "").upper().strip()

        recent = df.tail(5).copy()
        prior = df.iloc[-10:-5].copy()
        if len(recent) < 4 or len(prior) < 3:
            return out

        recent_range = pd.to_numeric(recent["high"] - recent["low"], errors="coerce").dropna()
        prior_range = pd.to_numeric(prior["high"] - prior["low"], errors="coerce").dropna()
        recent_body = pd.to_numeric((recent["close"] - recent["open"]).abs(), errors="coerce").dropna()
        prior_body = pd.to_numeric((prior["close"] - prior["open"]).abs(), errors="coerce").dropna()
        if len(recent_range) < 3 or len(prior_range) < 3:
            return out

        rr_ratio = float(recent_range.mean() / max(1e-9, float(prior_range.mean())))
        rb_ratio = float(recent_body.mean() / max(1e-9, float(prior_body.mean()))) if len(prior_body) else float("nan")
        out["recent_range_ratio"] = rr_ratio
        out["recent_body_ratio"] = rb_ratio if np.isfinite(rb_ratio) else None

        prior_highs = pd.to_numeric(prior["high"], errors="coerce").dropna()
        prior_lows = pd.to_numeric(prior["low"], errors="coerce").dropna()
        prior_opens = pd.to_numeric(prior["open"], errors="coerce").dropna()
        prior_closes = pd.to_numeric(prior["close"], errors="coerce").dropna()
        if not len(prior_highs) or not len(prior_lows) or not len(prior_opens) or not len(prior_closes):
            return out
        prior_span_atr = float((prior_highs.max() - prior_lows.min()) / max(1e-9, atr_val))
        if direction == "LONG":
            prior_push_atr = float((prior_closes.iloc[-1] - prior_opens.iloc[0]) / max(1e-9, atr_val))
            impulse_floor_ok = bool(prior_span_atr >= 1.10 and prior_push_atr >= 0.35)
        else:
            prior_push_atr = float((prior_opens.iloc[0] - prior_closes.iloc[-1]) / max(1e-9, atr_val))
            impulse_floor_ok = bool(prior_span_atr >= 1.10 and prior_push_atr >= 0.35)
        out["prior_impulse_span_atr"] = float(prior_span_atr)
        out["prior_impulse_push_atr"] = float(prior_push_atr)
        out["impulse_floor_ok"] = bool(impulse_floor_ok)
        if not impulse_floor_ok:
            return out

        tightening = 0.0
        if rr_ratio <= 0.82:
            tightening += 1.0
        if np.isfinite(rb_ratio) and rb_ratio <= 0.88:
            tightening += 0.5
        if len(recent_range) >= 3 and float(recent_range.iloc[-3:].mean()) <= float(recent_range.iloc[:2].mean()) * 0.92:
            tightening += 0.5
        tightening = float(np.clip(tightening, 0.0, 2.0))

        lows = pd.to_numeric(recent["low"], errors="coerce")
        highs = pd.to_numeric(recent["high"], errors="coerce")
        closes = pd.to_numeric(recent["close"], errors="coerce")
        macd_tail = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(4), errors="coerce").dropna()
        rsi5_tail = pd.to_numeric(df.get("rsi5", pd.Series(index=df.index, dtype=float)).tail(3), errors="coerce").dropna()
        rsi14_tail = pd.to_numeric(df.get("rsi14", pd.Series(index=df.index, dtype=float)).tail(3), errors="coerce").dropna()

        structural_hold = 0.0
        pressure = 0.0
        macd_directional_build = False
        if direction == "LONG":
            if len(lows.dropna()) >= 4 and int((lows.diff().fillna(0.0) >= (-0.12 * atr_val)).sum()) >= 4:
                structural_hold += 1.0
            if float(closes.min()) >= float(structural) - 0.35 * atr_val:
                structural_hold += 1.0
            if len(macd_tail) >= 3 and bool(macd_tail.iloc[-1] > macd_tail.iloc[-2] > macd_tail.iloc[-3]):
                pressure += 1.0
                macd_directional_build = True
            recent_span = max(1e-9, float(highs.max() - lows.min()))
            if float(closes.mean()) >= float(lows.min()) + 0.58 * recent_span:
                pressure += 0.5
            retrace_span = max(1e-9, float(df["close"].tail(6).max() - df["close"].tail(6).min()))
            if float(df["close"].tail(3).min()) >= float(df["close"].tail(6).max()) - 0.75 * retrace_span:
                pressure += 0.5
            if macd_directional_build and len(rsi5_tail) and len(rsi14_tail):
                if float(rsi5_tail.iloc[-1]) >= 42.0 and float(rsi14_tail.iloc[-1]) >= 40.0:
                    pressure += 0.25
        else:
            if len(highs.dropna()) >= 4 and int((highs.diff().fillna(0.0) <= (0.12 * atr_val)).sum()) >= 4:
                structural_hold += 1.0
            if float(closes.max()) <= float(structural) + 0.35 * atr_val:
                structural_hold += 1.0
            if len(macd_tail) >= 3 and bool(macd_tail.iloc[-1] < macd_tail.iloc[-2] < macd_tail.iloc[-3]):
                pressure += 1.0
                macd_directional_build = True
            recent_span = max(1e-9, float(highs.max() - lows.min()))
            if float(closes.mean()) <= float(highs.max()) - 0.58 * recent_span:
                pressure += 0.5
            retrace_span = max(1e-9, float(df["close"].tail(6).max() - df["close"].tail(6).min()))
            if float(df["close"].tail(3).max()) <= float(df["close"].tail(6).min()) + 0.75 * retrace_span:
                pressure += 0.5
            if macd_directional_build and len(rsi5_tail) and len(rsi14_tail):
                if float(rsi5_tail.iloc[-1]) <= 58.0 and float(rsi14_tail.iloc[-1]) <= 60.0:
                    pressure += 0.25
        structural_hold = float(np.clip(structural_hold, 0.0, 2.0))
        pressure = float(np.clip(pressure, 0.0, 2.0))
        out["macd_directional_build"] = bool(macd_directional_build)

        last_close = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
        release_dist_atr = abs(last_close - release) / max(1e-9, atr_val)
        out["release_dist_atr"] = float(release_dist_atr)
        release_prox = 0.0
        if trigger_near or release_dist_atr <= 0.75:
            release_prox += 1.0
        if release_dist_atr <= 0.40:
            release_prox += 1.0
        release_prox = float(np.clip(release_prox, 0.0, 2.0))

        readiness = 0.0
        pressure_floor_ok = bool(pressure >= 1.0)
        high_readiness_ok = bool((pressure >= 1.5) and macd_directional_build)
        if tightening >= 0.5 and structural_hold >= 0.5 and pressure_floor_ok and release_prox >= 0.5:
            readiness = float(np.clip(tightening + structural_hold + pressure + release_prox, 0.0, 8.0))
            if readiness >= 6.0 and not high_readiness_ok:
                readiness = min(readiness, 5.5)

        out.update({
            "eligible": bool(readiness > 0.0),
            "tightening": float(tightening),
            "structural_hold": float(structural_hold),
            "pressure": float(pressure),
            "release_proximity": float(release_prox),
            "readiness": float(readiness),
        })
        return out
    except Exception:
        return out



def _compute_scalp_reversal_stabilization(
    df: pd.DataFrame,
    *,
    direction: str,
    ref_level: float | None,
    atr_last: float | None,
) -> Dict[str, float | bool | None]:
    """Reversal-specific stabilizer for SCALP PRE assistance.

    Lets SCALP benefit from post-flush stabilization / reclaim lean without
    demanding the same continuation-style pressure required by RIDE breakout
    selection.
    """
    out: Dict[str, float | bool | None] = {
        "stabilizing": False,
        "flush_present": False,
        "cluster_improving": False,
        "reclaim_lean": False,
        "bonus": 0.0,
    }
    try:
        if df is None or len(df) < 8:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        ref = float(ref_level) if ref_level is not None and np.isfinite(ref_level) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(ref):
            return out
        recent = df.tail(4).copy()
        prior = df.iloc[-8:-4].copy()
        if len(recent) < 4 or len(prior) < 4:
            return out
        o_r = pd.to_numeric(recent["open"], errors="coerce").dropna()
        h_r = pd.to_numeric(recent["high"], errors="coerce").dropna()
        l_r = pd.to_numeric(recent["low"], errors="coerce").dropna()
        c_r = pd.to_numeric(recent["close"], errors="coerce").dropna()
        o_p = pd.to_numeric(prior["open"], errors="coerce").dropna()
        h_p = pd.to_numeric(prior["high"], errors="coerce").dropna()
        l_p = pd.to_numeric(prior["low"], errors="coerce").dropna()
        c_p = pd.to_numeric(prior["close"], errors="coerce").dropna()
        macd_tail = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(4), errors="coerce").dropna()
        if min(len(o_r), len(h_r), len(l_r), len(c_r), len(o_p), len(h_p), len(l_p), len(c_p)) < 3:
            return out
        direction = str(direction or "").upper().strip()
        if direction == "LONG":
            flush_present = bool(((o_p.iloc[0] - l_p.min()) / max(1e-9, atr_val) >= 0.50) or (c_p.iloc[-1] < o_p.iloc[0] - 0.30 * atr_val))
            lows_stable = bool(l_r.min() >= l_p.min() - 0.10 * atr_val)
            closes_improving = bool(c_r.iloc[-1] >= c_r.iloc[0] - 0.04 * atr_val and c_r.mean() >= l_r.min() + 0.52 * max(1e-9, float(h_r.max() - l_r.min())))
            reclaim_lean = bool(c_r.iloc[-1] >= ref - 0.34 * atr_val or h_r.max() >= ref - 0.15 * atr_val)
            macd_improving = bool(len(macd_tail) >= 3 and macd_tail.iloc[-1] >= macd_tail.iloc[-2] >= macd_tail.iloc[-3])
        else:
            flush_present = bool(((h_p.max() - o_p.iloc[0]) / max(1e-9, atr_val) >= 0.50) or (c_p.iloc[-1] > o_p.iloc[0] + 0.30 * atr_val))
            lows_stable = bool(h_r.max() <= h_p.max() + 0.10 * atr_val)
            closes_improving = bool(c_r.iloc[-1] <= c_r.iloc[0] + 0.04 * atr_val and c_r.mean() <= h_r.max() - 0.52 * max(1e-9, float(h_r.max() - l_r.min())))
            reclaim_lean = bool(c_r.iloc[-1] <= ref + 0.34 * atr_val or l_r.min() <= ref + 0.15 * atr_val)
            macd_improving = bool(len(macd_tail) >= 3 and macd_tail.iloc[-1] <= macd_tail.iloc[-2] <= macd_tail.iloc[-3])
        bonus = 0.0
        if flush_present:
            bonus += 0.30
        if lows_stable:
            bonus += 0.25
        if closes_improving:
            bonus += 0.25
        if reclaim_lean:
            bonus += 0.20
        if macd_improving:
            bonus += 0.15
        bonus = float(min(1.0, bonus))
        out.update({
            "stabilizing": bool(flush_present and lows_stable and closes_improving),
            "flush_present": bool(flush_present),
            "cluster_improving": bool(closes_improving and macd_improving),
            "reclaim_lean": bool(reclaim_lean),
            "bonus": float(bonus),
        })
        return out
    except Exception:
        return out


def _compute_release_rejection_penalty(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    release_level: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {"penalty": 0.0, "stuffing": False, "wick_ratio": 0.0, "close_finish": 0.0}
    try:
        if df is None or len(df) < 3:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        rel = float(release_level) if release_level is not None and np.isfinite(release_level) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(rel):
            return out
        recent = df.tail(3).copy()
        o = pd.to_numeric(recent["open"], errors="coerce")
        h = pd.to_numeric(recent["high"], errors="coerce")
        l = pd.to_numeric(recent["low"], errors="coerce")
        c = pd.to_numeric(recent["close"], errors="coerce")
        if min(o.notna().sum(), h.notna().sum(), l.notna().sum(), c.notna().sum()) < 3:
            return out
        direction = str(direction or "").upper().strip()
        ranges = (h - l).replace(0, np.nan)
        if direction == "LONG":
            wick = (h - np.maximum(o, c)) / ranges
            close_finish = (c - l) / ranges
            near_release = bool(float(h.max()) >= rel - 0.18 * atr_val)
            stuffing = bool(near_release and float(wick.fillna(0).mean()) >= 0.34 and float(close_finish.fillna(0.5).mean()) <= 0.62)
        else:
            wick = (np.minimum(o, c) - l) / ranges
            close_finish = (h - c) / ranges
            near_release = bool(float(l.min()) <= rel + 0.18 * atr_val)
            stuffing = bool(near_release and float(wick.fillna(0).mean()) >= 0.34 and float(close_finish.fillna(0.5).mean()) <= 0.62)
        penalty = 0.0
        if near_release and float(wick.fillna(0).mean()) >= 0.26:
            penalty += 0.5
        if stuffing:
            penalty += 0.5
        out.update({
            "penalty": float(min(1.0, penalty)),
            "stuffing": bool(stuffing),
            "wick_ratio": float(wick.fillna(0).mean()),
            "close_finish": float(close_finish.fillna(0.5).mean()),
        })
        return out
    except Exception:
        return out


def _compute_breakout_urgency(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    release_level: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {"score": 0.0, "urgent": False}
    try:
        if df is None or len(df) < 4:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        rel = float(release_level) if release_level is not None and np.isfinite(release_level) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(rel):
            return out
        direction = str(direction or "").upper().strip()
        recent = df.tail(4).copy()
        c = pd.to_numeric(recent["close"], errors="coerce").dropna()
        h = pd.to_numeric(recent["high"], errors="coerce").dropna()
        l = pd.to_numeric(recent["low"], errors="coerce").dropna()
        macd = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(4), errors="coerce").dropna()
        if min(len(c), len(h), len(l)) < 3:
            return out
        score = 0.0
        if direction == "LONG":
            if len(macd) >= 3 and macd.iloc[-1] > macd.iloc[-2] > macd.iloc[-3]:
                score += 0.5
            recent_span = max(1e-9, float(h.max() - l.min()))
            if float(c.iloc[-2:].mean()) >= float(l.min()) + 0.68 * recent_span:
                score += 0.5
            if float(c.iloc[-1]) >= rel - 0.22 * atr_val:
                score += 0.5
            if float(l.tail(2).min()) >= rel - 0.55 * atr_val:
                score += 0.5
        else:
            if len(macd) >= 3 and macd.iloc[-1] < macd.iloc[-2] < macd.iloc[-3]:
                score += 0.5
            recent_span = max(1e-9, float(h.max() - l.min()))
            if float(c.iloc[-2:].mean()) <= float(h.max()) - 0.68 * recent_span:
                score += 0.5
            if float(c.iloc[-1]) <= rel + 0.22 * atr_val:
                score += 0.5
            if float(h.tail(2).max()) <= rel + 0.55 * atr_val:
                score += 0.5
        score = float(min(2.0, score))
        out.update({"score": score, "urgent": bool(score >= 1.5)})
        return out
    except Exception:
        return out


def _compute_pullback_unlikelihood(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    accept_line: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {"score": 0.0, "unlikely": False}
    try:
        if df is None or len(df) < 6:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept):
            return out
        direction = str(direction or "").upper().strip()
        c6 = pd.to_numeric(df["close"].tail(6), errors="coerce").dropna()
        l4 = pd.to_numeric(df["low"].tail(4), errors="coerce").dropna()
        h4 = pd.to_numeric(df["high"].tail(4), errors="coerce").dropna()
        if min(len(c6), len(l4), len(h4)) < 4:
            return out
        score = 0.0
        if direction == "LONG":
            retrace_depth = float(c6.max() - l4.min()) / max(1e-9, atr_val)
            if retrace_depth <= 0.65:
                score += 0.75
            if float(l4.min()) >= accept - 0.18 * atr_val:
                score += 0.75
            if float(c6.tail(3).mean()) >= float(c6.max()) - 0.28 * atr_val:
                score += 0.5
        else:
            retrace_depth = float(h4.max() - c6.min()) / max(1e-9, atr_val)
            if retrace_depth <= 0.65:
                score += 0.75
            if float(h4.max()) <= accept + 0.18 * atr_val:
                score += 0.75
            if float(c6.tail(3).mean()) <= float(c6.min()) + 0.28 * atr_val:
                score += 0.5
        score = float(min(2.0, score))
        out.update({"score": score, "unlikely": bool(score >= 1.25)})
        return out
    except Exception:
        return out

def _compute_breakout_acceptance_quality(
    df: pd.DataFrame,
    *,
    direction: str,
    breakout_ref: float | None,
    atr_last: float | None,
    buffer: float = 0.0,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {
        "accepted": False,
        "clean_accept": False,
        "rejection": False,
        "touch": False,
        "wick_ratio": 0.0,
        "close_finish": 0.5,
        "last_close_vs_ref": 0.0,
    }
    try:
        if df is None or len(df) < 2:
            return out
        direction = str(direction or "").upper().strip()
        ref = float(breakout_ref) if breakout_ref is not None and np.isfinite(breakout_ref) else float("nan")
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        if not np.isfinite(ref) or not np.isfinite(atr_val) or atr_val <= 0:
            return out
        recent = df.tail(2).copy()
        o = pd.to_numeric(recent["open"], errors="coerce")
        h = pd.to_numeric(recent["high"], errors="coerce")
        l = pd.to_numeric(recent["low"], errors="coerce")
        c = pd.to_numeric(recent["close"], errors="coerce")
        if min(o.notna().sum(), h.notna().sum(), l.notna().sum(), c.notna().sum()) < 2:
            return out
        ranges = (h - l).replace(0, np.nan)
        buf = float(buffer or 0.0)
        if direction == "LONG":
            wick = (h - np.maximum(o, c)) / ranges
            close_finish = (c - l) / ranges
            touch = bool((h >= ref - buf).any())
            last_close_ok = bool(float(c.iloc[-1]) >= ref - max(buf, 0.03 * atr_val))
            avg_close_ok = bool(float(c.mean()) >= ref - 0.01 * atr_val)
            clean_accept = bool(float(c.iloc[-1]) >= ref + 0.02 * atr_val and float(close_finish.iloc[-1]) >= 0.58)
            rejection = bool(
                touch and (
                    (float(wick.fillna(0).iloc[-1]) >= 0.33 and float(c.iloc[-1]) < ref + 0.02 * atr_val)
                    or (float(h.iloc[-1]) >= ref + 0.08 * atr_val and float(c.iloc[-1]) <= ref - 0.02 * atr_val)
                    or (float(wick.fillna(0).mean()) >= 0.30 and float(close_finish.fillna(0.5).mean()) <= 0.58 and float(c.mean()) < ref + 0.01 * atr_val)
                )
            )
            accepted = bool(touch and last_close_ok and avg_close_ok and not rejection)
            last_close_vs_ref = float((float(c.iloc[-1]) - ref) / atr_val)
        else:
            wick = (np.minimum(o, c) - l) / ranges
            close_finish = (h - c) / ranges
            touch = bool((l <= ref + buf).any())
            last_close_ok = bool(float(c.iloc[-1]) <= ref + max(buf, 0.03 * atr_val))
            avg_close_ok = bool(float(c.mean()) <= ref + 0.01 * atr_val)
            clean_accept = bool(float(c.iloc[-1]) <= ref - 0.02 * atr_val and float(close_finish.iloc[-1]) >= 0.58)
            rejection = bool(
                touch and (
                    (float(wick.fillna(0).iloc[-1]) >= 0.33 and float(c.iloc[-1]) > ref - 0.02 * atr_val)
                    or (float(l.iloc[-1]) <= ref - 0.08 * atr_val and float(c.iloc[-1]) >= ref + 0.02 * atr_val)
                    or (float(wick.fillna(0).mean()) >= 0.30 and float(close_finish.fillna(0.5).mean()) <= 0.58 and float(c.mean()) > ref - 0.01 * atr_val)
                )
            )
            accepted = bool(touch and last_close_ok and avg_close_ok and not rejection)
            last_close_vs_ref = float((ref - float(c.iloc[-1])) / atr_val)
        out.update({
            "accepted": bool(accepted),
            "clean_accept": bool(clean_accept and accepted),
            "rejection": bool(rejection),
            "touch": bool(touch),
            "wick_ratio": float(wick.fillna(0).mean()),
            "close_finish": float(close_finish.fillna(0.5).mean()),
            "last_close_vs_ref": float(last_close_vs_ref),
        })
        return out
    except Exception:
        return out


def _compute_breakout_extension_state(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    accept_line: float | None,
    ref_vwap: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {
        "penalty": 0.0,
        "extended": False,
        "exhausted": False,
        "dist_accept_atr": 0.0,
        "dist_vwap_atr": 0.0,
        "momentum_fade": False,
        "stalling": False,
    }
    try:
        if df is None or len(df) < 6:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float("nan")
        vwap = float(ref_vwap) if ref_vwap is not None and np.isfinite(ref_vwap) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept):
            return out
        direction = str(direction or "").upper().strip()
        recent = df.tail(6).copy()
        c = pd.to_numeric(recent["close"], errors="coerce").dropna()
        h = pd.to_numeric(recent["high"], errors="coerce").dropna()
        l = pd.to_numeric(recent["low"], errors="coerce").dropna()
        macd = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(5), errors="coerce").dropna()
        if min(len(c), len(h), len(l)) < 5:
            return out

        if direction == "LONG":
            last_px = float(c.iloc[-1])
            dist_accept_atr = max(0.0, (last_px - accept) / atr_val)
            dist_vwap_atr = max(0.0, (last_px - vwap) / atr_val) if np.isfinite(vwap) else max(0.0, dist_accept_atr)
            recent_high = float(h.max())
            prev_high = float(h.iloc[:-2].max()) if len(h) > 2 else recent_high
            close_slip = bool(last_px <= recent_high - 0.35 * atr_val)
            stall = bool((recent_high - prev_high) <= 0.12 * atr_val and close_slip)
            fade = bool(len(macd) >= 3 and (float(macd.iloc[-1]) < float(macd.iloc[-2]) <= float(macd.iloc[-3])))
            extended = bool(dist_accept_atr >= 0.80 or dist_vwap_atr >= 1.20)
            exhausted = bool(extended and (stall or fade) and dist_accept_atr >= 0.70)
        else:
            last_px = float(c.iloc[-1])
            dist_accept_atr = max(0.0, (accept - last_px) / atr_val)
            dist_vwap_atr = max(0.0, (vwap - last_px) / atr_val) if np.isfinite(vwap) else max(0.0, dist_accept_atr)
            recent_low = float(l.min())
            prev_low = float(l.iloc[:-2].min()) if len(l) > 2 else recent_low
            close_slip = bool(last_px >= recent_low + 0.35 * atr_val)
            stall = bool((prev_low - recent_low) <= 0.12 * atr_val and close_slip)
            fade = bool(len(macd) >= 3 and (float(macd.iloc[-1]) > float(macd.iloc[-2]) >= float(macd.iloc[-3])))
            extended = bool(dist_accept_atr >= 0.80 or dist_vwap_atr >= 1.20)
            exhausted = bool(extended and (stall or fade) and dist_accept_atr >= 0.70)

        penalty = 0.0
        if extended:
            penalty += 0.5
        if max(dist_accept_atr, dist_vwap_atr) >= 1.55:
            penalty += 0.5
        if stall:
            penalty += 0.5
        if fade:
            penalty += 0.5
        if exhausted:
            penalty += 0.25
        out.update({
            "penalty": float(min(1.5, penalty)),
            "extended": bool(extended),
            "exhausted": bool(exhausted),
            "dist_accept_atr": float(dist_accept_atr),
            "dist_vwap_atr": float(dist_vwap_atr),
            "momentum_fade": bool(fade),
            "stalling": bool(stall),
        })
        return out
    except Exception:
        return out




def _anchor_recent_interaction_score(
    df: pd.DataFrame,
    *,
    direction: str,
    anchor: float | None,
    atr_last: float | None,
    lookback: int = 8,
) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {"score": 0.0, "touches": 0, "defended": 0, "wick_quality": 0.0, "close_quality": 0.0}
    try:
        if df is None or len(df) < 3:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        anch = float(anchor) if anchor is not None and np.isfinite(anchor) else float("nan")
        if not np.isfinite(anch) or not np.isfinite(atr_val) or atr_val <= 0:
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(int(max(3, min(lookback, len(df))))).copy()
        h = pd.to_numeric(recent['high'], errors='coerce')
        l = pd.to_numeric(recent['low'], errors='coerce')
        c = pd.to_numeric(recent['close'], errors='coerce')
        o = pd.to_numeric(recent['open'], errors='coerce')
        band = max(0.08 * atr_val, 1e-9)
        touches = 0
        defended = 0
        wick_scores = []
        close_scores = []
        for hh, ll, cc, oo in zip(h.tolist(), l.tolist(), c.tolist(), o.tolist()):
            if not all(np.isfinite(v) for v in [hh, ll, cc, oo]):
                continue
            touched = bool((ll <= anch + band) and (hh >= anch - band))
            if not touched:
                continue
            touches += 1
            rng = max(1e-9, hh - ll)
            if direction == 'LONG':
                wick_pen = max(0.0, anch - ll) / max(1e-9, 0.28 * atr_val)
                wick_q = float(np.clip(1.0 - wick_pen, 0.0, 1.0))
                close_q = float(np.clip((cc - anch) / max(1e-9, 0.30 * atr_val), -1.0, 1.0))
                defended_bar = bool(cc >= anch - 0.06 * atr_val and cc >= oo - 0.10 * rng)
            else:
                wick_pen = max(0.0, hh - anch) / max(1e-9, 0.28 * atr_val)
                wick_q = float(np.clip(1.0 - wick_pen, 0.0, 1.0))
                close_q = float(np.clip((anch - cc) / max(1e-9, 0.30 * atr_val), -1.0, 1.0))
                defended_bar = bool(cc <= anch + 0.06 * atr_val and cc <= oo + 0.10 * rng)
            wick_scores.append(wick_q)
            close_scores.append(max(0.0, close_q))
            if defended_bar:
                defended += 1
        if touches <= 0:
            return out
        wick_quality = float(np.mean(wick_scores)) if wick_scores else 0.0
        close_quality = float(np.mean(close_scores)) if close_scores else 0.0
        score = float(min(1.0, 0.28 * min(touches, 3) + 0.32 * min(defended, 3) / 3.0 + 0.20 * wick_quality + 0.20 * close_quality))
        out.update({
            'score': score,
            'touches': int(touches),
            'defended': int(defended),
            'wick_quality': float(wick_quality),
            'close_quality': float(close_quality),
        })
        return out
    except Exception:
        return out


def _compute_multibar_extension_profile(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    accept_line: float | None,
) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {"penalty": 0.0, "extended": False, "stalling": False, "fading": False, "path_stretched": False}
    try:
        if df is None or len(df) < 5:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float('nan')
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept):
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(5).copy()
        c = pd.to_numeric(recent['close'], errors='coerce').dropna()
        o = pd.to_numeric(recent['open'], errors='coerce').dropna()
        h = pd.to_numeric(recent['high'], errors='coerce').dropna()
        l = pd.to_numeric(recent['low'], errors='coerce').dropna()
        v = pd.to_numeric(recent.get('volume', pd.Series(index=recent.index, dtype=float)), errors='coerce').dropna()
        if min(len(c), len(o), len(h), len(l)) < 4:
            return out
        if direction == 'LONG':
            dists = np.maximum(0.0, (c.astype(float).values - accept) / atr_val)
            progresses = np.diff(c.astype(float).values)
            body_sign = (c.astype(float).values - o.astype(float).values)
            wick_reject = ((h.astype(float).values - np.maximum(c.astype(float).values, o.astype(float).values)) / np.maximum(1e-9, (h.astype(float).values - l.astype(float).values)))
        else:
            dists = np.maximum(0.0, (accept - c.astype(float).values) / atr_val)
            progresses = np.diff(-c.astype(float).values)
            body_sign = (o.astype(float).values - c.astype(float).values)
            wick_reject = ((np.minimum(c.astype(float).values, o.astype(float).values) - l.astype(float).values) / np.maximum(1e-9, (h.astype(float).values - l.astype(float).values)))
        avg_dist = float(np.mean(dists[-3:])) if len(dists) >= 3 else float(np.mean(dists))
        path_stretched = bool(avg_dist >= 0.82 or (len(dists) >= 2 and float(np.max(dists[-2:])) >= 1.05))
        prog_tail = progresses[-3:] if len(progresses) >= 3 else progresses
        stalling = bool(len(prog_tail) >= 2 and np.isfinite(prog_tail).all() and float(np.mean(prog_tail)) <= 0.06 * atr_val)
        fading = bool(len(body_sign) >= 3 and float(np.mean(body_sign[-2:])) <= float(np.mean(body_sign[:2])) * 0.70)
        if len(v) >= 4:
            vol_tail = v.astype(float).values
            fading = bool(fading or (np.mean(vol_tail[-2:]) <= 0.82 * np.mean(vol_tail[:2]) and avg_dist >= 0.70))
        rejection_rising = bool(len(wick_reject) >= 3 and float(np.mean(wick_reject[-2:])) >= max(0.38, float(np.mean(wick_reject[:2])) + 0.08))
        penalty = 0.0
        if path_stretched:
            penalty += 0.35
        if stalling:
            penalty += 0.30
        if fading:
            penalty += 0.25
        if rejection_rising:
            penalty += 0.25
        out.update({
            'penalty': float(min(1.2, penalty)),
            'extended': bool(avg_dist >= 0.70),
            'stalling': bool(stalling or rejection_rising),
            'fading': bool(fading),
            'path_stretched': bool(path_stretched),
        })
        return out
    except Exception:
        return out


def _assess_scalp_weak_tape_turn(
    df: pd.DataFrame,
    *,
    direction: str,
    trigger_line: float | None,
    atr_last: float | None,
) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {"score": 0.0, "ok": False, "stall": False, "rejection": False}
    try:
        if df is None or len(df) < 4:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        trigger = float(trigger_line) if trigger_line is not None and np.isfinite(trigger_line) else float('nan')
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(trigger):
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(4).copy()
        c = pd.to_numeric(recent['close'], errors='coerce').dropna()
        o = pd.to_numeric(recent['open'], errors='coerce').dropna()
        h = pd.to_numeric(recent['high'], errors='coerce').dropna()
        l = pd.to_numeric(recent['low'], errors='coerce').dropna()
        if min(len(c), len(o), len(h), len(l)) < 4:
            return out
        ranges = np.maximum(1e-9, (h.values - l.values))
        if direction == 'LONG':
            lower_wicks = (np.minimum(c.values, o.values) - l.values) / ranges
            close_progress = np.diff(c.values)
            stall = bool(np.mean(close_progress[-2:]) >= -0.03 * atr_val)
            reclaim_fail = int(np.sum((h.values >= trigger - 0.03 * atr_val) & (c.values <= trigger - 0.04 * atr_val)))
            rejection = bool(reclaim_fail >= 2)
            score = float(0.50 * stall + 0.30 * (np.mean(lower_wicks[-2:]) >= 0.28) + 0.20 * (c.values[-1] >= c.values[-2] - 0.02 * atr_val))
        else:
            upper_wicks = (h.values - np.maximum(c.values, o.values)) / ranges
            close_progress = np.diff(-c.values)
            stall = bool(np.mean(close_progress[-2:]) >= -0.03 * atr_val)
            reclaim_fail = int(np.sum((l.values <= trigger + 0.03 * atr_val) & (c.values >= trigger + 0.04 * atr_val)))
            rejection = bool(reclaim_fail >= 2)
            score = float(0.50 * stall + 0.30 * (np.mean(upper_wicks[-2:]) >= 0.28) + 0.20 * (c.values[-1] <= c.values[-2] + 0.02 * atr_val))
        out.update({'score': float(np.clip(score, 0.0, 1.0)), 'ok': bool(score >= 0.55 and not rejection), 'stall': bool(stall), 'rejection': bool(rejection)})
        return out
    except Exception:
        return out


def _classify_ride_structure_phase(
    *,
    direction: str,
    df: pd.DataFrame,
    accept_line: float | None,
    break_trigger: float | None,
    atr_last: float | None,
) -> str:
    try:
        if df is None or len(df) < 4:
            return 'UNSET'
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float('nan')
        br = float(break_trigger) if break_trigger is not None and np.isfinite(break_trigger) else float('nan')
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept) or not np.isfinite(br):
            return 'UNSET'
        direction = str(direction or '').upper().strip()
        recent = df.tail(4).copy()
        c = pd.to_numeric(recent['close'], errors='coerce').dropna()
        l = pd.to_numeric(recent['low'], errors='coerce').dropna()
        h = pd.to_numeric(recent['high'], errors='coerce').dropna()
        if min(len(c), len(l), len(h)) < 4:
            return 'UNSET'
        if direction == 'LONG':
            compressed = bool(float(h.max()) - float(l.min()) <= 1.15 * atr_val and float(c.iloc[-1]) <= br + 0.22 * atr_val)
            defended = bool(float(l.tail(3).min()) >= accept - 0.16 * atr_val)
            extended = bool(float(c.iloc[-1]) >= br + 0.55 * atr_val)
            failed = bool(extended and float(c.iloc[-1]) <= float(c.max()) - 0.32 * atr_val)
        else:
            compressed = bool(float(h.max()) - float(l.min()) <= 1.15 * atr_val and float(c.iloc[-1]) >= br - 0.22 * atr_val)
            defended = bool(float(h.tail(3).max()) <= accept + 0.16 * atr_val)
            extended = bool(float(c.iloc[-1]) <= br - 0.55 * atr_val)
            failed = bool(extended and float(c.iloc[-1]) >= float(c.min()) + 0.32 * atr_val)
        if compressed and defended:
            return 'BREAK_AND_HOLD'
        if defended and not extended:
            return 'ACCEPT_AND_GO'
        if extended and not failed:
            return 'EXTEND_THEN_PULLBACK'
        if failed:
            return 'FAILED_EXTENSION'
        return 'ACCEPT_AND_GO'
    except Exception:
        return 'UNSET'

def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,

    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",

    # VWAP / Fib / HTF
    lookback_bars: int = 180,
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    session_vwap_include_afterhours: bool = False,
    fib_lookback_bars: int = 120,
    htf_bias: Optional[Dict[str, object]] = None,   # {bias, score, details}
    htf_strict: bool = False,

    # Liquidity / ORB / execution model
    killzone_preset: str = "Custom (use toggles)",
    liquidity_weighting: float = 0.55,
    orb_minutes: int = 15,
    entry_model: str = "VWAP reclaim limit",
    slippage_mode: str = "Off",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,

    # Score normalization
    target_atr_pct: float | None = None,
    tape_mode_enabled: bool = False,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    # --- Interval parsing ---
    # interval is typically like "1min", "5min", "15min", "30min", "60min"
    interval_mins = 1
    try:
        s = str(interval).lower().strip()
        if s.endswith("min"):
            interval_mins = int(float(s.replace("min", "").strip()))
        elif s.endswith("m"):
            interval_mins = int(float(s.replace("m", "").strip()))
        else:
            interval_mins = int(float(s))
    except Exception:
        interval_mins = 1

    # --- Killzone presets ---
    # Presets can optionally override the time-of-day allow toggles.
    kz = (killzone_preset or "Custom (use toggles)").strip()
    if kz == "Opening Drive":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = True, False, False, False, False
    elif kz == "Lunch Chop":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, True, False, False, False
    elif kz == "Power Hour":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, True, False, False
    elif kz == "Pre-market":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, False, True, False

    # --- Snapshot / bar-closed guards ---
    try:
        df_asof = _asof_slice(ohlcv.copy(), interval_mins=interval_mins, use_last_closed_only=use_last_closed_only, bar_closed_guard=bar_closed_guard)
    except Exception:
        df_asof = ohlcv.copy()

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = df_asof.copy().tail(int(lookback_bars)).copy()
    # --- Attach indicator series onto df for downstream helpers that expect columns ---
    # Some callers pass RSI/MACD as separate Series; downstream logic may reference df["rsi5"]/df["rsi14"]/df["macd_hist"].
    # Align by index when possible; otherwise fall back to tail-alignment by length.
    def _attach_series(_df: pd.DataFrame, col: str, s) -> None:
        if s is None:
            return
        try:
            if isinstance(s, pd.Series):
                # Prefer index alignment
                if _df.index.equals(s.index):
                    _df[col] = s
                else:
                    _df[col] = s.reindex(_df.index)
                    # If reindex produced all-NaN (e.g., different tz), tail-align values
                    if _df[col].isna().all() and len(s) >= len(_df):
                        _df[col] = pd.Series(s.values[-len(_df):], index=_df.index)
            else:
                # list/np array
                arr = list(s)
                if len(arr) >= len(_df):
                    _df[col] = pd.Series(arr[-len(_df):], index=_df.index)
        except Exception:
            # Last resort: do nothing
            return

    _attach_series(df, "rsi5", rsi_fast)
    _attach_series(df, "rsi14", rsi_slow)
    _attach_series(df, "macd_hist", macd_hist)
    # Session VWAP windows are session-dependent. If the user enables scanning PM/AH but keeps
    # session VWAP restricted to RTH, VWAP-based logic becomes NaN during those windows.
    # As a product guardrail, automatically extend session VWAP to the scanned session(s).
    auto_vwap_fix = False
    if vwap_logic == "session":
        if allow_premarket and not session_vwap_include_premarket:
            session_vwap_include_premarket = True
            auto_vwap_fix = True
        if allow_afterhours and not session_vwap_include_afterhours:
            session_vwap_include_afterhours = True
            auto_vwap_fix = True

    df["vwap_cum"] = calc_vwap(df)
    df["vwap_sess"] = calc_session_vwap(
        df,
        include_premarket=session_vwap_include_premarket,
        include_afterhours=session_vwap_include_afterhours,
    )
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    # Pro: Trend strength (ADX) + direction (DI+/DI-)
    adx14 = plus_di = minus_di = None
    try:
        adx_s, pdi_s, mdi_s = calc_adx(df, 14)
        df["adx14"] = adx_s
        df["plus_di14"] = pdi_s
        df["minus_di14"] = mdi_s
        adx14 = float(adx_s.iloc[-1]) if len(adx_s) and np.isfinite(adx_s.iloc[-1]) else None
        plus_di = float(pdi_s.iloc[-1]) if len(pdi_s) and np.isfinite(pdi_s.iloc[-1]) else None
        minus_di = float(mdi_s.iloc[-1]) if len(mdi_s) and np.isfinite(mdi_s.iloc[-1]) else None
    except Exception:
        adx14 = plus_di = minus_di = None

    # Keep a stable local alias for weak-tape gating and any upstream helpers
    # that expect the most recent ADX reading by this name.
    adx_last = float(adx14) if isinstance(adx14, (int, float)) and np.isfinite(adx14) else None

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap_use = df["vwap_sess"] if vwap_logic == "session" else df["vwap_cum"]
    df["vwap_use"] = vwap_use  # unify VWAP ref for downstream TP/expected-excursion logic

    last_ts = df.index[-1]
    # Feed freshness diagnostics (ET): this helps catch the "AsOf is yesterday" case.
    try:
        now_et = pd.Timestamp.now(tz="America/New_York")
        ts_et = last_ts.tz_convert("America/New_York") if last_ts.tzinfo is not None else last_ts.tz_localize("America/New_York")
        data_age_min = float((now_et - ts_et).total_seconds() / 60.0)
        extras_feed = {"data_age_min": data_age_min, "data_date": str(ts_et.date())}
    except Exception:
        extras_feed = {"data_age_min": None, "data_date": None}
    session = classify_session(last_ts)
    phase = classify_liquidity_phase(last_ts)

    # IMPORTANT PRODUCT RULE:
    # Time-of-day toggles should NOT *block* scoring/alerts.
    # They are preference hints used for liquidity weighting and optional UI filtering.
    # A great setup is a great setup regardless of clock-time.
    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
        or (session == "PREMARKET" and allow_premarket)
        or (session == "AFTERHOURS" and allow_afterhours)
    )
    last_price = float(close.iloc[-1])

    # --- Safety: define reference VWAP early so it is always in-scope ---
    # The PRE-alert logic and entry/TP models reference `ref_vwap`. In some code paths
    # (depending on toggles/returns), `ref_vwap` can otherwise be referenced before it
    # is assigned, causing UnboundLocalError.
    try:
        _rv = vwap_use.iloc[-1]
        ref_vwap: float | None = float(_rv) if _rv is not None and np.isfinite(_rv) else None
    except Exception:
        ref_vwap = None

    atr_last = float(df["atr14"].iloc[-1]) if np.isfinite(df["atr14"].iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0
    atr_pct = (atr_last / last_price) if last_price else 0.0

    # Liquidity weighting: scale contributions based on the current liquidity phase.
    # liquidity_weighting in [0..1] controls how strongly we care about time-of-day liquidity.
    #  - OPENING / POWER: boost
    #  - MIDDAY: discount
    #  - PREMARKET / AFTERHOURS: heavier discount
    base = 1.0
    if phase in ("OPENING", "POWER"):
        base = 1.15
    elif phase in ("MIDDAY",):
        base = 0.85
    elif phase in ("PREMARKET", "AFTERHOURS"):
        base = 0.75
    try:
        w = max(0.0, min(1.0, float(liquidity_weighting)))
    except Exception:
        w = 0.55
    liquidity_mult = 1.0 + w * (base - 1.0)

    extras: Dict[str, Any] = {
        "vwap_logic": vwap_logic,
        "session_vwap_include_premarket": bool(session_vwap_include_premarket),
        "session_vwap_include_afterhours": bool(session_vwap_include_afterhours),
        "auto_vwap_session_fix": bool(auto_vwap_fix),
        "vwap_session": float(df["vwap_sess"].iloc[-1]) if np.isfinite(df["vwap_sess"].iloc[-1]) else None,
        "vwap_cumulative": float(df["vwap_cum"].iloc[-1]) if np.isfinite(df["vwap_cum"].iloc[-1]) else None,
        "ema20": float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None,
        "ema50": float(df["ema50"].iloc[-1]) if np.isfinite(df["ema50"].iloc[-1]) else None,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "atr14": atr_last,
        "atr_pct": atr_pct,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "liquidity_phase": phase,
        "liquidity_mult": liquidity_mult,
        "fib_lookback_bars": int(fib_lookback_bars),
        "htf_bias": htf_bias,
        "htf_strict": bool(htf_strict),
        "target_atr_pct": (float(target_atr_pct) if target_atr_pct is not None else None),
        # Diagnostics: whether the current session is inside the user's preferred windows.
        # This is NEVER used to block actionability.
        "time_filter_allowed": bool(allowed),
    }

    # Attach feed diagnostics (age/date) to every result.
    try:
        extras.update(extras_feed)
    except Exception:
        pass

    # merge feed freshness fields
    extras.update(extras_feed)

    # Do not early-return when outside preferred windows.
    # We keep scoring normally and simply annotate the result.

    # VWAP event
    was_below_vwap = (close.shift(3) < vwap_use.shift(3)).iloc[-1] or (close.shift(5) < vwap_use.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap_use.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap_use.shift(3)).iloc[-1] or (close.shift(5) > vwap_use.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap_use.shift(1).iloc[-1])

    # RSI + MACD events
    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    # Pro: RSI divergence (RSI-5 vs price pivots)
    rsi_div = None
    if pro_mode:
        try:
            rsi_div = _detect_rsi_divergence(df, rsi_fast, rsi_slow, lookback=int(min(220, max(80, lookback_bars))))
        except Exception:
            rsi_div = None
    extras["rsi_divergence"] = rsi_div

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    def _macd_build_bonus(hist: pd.Series, side: str, structure_ok: bool) -> tuple[int, str | None]:
        """Small capped early-build momentum bonus using existing MACD histogram only.

        This is intentionally subordinate to the existing MACD turn logic. It is only meant
        to help borderline reversal setups surface a little earlier when momentum is clearly
        improving and local structure is already at least minimally credible.
        """
        try:
            h0 = float(hist.iloc[-1]); h1 = float(hist.shift(1).iloc[-1]); h2 = float(hist.shift(2).iloc[-1]); h3 = float(hist.shift(3).iloc[-1])
        except Exception:
            return 0, None
        if not (np.isfinite(h0) and np.isfinite(h1) and np.isfinite(h2) and np.isfinite(h3)):
            return 0, None
        if not bool(structure_ok):
            return 0, None

        if str(side).upper() == "LONG":
            rising3 = bool(h0 > h1 > h2)
            rising4 = bool(rising3 and h2 > h3)
            if h0 > 0 and h1 > 0 and rising3:
                return 6, "MACD build expanding positive"
            if h0 > 0 and h1 <= 0 and h0 > h1:
                return 5, "MACD energy flip positive"
            if h0 <= 0 and rising4:
                return 4, "MACD red bars shrinking"
            if h0 <= 0 and rising3:
                return 3, "MACD build improving"
            return 0, None

        falling3 = bool(h0 < h1 < h2)
        falling4 = bool(falling3 and h2 < h3)
        if h0 < 0 and h1 < 0 and falling3:
            return 6, "MACD build expanding negative"
        if h0 < 0 and h1 >= 0 and h0 < h1:
            return 5, "MACD energy flip negative"
        if h0 >= 0 and falling4:
            return 4, "MACD green bars shrinking"
        if h0 >= 0 and falling3:
            return 3, "MACD build weakening"
        return 0, None

    # Volume confirmation (liquidity weighted)
    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Swings
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    # Trend context (EMA)
    trend_long_ok = bool((close.iloc[-1] >= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] >= df["ema50"].iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] <= df["ema50"].iloc[-1]))
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    # Fib context (scoring + fib-anchored take profits)
    seg = df.tail(int(min(max(60, fib_lookback_bars), len(df))))
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    rng = hi - lo

    fib_name = fib_level = fib_dist = None
    fib_near_long = fib_near_short = False
    fib_bias = "range"
    retr = _fib_retracement_levels(hi, lo) if rng > 0 else []
    fib_name, fib_level, fib_dist = _closest_level(last_price, retr)

    if rng > 0:
        pos = (last_price - lo) / rng
        if pos >= 0.60:
            fib_bias = "up"
        elif pos <= 0.40:
            fib_bias = "down"
        else:
            fib_bias = "range"

    if fib_level is not None and fib_dist is not None:
        # Volatility-aware proximity: tighter when ATR is small, wider when ATR is large.
        # For scalping, we don't want "near fib" firing when price is far away in ATR terms.
        prox = None
        if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
            prox = max(0.35 * float(atr_last), 0.0015 * float(last_price))
        else:
            prox = 0.002 * float(last_price)
        near = float(fib_dist) <= max(float(buffer), float(prox))
        if near:
            if fib_bias == "up":
                fib_near_long = True
            elif fib_bias == "down":
                fib_near_short = True

    extras["fib_hi"] = hi if rng > 0 else None
    extras["fib_lo"] = lo if rng > 0 else None
    extras["fib_bias"] = fib_bias
    extras["fib_closest"] = {"name": fib_name, "level": fib_level, "dist": fib_dist}
    extras["fib_near_long"] = fib_near_long
    extras["fib_near_short"] = fib_near_short

    # Liquidity sweeps + ORB context
    # Use session-aware levels (prior day high/low, premarket high/low, ORB high/low) when possible.
    try:
        levels = _session_liquidity_levels(df, interval_mins=interval_mins, orb_minutes=int(orb_minutes))
    except Exception:
        levels = {}

    extras["liq_levels"] = levels

    # Fallback swing-based levels (always available)
    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())

    # Sweep definition:
    # - Primary: wick through a key level, then close back inside (ICT-style)
    # - Secondary fallback: take + reclaim against recent swing
    bull_sweep = False
    bear_sweep = False
    if pro_mode and levels:
        sweep = _detect_liquidity_sweep(df, levels, atr_last=atr_last, buffer=buffer)
        extras["liquidity_sweep"] = sweep
        if isinstance(sweep, dict) and sweep.get("type"):
            stype = str(sweep.get("type")).lower()
            bull_sweep = stype.startswith("bull")
            bear_sweep = stype.startswith("bear")
    else:
        # Fallback sweeps should still require meaningful displacement so a tiny poke-through
        # does not masquerade as a real liquidity raid.
        fallback_disp_ok = True
        try:
            if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
                fallback_disp_ok = float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 0.8 * float(atr_last)
        except Exception:
            fallback_disp_ok = True
        bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low) and fallback_disp_ok)
        bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high) and fallback_disp_ok)

    extras["bull_liquidity_sweep"] = bool(bull_sweep)
    extras["bear_liquidity_sweep"] = bool(bear_sweep)

    # ORB bias (upgraded): 3-stage sequence (break → accept → retest)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    extras["orb_high"] = orb_high
    extras["orb_low"] = orb_low

    orb_seq = _orb_three_stage(
        df,
        orb_high=float(orb_high) if orb_high is not None else None,
        orb_low=float(orb_low) if orb_low is not None else None,
        buffer=float(buffer),
        lookback_bars=int(max(24, orb_minutes * 3)),  # ~last ~2 hours on 5m, ~6 bars on 1m
        accept_bars=2,
    )
    orb_bull = bool(orb_seq.get("bull_orb_seq"))
    orb_bear = bool(orb_seq.get("bear_orb_seq"))
    # keep break-only flags for diagnostics/UI
    extras["orb_bull_break"] = bool(orb_seq.get("bull_break"))
    extras["orb_bear_break"] = bool(orb_seq.get("bear_break"))
    extras["orb_bull_seq"] = orb_bull
    extras["orb_bear_seq"] = orb_bear


    # FVG + OB + Breaker
    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, df["atr14"], side="bull", lookback=35)
    ob_bear = find_order_block(df, df["atr14"], side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear
    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    brk_bull = find_breaker_block(df, df["atr14"], side="bull", lookback=60)
    brk_bear = find_breaker_block(df, df["atr14"], side="bear", lookback=60)
    extras["bull_breaker"] = brk_bull
    extras["bear_breaker"] = brk_bear
    bull_breaker_retest = bool(brk_bull[0] is not None and in_zone(last_price, brk_bull[0], brk_bull[1], buffer=buffer))
    bear_breaker_retest = bool(brk_bear[0] is not None and in_zone(last_price, brk_bear[0], brk_bear[1], buffer=buffer))
    extras["bull_breaker_retest"] = bull_breaker_retest
    extras["bear_breaker_retest"] = bear_breaker_retest

    displacement = bool(atr_last and float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 1.5 * atr_last)
    extras["displacement"] = displacement

    # HTF bias overlay
    htf_b = None
    if isinstance(htf_bias, dict):
        htf_b = htf_bias.get("bias")
    extras["htf_bias_value"] = htf_b

    # --- Scoring (raw) ---
    contrib: Dict[str, Dict[str, int]] = {"LONG": {}, "SHORT": {}}

    def _add(side: str, key: str, pts: int, why: str | None = None):
        nonlocal long_points, short_points
        if side == "LONG":
            long_points += int(pts)
            contrib["LONG"][key] = contrib["LONG"].get(key, 0) + int(pts)
            if why:
                long_reasons.append(why)
        else:
            short_points += int(pts)
            contrib["SHORT"][key] = contrib["SHORT"].get(key, 0) + int(pts)
            if why:
                short_reasons.append(why)

    micro_hl_pre = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh_pre = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    macd_build_long_pts, macd_build_long_why = _macd_build_bonus(macd_hist, "LONG", micro_hl_pre)
    macd_build_short_pts, macd_build_short_why = _macd_build_bonus(macd_hist, "SHORT", micro_lh_pre)
    extras["macd_build_long_bonus"] = int(macd_build_long_pts)
    extras["macd_build_short_bonus"] = int(macd_build_short_pts)

    long_points = 0
    long_reasons: List[str] = []
    if was_below_vwap and reclaim_vwap:
        _add("LONG", "vwap_event", 35, f"VWAP reclaim ({vwap_logic})")
    if rsi_snap and rsi14 < 60:
        _add("LONG", "rsi_snap", 20, "RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        _add("LONG", "macd_turn", 20, "MACD hist turning up")
    if macd_build_long_pts > 0:
        _add("LONG", "macd_build", int(macd_build_long_pts), macd_build_long_why)
    if vol_ok:
        _add("LONG", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if micro_hl_pre:
        _add("LONG", "micro_structure", 10, "Higher-low micro structure")

    short_points = 0
    short_reasons: List[str] = []
    if was_above_vwap and reject_vwap:
        _add("SHORT", "vwap_event", 35, f"VWAP rejection ({vwap_logic})")
    if rsi_downshift and rsi14 > 40:
        _add("SHORT", "rsi_downshift", 20, "RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        _add("SHORT", "macd_turn", 20, "MACD hist turning down")
    if macd_build_short_pts > 0:
        _add("SHORT", "macd_build", int(macd_build_short_pts), macd_build_short_why)
    if vol_ok:
        _add("SHORT", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if micro_lh_pre:
        _add("SHORT", "micro_structure", 10, "Lower-high micro structure")

    # Fib scoring (volatility-aware, cluster-gated)
    # Fib/FVG should only matter when clustered with structure + volatility context.
    micro_hl = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    long_structure_ok = bool((was_below_vwap and reclaim_vwap) or micro_hl or orb_bull)
    short_structure_ok = bool((was_above_vwap and reject_vwap) or micro_lh or orb_bear)
    vol_context_ok = bool(vol_ok or displacement)

    if fib_near_long and fib_name is not None and long_structure_ok and vol_context_ok:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("LONG", "fib", add, f"Fib cluster ({fib_name})")
    if fib_near_short and fib_name is not None and short_structure_ok and vol_context_ok:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("SHORT", "fib", add, f"Fib cluster ({fib_name})")


    # Pro structure scoring
    if pro_mode:
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bull":
            _add("LONG", "rsi_divergence", 22, "RSI bullish divergence")
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bear":
            _add("SHORT", "rsi_divergence", 22, "RSI bearish divergence")
        if bull_sweep:
            _add("LONG", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (low)")
        if bear_sweep:
            _add("SHORT", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (high)")
        if orb_bull:
            _add("LONG", "orb", int(round(12 * liquidity_mult)), f"ORB seq (break→accept→retest, {orb_minutes}m)")
        if orb_bear:
            _add("SHORT", "orb", int(round(12 * liquidity_mult)), f"ORB seq (break→accept→retest, {orb_minutes}m)")
        if bull_ob_retest:
            _add("LONG", "order_block", 15, "Bullish order block retest")
        if bear_ob_retest:
            _add("SHORT", "order_block", 15, "Bearish order block retest")
                # FVG only matters when price is actually interacting with the gap AND structure/vol context agrees.
        if bull_fvg is not None and isinstance(bull_fvg, (tuple, list)) and len(bull_fvg) == 2:
            z0, z1 = float(min(bull_fvg)), float(max(bull_fvg))
            near_fvg = (last_price >= z0 - buffer) and (last_price <= z1 + buffer)
            if near_fvg and long_structure_ok and vol_context_ok:
                _add("LONG", "fvg", 10, "Bullish FVG cluster")
        if bear_fvg is not None and isinstance(bear_fvg, (tuple, list)) and len(bear_fvg) == 2:
            z0, z1 = float(min(bear_fvg)), float(max(bear_fvg))
            near_fvg = (last_price >= z0 - buffer) and (last_price <= z1 + buffer)
            if near_fvg and short_structure_ok and vol_context_ok:
                _add("SHORT", "fvg", 10, "Bearish FVG cluster")
        if bull_breaker_retest:
            _add("LONG", "breaker", 20, "Bullish breaker retest")
        if bear_breaker_retest:
            _add("SHORT", "breaker", 20, "Bearish breaker retest")
        if displacement:
            _add("LONG", "displacement", 5, None)
            _add("SHORT", "displacement", 5, None)

        # ADX trend-strength bonus (directional): helps avoid low-energy chop.
        # - If ADX is strong and DI agrees with direction => small bonus.
        # - If ADX is very low => mild penalty (but don't over-filter reversal setups).
        try:
            adx_val = float(adx14) if adx14 is not None else None
            pdi_val = float(plus_di) if plus_di is not None else None
            mdi_val = float(minus_di) if minus_di is not None else None
        except Exception:
            adx_val = pdi_val = mdi_val = None

        if adx_val is not None and np.isfinite(adx_val):
            if adx_val >= 20 and pdi_val is not None and mdi_val is not None:
                if pdi_val > mdi_val:
                    _add("LONG", "adx_trend", 8, "ADX trend strength (DI+)")
                elif mdi_val > pdi_val:
                    _add("SHORT", "adx_trend", 8, "ADX trend strength (DI-)")
            elif adx_val <= 15:
                # Penalize both slightly during very low trend strength
                long_points = max(0, long_points - 5)
                short_points = max(0, short_points - 5)
                contrib["LONG"]["adx_chop_penalty"] = contrib["LONG"].get("adx_chop_penalty", 0) - 5
                contrib["SHORT"]["adx_chop_penalty"] = contrib["SHORT"].get("adx_chop_penalty", 0) - 5

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # HTF overlay scoring
    if htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL":
            long_points += 10; long_reasons.append("HTF bias bullish")
            short_points = max(0, short_points - 10)
        elif htf_b == "BEAR":
            short_points += 10; short_reasons.append("HTF bias bearish")
            long_points = max(0, long_points - 10)

    # Requirements / Gatekeeping (product-safe)
    #
    # Product philosophy:
    #   - Score represents *setup quality*.
    #   - Actionability represents *tradeability* (do we have enough confirmation to plan an entry/stop/targets).
    #
    # We do this with a "confirmation score" (count of independent confirmations) and a
    # "soft-hard" volume requirement:
    #   - Volume is still required for alerting *unless* we have strong Pro confluence
    #     (sweep/OB/breaker/ORB + divergence), so we don't miss real money-makers.
    #
    # Confirmation components are boolean (0/1) and deliberately simple:
    #   confirmation_score = vwap + orb + rsi + micro_structure + volume + divergence + liquidity + fib
    #
    # NOTE: Time-of-day filters do NOT block actionability. They only affect liquidity weighting
    # (via liquidity_mult) and UI display.

    vwap_event = bool((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap))
    rsi_event = bool(rsi_snap or rsi_downshift)
    macd_event = bool(macd_turn_up or macd_turn_down)
    volume_event = bool(vol_ok)

    # Micro-structure flags (used for confirmation, not direction)
    micro_hl = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    micro_structure_event = bool(micro_hl or micro_lh)

    is_extended_session = session in ("PREMARKET", "AFTERHOURS")

    # Pro structural trigger (if enabled)
    pro_trigger = False
    divergence_event = False
    if pro_mode:
        divergence_event = bool(isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
        pro_trigger = bool(
            bull_sweep or bear_sweep
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or orb_bull or orb_bear
            or divergence_event
        )
    extras["pro_trigger"] = bool(pro_trigger)

    # Strong Pro confluence: 2+ independent Pro triggers (plus divergence counts as a trigger)
    # This is the override that can allow alerts even without the simplistic volume flag.
    pro_triggers_count = 0
    if pro_mode:
        pro_triggers_count += 1 if (bull_sweep or bear_sweep) else 0
        pro_triggers_count += 1 if (bull_ob_retest or bear_ob_retest) else 0
        pro_triggers_count += 1 if (bull_breaker_retest or bear_breaker_retest) else 0
        pro_triggers_count += 1 if (orb_bull or orb_bear) else 0
        pro_triggers_count += 1 if divergence_event else 0
    strong_pro_confluence = bool(pro_mode and pro_triggers_count >= 2)

    # Confirmation score (0..8)
    orb_event = bool(orb_bull or orb_bear)
    liquidity_event = bool((bull_sweep or bear_sweep) or (bull_ob_retest or bear_ob_retest) or (bull_breaker_retest or bear_breaker_retest))
    fib_event = bool(fib_near_long or fib_near_short)

    confirmation_components = {
        "vwap": int(vwap_event),
        "orb": int(orb_event),
        "rsi": int(rsi_event),
        "micro_structure": int(micro_structure_event),
        "volume": int(volume_event),
        "divergence": int(divergence_event),
        "liquidity": int(liquidity_event),
        "fib": int(fib_event),
    }
    confirmation_score = int(sum(confirmation_components.values()))
    extras["confirmation_components"] = confirmation_components
    extras["confirmation_score"] = confirmation_score
    extras["strong_pro_confluence"] = bool(strong_pro_confluence)

    # Preserve gate diagnostics (used in UI/why strings)
    extras["gates"] = {
        "vwap_event": vwap_event,
        "rsi_event": rsi_event,
        "macd_event": macd_event,
        "volume_event": volume_event,
        "extended_session": bool(is_extended_session),
        "confirmation_score": confirmation_score,
        "strong_pro_confluence": bool(strong_pro_confluence),
    }

    # Confirm threshold: require multiple independent confirmations before we emit entry/TP or alert.
    # Pro mode gets a slightly lower threshold because we have more independent features.
    confirm_threshold = 4 if not pro_mode else 3
    extras["confirm_threshold"] = int(confirm_threshold)

    # PRE vs CONFIRMED stages
    # ----------------------
    # Goal: fire *earlier* (pre-trigger) alerts when a high-quality setup is forming,
    # without removing the confirmed (fully gated) alert. We do this by allowing a
    # PRE stage when price is approaching the planned trigger (usually VWAP) with
    # supportive momentum/structure, but before the reclaim/rejection event prints.
    #
    # Stages are stored in extras["stage"]:
    #   - "PRE"        : forming setup, provides an entry/stop/TP plan
    #   - "CONFIRMED"  : classic gated setup (confirm_threshold met + hard gates)
    stage: str | None = None
    stage_note: str = ""

    # Trigger-proximity used for PRE alerts
    # -------------------------------
    # PRE alerts should be *trigger proximity* driven (distance to the trigger line, normalized by ATR),
    # not only score thresholds or "actionable transition".
    #
    # Today the most common trigger line is VWAP (session or cumulative). If VWAP is unavailable (NaN)
    # we still allow PRE when Pro structural trigger exists, but proximity math is skipped.
    prox_atr = None
    prox_abs = None
    try:
        prox_abs = max(0.35 * float(atr_last or 0.0), 0.0008 * float(last_price or 0.0))
    except Exception:
        prox_abs = None

    trigger_near = False
    dist = None
    if isinstance(ref_vwap, (float, int)) and isinstance(last_price, (float, int)) and isinstance(prox_abs, (float, int)) and prox_abs > 0:
        dist = abs(float(last_price) - float(ref_vwap))
        trigger_near = bool(dist <= float(prox_abs))
        try:
            if atr_last and float(atr_last) > 0:
                prox_atr = float(dist) / float(atr_last)
        except Exception:
            prox_atr = None

    extras["trigger_proximity_atr"] = prox_atr
    extras["trigger_proximity_abs"] = float(prox_abs) if isinstance(prox_abs, (float, int)) else None
    extras["trigger_near"] = bool(trigger_near)

    # Momentum/structure "pre" hints
    rsi_pre_long = bool(_is_rising(df["rsi5"], 3) and float(df["rsi5"].iloc[-1]) < 60)
    rsi_pre_short = bool(_is_falling(df["rsi5"], 3) and float(df["rsi5"].iloc[-1]) > 40)
    macd_pre_long = bool(_is_rising(df["macd_hist"], 3))
    macd_pre_short = bool(_is_falling(df["macd_hist"], 3))
    struct_pre_long = bool(micro_hl)
    struct_pre_short = bool(micro_lh)

    # Small early-reversal PRE pathway:
    # allow an approaching / attempted reclaim to surface as PRE when momentum is already
    # improving and local structure has stopped cleanly trending against the trade, without
    # requiring the full reclaim-hold event yet.
    pre_trigger_near = bool(trigger_near)
    try:
        if not pre_trigger_near and isinstance(dist, (float, int)) and isinstance(prox_abs, (float, int)) and np.isfinite(dist) and np.isfinite(prox_abs):
            pre_trigger_near = bool(float(dist) <= max(1.35 * float(prox_abs), 0.55 * float(atr_last or 0.0)))
    except Exception:
        pass
    try:
        reclaim_attempt_long = bool(
            isinstance(ref_vwap, (float, int))
            and float(last_price) < float(ref_vwap)
            and (float(df["high"].tail(3).max()) >= float(ref_vwap) - 0.10 * float(buffer))
        )
        reclaim_attempt_short = bool(
            isinstance(ref_vwap, (float, int))
            and float(last_price) > float(ref_vwap)
            and (float(df["low"].tail(3).min()) <= float(ref_vwap) + 0.10 * float(buffer))
        )
    except Exception:
        reclaim_attempt_long = False
        reclaim_attempt_short = False
    early_pre_long = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) < float(ref_vwap)
        and pre_trigger_near
        and reclaim_attempt_long
        and ((macd_build_long_pts >= 4) or macd_pre_long or macd_event)
        and (struct_pre_long or liquidity_event or orb_event)
        and (confirmation_score >= max(2, confirm_threshold - 2))
    )
    early_pre_short = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) > float(ref_vwap)
        and pre_trigger_near
        and reclaim_attempt_short
        and ((macd_build_short_pts >= 4) or macd_pre_short or macd_event)
        and (struct_pre_short or liquidity_event or orb_event)
        and (confirmation_score >= max(2, confirm_threshold - 2))
    )
    extras["pre_trigger_near"] = bool(pre_trigger_near)
    extras["reclaim_attempt_long"] = bool(reclaim_attempt_long)
    extras["reclaim_attempt_short"] = bool(reclaim_attempt_short)
    extras["early_pre_long"] = bool(early_pre_long)
    extras["early_pre_short"] = bool(early_pre_short)

    # Acceptance progression (SCALP): a small capped energy-build bonus that helps PRE
    # setups surface a bit earlier when momentum is improving and price is starting
    # to interact with reclaim territory. This stays subordinate to the core logic.
    scalp_accept_progress_long_bonus = 0
    scalp_accept_progress_short_bonus = 0
    try:
        hist_tail = pd.to_numeric(df["macd_hist"].tail(int(min(4, len(df)))), errors="coerce").dropna()
        hist_last = float(hist_tail.iloc[-1]) if len(hist_tail) else float("nan")
        recent_abs = float(hist_tail.abs().tail(int(min(12, len(hist_tail)))).median()) if len(hist_tail) else float("nan")
        near_zero_band = max(0.0001, 0.60 * recent_abs) if np.isfinite(recent_abs) and recent_abs > 0 else 0.02
        hist_near_zero = bool(np.isfinite(hist_last) and abs(hist_last) <= near_zero_band)
        hist_green_build = bool(len(hist_tail) >= 3 and hist_tail.iloc[-1] > hist_tail.iloc[-2] > hist_tail.iloc[-3] and hist_tail.iloc[-1] > 0)
        hist_red_build = bool(len(hist_tail) >= 3 and hist_tail.iloc[-1] < hist_tail.iloc[-2] < hist_tail.iloc[-3] and hist_tail.iloc[-1] < 0)
        if early_pre_long:
            scalp_accept_progress_long_bonus += 2
            if hist_near_zero or hist_last > 0:
                scalp_accept_progress_long_bonus += 1
            if hist_green_build:
                scalp_accept_progress_long_bonus += 1
            if rsi_pre_long:
                scalp_accept_progress_long_bonus += 1
        if early_pre_short:
            scalp_accept_progress_short_bonus += 2
            if hist_near_zero or hist_last < 0:
                scalp_accept_progress_short_bonus += 1
            if hist_red_build:
                scalp_accept_progress_short_bonus += 1
            if rsi_pre_short:
                scalp_accept_progress_short_bonus += 1
    except Exception:
        pass
    scalp_accept_progress_long_bonus = int(min(4, max(0, scalp_accept_progress_long_bonus)))
    scalp_accept_progress_short_bonus = int(min(4, max(0, scalp_accept_progress_short_bonus)))
    if scalp_accept_progress_long_bonus:
        long_points += scalp_accept_progress_long_bonus
        long_reasons.append(f"Acceptance build +{scalp_accept_progress_long_bonus}")
    if scalp_accept_progress_short_bonus:
        short_points += scalp_accept_progress_short_bonus
        short_reasons.append(f"Acceptance build +{scalp_accept_progress_short_bonus}")
    extras["scalp_accept_progress_long_bonus"] = int(scalp_accept_progress_long_bonus)
    extras["scalp_accept_progress_short_bonus"] = int(scalp_accept_progress_short_bonus)

    tape_long = {"eligible": False, "readiness": 0.0, "tightening": 0.0, "structural_hold": 0.0, "pressure": 0.0, "release_proximity": 0.0}
    tape_short = {"eligible": False, "readiness": 0.0, "tightening": 0.0, "structural_hold": 0.0, "pressure": 0.0, "release_proximity": 0.0}
    tape_long_bonus = 0
    tape_short_bonus = 0
    tape_pre_long_assist = False
    tape_pre_short_assist = False
    if tape_mode_enabled:
        tape_long = _compute_tape_readiness(
            df,
            direction="LONG",
            atr_last=float(atr_last) if atr_last is not None else None,
            release_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            structural_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            trigger_near=bool(pre_trigger_near),
            baseline_ok=bool(pre_trigger_near and reclaim_attempt_long and (struct_pre_long or liquidity_event or orb_event) and ((macd_build_long_pts >= 3) or macd_pre_long or rsi_pre_long)),
        )
        tape_short = _compute_tape_readiness(
            df,
            direction="SHORT",
            atr_last=float(atr_last) if atr_last is not None else None,
            release_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            structural_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            trigger_near=bool(pre_trigger_near),
            baseline_ok=bool(pre_trigger_near and reclaim_attempt_short and (struct_pre_short or liquidity_event or orb_event) and ((macd_build_short_pts >= 3) or macd_pre_short or rsi_pre_short)),
        )
        tape_reversal_long = _compute_scalp_reversal_stabilization(
            df,
            direction="LONG",
            ref_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        tape_reversal_short = _compute_scalp_reversal_stabilization(
            df,
            direction="SHORT",
            ref_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        effective_readiness_long = float(tape_long.get("readiness") or 0.0) + 0.90 * float(tape_reversal_long.get("bonus") or 0.0)
        effective_readiness_short = float(tape_short.get("readiness") or 0.0) + 0.90 * float(tape_reversal_short.get("bonus") or 0.0)
        tape_long_bonus = _tape_bonus_from_readiness(
            effective_readiness_long,
            cap=3,
            thresholds=(4.8, 5.8, 6.8, 7.8),
        )
        tape_short_bonus = _tape_bonus_from_readiness(
            effective_readiness_short,
            cap=3,
            thresholds=(4.8, 5.8, 6.8, 7.8),
        )
        if tape_long_bonus:
            long_points += tape_long_bonus
            long_reasons.append(f"Tape readiness +{tape_long_bonus}")
        if tape_short_bonus:
            short_points += tape_short_bonus
            short_reasons.append(f"Tape readiness +{tape_short_bonus}")
        tape_pre_long_assist = bool(
            pre_trigger_near
            and reclaim_attempt_long
            and (struct_pre_long or liquidity_event or orb_event)
            and (confirmation_score >= max(1, confirm_threshold - 2))
            and (not vwap_event)
            and (
                effective_readiness_long >= 5.75
                or (
                    effective_readiness_long >= 5.15
                    and bool(tape_reversal_long.get("stabilizing") or False)
                    and bool(tape_reversal_long.get("reclaim_lean") or False)
                    and ((macd_build_long_pts >= 2) or macd_pre_long or rsi_pre_long)
                )
            )
        )
        tape_pre_short_assist = bool(
            pre_trigger_near
            and reclaim_attempt_short
            and (struct_pre_short or liquidity_event or orb_event)
            and (confirmation_score >= max(1, confirm_threshold - 2))
            and (not vwap_event)
            and (
                effective_readiness_short >= 5.75
                or (
                    effective_readiness_short >= 5.15
                    and bool(tape_reversal_short.get("stabilizing") or False)
                    and bool(tape_reversal_short.get("reclaim_lean") or False)
                    and ((macd_build_short_pts >= 2) or macd_pre_short or rsi_pre_short)
                )
            )
        )
    else:
        tape_reversal_long = {"bonus": 0.0, "stabilizing": False, "reclaim_lean": False}
        tape_reversal_short = {"bonus": 0.0, "stabilizing": False, "reclaim_lean": False}
        effective_readiness_long = float(tape_long.get("readiness") or 0.0)
        effective_readiness_short = float(tape_short.get("readiness") or 0.0)
    extras["tape_mode_enabled"] = bool(tape_mode_enabled)
    extras["tape_readiness_long"] = float(tape_long.get("readiness") or 0.0)
    extras["tape_readiness_short"] = float(tape_short.get("readiness") or 0.0)
    extras["tape_tightening_long"] = float(tape_long.get("tightening") or 0.0)
    extras["tape_tightening_short"] = float(tape_short.get("tightening") or 0.0)
    extras["tape_hold_long"] = float(tape_long.get("structural_hold") or 0.0)
    extras["tape_hold_short"] = float(tape_short.get("structural_hold") or 0.0)
    extras["tape_pressure_long"] = float(tape_long.get("pressure") or 0.0)
    extras["tape_pressure_short"] = float(tape_short.get("pressure") or 0.0)
    extras["tape_release_proximity_long"] = float(tape_long.get("release_proximity") or 0.0)
    extras["tape_release_proximity_short"] = float(tape_short.get("release_proximity") or 0.0)
    extras["tape_bonus_applied_long"] = int(tape_long_bonus)
    extras["tape_bonus_applied_short"] = int(tape_short_bonus)
    extras["tape_pre_long_assist"] = bool(tape_pre_long_assist)
    extras["tape_pre_short_assist"] = bool(tape_pre_short_assist)
    extras["tape_effective_readiness_long"] = float(effective_readiness_long)
    extras["tape_effective_readiness_short"] = float(effective_readiness_short)
    extras["tape_reversal_stabilization_long"] = bool(tape_reversal_long.get("stabilizing") or False)
    extras["tape_reversal_stabilization_short"] = bool(tape_reversal_short.get("stabilizing") or False)
    extras["tape_reclaim_lean_long"] = bool(tape_reversal_long.get("reclaim_lean") or False)
    extras["tape_reclaim_lean_short"] = bool(tape_reversal_short.get("reclaim_lean") or False)
    extras["tape_reversal_bonus_long"] = float(tape_reversal_long.get("bonus") or 0.0)
    extras["tape_reversal_bonus_short"] = float(tape_reversal_short.get("bonus") or 0.0)

    # Primary trigger must exist (otherwise we have nothing to anchor a plan).
    # NOTE: this is used by both PRE and CONFIRMED routing.
    primary_trigger = bool(vwap_event or rsi_event or macd_event or pro_trigger or early_pre_long or early_pre_short or tape_pre_long_assist or tape_pre_short_assist)
    extras["primary_trigger"] = primary_trigger

    # PRE condition: near trigger line on the "wrong" side, with momentum/structure pointing toward a flip.
    pre_long_ok = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) < float(ref_vwap)
        and (trigger_near or early_pre_long or tape_pre_long_assist)
        and (rsi_event or rsi_pre_long or macd_event or macd_pre_long or pro_trigger or (macd_build_long_pts >= 4))
        and (struct_pre_long or liquidity_event or orb_event or reclaim_attempt_long)
        and (confirmation_score >= max(2, confirm_threshold - 1) or early_pre_long or tape_pre_long_assist)
    )
    pre_short_ok = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) > float(ref_vwap)
        and (trigger_near or early_pre_short or tape_pre_short_assist)
        and (rsi_event or rsi_pre_short or macd_event or macd_pre_short or pro_trigger or (macd_build_short_pts >= 4))
        and (struct_pre_short or liquidity_event or orb_event or reclaim_attempt_short)
        and (confirmation_score >= max(2, confirm_threshold - 1) or early_pre_short or tape_pre_short_assist)
    )

    # If we're near the trigger line and the setup quality is already strong, emit PRE even if we are
    # one confirmation short (so you don't get the alert *after* the move already started).
    # This is intentionally conservative: requires proximity + at least 2 confirmations + a real trigger anchor.
    try:
        setup_quality_points = float(max(long_points_cal, short_points_cal))
    except Exception:
        setup_quality_points = float(max(long_points, short_points))
    pre_proximity_quality = bool(
        trigger_near
        and primary_trigger
        and confirmation_score >= 2
        and setup_quality_points >= float(cfg.get("min_actionable_score", 60)) * 0.85
    )
    extras["pre_proximity_quality"] = bool(pre_proximity_quality)

    # "Soft-hard" volume requirement:
    # If volume is required and missing, defer the final decision until after
    # entry-zone context is known. Weak no-volume setups still die; strong
    # reversal-context setups can survive with a score penalty instead.
    volume_gate_active = bool(int(cfg.get("require_volume", 0)) == 1 and (not volume_event) and (not strong_pro_confluence))
    volume_missing_penalty = 0
    low_volume_override = False
    extras["volume_gate_active"] = bool(volume_gate_active)

    if not primary_trigger:
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No primary trigger (VWAP/RSI/MACD/Pro)", None, None, None, None, last_price, last_ts, session, extras)

    # Stage selection:
    #   - CONFIRMED requires full confirmation_score + hard gates.
    #   - PRE can be emitted one notch earlier (approaching VWAP) so traders can be ready.
    if confirmation_score < confirm_threshold:
        if pre_long_ok or pre_short_ok or pre_proximity_quality:
            stage = "PRE"
            stage_note = f"PRE: trigger proximity (confirmations {confirmation_score}/{confirm_threshold})"
        else:
            return SignalResult(
                symbol, "NEUTRAL", _cap_score(max(long_points, short_points)),
                f"Not enough confirmations ({confirmation_score}/{confirm_threshold})",
                None, None, None, None,
                last_price, last_ts, session, extras,
            )
    else:
        stage = "CONFIRMED"
        stage_note = f"CONFIRMED ({confirmation_score}/{confirm_threshold})"

    # Optional: keep classic hard requirements during RTH when Pro confluence is absent.
    # (These protect the "Cleaner signals" preset from becoming too loose.)
    hard_vwap = (int(cfg.get("require_vwap_event", 0)) == 1) and (not is_extended_session)
    hard_rsi  = (int(cfg.get("require_rsi_event", 0)) == 1) and (not is_extended_session)
    hard_macd = (int(cfg.get("require_macd_turn", 0)) == 1) and (not is_extended_session)

    # Hard gates apply to CONFIRMED only (PRE is allowed to form *before* these print).
    if stage == "CONFIRMED":
        if hard_vwap and (not vwap_event) and (not pro_trigger):
            # If the setup is *almost* there, degrade to PRE instead of dropping it.
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: VWAP event not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
        if hard_rsi and (not rsi_event) and (not pro_trigger):
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: RSI event not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
        if hard_macd and (not macd_event) and (not pro_trigger):
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: MACD turn not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)

    # For extended sessions (PM/AH), mark missing classic triggers for transparency.
    if is_extended_session:
        if int(cfg.get("require_vwap_event", 0)) == 1 and (not vwap_event) and (not pro_trigger):
            extras["soft_gate_missing_vwap"] = True
        if int(cfg.get("require_rsi_event", 0)) == 1 and (not rsi_event) and (not pro_trigger):
            extras["soft_gate_missing_rsi"] = True
        if int(cfg.get("require_macd_turn", 0)) == 1 and (not macd_event) and (not pro_trigger):
            extras["soft_gate_missing_macd"] = True

    # ATR-normalized score calibration (per ticker)
    # If target_atr_pct is None => auto-tune per ticker using median ATR% over a recent window.
    # Otherwise => use the manual target ATR% as a global anchor.
    scale = 1.0
    ref_atr_pct = None
    if atr_pct:
        if target_atr_pct is None:
            atr_series = df["atr14"].tail(120)
            close_series = df["close"].tail(120).replace(0, np.nan)
            atr_pct_series = (atr_series / close_series).replace([np.inf, -np.inf], np.nan).dropna()
            if len(atr_pct_series) >= 20:
                ref_atr_pct = float(np.nanmedian(atr_pct_series.values))
        else:
            ref_atr_pct = float(target_atr_pct)

        if ref_atr_pct and ref_atr_pct > 0:
            scale = ref_atr_pct / atr_pct
            # Keep calibration gentle; we want comparability, not distortion.
            scale = float(np.clip(scale, 0.75, 1.25))

    extras["atr_score_scale"] = scale
    extras["atr_ref_pct"] = ref_atr_pct

    long_points_cal = int(round(long_points * scale))
    short_points_cal = int(round(short_points * scale))
    extras["long_points_raw"] = long_points
    extras["short_points_raw"] = short_points
    extras["long_points_cal"] = long_points_cal
    extras["short_points_cal"] = short_points_cal
    extras["contrib_points"] = contrib

    min_score = int(cfg["min_actionable_score"])

    # Entry/stop + targets
    tighten_factor = 1.0
    if pro_mode:
        # Tighten stops a bit when we have structural confluence.
        # NOTE: We intentionally do NOT mutate the setup_score here; scoring is handled above.
        confluence = bool(
            (isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
            or bull_sweep or bear_sweep
            or orb_bull or orb_bear
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or (bull_fvg is not None) or (bear_fvg is not None)
        )
        if confluence:
            tighten_factor = 0.85
        extras["stop_tighten_factor"] = float(tighten_factor)

    def _fib_take_profits_long(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        exts = _fib_extensions(hi, lo)
        # Partial at recent high if above entry, else at ext 1.272
        tp1 = hi if entry_px < hi else next((lvl for _, lvl in exts if lvl > entry_px), None)
        tp2 = next((lvl for _, lvl in exts if lvl and tp1 and lvl > tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _fib_take_profits_short(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        # Mirror extensions below lo
        ratios = [1.0, 1.272, 1.618]
        exts_dn = [ (f"Ext -{r:g}", lo - (r - 1.0) * rng) for r in ratios ]
        tp1 = lo if entry_px > lo else next((lvl for _, lvl in exts_dn if lvl < entry_px), None)
        tp2 = next((lvl for _, lvl in exts_dn if lvl and tp1 and lvl < tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _long_entry_stop(entry_px: float):
        stop_px = float(min(recent_swing_low, entry_px - max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px - (entry_px - stop_px) * tighten_factor)
        if bull_breaker_retest and brk_bull[0] is not None:
            stop_px = float(min(stop_px, brk_bull[0] - buffer))
        if fib_near_long and fib_level is not None:
            stop_px = float(min(stop_px, fib_level - buffer))
        return entry_px, stop_px

    def _short_entry_stop(entry_px: float):
        stop_px = float(max(recent_swing_high, entry_px + max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px + (stop_px - entry_px) * tighten_factor)
        if bear_breaker_retest and brk_bear[1] is not None:
            stop_px = float(max(stop_px, brk_bear[1] + buffer))
        if fib_near_short and fib_level is not None:
            stop_px = float(max(stop_px, fib_level + buffer))
        return entry_px, stop_px
    # Final decision + trade levels
    long_score = int(round(float(long_points_cal))) if 'long_points_cal' in locals() else int(round(float(long_points)))
    short_score = int(round(float(short_points_cal))) if 'short_points_cal' in locals() else int(round(float(short_points)))

    # Never allow scores outside 0..100.
    long_score = _cap_score(long_score)
    short_score = _cap_score(short_score)

    # NOTE: entry-zone context is evaluated later, after we have a concrete executable entry_limit.
    # So do not apply any zone adjustment or min-score return here.
    extras["entry_zone_score_adj"] = 0

    # Stage + direction
    extras["stage"] = stage
    extras["stage_note"] = stage_note

    # For PRE alerts, prefer the directional pre-condition when it is unambiguous.
    if stage == "PRE" and pre_long_ok and not pre_short_ok:
        bias = "LONG"
    elif stage == "PRE" and pre_short_ok and not pre_long_ok:
        bias = "SHORT"
    else:
        bias = "LONG" if long_score >= short_score else "SHORT"
    setup_score = _cap_score(max(long_score, short_score))

    # Assemble reason text from the winning side
    if bias == "LONG":
        reasons = long_reasons[:] if 'long_reasons' in locals() else []
    else:
        reasons = short_reasons[:] if 'short_reasons' in locals() else []
    try:
        if isinstance(extras.get("entry_zone_context"), dict):
            ez = extras.get("entry_zone_context") or {}
            if ez.get("favorable") and ez.get("favorable_type"):
                reasons.append(f"Entry near {ez.get('favorable_type')}")
            if ez.get("hostile") and ez.get("hostile_type"):
                reasons.append(f"Entry near {ez.get('hostile_type')}")
    except Exception:
        pass

    core_reason = "; ".join(reasons) if reasons else "Actionable setup"
    reason = (stage_note + " — " if stage_note else "") + core_reason

    # Entry model context
    ref_vwap = None
    try:
        ref_vwap = float(vwap_use.iloc[-1])
    except Exception:
        ref_vwap = None

    mid_price = None
    try:
        mid_price = float((df["high"].iloc[-1] + df["low"].iloc[-1]) / 2.0)
    except Exception:
        mid_price = None

    # Adaptive SCALP acceptance line:
    # blend reclaim/rejection anchors with nearby structure, but favor the level
    # the market has defended most recently so entry logic follows the live shelf.
    scalp_accept_line = ref_vwap
    scalp_accept_src = "VWAP"
    scalp_accept_diag = {}
    try:
        ema20_ref = float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None
    except Exception:
        ema20_ref = None
    try:
        if bias == "LONG":
            hold_count = int((close.astype(float).tail(int(min(3, len(close)))) >= float(ref_vwap) - 0.10 * float(buffer)).sum()) if isinstance(ref_vwap, (float, int)) else 0
            pivot_anchor = float(recent_swing_low)
            anchors = []
            weights = []
            diag = {}
            if isinstance(ref_vwap, (float, int)):
                base_w = 0.58 + 0.10 * hold_count
                rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["VWAP"] = rx
                anchors.append(float(ref_vwap)); weights.append(base_w + 0.30 * float(rx.get("score") or 0.0))
            if isinstance(pivot_anchor, (float, int)):
                pivot_near = max(0.0, 1.0 - (abs(float(last_price) - float(pivot_anchor)) / max(1e-9, 1.25 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(pivot_anchor), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["PIVOT"] = rx
                weights.append(0.22 + 0.18 * float(micro_hl) + 0.10 * pivot_near + 0.34 * float(rx.get("score") or 0.0))
                anchors.append(float(pivot_anchor))
            if isinstance(ema20_ref, (float, int)):
                ema_near = max(0.0, 1.0 - (abs(float(last_price) - float(ema20_ref)) / max(1e-9, 1.50 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
                diag["EMA20"] = rx
                anchors.append(float(ema20_ref)); weights.append(0.12 + 0.08 * float(trend_long_ok) + 0.05 * ema_near + 0.20 * float(rx.get("score") or 0.0))
            if anchors and sum(weights) > 0:
                scalp_accept_line = float(np.average(np.asarray(anchors, dtype=float), weights=np.asarray(weights, dtype=float)))
                scalp_accept_line = float(min(float(last_price), max(float(min(anchors)), scalp_accept_line)))
                scalp_accept_src = "BLEND" if len(anchors) > 1 else "VWAP"
            scalp_accept_diag = diag
        else:
            hold_count = int((close.astype(float).tail(int(min(3, len(close)))) <= float(ref_vwap) + 0.10 * float(buffer)).sum()) if isinstance(ref_vwap, (float, int)) else 0
            pivot_anchor = float(recent_swing_high)
            anchors = []
            weights = []
            diag = {}
            if isinstance(ref_vwap, (float, int)):
                base_w = 0.58 + 0.10 * hold_count
                rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["VWAP"] = rx
                anchors.append(float(ref_vwap)); weights.append(base_w + 0.30 * float(rx.get("score") or 0.0))
            if isinstance(pivot_anchor, (float, int)):
                pivot_near = max(0.0, 1.0 - (abs(float(last_price) - float(pivot_anchor)) / max(1e-9, 1.25 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(pivot_anchor), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["PIVOT"] = rx
                anchors.append(float(pivot_anchor)); weights.append(0.22 + 0.18 * float(micro_lh) + 0.10 * pivot_near + 0.34 * float(rx.get("score") or 0.0))
            if isinstance(ema20_ref, (float, int)):
                ema_near = max(0.0, 1.0 - (abs(float(last_price) - float(ema20_ref)) / max(1e-9, 1.50 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
                diag["EMA20"] = rx
                anchors.append(float(ema20_ref)); weights.append(0.12 + 0.08 * float(trend_short_ok) + 0.05 * ema_near + 0.20 * float(rx.get("score") or 0.0))
            if anchors and sum(weights) > 0:
                scalp_accept_line = float(np.average(np.asarray(anchors, dtype=float), weights=np.asarray(weights, dtype=float)))
                scalp_accept_line = float(max(float(last_price), min(float(max(anchors)), scalp_accept_line)))
                scalp_accept_src = "BLEND" if len(anchors) > 1 else "VWAP"
            scalp_accept_diag = diag
    except Exception:
        scalp_accept_line = ref_vwap
        scalp_accept_src = "VWAP"
        scalp_accept_diag = {}

    extras["accept_line"] = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else None
    extras["accept_src"] = scalp_accept_src
    extras["accept_line_raw"] = float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None
    extras["accept_line_recent_diag"] = scalp_accept_diag

    entry_px = _entry_from_model(
        bias,
        entry_model=entry_model,
        last_price=float(last_price),
        ref_vwap=(float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else ref_vwap),
        mid_price=mid_price,
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    # Entry model upgrade: expose both a limit entry and a chase-line.
    entry_limit, chase_line = _entry_limit_and_chase(
        bias,
        entry_px=float(entry_px),
        last_px=float(last_price),
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    # Priority #2 — SCALP timing improvement:
    # On fast snap reversals, the old flow could wait for reclaim/confirmation and end up
    # entering after a meaningful portion of the first impulse had already occurred.
    # Here we allow a modest "early reversal" entry improvement when price has clearly
    # stabilized and started reclaiming, but before the move becomes fully comfortable.
    scalp_early_entry_applied = False
    scalp_early_entry_reason = None
    scalp_early_entry_anchor = None
    try:
        atr_ref = float(atr_last or 0.0)
        if isinstance(entry_limit, (float, int)) and isinstance(last_price, (float, int)) and atr_ref > 0:
            recent = df.tail(int(min(6, len(df))))
            lows = pd.to_numeric(recent.get("low"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            highs = pd.to_numeric(recent.get("high"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            closes = pd.to_numeric(recent.get("close"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            opens = pd.to_numeric(recent.get("open"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            vols = pd.to_numeric(recent.get("volume"), errors="coerce") if len(recent) and "volume" in recent else pd.Series(dtype=float)
            if len(closes) >= 4 and len(lows) >= 4 and len(highs) >= 4 and len(opens) >= 4:
                body_last = abs(float(closes.iloc[-1]) - float(opens.iloc[-1]))
                body_prev = abs(float(closes.iloc[-2]) - float(opens.iloc[-2]))
                rng_last = max(1e-9, float(highs.iloc[-1]) - float(lows.iloc[-1]))
                rng_prev = max(1e-9, float(highs.iloc[-2]) - float(lows.iloc[-2]))
                upper_wick_last = float(highs.iloc[-1]) - max(float(closes.iloc[-1]), float(opens.iloc[-1]))
                lower_wick_last = min(float(closes.iloc[-1]), float(opens.iloc[-1])) - float(lows.iloc[-1])
                upper_wick_prev = float(highs.iloc[-2]) - max(float(closes.iloc[-2]), float(opens.iloc[-2]))
                lower_wick_prev = min(float(closes.iloc[-2]), float(opens.iloc[-2])) - float(lows.iloc[-2])
                close_pos_last = (float(closes.iloc[-1]) - float(lows.iloc[-1])) / rng_last
                close_pos_last_short = (float(highs.iloc[-1]) - float(closes.iloc[-1])) / rng_last
                local_low = float(lows.tail(4).min())
                local_high = float(highs.tail(4).max())
                low_hold = min(float(lows.iloc[-1]), float(lows.iloc[-2])) >= local_low - 0.06 * atr_ref
                high_hold = max(float(highs.iloc[-1]), float(highs.iloc[-2])) <= local_high + 0.06 * atr_ref
                reclaim_progress_long = float(closes.iloc[-1]) >= local_low + 0.32 * atr_ref
                reclaim_progress_short = float(closes.iloc[-1]) <= local_high - 0.32 * atr_ref
                vol_support_long = True
                vol_support_short = True
                if len(vols) >= 4 and vols.notna().sum() >= 4:
                    v_last = float(vols.iloc[-1])
                    v_prev = float(vols.iloc[-2])
                    v_base = float(vols.iloc[:-2].replace(0, pd.NA).dropna().mean()) if len(vols) > 2 and not vols.iloc[:-2].replace(0, pd.NA).dropna().empty else 0.0
                    if v_base > 0:
                        vol_support_long = (v_last >= 0.70 * v_base) or (v_last >= 0.90 * v_prev)
                        vol_support_short = vol_support_long
                fast_rev_long = bool(
                    str(bias).upper() == "LONG"
                    and (bool(early_pre_long) or bool(tape_pre_long_assist) or (str(stage).upper() == "CONFIRMED" and bool(rsi_pre_long) and bool(macd_pre_long) and bool(struct_pre_long)))
                    and float(closes.iloc[-1]) > float(closes.iloc[-2])
                    and low_hold
                    and reclaim_progress_long
                    and body_last >= max(0.08 * atr_ref, 0.85 * max(1e-9, body_prev))
                    and body_last >= 0.42 * rng_last
                    and close_pos_last >= 0.58
                    and upper_wick_last <= 0.52 * rng_last
                    and lower_wick_last + 0.03 * atr_ref >= 0.85 * max(0.0, lower_wick_prev)
                    and vol_support_long
                    and float(last_price) <= float(entry_limit) + 0.32 * atr_ref
                )
                fast_rev_short = bool(
                    str(bias).upper() == "SHORT"
                    and (bool(early_pre_short) or bool(tape_pre_short_assist) or (str(stage).upper() == "CONFIRMED" and bool(rsi_pre_short) and bool(macd_pre_short) and bool(struct_pre_short)))
                    and float(closes.iloc[-1]) < float(closes.iloc[-2])
                    and high_hold
                    and reclaim_progress_short
                    and body_last >= max(0.08 * atr_ref, 0.85 * max(1e-9, body_prev))
                    and body_last >= 0.42 * rng_last
                    and close_pos_last_short >= 0.58
                    and lower_wick_last <= 0.52 * rng_last
                    and upper_wick_last + 0.03 * atr_ref >= 0.85 * max(0.0, upper_wick_prev)
                    and vol_support_short
                    and float(last_price) >= float(entry_limit) - 0.32 * atr_ref
                )
                if fast_rev_long:
                    accept_ref = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(entry_limit)
                    reclaim_anchor = max(local_low + 0.28 * atr_ref, accept_ref + 0.01 * atr_ref)
                    candidate = min(float(entry_limit), max(reclaim_anchor, float(last_price) - 0.10 * atr_ref))
                    if candidate < float(entry_limit):
                        entry_limit = float(candidate)
                        chase_line = max(float(chase_line), float(entry_limit) + 0.12 * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit + 0.12 * atr_ref)
                        scalp_early_entry_applied = True
                        scalp_early_entry_reason = "FAST_REVERSAL_LONG"
                        scalp_early_entry_anchor = float(reclaim_anchor)
                elif fast_rev_short:
                    accept_ref = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(entry_limit)
                    reclaim_anchor = min(local_high - 0.28 * atr_ref, accept_ref - 0.01 * atr_ref)
                    candidate = max(float(entry_limit), min(reclaim_anchor, float(last_price) + 0.10 * atr_ref))
                    if candidate > float(entry_limit):
                        entry_limit = float(candidate)
                        chase_line = min(float(chase_line), float(entry_limit) - 0.12 * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit - 0.12 * atr_ref)
                        scalp_early_entry_applied = True
                        scalp_early_entry_reason = "FAST_REVERSAL_SHORT"
                        scalp_early_entry_anchor = float(reclaim_anchor)
    except Exception:
        scalp_early_entry_applied = False
        scalp_early_entry_reason = None
        scalp_early_entry_anchor = None

    extras["scalp_early_entry_applied"] = bool(scalp_early_entry_applied)
    extras["scalp_early_entry_reason"] = scalp_early_entry_reason
    extras["scalp_early_entry_anchor"] = float(scalp_early_entry_anchor) if isinstance(scalp_early_entry_anchor, (float, int)) else None

    # Entry model upgrade: adapt when the planned limit is already stale.
    # If price has already moved beyond the limit by a meaningful fraction of ATR,
    # we flip the plan to a chase-based execution so we don't alert *after* the move.
    #
    # - LONG: if last is above the limit by > stale_buffer => use chase line as the new entry.
    # - SHORT: if last is below the limit by > stale_buffer => use chase line as the new entry.
    #
    # This keeps entry/stop/TP coherent (all are computed off entry_limit) while preserving
    # the informational chase line for the trader.
    stale_buffer = None
    try:
        stale_buffer = max(0.25 * float(atr_last or 0.0), 0.0006 * float(last_price or 0.0))
    except Exception:
        stale_buffer = None

    exec_mode = "LIMIT"
    entry_stale = False
    if isinstance(stale_buffer, (float, int)) and stale_buffer and stale_buffer > 0:
        try:
            if bias == "LONG" and float(last_price) > float(entry_limit) + float(stale_buffer):
                exec_mode = "CHASE"; entry_stale = True
                entry_limit = float(chase_line)
            elif bias == "SHORT" and float(last_price) < float(entry_limit) - float(stale_buffer):
                exec_mode = "CHASE"; entry_stale = True
                entry_limit = float(chase_line)
        except Exception:
            pass

    extras["execution_mode"] = exec_mode
    extras["entry_stale"] = bool(entry_stale)
    extras["entry_stale_buffer"] = float(stale_buffer) if isinstance(stale_buffer, (float, int)) else None
    extras["entry_limit"] = float(entry_limit)
    extras["entry_chase_line"] = float(chase_line)

    # Entry-zone context: small local demand/supply tilt around the proposed executable entry.
    scalp_zone_ctx = _evaluate_entry_zone_context(
        df, entry_price=float(entry_limit), direction=str(bias), atr_last=float(atr_last) if atr_last is not None else None, lookback=10
    )
    scalp_zone_adj = 0.0
    fav_q = float(scalp_zone_ctx.get("favorable_quality") or 0.0)
    host_q = float(scalp_zone_ctx.get("hostile_quality") or 0.0)
    fav_inside = bool(scalp_zone_ctx.get("favorable_inside"))
    host_inside = bool(scalp_zone_ctx.get("hostile_inside"))
    if bool(scalp_zone_ctx.get("favorable")):
        favorable_boost = 3.0 + 2.0 * fav_q
        if fav_inside:
            favorable_boost += 1.25 + 1.25 * fav_q
        scalp_zone_adj += favorable_boost
    if bool(scalp_zone_ctx.get("hostile")):
        hostile_pen = 4.0 + 3.0 * host_q
        if host_inside:
            hostile_pen += 1.5 + 1.5 * host_q
        if stage == "PRE":
            hostile_pen += 1.5 + 1.5 * host_q
        if exec_mode == "CHASE":
            hostile_pen += 0.75
        scalp_zone_adj -= hostile_pen
    extras["entry_zone_context"] = scalp_zone_ctx
    extras["entry_zone_score_adj"] = int(round(float(scalp_zone_adj))) if isinstance(scalp_zone_adj, (int, float)) else 0

    # Apply the zone tilt to the currently selected direction BEFORE final threshold gating.
    try:
        if isinstance(scalp_zone_adj, (int, float)) and scalp_zone_adj != 0:
            if str(bias).upper() == "LONG":
                long_score = _cap_score(long_score + int(round(float(scalp_zone_adj))))
            elif str(bias).upper() == "SHORT":
                short_score = _cap_score(short_score + int(round(float(scalp_zone_adj))))
    except Exception:
        pass
    setup_score = _cap_score(max(long_score, short_score))

    scalp_extension_profile = _compute_multibar_extension_profile(
        df,
        direction=str(bias),
        atr_last=float(atr_last) if atr_last is not None else None,
        accept_line=float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else None,
    )
    extras["scalp_extension_profile"] = scalp_extension_profile
    scalp_ext_penalty = int(round(4.0 * float(scalp_extension_profile.get("penalty") or 0.0)))
    if scalp_ext_penalty > 0:
        if str(bias).upper() == "LONG":
            long_score = _cap_score(long_score - scalp_ext_penalty)
        else:
            short_score = _cap_score(short_score - scalp_ext_penalty)
        setup_score = _cap_score(max(long_score, short_score))
        reason = (reason + "; " if reason else "") + f"Extension guard (-{int(scalp_ext_penalty)})"

    scalp_weak_tape_diag = {"score": 1.0, "ok": True, "stall": False, "rejection": False}
    weak_tape_env = bool((not strong_pro_confluence) and ((adx_last is None) or (float(adx_last) < 18.0) or (not volume_event)))
    if stage == "PRE" and weak_tape_env:
        scalp_weak_tape_diag = _assess_scalp_weak_tape_turn(
            df,
            direction=str(bias),
            trigger_line=float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        extras["scalp_weak_tape_diag"] = scalp_weak_tape_diag
        if bool(scalp_weak_tape_diag.get("rejection")):
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "PRE blocked: repeated trigger rejection in weak tape", None, None, None, None, last_price, last_ts, session, extras)
        if not bool(scalp_weak_tape_diag.get("ok")):
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "PRE blocked: weak tape lacks stall/turn quality", None, None, None, None, last_price, last_ts, session, extras)
    else:
        extras["scalp_weak_tape_diag"] = scalp_weak_tape_diag

    try:
        if isinstance(scalp_zone_ctx, dict):
            if scalp_zone_ctx.get("favorable") and scalp_zone_ctx.get("favorable_type"):
                reason = (reason + "; " if reason else "") + f"Entry near {scalp_zone_ctx.get('favorable_type')}"
            if scalp_zone_ctx.get("hostile") and scalp_zone_ctx.get("hostile_type"):
                reason = (reason + "; " if reason else "") + f"Entry near {scalp_zone_ctx.get('hostile_type')}"
    except Exception:
        pass

    # Deferred low-volume handling: allow only strong reversal-context setups to survive.
    if volume_gate_active:
        reversal_flags = [liquidity_event, micro_structure_event, rsi_event, macd_event, orb_event]
        reversal_structure_count = int(sum(bool(x) for x in reversal_flags))
        reversal_structure_ok = bool(reversal_structure_count > 0)
        fav_q_for_override = float(scalp_zone_ctx.get("favorable_quality") or 0.0) if isinstance(scalp_zone_ctx, dict) else 0.0
        favorable_zone_ok = bool(isinstance(scalp_zone_ctx, dict) and scalp_zone_ctx.get("favorable") and fav_q_for_override >= 0.40)
        if favorable_zone_ok and reversal_structure_ok:
            low_volume_override = True
            base_penalty = float(5.0 * float(liquidity_mult)) if isinstance(liquidity_mult, (int, float)) else 5.0
            structure_quality = float(np.clip(reversal_structure_count / 4.0, 0.0, 1.0))
            proximity_quality = 1.0 if bool(trigger_near) else 0.0
            override_quality = float(np.clip((0.50 * fav_q_for_override) + (0.30 * structure_quality) + (0.20 * proximity_quality), 0.0, 1.0))
            penalty_scale = float(np.clip(1.15 - 0.45 * override_quality, 0.70, 1.15))
            volume_missing_penalty = int(np.clip(round(base_penalty * penalty_scale), 2, 8))
            if str(bias).upper() == "LONG":
                long_score = _cap_score(long_score - int(volume_missing_penalty))
            elif str(bias).upper() == "SHORT":
                short_score = _cap_score(short_score - int(volume_missing_penalty))
            setup_score = _cap_score(max(long_score, short_score))
            extras["volume_override_quality"] = float(round(override_quality, 4))
            extras["volume_override_structure_count"] = int(reversal_structure_count)
            reason = (reason + "; " if reason else "") + f"Low-volume override (-{int(volume_missing_penalty)})"
        else:
            extras["low_volume_override"] = False
            extras["volume_missing_penalty"] = 0
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "No volume confirmation", None, None, None, None, last_price, last_ts, session, extras)

    extras["low_volume_override"] = bool(low_volume_override)
    extras["volume_missing_penalty"] = int(volume_missing_penalty)

    if long_score < min_score and short_score < min_score:
        extras["decision"] = {"bias": bias, "long": long_score, "short": short_score, "min": min_score}
        neutral_reason = reason if reason else "Score below threshold"
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), neutral_reason, None, None, None, None, last_price, last_ts, session, extras)

    # PRE tier risk tightening: smaller risk ⇒ closer TP ⇒ more hits.
    interval_mins_i = int(interval_mins) if isinstance(interval_mins, (int, float)) else 1
    pre_stop_tighten = 0.70 if stage == "PRE" else 1.0
    extras["pre_stop_tighten"] = float(pre_stop_tighten)

    if bias == "LONG":
        entry_px, stop_px = _long_entry_stop(float(entry_limit))
        if stage == "PRE":
            stop_px = float(entry_px - (entry_px - stop_px) * pre_stop_tighten)
        risk = max(1e-9, entry_px - stop_px)
        # Targeting overhaul (structure-first): TP0/TP1/TP2
        lvl_map = _candidate_levels_from_context(
            levels=levels if isinstance(levels, dict) else {},
            recent_swing_high=float(recent_swing_high),
            recent_swing_low=float(recent_swing_low),
            hi=float(hi),
            lo=float(lo),
        )
        tp0 = _pick_tp0("LONG", entry_px=entry_px, last_px=float(last_price), atr_last=float(atr_last), levels=lvl_map)
        tp1 = (tp0 + 0.9 * risk) if tp0 is not None else (entry_px + risk)
        tp2 = (tp0 + 1.8 * risk) if tp0 is not None else (entry_px + 2 * risk)
        # Optional TP3: expected excursion (rolling MFE) for similar historical signatures
        sig_key = {
            "rsi_event": bool(rsi_snap and rsi14 < 60),
            "macd_event": bool(macd_turn_up),
            "vol_event": bool(vol_ok),
            "struct_event": bool(micro_hl),
            "vol_mult": float(cfg.get("vol_multiplier", 1.25)),
        }
        tp3, tp3_diag = _tp3_from_expected_excursion(
            df, direction="LONG", signature=sig_key, entry_px=float(entry_px), interval_mins=int(interval_mins_i)
        )
        extras["tp3"] = float(tp3) if tp3 is not None else None
        extras["tp3_diag"] = tp3_diag

        # If fib extension helper is available, prefer it for pro mode.
        if pro_mode and "_fib_take_profits_long" in locals():
            f1, f2 = _fib_take_profits_long(entry_px)
            # Use fib as TP2 (runner) when it is further than our structure target.
            if f1 is not None and (tp0 is None or float(f1) > float(tp0)):
                tp1 = float(f1)
            if f2 is not None and float(f2) > float(tp1):
                tp2 = float(f2)
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
    else:
        entry_px, stop_px = _short_entry_stop(float(entry_limit))
        if stage == "PRE":
            stop_px = float(entry_px + (stop_px - entry_px) * pre_stop_tighten)
        risk = max(1e-9, stop_px - entry_px)
        lvl_map = _candidate_levels_from_context(
            levels=levels if isinstance(levels, dict) else {},
            recent_swing_high=float(recent_swing_high),
            recent_swing_low=float(recent_swing_low),
            hi=float(hi),
            lo=float(lo),
        )
        tp0 = _pick_tp0("SHORT", entry_px=entry_px, last_px=float(last_price), atr_last=float(atr_last), levels=lvl_map)
        tp1 = (tp0 - 0.9 * risk) if tp0 is not None else (entry_px - risk)
        tp2 = (tp0 - 1.8 * risk) if tp0 is not None else (entry_px - 2 * risk)
        sig_key = {
            "rsi_event": bool(rsi_downshift and rsi14 > 40),
            "macd_event": bool(macd_turn_down),
            "vol_event": bool(vol_ok),
            "struct_event": bool(micro_lh),
            "vol_mult": float(cfg.get("vol_multiplier", 1.25)),
        }
        tp3, tp3_diag = _tp3_from_expected_excursion(
            df, direction="SHORT", signature=sig_key, entry_px=float(entry_px), interval_mins=int(interval_mins_i)
        )
        extras["tp3"] = float(tp3) if tp3 is not None else None
        extras["tp3_diag"] = tp3_diag

        if pro_mode and "_fib_take_profits_short" in locals():
            f1, f2 = _fib_take_profits_short(entry_px)
            if f1 is not None and (tp0 is None or float(f1) < float(tp0)):
                tp1 = float(f1)
            if f2 is not None and float(f2) < float(tp1):
                tp2 = float(f2)
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None

    # Expected time-to-TP0 UI helper
    extras["tp0"] = float(tp0) if "tp0" in locals() and tp0 is not None else None
    extras["eta_tp0_min"] = _eta_minutes_to_tp0(
        last_px=float(last_price),
        tp0=tp0 if "tp0" in locals() else None,
        atr_last=float(atr_last) if atr_last else 0.0,
        interval_mins=interval_mins_i,
        liquidity_mult=float(liquidity_mult) if "liquidity_mult" in locals() else 1.0,
    )

    extras["decision"] = {"bias": bias, "long": long_score, "short": short_score, "min": min_score}
    return SignalResult(
        symbol,
        bias,
        setup_score,
        reason,
        float(entry_px),
        float(stop_px),
        float(tp1) if tp1 is not None else None,
        float(tp2) if tp2 is not None else None,
        last_price,
        last_ts,
        session,
        extras,
    )

def _slip_amount(*, slippage_mode: str, fixed_slippage_cents: float, atr_last: float, atr_fraction_slippage: float) -> float:
    """Return slippage amount in price units (not percent)."""
    try:
        mode = (slippage_mode or "Off").strip()
    except Exception:
        mode = "Off"

    if mode == "Off":
        return 0.0

    if mode == "Fixed cents":
        try:
            return max(0.0, float(fixed_slippage_cents)) / 100.0
        except Exception:
            return 0.0

    if mode == "ATR fraction":
        try:
            return max(0.0, float(atr_last)) * max(0.0, float(atr_fraction_slippage))
        except Exception:
            return 0.0

    return 0.0
def _entry_from_model(
    direction: str,
    *,
    entry_model: str,
    last_price: float,
    ref_vwap: float | None,
    mid_price: float | None,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> float:
    """Compute an execution-realistic entry based on the selected entry model."""
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=atr_last,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    model = (entry_model or "Last price").strip()

    # 1) VWAP-based: place a limit slightly beyond VWAP in the adverse direction (more realistic fills).
    if model == "VWAP reclaim limit" and isinstance(ref_vwap, (float, int)):
        return (float(ref_vwap) + slip) if direction == "LONG" else (float(ref_vwap) - slip)

    # 2) Midpoint of the last completed bar
    if model == "Midpoint (last closed bar)" and isinstance(mid_price, (float, int)):
        return (float(mid_price) + slip) if direction == "LONG" else (float(mid_price) - slip)

    # 3) Default: last price with slippage in the adverse direction
    return (float(last_price) + slip) if direction == "LONG" else (float(last_price) - slip)

# ===========================
# RIDE / Continuation signals
# ===========================

def _last_swing_level(series: pd.Series, *, kind: str, lookback: int = 60) -> float | None:
    """Return the most recent swing high/low level in the lookback window (excluding the last bar)."""
    if series is None or len(series) < 10:
        return None
    s = series.astype(float).tail(int(min(len(series), max(12, lookback))))
    flags = rolling_swing_highs(s, left=3, right=3) if kind == "high" else rolling_swing_lows(s, left=3, right=3)

    # exclude last bar (cannot be a confirmed pivot yet)
    flags = flags.iloc[:-1]
    s2 = s.iloc[:-1]

    idx = None
    for i in range(len(flags) - 1, -1, -1):
        if bool(flags.iloc[i]):
            idx = flags.index[i]
            break
    if idx is None:
        return None
    try:
        return float(s2.loc[idx])
    except Exception:
        return None


def compute_ride_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi5: pd.Series,
    rsi14: pd.Series,
    macd_hist: pd.Series,
    *,
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    entry_model: str = "Last price",
    slippage_mode: str = "Off",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,
    fib_lookback_bars: int = 200,
    killzone_preset: str = "none",
    target_atr_pct: float = 0.004,
    htf_bias: dict | None = None,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    tape_mode_enabled: bool = False,
    **_ignored: object,
) -> SignalResult:
    """Continuation / Drive signal family.

    Returns bias:
      - RIDE_LONG / RIDE_SHORT when trend + impulse/acceptance exists (actionable proximity)
      - CHOP when trend is insufficient or setup is not actionable yet
    """
    try:
        df = ohlcv.sort_index().copy()
    except Exception:
        df = ohlcv.copy()

    # interval mins
    try:
        interval_mins = int(str(interval).replace("min", "").strip())
    except Exception:
        interval_mins = 1

    # bar-closed guard (avoid partial last bar)
    df = _asof_slice(df, interval_mins, use_last_closed_only, bar_closed_guard)

    if df is None or len(df) < 60:
        return SignalResult(symbol, "CHOP", 0, "Not enough data for continuation scan.", None, None, None, None, None, None, "OFF", {"mode": "RIDE"})

    # attach indicators (aligned)
    df["rsi5"] = pd.to_numeric(rsi5.reindex(df.index).ffill(), errors="coerce")
    df["rsi14"] = pd.to_numeric(rsi14.reindex(df.index).ffill(), errors="coerce")
    df["macd_hist"] = pd.to_numeric(macd_hist.reindex(df.index).ffill(), errors="coerce")

    session = classify_session(df.index[-1])
    liquidity_phase = classify_liquidity_phase(df.index[-1])
    liquidity_mult = float(np.clip(0.75 + liquidity_weighting, 0.75, 1.25))

    last_ts = pd.to_datetime(df.index[-1])
    last_price = float(df["close"].iloc[-1])

    # VWAP reference
    vwap_sess = calc_session_vwap(df, include_premarket=session_vwap_include_premarket, include_afterhours=allow_afterhours)
    vwap_cum = calc_vwap(df)
    ref_vwap_series = vwap_sess if str(vwap_logic).lower() == "session" else vwap_cum
    ref_vwap = float(ref_vwap_series.iloc[-1]) if len(ref_vwap_series) else None

    # ATR + trend stats
    atr_s = calc_atr(df, period=14).reindex(df.index).ffill()
    atr_last = float(atr_s.iloc[-1]) if len(atr_s) else None
    if atr_last is None or not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = max(1e-6, float(df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()) / 10.0)

    close = df["close"].astype(float)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    adx, di_plus, di_minus = calc_adx(df, period=14)

    adx_ff = adx.reindex(df.index).ffill() if len(adx) else pd.Series(dtype=float)
    adx_last = float(adx_ff.iloc[-1]) if len(adx_ff) else float("nan")
    di_p = float(di_plus.reindex(df.index).ffill().iloc[-1]) if len(di_plus) else float("nan")
    di_m = float(di_minus.reindex(df.index).ffill().iloc[-1]) if len(di_minus) else float("nan")

    def _ride_adx_modifier(adx_series: pd.Series) -> tuple[float, str | None]:
        try:
            s = pd.to_numeric(adx_series, errors="coerce").dropna()
        except Exception:
            s = pd.Series(dtype=float)
        if s.empty:
            return 0.0, None
        last = float(s.iloc[-1])
        level_adj = 0.0
        if last < 20.0:
            level_adj = -5.0
        elif last < 25.0:
            level_adj = 0.0
        elif last < 30.0:
            level_adj = 3.0
        else:
            level_adj = 6.0
        slope_adj = 0.0
        if len(s) >= 3:
            a, b, c = float(s.iloc[-3]), float(s.iloc[-2]), float(s.iloc[-1])
            eps = 0.15
            if c > b + eps and b > a + eps:
                slope_adj = 2.0
            elif c < b - eps and b < a - eps:
                slope_adj = -2.0
        mod = float(np.clip(level_adj + slope_adj, -7.0, 8.0))
        note = None
        if mod > 0:
            note = f"ADX tailwind (+{mod:.0f})"
        elif mod < 0:
            note = f"ADX headwind ({mod:.0f})"
        return mod, note

    adx_modifier, adx_modifier_note = _ride_adx_modifier(adx_ff)

    adx_floor = 20.0 if interval_mins <= 1 else 18.0
    di_gap_floor = 6.0 if interval_mins <= 1 else 5.0

    pass_adx = bool(np.isfinite(adx_last) and adx_last >= adx_floor)
    pass_di_gap = bool(np.isfinite(di_p) and np.isfinite(di_m) and abs(di_p - di_m) >= di_gap_floor)
    pass_ema_up = bool(float(ema20.iloc[-1]) > float(ema50.iloc[-1]))
    pass_ema_dn = bool(float(ema20.iloc[-1]) < float(ema50.iloc[-1]))

    trend_votes = int(pass_adx) + int(pass_di_gap) + int(pass_ema_up or pass_ema_dn)
    trend_ok = trend_votes >= 2

    if not trend_ok:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason=f"Too choppy for RIDE (trend {trend_votes}/3).",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "adx": adx_last, "di_plus": di_p, "di_minus": di_m, "liquidity_phase": liquidity_phase},
        )

    # ORB / pivots / displacement
    levels = _session_liquidity_levels(df, interval_mins, orb_minutes)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    buffer = 0.15 * float(atr_last)

    orb_seq = _orb_three_stage(df, orb_high=orb_high, orb_low=orb_low, buffer=buffer, lookback_bars=60, accept_bars=2)
    swing_hi = _last_swing_level(df["high"], kind="high", lookback=60)
    swing_lo = _last_swing_level(df["low"], kind="low", lookback=60)

    # --- Impulse legitimacy inputs ---
    # We want more than a retail-visible line break. Reward sequence quality,
    # held reclaims, and breaks that emerge from compression/sweep context.
    close_s = df["close"].astype(float)
    high_s = df["high"].astype(float)
    low_s = df["low"].astype(float)
    open_s = df["open"].astype(float)

    last_range = float(high_s.iloc[-1] - low_s.iloc[-1])
    disp_ratio = float(last_range / max(1e-9, float(atr_last)))
    disp_ok = disp_ratio >= 1.2
    prev_close = float(close_s.iloc[-2])
    prev_low_min = float(low_s.iloc[-6:-1].min()) if len(low_s) >= 6 else float(low_s.iloc[:-1].min())
    prev_high_max = float(high_s.iloc[-6:-1].max()) if len(high_s) >= 6 else float(high_s.iloc[:-1].max())
    prior_rng = float(high_s.tail(6).max() - low_s.tail(6).min()) if len(df) >= 6 else float(last_range)
    compression_ok = bool(prior_rng <= 1.35 * float(atr_last))
    body_ratio = float(abs(close_s.iloc[-1] - open_s.iloc[-1]) / max(1e-9, last_range))
    close_pos = float((close_s.iloc[-1] - low_s.iloc[-1]) / max(1e-9, last_range))
    close_q_long = float(np.clip(close_pos, 0.0, 1.0))
    close_q_short = float(np.clip(1.0 - close_pos, 0.0, 1.0))

    # VWAP reclaim/reject legitimacy: cross + hold + sweep awareness.
    vwap_reclaim_cross = bool(ref_vwap is not None and prev_close <= ref_vwap and last_price > ref_vwap and disp_ok)
    vwap_reject_cross = bool(ref_vwap is not None and prev_close >= ref_vwap and last_price < ref_vwap and disp_ok)
    if ref_vwap is not None and len(close_s) >= 2:
        vwap_reclaim_hold = bool((close_s.tail(2) > float(ref_vwap) - 0.10 * buffer).all())
        vwap_reject_hold = bool((close_s.tail(2) < float(ref_vwap) + 0.10 * buffer).all())
    else:
        vwap_reclaim_hold = False
        vwap_reject_hold = False
    swept_low_then_reclaim = bool(ref_vwap is not None and low_s.iloc[-1] < (prev_low_min - 0.05 * atr_last) and last_price > ref_vwap and close_q_long >= 0.55)
    swept_high_then_reject = bool(ref_vwap is not None and high_s.iloc[-1] > (prev_high_max + 0.05 * atr_last) and last_price < ref_vwap and close_q_short >= 0.55)
    vwap_reclaim = bool(vwap_reclaim_cross and (vwap_reclaim_hold or swept_low_then_reclaim))
    vwap_reject = bool(vwap_reject_cross and (vwap_reject_hold or swept_high_then_reject))
    vwap_score_up = 0.0
    vwap_score_dn = 0.0
    if vwap_reclaim_cross:
        vwap_score_up += 0.45
    if vwap_reclaim_hold:
        vwap_score_up += 0.30
    if swept_low_then_reclaim:
        vwap_score_up += 0.25
    if vwap_reject_cross:
        vwap_score_dn += 0.45
    if vwap_reject_hold:
        vwap_score_dn += 0.30
    if swept_high_then_reject:
        vwap_score_dn += 0.25

    # Pivot legitimacy: break + hold + compression context.
    pivot_break_up = bool(swing_hi is not None and last_price > float(swing_hi) + buffer)
    pivot_break_dn = bool(swing_lo is not None and last_price < float(swing_lo) - buffer)
    pivot_hold_up = bool(swing_hi is not None and len(close_s) >= 2 and (close_s.tail(2) > float(swing_hi) + 0.05 * buffer).all())
    pivot_hold_dn = bool(swing_lo is not None and len(close_s) >= 2 and (close_s.tail(2) < float(swing_lo) - 0.05 * buffer).all())
    pivot_score_up = (0.45 if pivot_break_up else 0.0) + (0.30 if pivot_hold_up else 0.0) + (0.25 if (pivot_break_up and compression_ok and close_q_long >= 0.60) else 0.0)
    pivot_score_dn = (0.45 if pivot_break_dn else 0.0) + (0.30 if pivot_hold_dn else 0.0) + (0.25 if (pivot_break_dn and compression_ok and close_q_short >= 0.60) else 0.0)

    # ORB legitimacy: sequence beats break-only.
    orb_break_up = bool(orb_high is not None and orb_seq.get("bull_break") and last_price > float(orb_high) + buffer)
    orb_break_dn = bool(orb_low is not None and orb_seq.get("bear_break") and last_price < float(orb_low) - buffer)
    orb_accept_up = bool(orb_high is not None and orb_break_up and len(close_s) >= 2 and (close_s.tail(2) > float(orb_high) + 0.05 * buffer).all())
    orb_accept_dn = bool(orb_low is not None and orb_break_dn and len(close_s) >= 2 and (close_s.tail(2) < float(orb_low) - 0.05 * buffer).all())
    orb_retest_up = bool(orb_seq.get("bull_orb_seq"))
    orb_retest_dn = bool(orb_seq.get("bear_orb_seq"))
    orb_score_up = (0.35 if orb_break_up else 0.0) + (0.30 if orb_accept_up else 0.0) + (0.35 if orb_retest_up else 0.0)
    orb_score_dn = (0.35 if orb_break_dn else 0.0) + (0.30 if orb_accept_dn else 0.0) + (0.35 if orb_retest_dn else 0.0)

    impulse_scores_long = {"ORB": float(np.clip(orb_score_up, 0.0, 1.0)), "PIVOT": float(np.clip(pivot_score_up, 0.0, 1.0)), "VWAP": float(np.clip(vwap_score_up, 0.0, 1.0))}
    impulse_scores_short = {"ORB": float(np.clip(orb_score_dn, 0.0, 1.0)), "PIVOT": float(np.clip(pivot_score_dn, 0.0, 1.0)), "VWAP": float(np.clip(vwap_score_dn, 0.0, 1.0))}
    long_best_type = max(impulse_scores_long, key=impulse_scores_long.get)
    short_best_type = max(impulse_scores_short, key=impulse_scores_short.get)
    long_legitimacy = float(impulse_scores_long.get(long_best_type, 0.0))
    short_legitimacy = float(impulse_scores_short.get(short_best_type, 0.0))

    long_legit_trigger = bool((orb_score_up > 0.45) or (pivot_score_up > 0.45) or vwap_reclaim)
    short_legit_trigger = bool((orb_score_dn > 0.45) or (pivot_score_dn > 0.45) or vwap_reject)
    impulse_long = bool(long_legit_trigger and long_legitimacy >= 0.45)
    impulse_short = bool(short_legit_trigger and short_legitimacy >= 0.45)

    if not impulse_long and not impulse_short:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason="Trend present but no legitimate impulse/drive signature yet.",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "liquidity_phase": liquidity_phase,
                    "long_legitimacy": long_legitimacy, "short_legitimacy": short_legitimacy},
        )

    direction = None
    if impulse_long and not impulse_short:
        direction = "LONG"
    elif impulse_short and not impulse_long:
        direction = "SHORT"
    else:
        dir_edge = float((di_p if np.isfinite(di_p) else 0.0) - (di_m if np.isfinite(di_m) else 0.0))
        long_total = long_legitimacy + (0.05 if dir_edge >= 0 else 0.0)
        short_total = short_legitimacy + (0.05 if dir_edge <= 0 else 0.0)
        direction = "LONG" if long_total >= short_total else "SHORT"
    impulse_legitimacy = long_legitimacy if direction == "LONG" else short_legitimacy
    impulse_type_hint = long_best_type if direction == "LONG" else short_best_type

    # Adaptive RIDE acceptance line:
    # weight structural anchors by source confidence, but also by *recently defended*
    # interaction quality so stale ORB/pivot references fade when VWAP/EMA20 becomes
    # the line the market is actually holding now.
    accept_components: Dict[str, float] = {}
    accept_component_weights: Dict[str, float] = {}
    accept_recent_diag: Dict[str, Dict[str, float | int]] = {}
    try:
        ema20_ref = float(ema20.iloc[-1]) if np.isfinite(ema20.iloc[-1]) else None
    except Exception:
        ema20_ref = None

    if direction == "LONG":
        if ref_vwap is not None and np.isfinite(ref_vwap):
            accept_components["VWAP"] = float(ref_vwap)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["VWAP"] = rx
            accept_component_weights["VWAP"] = max(0.05, 0.30 + 0.55 * float(np.clip(vwap_score_up, 0.0, 1.0)) + 0.20 * float(vwap_reclaim_hold) - 0.12 * float(vwap_reclaim_cross and not vwap_reclaim_hold) + 0.32 * float(rx.get("score") or 0.0))
        if orb_high is not None and np.isfinite(orb_high):
            accept_components["ORB"] = float(orb_high)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(orb_high), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["ORB"] = rx
            accept_component_weights["ORB"] = max(0.05, 0.28 + 0.55 * float(np.clip(orb_score_up, 0.0, 1.0)) + 0.18 * float(orb_accept_up) + 0.14 * float(orb_retest_up) - 0.14 * float(orb_break_up and not orb_accept_up) + 0.28 * float(rx.get("score") or 0.0))
        if swing_hi is not None and np.isfinite(swing_hi):
            accept_components["PIVOT"] = float(swing_hi)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(swing_hi), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["PIVOT"] = rx
            accept_component_weights["PIVOT"] = max(0.05, 0.24 + 0.55 * float(np.clip(pivot_score_up, 0.0, 1.0)) + 0.18 * float(pivot_hold_up) - 0.12 * float(pivot_break_up and not pivot_hold_up) + 0.26 * float(rx.get("score") or 0.0))
        if ema20_ref is not None and np.isfinite(ema20_ref):
            accept_components["EMA20"] = float(ema20_ref)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
            accept_recent_diag["EMA20"] = rx
            accept_component_weights["EMA20"] = 0.10 + 0.08 * float(trend_votes >= 2) + 0.18 * float(rx.get("score") or 0.0)
    else:
        if ref_vwap is not None and np.isfinite(ref_vwap):
            accept_components["VWAP"] = float(ref_vwap)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["VWAP"] = rx
            accept_component_weights["VWAP"] = max(0.05, 0.30 + 0.55 * float(np.clip(vwap_score_dn, 0.0, 1.0)) + 0.20 * float(vwap_reject_hold) - 0.12 * float(vwap_reject_cross and not vwap_reject_hold) + 0.32 * float(rx.get("score") or 0.0))
        if orb_low is not None and np.isfinite(orb_low):
            accept_components["ORB"] = float(orb_low)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(orb_low), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["ORB"] = rx
            accept_component_weights["ORB"] = max(0.05, 0.28 + 0.55 * float(np.clip(orb_score_dn, 0.0, 1.0)) + 0.18 * float(orb_accept_dn) + 0.14 * float(orb_retest_dn) - 0.14 * float(orb_break_dn and not orb_accept_dn) + 0.28 * float(rx.get("score") or 0.0))
        if swing_lo is not None and np.isfinite(swing_lo):
            accept_components["PIVOT"] = float(swing_lo)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(swing_lo), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["PIVOT"] = rx
            accept_component_weights["PIVOT"] = max(0.05, 0.24 + 0.55 * float(np.clip(pivot_score_dn, 0.0, 1.0)) + 0.18 * float(pivot_hold_dn) - 0.12 * float(pivot_break_dn and not pivot_hold_dn) + 0.26 * float(rx.get("score") or 0.0))
        if ema20_ref is not None and np.isfinite(ema20_ref):
            accept_components["EMA20"] = float(ema20_ref)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
            accept_recent_diag["EMA20"] = rx
            accept_component_weights["EMA20"] = 0.10 + 0.08 * float(trend_votes >= 2) + 0.18 * float(rx.get("score") or 0.0)

    if accept_components:
        valid_keys = [
            k for k in accept_components
            if np.isfinite(accept_components.get(k, float("nan")))
            and np.isfinite(accept_component_weights.get(k, float("nan")))
            and float(accept_component_weights.get(k, 0.0)) > 0.0
        ]
        if valid_keys:
            anchors = np.asarray([accept_components[k] for k in valid_keys], dtype=float)
            weights = np.asarray([max(0.05, accept_component_weights[k]) for k in valid_keys], dtype=float)
            accept_line = float(np.average(anchors, weights=weights))
            if len(valid_keys) == 1:
                accept_src = str(valid_keys[0])
            else:
                ranked = sorted(
                    ((str(k), float(max(0.05, accept_component_weights.get(k, 0.0)))) for k in valid_keys),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                total_w = float(sum(w for _, w in ranked)) or 1.0
                top1_k, top1_w = ranked[0]
                top2_k, top2_w = ranked[1]
                top1_frac = top1_w / total_w
                top2_frac = top2_w / total_w
                gap_frac = top1_frac - top2_frac
                if top1_frac >= 0.56 and gap_frac >= 0.12:
                    accept_src = top1_k
                elif top1_frac >= 0.42 and top2_frac >= 0.25 and gap_frac <= 0.12:
                    accept_src = f"{top1_k}+{top2_k}"
                else:
                    accept_src = "BLEND"
        else:
            accept_line = float(ema20_ref if isinstance(ema20_ref, (float, int)) and np.isfinite(ema20_ref) else last_price)
            accept_src = "EMA20"
    else:
        accept_line = float(ema20_ref if isinstance(ema20_ref, (float, int)) and np.isfinite(ema20_ref) else last_price)
        accept_src = "EMA20"

    # --- Acceptance / retest logic ---
    # NOTE: In live tape, ORB/VWAP levels can be *valid* but still too far from price
    # to be a realistic pullback entry for a continuation scalp.
    #
    # Example: price is actionable via break trigger proximity, but the selected
    # accept line sits far away (stale ORB from earlier in the session). In these
    # cases we still want to surface the breakout plan, and we clamp the accept
    # line used for pullback bands into a sane ATR window around last.
    accept_line_raw = float(accept_line)
    if direction == "LONG":
        # accept line should be below last, but not absurdly far.
        lo = float(last_price - 1.20 * atr_last)
        hi = float(last_price - 0.05 * atr_last)
        accept_line = float(np.clip(accept_line_raw, lo, hi))
    else:
        # accept line should be above last, but not absurdly far.
        lo = float(last_price + 0.05 * atr_last)
        hi = float(last_price + 1.20 * atr_last)
        accept_line = float(np.clip(accept_line_raw, lo, hi))
    # Accept = closes remain on the correct side of the accept line.
    look = int(min(3, len(df) - 1))
    recent_closes = df["close"].astype(float).iloc[-look:]
    if direction == "LONG":
        accept_ok = bool((recent_closes > float(accept_line) - buffer).all())
    else:
        accept_ok = bool((recent_closes < float(accept_line) + buffer).all())

    # Retest/hold = within the last few bars, price *tests* the accept line band and holds.
    retest_look = int(min(6, len(df) - 1))
    recent_lows = df["low"].astype(float).iloc[-retest_look:]
    recent_highs = df["high"].astype(float).iloc[-retest_look:]
    if direction == "LONG":
        retest_seen = bool((recent_lows <= float(accept_line) + buffer).any())
        hold_ok = bool((recent_closes >= float(accept_line) - buffer).all())
    else:
        retest_seen = bool((recent_highs >= float(accept_line) - buffer).any())
        hold_ok = bool((recent_closes <= float(accept_line) + buffer).all())

    stage = "CONFIRMED" if (accept_ok and retest_seen and hold_ok) else "PRE"

    # volume pattern: impulse expansion + hold compression
    vol = df["volume"].astype(float)
    med30 = float(vol.tail(60).rolling(30).median().iloc[-1]) if len(vol) >= 30 else float(vol.median())
    vol_impulse = float(vol.iloc[-1])
    vol_hold = float(vol.tail(3).mean()) if len(vol) >= 3 else vol_impulse
    vol_ok = bool(med30 > 0 and (vol_impulse >= 1.5 * med30) and (vol_hold <= 1.1 * vol_impulse))

    # exhaustion guard
    r5 = float(df["rsi5"].iloc[-1]) if np.isfinite(df["rsi5"].iloc[-1]) else None
    r14 = float(df["rsi14"].iloc[-1]) if np.isfinite(df["rsi14"].iloc[-1]) else None
    exhausted = False
    if direction == "LONG" and r5 is not None and r14 is not None:
        exhausted = bool(r5 > 85 and r14 > 70)
    if direction == "SHORT" and r5 is not None and r14 is not None:
        exhausted = bool(r5 < 15 and r14 < 30)

    # RSI rideability context (not a trigger, a realism guard):
    # - Continuations are best when short-term momentum is strong *but not blown out*.
    # - We use RSI-5 for timing and RSI-14 as a validation/backdrop.
    rsi_q = 0.5
    try:
        if r5 is not None and r14 is not None:
            if direction == "LONG":
                # Ideal: RSI-5 45..78 with RSI-14 >= ~45.
                base = 1.0 if (45.0 <= r5 <= 78.0 and r14 >= 45.0) else 0.6
                # If it's very hot, require pullback/retest; otherwise reduce quality.
                if r5 >= 85.0 and r14 >= 70.0:
                    base = 0.2
                rsi_q = float(np.clip(base, 0.0, 1.0))
            else:
                base = 1.0 if (22.0 <= r5 <= 55.0 and r14 <= 55.0) else 0.6
                if r5 <= 15.0 and r14 <= 30.0:
                    base = 0.2
                rsi_q = float(np.clip(base, 0.0, 1.0))
    except Exception:
        rsi_q = 0.5

    # --- Impulse/Hold quality score (0..1) ---
    # We want Score 100 to *mean* something tradeable:
    #   - displacement strength
    #   - close in the direction of travel
    #   - impulse volume expansion + hold compression
    #   - (for CONFIRMED) accept+retest/hold quality
    try:
        close_pos = (float(df["close"].iloc[-1]) - float(df["low"].iloc[-1])) / max(1e-9, last_range)
    except Exception:
        close_pos = 0.5

    # Directional close quality: long wants close near highs; short near lows.
    close_q = float(close_pos) if direction == "LONG" else float(1.0 - close_pos)
    close_q = float(np.clip(close_q, 0.0, 1.0))

    disp_q = float(np.clip((disp_ratio - 1.0) / 1.5, 0.0, 1.0))  # ~0 at 1.0ATR, ~1 at 2.5ATR
    vol_q = 1.0 if vol_ok else 0.0
    retest_q = 1.0 if (stage == "CONFIRMED") else (0.5 if accept_ok else 0.0)
    impulse_quality = float(np.clip(0.25 * disp_q + 0.20 * close_q + 0.15 * vol_q + 0.15 * retest_q + 0.25 * float(np.clip(impulse_legitimacy, 0.0, 1.0)), 0.0, 1.0))

    # Fold in RSI rideability (timing realism). This does NOT create the signal;
    # it just prevents weak/overextended moves from scoring like perfect rides.
    # Keep it gentle: at most ~15% adjustment.
    impulse_quality = float(np.clip(impulse_quality * (0.85 + 0.15 * float(rsi_q)), 0.0, 1.0))

    # If we're exhausted, don't allow CONFIRMED without a retest/hold.
    if exhausted and stage == "CONFIRMED":
        stage = "PRE"

    # If the impulse/accept sequence is low quality, don't label it "rideable".
    # This keeps 100 scores from appearing on flimsy moves.
    if impulse_quality < 0.35:
        # Too weak to trade as continuation.
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0.0,
            reason="Not rideable (low impulse quality)",
            entry=None,
            stop=None,
            target_1r=None,
            target_2r=None,
            last_price=last_price,
            timestamp=last_ts,
            session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "impulse_quality": impulse_quality,
                    "disp_ratio": disp_ratio, "liquidity_phase": liquidity_phase},
        )

    # If quality is mediocre, allow PRE but not CONFIRMED.
    if stage == "CONFIRMED" and impulse_quality < 0.55:
        stage = "PRE"

    # --- Robustness guardrails ---
    # If the impulse/hold quality is weak, a RIDE alert isn't "rideable".
    # - Very weak quality -> CHOP (no alert)
    # - Borderline quality -> PRE only (avoid overconfident CONFIRMED labels)
    if impulse_quality < 0.35:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason=f"Trend present but impulse/hold quality too weak (Q={impulse_quality:.2f}).",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "adx": adx_last, "di_plus": di_p, "di_minus": di_m,
                   "liquidity_phase": liquidity_phase, "impulse_quality": impulse_quality, "disp_ratio": disp_ratio, "vol_ok": vol_ok,
                   "accept_src": accept_src, "accept_line": accept_line,
                   "accept_components": {k: float(v) for k, v in accept_components.items()},
                   "accept_component_weights": {k: float(v) for k, v in accept_component_weights.items()}},
        )
    if stage == "CONFIRMED" and impulse_quality < 0.55:
        stage = "PRE"

    # scoring (quality-weighted)
    pts = 0.0
    pts += 22.0  # base for being in a trend-filtered universe
    pts += 18.0 if pass_adx else 0.0
    pts += 12.0 if pass_di_gap else 0.0
    pts += 15.0 if (direction == "LONG" and pass_ema_up) or (direction == "SHORT" and pass_ema_dn) else 0.0

    # Impulse + acceptance are amplified by quality; weak impulses shouldn't look like 100s.
    pts += (26.0 * impulse_quality)
    pts += (14.0 * impulse_quality) if stage == "CONFIRMED" else (7.0 * impulse_quality)
    pts += (10.0 * liquidity_mult) if vol_ok else 0.0
    pts -= 12.0 if exhausted else 0.0

    htf_effect = 0.0
    htf_label = None
    if isinstance(htf_bias, dict) and "bias" in htf_bias:
        hb = str(htf_bias.get("bias", "")).upper()
        htf_label = hb or None
        if direction == "LONG":
            if hb in ("BULL", "BULLISH"):
                htf_effect = 6.0
            elif hb in ("BEAR", "BEARISH"):
                htf_effect = -5.0
        elif direction == "SHORT":
            if hb in ("BEAR", "BEARISH"):
                htf_effect = 6.0
            elif hb in ("BULL", "BULLISH"):
                htf_effect = -5.0
        pts += htf_effect

    score = _cap_score(pts)

    # --- Entries: pullback band (PB1/PB2) + break trigger ---
    # A single-line pullback is too brittle. Bands are more realistic for continuation execution.
    #
    # Phase 2 upgrade: keep payload names the same, but make the pullback band width adaptive
    # to ATR and impulse quality. Stronger impulses deserve shallower pullback bands; weaker
    # setups require deeper pullbacks before we call them attractive.
    #
    # Additional refinement: use the *final* RIDE score as the conviction signal for banding.
    # High-conviction (score >= 87) setups get tighter/shallower pullback bands; everything
    # else gets slightly deeper bands. We compute a provisional geometry to score the entry zone,
    # then rebuild the final band from the final score so the displayed setup and final score agree.
    #
    # Also: if the setup is actionable because we're *near the break trigger* (not near pullback),
    # we should not keep showing a stale pullback limit far away. In that case we surface a
    # breakout-style entry (stop/trigger + a small chase line).
    q_weak = float(np.clip(1.0 - float(impulse_quality), 0.0, 1.0))

    # Wire the existing slippage controls into RIDE in a controlled way.
    # This should only add small tactical elasticity to the executable entry / proximity checks;
    # it should not rewrite the overall trade thesis.
    slip_amt = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=float(atr_last or 0.0),
        atr_fraction_slippage=float(atr_fraction_slippage or 0.0),
    )
    entry_pad = float(min(max(0.0, float(slip_amt)), 0.30 * float(atr_last)))

    # IMPORTANT: break_trigger must be stable across refreshes.
    # Anchor it to the strongest legitimate impulse source, not just the first raw break we see.
    if direction == "LONG":
        if impulse_type_hint == "ORB" and isinstance(orb_high, (float, int)):
            impulse_type, impulse_level = "ORB", float(orb_high)
        elif impulse_type_hint == "PIVOT" and isinstance(swing_hi, (float, int)):
            impulse_type, impulse_level = "PIVOT", float(swing_hi)
        else:
            impulse_type, impulse_level = "VWAP", float(ref_vwap) if isinstance(ref_vwap, (float, int)) else float(accept_line)
    else:
        if impulse_type_hint == "ORB" and isinstance(orb_low, (float, int)):
            impulse_type, impulse_level = "ORB", float(orb_low)
        elif impulse_type_hint == "PIVOT" and isinstance(swing_lo, (float, int)):
            impulse_type, impulse_level = "PIVOT", float(swing_lo)
        else:
            impulse_type, impulse_level = "VWAP", float(ref_vwap) if isinstance(ref_vwap, (float, int)) else float(accept_line)

    impulse_idx: Optional[int] = None
    try:
        lvl = float(impulse_level)
        if direction == "LONG":
            crossed = (df["close"].astype(float) > lvl) & (df["close"].astype(float).shift(1) <= lvl)
        else:
            crossed = (df["close"].astype(float) < lvl) & (df["close"].astype(float).shift(1) >= lvl)
        cross_locs = np.flatnonzero(crossed.fillna(False).to_numpy())
        if len(cross_locs):
            impulse_idx = int(cross_locs[-1])
    except Exception:
        impulse_idx = None

    def _build_ride_entry_geometry(conviction_score: float) -> dict[str, object]:
        high_conviction_local = bool(float(conviction_score or 0.0) >= 87.0)

        # Make the pullback band adapt not only to weak/strong impulse quality,
        # but also to how well the move is actually being accepted/held.
        src_weight_raw = float(accept_component_weights.get(accept_src, 0.18)) if isinstance(accept_component_weights, dict) else 0.18
        src_conf_local = float(np.clip((src_weight_raw - 0.05) / 1.05, 0.0, 1.0))
        accept_quality_local = float(np.clip(
            0.42 * float(accept_ok) + 0.23 * float(retest_seen) + 0.35 * float(hold_ok),
            0.0,
            1.0,
        ))
        acceptance_loose_local = float(np.clip(1.0 - accept_quality_local, 0.0, 1.0))
        source_loose_local = float(np.clip(1.0 - src_conf_local, 0.0, 1.0))
        extension_from_accept_local = (
            max(0.0, float(last_price - accept_line)) if direction == "LONG" else max(0.0, float(accept_line - last_price))
        )
        extension_loose_local = float(np.clip(extension_from_accept_local / max(1e-9, 0.85 * float(atr_last)), 0.0, 1.0))

        width_bias_inner = (0.00 if high_conviction_local else 0.03)
        width_bias_outer = (0.00 if high_conviction_local else 0.05)
        width_bias_inner += 0.025 * acceptance_loose_local + 0.015 * source_loose_local + 0.010 * extension_loose_local
        width_bias_outer += 0.055 * acceptance_loose_local + 0.030 * source_loose_local + 0.025 * extension_loose_local
        width_bias_inner -= 0.015 * accept_quality_local + 0.010 * src_conf_local
        width_bias_outer -= 0.030 * accept_quality_local + 0.015 * src_conf_local
        width_bias_inner = float(np.clip(width_bias_inner, -0.02, 0.08))
        width_bias_outer = float(np.clip(width_bias_outer, -0.03, 0.14))

        pb_inner_mult_local = float(np.clip(0.18 + 0.14 * q_weak + width_bias_inner, 0.15, 0.35))
        pb_outer_mult_local = float(np.clip(0.42 + 0.30 * q_weak + width_bias_outer, 0.35, 0.80))
        pb_inner_local = float(pb_inner_mult_local * float(atr_last))
        pb_outer_local = float(pb_outer_mult_local * float(atr_last))

        if direction == "LONG":
            break_trigger_local = float(max(float(impulse_level), float(df["high"].iloc[impulse_idx]))) if impulse_idx is not None else float(impulse_level)
            pb1_local = float(accept_line) + pb_inner_local
            pb2_local = float(accept_line) - pb_outer_local
            pullback_entry_local = float(np.clip(float(accept_line), pb2_local, pb1_local))
            stop_mult_local = 0.55 if stage == "PRE" else 0.80
            stop_local = float(pullback_entry_local - stop_mult_local * atr_last)
        else:
            break_trigger_local = float(min(float(impulse_level), float(df["low"].iloc[impulse_idx]))) if impulse_idx is not None else float(impulse_level)
            pb1_local = float(accept_line) - pb_inner_local
            pb2_local = float(accept_line) + pb_outer_local
            pullback_entry_local = float(np.clip(float(accept_line), pb1_local, pb2_local))
            stop_mult_local = 0.55 if stage == "PRE" else 0.80
            stop_local = float(pullback_entry_local + stop_mult_local * atr_last)

        structure_phase_local = _classify_ride_structure_phase(
            direction=str(direction),
            df=df,
            accept_line=float(accept_line),
            break_trigger=float(break_trigger_local),
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        prefers_breakout_phase_local = bool(structure_phase_local in ("ACCEPT_AND_GO", "BREAK_AND_HOLD"))
        prefers_pullback_phase_local = bool(structure_phase_local in ("EXTEND_THEN_PULLBACK", "FAILED_EXTENSION"))
        multibar_extension_profile_local = _compute_multibar_extension_profile(
            df,
            direction=str(direction),
            atr_last=float(atr_last) if atr_last is not None else None,
            accept_line=float(accept_line),
        )

        prox_atr_local = 0.45
        try:
            hist_tail_prox = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(int(min(4, len(df)))), errors="coerce").dropna()
        except Exception:
            hist_tail_prox = pd.Series(dtype=float)
        strong_pressure_local = False
        try:
            if len(hist_tail_prox) >= 3:
                if direction == "LONG":
                    strong_pressure_local = bool(hist_tail_prox.iloc[-1] > hist_tail_prox.iloc[-2] > hist_tail_prox.iloc[-3])
                else:
                    strong_pressure_local = bool(hist_tail_prox.iloc[-1] < hist_tail_prox.iloc[-2] < hist_tail_prox.iloc[-3])
        except Exception:
            strong_pressure_local = False
        if bool(tape_mode_enabled) and strong_pressure_local and float(impulse_quality or 0.0) >= 0.48 and float(adx_modifier or 0.0) >= 1.0:
            prox_atr_local = 0.50
        if prefers_breakout_phase_local and float(impulse_quality or 0.0) >= 0.45 and not bool(multibar_extension_profile_local.get("path_stretched") or False):
            prox_atr_local = max(prox_atr_local, 0.50)
        elif prefers_pullback_phase_local:
            prox_atr_local = max(prox_atr_local, 0.48)
        prox_dist_local = prox_atr_local * float(atr_last) + entry_pad
        breakout_stale_mult_local = 0.50 if bool(multibar_extension_profile_local.get("path_stretched") or False) else 0.60
        if direction == "LONG":
            dist_pb_band_local = 0.0 if (last_price >= pb2_local and last_price <= pb1_local) else min(abs(last_price - pb2_local), abs(last_price - pb1_local))
            stale_breakout_local = bool(last_price > break_trigger_local + breakout_stale_mult_local * atr_last + entry_pad and dist_pb_band_local > prox_dist_local)
        else:
            dist_pb_band_local = 0.0 if (last_price <= pb2_local and last_price >= pb1_local) else min(abs(last_price - pb2_local), abs(last_price - pb1_local))
            stale_breakout_local = bool(last_price < break_trigger_local - breakout_stale_mult_local * atr_last - entry_pad and dist_pb_band_local > prox_dist_local)

        dist_br_local = abs(last_price - break_trigger_local)
        near_pullback_local = bool(dist_pb_band_local <= prox_dist_local)
        near_break_local = bool(dist_br_local <= prox_dist_local)
        if not near_break_local and prefers_breakout_phase_local and float(impulse_quality or 0.0) >= 0.45 and dist_br_local <= 1.10 * prox_dist_local and not stale_breakout_local and not bool(multibar_extension_profile_local.get("path_stretched") or False):
            near_break_local = True
        if not near_pullback_local and prefers_pullback_phase_local and dist_pb_band_local <= 1.10 * prox_dist_local:
            near_pullback_local = True
        actionable_local = bool((near_pullback_local or near_break_local) and not stale_breakout_local)
        elite_runaway_local = False

        breakout_ref_local = None
        if direction == "LONG":
            if impulse_type_hint == "ORB" and orb_high is not None:
                breakout_ref_local = float(orb_high)
            elif impulse_type_hint == "VWAP" and ref_vwap is not None:
                breakout_ref_local = float(ref_vwap)
            elif impulse_type_hint == "PIVOT" and swing_hi is not None:
                breakout_ref_local = float(swing_hi)
        else:
            if impulse_type_hint == "ORB" and orb_low is not None:
                breakout_ref_local = float(orb_low)
            elif impulse_type_hint == "VWAP" and ref_vwap is not None:
                breakout_ref_local = float(ref_vwap)
            elif impulse_type_hint == "PIVOT" and swing_lo is not None:
                breakout_ref_local = float(swing_lo)

        recent_break_closes_local = df["close"].astype(float).iloc[-int(min(2, len(df))):]
        breakout_acceptance_quality_local = {"accepted": bool(accept_ok), "clean_accept": bool(accept_ok), "rejection": False, "wick_ratio": 0.0, "close_finish": 0.5, "last_close_vs_ref": 0.0}
        if breakout_ref_local is not None and len(recent_break_closes_local):
            breakout_acceptance_quality_local = _compute_breakout_acceptance_quality(
                df,
                direction=str(direction),
                breakout_ref=float(breakout_ref_local),
                atr_last=float(atr_last) if atr_last is not None else None,
                buffer=float(buffer),
            )
            breakout_acceptance_ok_local = bool(breakout_acceptance_quality_local.get("accepted") or False)
            breakout_clean_accept_local = bool(breakout_acceptance_quality_local.get("clean_accept") or False)
        else:
            breakout_acceptance_ok_local = bool(accept_ok)
            breakout_clean_accept_local = bool(accept_ok)

        # Controlled ignition override:
        # When an elite continuation is clearly running away without offering a normal
        # pullback, allow a breakout-style entry instead of collapsing to CHOP solely
        # because legacy near-entry geometry was missed.
        try:
            last_bar_high_local = float(df["high"].iloc[-1])
            last_bar_low_local = float(df["low"].iloc[-1])
            last_bar_open_local = float(df["open"].iloc[-1])
            last_bar_close_local = float(df["close"].iloc[-1])
            last_bar_range_local = max(1e-9, last_bar_high_local - last_bar_low_local)
            body_ratio_local = abs(last_bar_close_local - last_bar_open_local) / last_bar_range_local
            if direction == "LONG":
                opposite_wick_ratio_local = max(0.0, last_bar_high_local - max(last_bar_open_local, last_bar_close_local)) / last_bar_range_local
                directional_close_local = (last_bar_close_local - last_bar_low_local) / last_bar_range_local
                directional_dom_local = float(di_p or 0.0) - float(di_m or 0.0)
                hold_above_accept_local = bool(float(df["low"].astype(float).tail(int(min(3, len(df)))).min()) >= float(accept_line) - 0.15 * float(atr_last))
                no_immediate_failure_local = bool(last_bar_close_local >= float(break_trigger_local) - 0.10 * float(atr_last))
            else:
                opposite_wick_ratio_local = max(0.0, min(last_bar_open_local, last_bar_close_local) - last_bar_low_local) / last_bar_range_local
                directional_close_local = (last_bar_high_local - last_bar_close_local) / last_bar_range_local
                directional_dom_local = float(di_m or 0.0) - float(di_p or 0.0)
                hold_above_accept_local = bool(float(df["high"].astype(float).tail(int(min(3, len(df)))).max()) <= float(accept_line) + 0.15 * float(atr_last))
                no_immediate_failure_local = bool(last_bar_close_local <= float(break_trigger_local) + 0.10 * float(atr_last))
            vol_impulse_local = bool(vol_ok or (med30 > 0 and float(vol.iloc[-1]) >= 1.8 * float(med30)))
            elite_runaway_local = bool(
                (not actionable_local)
                and (not near_pullback_local)
                and (not prefers_pullback_phase_local)
                and high_conviction_local and float(conviction_score or 0.0) >= 95.0
                and float(impulse_quality or 0.0) >= 0.68
                and float(impulse_legitimacy or 0.0) >= 0.62
                and directional_dom_local >= 8.0
                and vol_impulse_local
                and (structure_phase_local in ("BREAK_AND_HOLD", "ACCEPT_AND_GO"))
                and (not bool(multibar_extension_profile_local.get("path_stretched") or False))
                and (not bool(multibar_extension_profile_local.get("stalling") or False))
                and (not bool(multibar_extension_profile_local.get("fading") or False))
                and body_ratio_local >= 0.55
                and directional_close_local >= 0.62
                and opposite_wick_ratio_local <= 0.22
                and hold_above_accept_local
                and no_immediate_failure_local
                and (dist_br_local <= 1.75 * max(prox_dist_local, 0.35 * float(atr_last)))
                and (float(multibar_extension_profile_local.get("dist_accept_atr") or 0.0) <= 2.8)
            )
            if elite_runaway_local:
                near_break_local = True
                stale_breakout_local = False
                actionable_local = True
        except Exception:
            elite_runaway_local = False

        entry_mode_local = None
        entry_price_local = None
        chase_line_local = None
        entry_base_local = None
        tape_metrics_local = {"eligible": False, "readiness": 0.0, "tightening": 0.0, "structural_hold": 0.0, "pressure": 0.0, "release_proximity": 0.0}
        tape_score_bonus_local = 0
        tape_breakout_bias_bonus_local = 0
        tape_prefers_breakout_local = False
        tape_rejection_penalty_local = {"penalty": 0.0, "stuffing": False}
        tape_breakout_urgency_local = {"score": 0.0, "urgent": False}
        tape_pullback_unlikelihood_local = {"score": 0.0, "unlikely": False}
        breakout_extension_state_local = {"penalty": float(multibar_extension_profile_local.get("penalty") or 0.0), "extended": False, "exhausted": bool((multibar_extension_profile_local.get("penalty") or 0.0) >= 1.0), "dist_accept_atr": float(multibar_extension_profile_local.get("dist_accept_atr") or 0.0), "dist_vwap_atr": 0.0, "momentum_fade": bool(multibar_extension_profile_local.get("fading") or False), "stalling": bool(multibar_extension_profile_local.get("stalling") or False), "path_stretched": bool(multibar_extension_profile_local.get("path_stretched") or False)}
        breakout_bias_score_local = 0
        extension_penalty_local = float(breakout_extension_state_local.get("penalty") or 0.0)
        extension_exhausted_local = bool(breakout_extension_state_local.get("exhausted") or False)
        combined_rejection_penalty_local = 0.0
        hard_rejection_local = False
        soft_rejection_local = False
        if actionable_local:
            ext_above_accept = max(0.0, float(last_price - accept_line)) if direction == "LONG" else max(0.0, float(accept_line - last_price))
            ext_above_pullback = max(0.0, float(last_price - pb1_local)) if direction == "LONG" else max(0.0, float(pb1_local - last_price))
            breakout_extension_ok = bool(ext_above_accept <= 0.60 * float(atr_last) and ext_above_pullback <= 0.35 * float(atr_last))
            breakout_elite_pre = bool(stage == "PRE" and (impulse_type_hint in ("ORB", "VWAP")) and impulse_legitimacy >= 0.86 and accept_ok and breakout_acceptance_ok_local and breakout_clean_accept_local and breakout_extension_ok)
            breakout_confirmed_ok = bool(stage == "CONFIRMED" and (impulse_type_hint in ("ORB", "VWAP")) and impulse_legitimacy >= 0.72 and breakout_acceptance_ok_local and breakout_clean_accept_local and breakout_extension_ok)
            breakout_margin_ok = bool(dist_br_local <= (0.75 * max(dist_pb_band_local, 1e-9)))

            breakout_bias_score_local = 0
            try:
                hist_tail_local = pd.to_numeric(df["macd_hist"].tail(int(min(4, len(df)))), errors="coerce").dropna()
            except Exception:
                hist_tail_local = pd.Series(dtype=float)
            if breakout_acceptance_ok_local:
                breakout_bias_score_local += 1
                if breakout_clean_accept_local:
                    breakout_bias_score_local += 1
            if float(adx_modifier or 0.0) >= 4.0:
                breakout_bias_score_local += 2
            elif float(adx_modifier or 0.0) > 0.0:
                breakout_bias_score_local += 1
            if len(hist_tail_local) >= 3:
                if direction == "LONG":
                    if hist_tail_local.iloc[-1] > hist_tail_local.iloc[-2] > hist_tail_local.iloc[-3] and hist_tail_local.iloc[-1] > 0:
                        breakout_bias_score_local += 2
                    elif hist_tail_local.iloc[-1] > hist_tail_local.iloc[-2] > hist_tail_local.iloc[-3]:
                        breakout_bias_score_local += 1
                else:
                    if hist_tail_local.iloc[-1] < hist_tail_local.iloc[-2] < hist_tail_local.iloc[-3] and hist_tail_local.iloc[-1] < 0:
                        breakout_bias_score_local += 2
                    elif hist_tail_local.iloc[-1] < hist_tail_local.iloc[-2] < hist_tail_local.iloc[-3]:
                        breakout_bias_score_local += 1
            recent_hold_closes_local = df["close"].astype(float).tail(int(min(3, len(df))))
            if direction == "LONG":
                shallow_retests_local = bool(float(df["low"].astype(float).tail(int(min(4, len(df)))).min()) >= float(accept_line) - 0.18 * float(atr_last))
                hold_progress_local = bool((recent_hold_closes_local >= float(accept_line) - buffer).all())
            else:
                shallow_retests_local = bool(float(df["high"].astype(float).tail(int(min(4, len(df)))).max()) <= float(accept_line) + 0.18 * float(atr_last))
                hold_progress_local = bool((recent_hold_closes_local <= float(accept_line) + buffer).all())
            if shallow_retests_local and hold_progress_local:
                breakout_bias_score_local += 1
            if near_break_local and dist_pb_band_local > max(1.10 * prox_dist_local, 0.28 * float(atr_last)):
                breakout_bias_score_local += 1

            if tape_mode_enabled:
                tape_metrics_local = _compute_tape_readiness(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    release_level=float(break_trigger_local),
                    structural_level=float(accept_line),
                    trigger_near=bool(near_break_local or (dist_br_local <= 1.20 * prox_dist_local)),
                    baseline_ok=bool(
                        (impulse_quality >= 0.45)
                        and (impulse_legitimacy >= 0.52)
                        and (accept_ok or hold_ok or breakout_acceptance_ok_local)
                        and (not stale_breakout_local)
                    ),
                )
                tape_score_bonus_local = _tape_bonus_from_readiness(
                    float(tape_metrics_local.get("readiness") or 0.0),
                    cap=4,
                    thresholds=(5.0, 6.0, 7.0, 8.0),
                )
                tape_rejection_penalty_local = _compute_release_rejection_penalty(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    release_level=float(break_trigger_local),
                )
                tape_breakout_urgency_local = _compute_breakout_urgency(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    release_level=float(break_trigger_local),
                )
                tape_pullback_unlikelihood_local = _compute_pullback_unlikelihood(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    accept_line=float(accept_line),
                )
                breakout_extension_state_local = _compute_breakout_extension_state(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    accept_line=float(accept_line),
                    ref_vwap=float(ref_vwap) if ref_vwap is not None else None,
                )
                breakout_extension_state_local["penalty"] = float(min(1.5, float(breakout_extension_state_local.get("penalty") or 0.0) + 0.55 * float(multibar_extension_profile_local.get("penalty") or 0.0)))
                breakout_extension_state_local["stalling"] = bool(breakout_extension_state_local.get("stalling") or multibar_extension_profile_local.get("stalling"))
                breakout_extension_state_local["momentum_fade"] = bool(breakout_extension_state_local.get("momentum_fade") or multibar_extension_profile_local.get("fading"))
                breakout_extension_state_local["path_stretched"] = bool(multibar_extension_profile_local.get("path_stretched") or False)
                acceptance_rejection_flag_local = bool(breakout_acceptance_quality_local.get("rejection") or False)
                acceptance_wick_local = float(breakout_acceptance_quality_local.get("wick_ratio") or 0.0)
                acceptance_close_finish_local = float(breakout_acceptance_quality_local.get("close_finish") or 0.5)
                acceptance_penalty_local = 0.0
                if acceptance_rejection_flag_local:
                    acceptance_penalty_local = 0.65 if (acceptance_wick_local < 0.42 and acceptance_close_finish_local >= 0.42) else 1.0
                combined_rejection_penalty_local = float(max(
                    float(tape_rejection_penalty_local.get("penalty") or 0.0),
                    acceptance_penalty_local,
                ))
                extension_penalty_local = float(breakout_extension_state_local.get("penalty") or 0.0)
                extension_exhausted_local = bool(breakout_extension_state_local.get("exhausted") or False)
                hard_rejection_local = bool(combined_rejection_penalty_local >= 0.95)
                soft_rejection_local = bool((combined_rejection_penalty_local >= 0.35) and not hard_rejection_local)
                tape_breakout_ready_local = bool(
                    float(tape_metrics_local.get("readiness") or 0.0) >= 6.0
                    and float(tape_metrics_local.get("pressure") or 0.0) >= 1.25
                    and float(tape_metrics_local.get("release_proximity") or 0.0) >= 1.0
                    and near_break_local
                    and shallow_retests_local
                    and hold_progress_local
                    and breakout_acceptance_ok_local
                    and breakout_clean_accept_local
                    and breakout_extension_ok
                    and not stale_breakout_local
                    and not hard_rejection_local
                    and combined_rejection_penalty_local <= 0.65
                    and extension_penalty_local < 1.0
                    and not extension_exhausted_local
                )
                tape_tiebreak_breakout_local = bool(
                    tape_breakout_ready_local
                    and float(tape_breakout_urgency_local.get("score") or 0.0) >= 1.0
                    and float(tape_pullback_unlikelihood_local.get("score") or 0.0) >= 1.0
                    and dist_br_local <= min(max(dist_pb_band_local, 1e-9), 0.85 * prox_dist_local)
                )
                if tape_breakout_ready_local and bool(tape_breakout_urgency_local.get("urgent") or False):
                    tape_breakout_bias_bonus_local = 1
                if hard_rejection_local:
                    breakout_bias_score_local -= 2
                elif soft_rejection_local:
                    breakout_bias_score_local -= 1
                if extension_penalty_local >= 1.0:
                    breakout_bias_score_local -= 2
                elif extension_penalty_local >= 0.5:
                    breakout_bias_score_local -= 1
                breakout_bias_score_local += int(tape_breakout_bias_bonus_local)
                tape_prefers_breakout_local = bool(
                    tape_tiebreak_breakout_local
                    and bool(tape_pullback_unlikelihood_local.get("unlikely") or False)
                    and not hard_rejection_local
                    and combined_rejection_penalty_local < 0.80
                    and extension_penalty_local < 1.0
                    and not extension_exhausted_local
                )

            breakout_bias = bool((breakout_confirmed_ok or breakout_elite_pre) and near_break_local and not stale_breakout_local and breakout_margin_ok)
            if breakout_bias and (hard_rejection_local or extension_exhausted_local or float(extension_penalty_local) >= 1.0):
                breakout_bias = False
            if not breakout_bias:
                breakout_bias = bool(
                    near_break_local
                    and not stale_breakout_local
                    and breakout_bias_score_local >= (5 if stage == "PRE" else 4)
                    and breakout_acceptance_ok_local
                    and (breakout_clean_accept_local or breakout_bias_score_local >= 6)
                    and not hard_rejection_local
                    and not extension_exhausted_local
                    and float(extension_penalty_local) < 1.0
                )
            if not breakout_bias and tape_prefers_breakout_local and near_break_local and (near_pullback_local or breakout_bias_score_local >= 4) and not hard_rejection_local and not extension_exhausted_local and float(extension_penalty_local) < 1.0:
                breakout_bias = True
            if breakout_bias and ((soft_rejection_local and near_pullback_local and dist_pb_band_local <= 1.10 * prox_dist_local) or extension_exhausted_local or (float(extension_penalty_local) >= 1.0 and near_pullback_local)):
                breakout_bias = False

            if elite_runaway_local:
                choose_pullback = False
            elif near_pullback_local and not near_break_local:
                choose_pullback = True
            elif near_break_local and not near_pullback_local:
                choose_pullback = bool((not breakout_bias) or prefers_pullback_phase_local)
            elif near_pullback_local and near_break_local:
                if prefers_breakout_phase_local and breakout_bias and not extension_exhausted_local and float(extension_penalty_local) < 1.0:
                    choose_pullback = False
                elif prefers_pullback_phase_local:
                    choose_pullback = True
                else:
                    choose_pullback = bool(not breakout_bias)
            else:
                choose_pullback = bool(near_pullback_local or prefers_pullback_phase_local)

            if choose_pullback:
                entry_mode_local = "PULLBACK"
                entry_base_local = float(pullback_entry_local)
                entry_price_local = float(entry_base_local + entry_pad) if direction == "LONG" else float(entry_base_local - entry_pad)
                chase_line_local = float(break_trigger_local + entry_pad) if direction == "LONG" else float(break_trigger_local - entry_pad)
            else:
                entry_mode_local = "BREAKOUT"
                entry_base_local = float(max(break_trigger_local, last_price)) if direction == "LONG" else float(min(break_trigger_local, last_price))
                entry_price_local = float(entry_base_local + entry_pad) if direction == "LONG" else float(entry_base_local - entry_pad)
                chase_line_local = float(entry_price_local + 0.10 * atr_last + entry_pad) if direction == "LONG" else float(entry_price_local - 0.10 * atr_last - entry_pad)
                if direction == "LONG":
                    stop_local = float(min(stop_local, accept_line - 0.80 * atr_last))
                else:
                    stop_local = float(max(stop_local, accept_line + 0.80 * atr_last))

        return {
            "high_conviction": bool(high_conviction_local),
            "accept_quality": float(accept_quality_local),
            "accept_src_confidence": float(src_conf_local),
            "accept_extension_ratio": float(extension_loose_local),
            "pb_inner_mult": float(pb_inner_mult_local),
            "pb_outer_mult": float(pb_outer_mult_local),
            "break_trigger": float(break_trigger_local),
            "pb1": float(pb1_local),
            "pb2": float(pb2_local),
            "pullback_entry": float(pullback_entry_local),
            "stop": float(stop_local),
            "dist_pb_band": float(dist_pb_band_local),
            "dist_br": float(dist_br_local),
            "near_pullback": bool(near_pullback_local),
            "near_break": bool(near_break_local),
            "actionable": bool(actionable_local),
            "entry_mode": entry_mode_local,
            "entry_price": entry_price_local,
            "chase_line": chase_line_local,
            "breakout_acceptance_ok": bool(breakout_acceptance_ok_local),
            "breakout_bias_score": int(breakout_bias_score_local),
            "structure_phase": structure_phase_local,
            "elite_runaway": bool(elite_runaway_local),
            "extension_profile": multibar_extension_profile_local,
            "tape_readiness": float(tape_metrics_local.get("readiness") or 0.0),
            "tape_tightening": float(tape_metrics_local.get("tightening") or 0.0),
            "tape_hold": float(tape_metrics_local.get("structural_hold") or 0.0),
            "tape_pressure": float(tape_metrics_local.get("pressure") or 0.0),
            "tape_release_proximity": float(tape_metrics_local.get("release_proximity") or 0.0),
            "tape_score_bonus": int(tape_score_bonus_local),
            "tape_breakout_bias_bonus": int(tape_breakout_bias_bonus_local),
            "tape_prefers_breakout": bool(tape_prefers_breakout_local),
            "tape_rejection_penalty": float(tape_rejection_penalty_local.get("penalty") or 0.0),
            "tape_stuffing": bool(tape_rejection_penalty_local.get("stuffing") or False),
            "tape_breakout_urgency": float(tape_breakout_urgency_local.get("score") or 0.0),
            "tape_pullback_unlikelihood": float(tape_pullback_unlikelihood_local.get("score") or 0.0),
            "breakout_soft_rejection": bool(locals().get("soft_rejection_local", False)),
            "breakout_hard_rejection": bool(locals().get("hard_rejection_local", False)),
            "breakout_extension_penalty": float(breakout_extension_state_local.get("penalty") or 0.0),
            "breakout_extended": bool(breakout_extension_state_local.get("extended") or False),
            "breakout_exhausted": bool(breakout_extension_state_local.get("exhausted") or False),
            "breakout_dist_accept_atr": float(breakout_extension_state_local.get("dist_accept_atr") or 0.0),
            "breakout_dist_vwap_atr": float(breakout_extension_state_local.get("dist_vwap_atr") or 0.0),
            "breakout_momentum_fade": bool(breakout_extension_state_local.get("momentum_fade") or False),
            "breakout_stalling": bool(breakout_extension_state_local.get("stalling") or False),
            "prox_atr_local": float(prox_atr_local),
        }

    provisional_geometry = _build_ride_entry_geometry(score)
    ride_tape_bonus = int(provisional_geometry.get("tape_score_bonus") or 0) if tape_mode_enabled else 0
    if ride_tape_bonus:
        pts += float(ride_tape_bonus)
    ride_entry_context_px = float(provisional_geometry["entry_price"]) if isinstance(provisional_geometry.get("entry_price"), (float, int)) else (float(provisional_geometry["pullback_entry"]) if isinstance(provisional_geometry.get("pullback_entry"), (float, int)) else float(provisional_geometry["break_trigger"]))
    ride_zone_ctx = _evaluate_entry_zone_context(
        df, entry_price=ride_entry_context_px, direction=str(direction), atr_last=float(atr_last) if atr_last is not None else None, lookback=10
    )
    ride_zone_adj = 0.0
    fav_q = float(ride_zone_ctx.get("favorable_quality") or 0.0)
    host_q = float(ride_zone_ctx.get("hostile_quality") or 0.0)
    provisional_phase = str(provisional_geometry.get("structure_phase") or "UNSET")
    if bool(ride_zone_ctx.get("favorable")):
        ride_zone_adj += 4.0 + 3.0 * fav_q
        if provisional_phase in ("EXTEND_THEN_PULLBACK", "FAILED_EXTENSION") and str(provisional_geometry.get("entry_mode") or '').upper() == 'PULLBACK':
            ride_zone_adj += 1.0
    if bool(ride_zone_ctx.get("hostile")):
        hostile_pen = 6.0 + 4.0 * host_q
        if str(provisional_geometry.get("entry_mode") or '').upper() == 'BREAKOUT':
            hostile_pen += (2.5 + 2.0 * host_q)
        if provisional_phase in ("BREAK_AND_HOLD", "ACCEPT_AND_GO") and str(provisional_geometry.get("entry_mode") or '').upper() == 'BREAKOUT':
            hostile_pen += 1.0
        ride_zone_adj -= hostile_pen
    score = _cap_score(pts + ride_zone_adj + adx_modifier)

    final_geometry = _build_ride_entry_geometry(score)
    break_trigger = float(final_geometry["break_trigger"])
    pb1 = float(final_geometry["pb1"])
    pb2 = float(final_geometry["pb2"])
    pullback_entry = float(final_geometry["pullback_entry"])
    stop = float(final_geometry["stop"])
    dist_pb_band = float(final_geometry["dist_pb_band"])
    dist_br = float(final_geometry["dist_br"])
    near_pullback = bool(final_geometry["near_pullback"])
    near_break = bool(final_geometry["near_break"])
    actionable = bool(final_geometry["actionable"])
    entry_mode = final_geometry.get("entry_mode")
    entry_price = final_geometry.get("entry_price")
    chase_line = final_geometry.get("chase_line")
    breakout_acceptance_ok = bool(final_geometry["breakout_acceptance_ok"])
    high_conviction = bool(final_geometry["high_conviction"])
    pb_inner_mult = float(final_geometry["pb_inner_mult"])
    pb_outer_mult = float(final_geometry["pb_outer_mult"])

    ride_entry_context_px = float(entry_price) if isinstance(entry_price, (float, int)) else (float(pullback_entry) if isinstance(pullback_entry, (float, int)) else float(break_trigger))
    ride_zone_ctx = _evaluate_entry_zone_context(
        df, entry_price=ride_entry_context_px, direction=str(direction), atr_last=float(atr_last) if atr_last is not None else None, lookback=10
    )
    ride_zone_adj = 0.0
    fav_q = float(ride_zone_ctx.get("favorable_quality") or 0.0)
    host_q = float(ride_zone_ctx.get("hostile_quality") or 0.0)
    structure_phase = str(final_geometry.get("structure_phase") or "UNSET")
    if bool(ride_zone_ctx.get("favorable")):
        ride_zone_adj += 4.0 + 3.0 * fav_q
        if structure_phase in ("EXTEND_THEN_PULLBACK", "FAILED_EXTENSION") and str(entry_mode or '').upper() == 'PULLBACK':
            ride_zone_adj += 1.0
    if bool(ride_zone_ctx.get("hostile")):
        hostile_pen = 6.0 + 4.0 * host_q
        if str(entry_mode or '').upper() == 'BREAKOUT':
            hostile_pen += (2.5 + 2.0 * host_q)
        if structure_phase in ("BREAK_AND_HOLD", "ACCEPT_AND_GO") and str(entry_mode or '').upper() == 'BREAKOUT':
            hostile_pen += 1.0
        ride_zone_adj -= hostile_pen
        if str(entry_mode or '').upper() == 'BREAKOUT' and host_q >= 0.70:
            actionable = False
            entry_price = None
            chase_line = None
    score = _cap_score(pts + ride_zone_adj + adx_modifier)

    # --- Targets: structure-first + monotonicity ---
    # TP0 should be a *real* liquidity/structure level (not a tiny tick), and TP ordering
    # must be monotonic (TP0 -> TP1 -> TP2 in the trade direction).
    hold_rng = float(df["high"].tail(6).max() - df["low"].tail(6).min())
    min_step = max(0.60 * float(atr_last), 0.35 * float(hold_rng))

    if direction == "LONG":
        cands = [x for x in [levels.get("prior_high"), levels.get("premarket_high"), swing_hi] if isinstance(x, (float, int))]
        cands = [float(x) for x in cands if float(x) > break_trigger + 0.10 * atr_last]
        tp0 = float(min(cands)) if cands else float(break_trigger + 0.90 * atr_last)
        # ensure tp0 isn't a meaningless "tick" target
        if float(tp0) - float(last_price) < 0.25 * float(atr_last):
            tp0 = float(last_price + 0.80 * atr_last)

        tp1 = float(tp0 + max(min_step, 0.70 * hold_rng))
        tp2 = float(tp1 + max(1.00 * atr_last, 0.90 * hold_rng))
    else:
        cands = [x for x in [levels.get("prior_low"), levels.get("premarket_low"), swing_lo] if isinstance(x, (float, int))]
        cands = [float(x) for x in cands if float(x) < break_trigger - 0.10 * atr_last]
        tp0 = float(max(cands)) if cands else float(break_trigger - 0.90 * atr_last)
        if float(last_price) - float(tp0) < 0.25 * float(atr_last):
            tp0 = float(last_price - 0.80 * atr_last)

        tp1 = float(tp0 - max(min_step, 0.70 * hold_rng))
        tp2 = float(tp1 - max(1.00 * atr_last, 0.90 * hold_rng))

    # Optional runner target (TP3): simple, monotonic extension.
    if direction == "LONG":
        tp3 = float(tp2 + max(1.25 * atr_last, 1.10 * hold_rng))
    else:
        tp3 = float(tp2 - max(1.25 * atr_last, 1.10 * hold_rng))

    # ETA to TP0 (minutes)
    liq_factor = 1.0
    if str(liquidity_phase).upper() in ("AFTERHOURS", "PREMARKET"):
        liq_factor = 1.6
    elif str(liquidity_phase).upper() in ("MIDDAY",):
        liq_factor = 1.25
    elif str(liquidity_phase).upper() in ("OPENING", "POWER"):
        liq_factor = 0.9
    eta_min = None
    try:
        dist = abs(float(tp0) - float(last_price))
        bars = dist / max(1e-6, float(atr_last))
        eta_min = float(bars * float(interval_mins) * liq_factor)
    except Exception:
        eta_min = None

    why = []
    why.append(f"Trend {trend_votes}/3 (ADX {adx_last:.1f})")
    if adx_modifier_note:
        why.append(adx_modifier_note)
    why.append(f"Impulse: {impulse_type_hint} L={impulse_legitimacy:.2f}")
    try:
        if isinstance(ride_zone_ctx, dict):
            if ride_zone_ctx.get("favorable") and ride_zone_ctx.get("favorable_type"):
                why.append(f"Entry near {ride_zone_ctx.get('favorable_type')}")
            if ride_zone_ctx.get("hostile") and ride_zone_ctx.get("hostile_type"):
                why.append(f"Entry near {ride_zone_ctx.get('hostile_type')}")
                if str(entry_mode or '').upper() == 'BREAKOUT':
                    why.append("Hostile zone penalizes breakout")
    except Exception:
        pass
    why.append(f"Accept: {accept_src}" + (" + retest" if stage == "CONFIRMED" else ""))
    if vol_ok:
        why.append("Vol: expand→compress")
    if tape_mode_enabled and float(final_geometry.get("tape_readiness") or 0.0) >= 4.0:
        why.append(f"Tape R={float(final_geometry.get('tape_readiness') or 0.0):.1f}")
    if exhausted:
        why.append("Exhaustion guard")
    if not actionable:
        why.append("Not near entry lines yet")
    # compact quality hint
    why.append(f"Q={impulse_quality:.2f}")

    bias = "RIDE_LONG" if direction == "LONG" else "RIDE_SHORT"

    return SignalResult(
        symbol=symbol,
        bias=bias if actionable else "CHOP",
        setup_score=score,
        reason="; ".join(why),
        entry=float(entry_price) if (actionable and entry_price is not None) else None,
        stop=stop if actionable else None,
        target_1r=tp0 if actionable else None,
        target_2r=tp1 if actionable else None,
        last_price=last_price,
        timestamp=last_ts,
        session=session,
        extras={
            "mode": "RIDE",
            "stage": stage if actionable else None,
            "actionable": actionable,
            "accept_line": float(accept_line),
            "accept_src": accept_src,
            "accept_recent_diag": accept_recent_diag,
            "break_trigger": float(break_trigger),
            "pullback_entry": float(pullback_entry),
            "pb1": float(pb1),
            "pb2": float(pb2),
            "pb_inner_mult": float(pb_inner_mult),
            "pb_outer_mult": float(pb_outer_mult),
            "pb_high_conviction": bool(high_conviction),
            "tp0": float(tp0),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "entry_mode": entry_mode,
            "breakout_acceptance_ok": bool(breakout_acceptance_ok),
            "tape_mode_enabled": bool(tape_mode_enabled),
            "tape_readiness": float(final_geometry.get("tape_readiness") or 0.0),
            "tape_tightening": float(final_geometry.get("tape_tightening") or 0.0),
            "tape_hold": float(final_geometry.get("tape_hold") or 0.0),
            "tape_pressure": float(final_geometry.get("tape_pressure") or 0.0),
            "tape_release_proximity": float(final_geometry.get("tape_release_proximity") or 0.0),
            "tape_score_bonus": int(final_geometry.get("tape_score_bonus") or 0),
            "tape_breakout_bias_bonus": int(final_geometry.get("tape_breakout_bias_bonus") or 0),
            "tape_prefers_breakout": bool(final_geometry.get("tape_prefers_breakout") or False),
            "entry_zone_context": ride_zone_ctx,
            "entry_zone_score_adj": float(ride_zone_adj),
            "adx_score_adj": float(adx_modifier),
            "chase_line": float(chase_line) if chase_line is not None else None,
            "eta_tp0_min": eta_min,
            "liquidity_phase": liquidity_phase,
            "trend_votes": trend_votes,
            "adx": adx_last,
            "di_plus": di_p,
            "di_minus": di_m,
            "impulse_quality": impulse_quality,
            "disp_ratio": disp_ratio,
            "impulse_legitimacy": float(impulse_legitimacy),
            "orb_score": float(impulse_scores_long.get("ORB") if direction == "LONG" else impulse_scores_short.get("ORB")),
            "pivot_score": float(impulse_scores_long.get("PIVOT") if direction == "LONG" else impulse_scores_short.get("PIVOT")),
            "vwap_score": float(impulse_scores_long.get("VWAP") if direction == "LONG" else impulse_scores_short.get("VWAP")),
            "swept_low_then_reclaimed": bool(swept_low_then_reclaim),
            "swept_high_then_rejected": bool(swept_high_then_reject),
            "compression_ok": bool(compression_ok),
            "impulse_type": impulse_type,
            "structure_phase": structure_phase,
            "extension_profile": final_geometry.get("extension_profile"),
            "impulse_level": float(impulse_level),
            "impulse_idx": int(impulse_idx) if impulse_idx is not None else None,
            "break_anchor_fallback": bool(impulse_idx is None),
            "slippage_mode": slippage_mode,
            "entry_slip_amount": float(entry_pad),
            "entry_model": entry_model,
            "htf_bias_value": htf_label,
            "htf_bias_effect": float(htf_effect),
            "vwap_logic": vwap_logic,
            "session_vwap_include_premarket": session_vwap_include_premarket,
        },
    )

# =========================
# MSS / ICT (Strict) alerts
# =========================

def _last_pivot_level(df: pd.DataFrame, piv_bool: pd.Series, col: str, *, before_idx: int) -> Tuple[Optional[float], Optional[int]]:
    """Return the most recent pivot level and its index position strictly before `before_idx`."""
    try:
        idxs = np.where(piv_bool.values)[0]
        idxs = idxs[idxs < before_idx]
        if len(idxs) == 0:
            return None, None
        i = int(idxs[-1])
        return float(df[col].iloc[i]), i
    except Exception:
        return None, None


def _first_touch_after(df: pd.DataFrame, *, start_i: int, zone_low: float, zone_high: float) -> Optional[int]:
    """First index >= start_i where candle overlaps the zone."""
    try:
        h = df["high"].values
        l = df["low"].values
        for i in range(max(0, start_i), len(df)):
            if (l[i] <= zone_high) and (h[i] >= zone_low):
                return i
        return None
    except Exception:
        return None


def compute_mss_signal(
    symbol: str,
    df: pd.DataFrame,
    rsi5: Optional[pd.Series] = None,
    rsi14: Optional[pd.Series] = None,
    macd_hist: Optional[pd.Series] = None,
    *,
    interval: str = "1min",
    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    # VWAP config (for context + some POI ranking)
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    # Fib/vol knobs
    fib_lookback_bars: int = 240,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    target_atr_pct: float | None = None,
) -> SignalResult:
    """Strict MSS/ICT alert family.

    Philosophy:
      - Very selective: only fire when we can explicitly see
        raid -> displacement -> MSS break -> POI retest/accept.
      - Output is actionability-oriented (pullback band + trigger + monotonic targets).

    Returns SignalResult with bias in {MSS_LONG, MSS_SHORT, CHOP}.
    """

    if df is None or len(df) < 80:
        return SignalResult(symbol, "CHOP", 0, "Not enough data", None, None, None, None, None, None, None, {"family": "MSS"})

    # Use last closed bar if requested (prevents half-formed candle artifacts)
    dfx = df.copy()
    if use_last_closed_only and len(dfx) >= 2:
        dfx = dfx.iloc[:-1].copy()

    last_ts = dfx.index[-1]
    session = classify_session(last_ts)
    liquidity_phase = classify_liquidity_phase(last_ts)

    # Respect user time-of-day filters (same pattern as other engines).
    allow = {
        "OPENING": allow_opening,
        "MIDDAY": allow_midday,
        "POWER": allow_power,
        "PREMARKET": allow_premarket,
        "AFTERHOURS": allow_afterhours,
        "CLOSED": allow_afterhours,
    }.get(session, True)
    if not allow:
        return SignalResult(symbol, "CHOP", 0, f"Time filter blocks MSS ({session})", None, None, None, None, None, None, session, {"family": "MSS", "session": session})

    # --- Core series ---
    atr14 = calc_atr(dfx[["high", "low", "close"]], period=14)
    atr_last = float(atr14.iloc[-1]) if len(atr14) else 0.0

    # Vol normalization baseline (optional)
    atr_pct = float(atr_last / float(dfx["close"].iloc[-1])) if float(dfx["close"].iloc[-1]) else 0.0
    atr_score_scale = 1.0
    baseline_atr_pct = None
    if target_atr_pct is not None and isinstance(target_atr_pct, (float, int)) and target_atr_pct > 0:
        baseline_atr_pct = float(target_atr_pct)
        try:
            atr_score_scale = float(np.clip(baseline_atr_pct / max(atr_pct, 1e-9), 0.75, 1.25))
        except Exception:
            atr_score_scale = 1.0

    # --- Pivots: external (structure) + internal (MSS) ---
    ext_l = 6 if interval in ("1min", "5min") else 8
    ext_r = ext_l
    int_l = 2
    int_r = 2

    piv_low_ext = rolling_swing_lows(dfx["low"], left=ext_l, right=ext_r)
    piv_high_ext = rolling_swing_highs(dfx["high"], left=ext_l, right=ext_r)
    piv_low_int = rolling_swing_lows(dfx["low"], left=int_l, right=int_r)
    piv_high_int = rolling_swing_highs(dfx["high"], left=int_l, right=int_r)

    # --- Find most recent raid (liquidity sweep) ---
    raid_search = min(180, len(dfx) - 10)
    raid_i = None
    raid_side = None  # "bull" means swept lows
    raid_level = None

    # scan from near-end backwards for a clean sweep
    lows = dfx["low"].values
    highs = dfx["high"].values
    closes = dfx["close"].values

    for i in range(len(dfx) - 2, max(10, len(dfx) - raid_search), -1):
        # bullish raid: take external pivot low, wick below, close back above pivot (reclaim)
        pl, pl_i = _last_pivot_level(dfx, piv_low_ext, "low", before_idx=i)
        if pl is not None and pl_i is not None:
            if lows[i] < pl and closes[i] > pl:
                # require meaningful sweep size
                if atr_last > 0 and (pl - lows[i]) >= 0.15 * atr_last:
                    raid_i = i
                    raid_side = "bull"
                    raid_level = float(pl)
                    break
        # bearish raid: take external pivot high, wick above, close back below pivot
        ph, ph_i = _last_pivot_level(dfx, piv_high_ext, "high", before_idx=i)
        if ph is not None and ph_i is not None:
            if highs[i] > ph and closes[i] < ph:
                if atr_last > 0 and (highs[i] - ph) >= 0.15 * atr_last:
                    raid_i = i
                    raid_side = "bear"
                    raid_level = float(ph)
                    break

    if raid_i is None or raid_side is None:
        return SignalResult(symbol, "CHOP", 0, "No clean liquidity raid found", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "liquidity_phase": liquidity_phase})

    # --- Displacement after raid ---
    tr = (dfx["high"] - dfx["low"]).rolling(20).median().fillna(method="bfill")
    disp_i = None
    disp_ratio = None

    for j in range(raid_i + 1, min(len(dfx), raid_i + 15)):
        rng = float(dfx["high"].iloc[j] - dfx["low"].iloc[j])
        med = float(tr.iloc[j]) if float(tr.iloc[j]) else 0.0
        if med <= 0:
            continue
        body = float(abs(dfx["close"].iloc[j] - dfx["open"].iloc[j]))
        dr = rng / med
        # directionality
        bull_dir = dfx["close"].iloc[j] > dfx["open"].iloc[j]
        bear_dir = dfx["close"].iloc[j] < dfx["open"].iloc[j]
        if raid_side == "bull" and bull_dir and dr >= 1.35 and (body / max(rng, 1e-9)) >= 0.55:
            disp_i = j
            disp_ratio = dr
            break
        if raid_side == "bear" and bear_dir and dr >= 1.35 and (body / max(rng, 1e-9)) >= 0.55:
            disp_i = j
            disp_ratio = dr
            break

    if disp_i is None:
        return SignalResult(symbol, "CHOP", 0, "Raid found but no displacement", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "liquidity_phase": liquidity_phase, "raid_i": raid_i})

    # --- MSS break: break of internal pivot in displacement direction ---
    if raid_side == "bull":
        # internal pivot high between raid and displacement
        mss_level, mss_piv_i = _last_pivot_level(dfx, piv_high_int, "high", before_idx=disp_i)
        if mss_level is None:
            return SignalResult(symbol, "CHOP", 0, "No internal pivot for MSS", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF"})
        mss_break_i = None
        for k in range(disp_i, min(len(dfx), disp_i + 20)):
            if float(dfx["close"].iloc[k]) > float(mss_level):
                mss_break_i = k
                break
        if mss_break_i is None:
            return SignalResult(symbol, "CHOP", 0, "No MSS break yet", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "mss_level": float(mss_level)})
        bias = "MSS_LONG"
    else:
        mss_level, mss_piv_i = _last_pivot_level(dfx, piv_low_int, "low", before_idx=disp_i)
        if mss_level is None:
            return SignalResult(symbol, "CHOP", 0, "No internal pivot for MSS", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF"})
        mss_break_i = None
        for k in range(disp_i, min(len(dfx), disp_i + 20)):
            if float(dfx["close"].iloc[k]) < float(mss_level):
                mss_break_i = k
                break
        if mss_break_i is None:
            return SignalResult(symbol, "CHOP", 0, "No MSS break yet", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "mss_level": float(mss_level)})
        bias = "MSS_SHORT"

    # --- POI selection (order block / FVG / breaker) from raid->break window ---
    window_df = dfx.iloc[max(0, raid_i - 5): mss_break_i + 1].copy()

    poi_low = None
    poi_high = None
    poi_src = None

    # Order block
    try:
        ob = find_order_block(window_df, atr14.loc[window_df.index], side=("bull" if raid_side == "bull" else "bear"))
        if ob and isinstance(ob, dict):
            poi_low = float(ob.get("low"))
            poi_high = float(ob.get("high"))
            poi_src = "OB"
    except Exception:
        pass

    # FVG (prefer if present and tighter)
    try:
        fvg = detect_fvg(window_df)
        if fvg and isinstance(fvg, dict):
            fl = float(fvg.get("low"))
            fh = float(fvg.get("high"))
            if (poi_low is None) or (fh - fl) < (poi_high - poi_low):
                poi_low, poi_high, poi_src = fl, fh, "FVG"
    except Exception:
        pass

    # Breaker fallback
    if poi_low is None or poi_high is None:
        try:
            br = find_breaker_block(window_df, atr14.loc[window_df.index], side=("bull" if raid_side == "bull" else "bear"))
            if br and isinstance(br, dict):
                poi_low = float(br.get("low"))
                poi_high = float(br.get("high"))
                poi_src = "BREAKER"
        except Exception:
            pass

    if poi_low is None or poi_high is None:
        # last resort: midpoint of displacement candle
        poi_low = float(min(window_df["open"].iloc[-1], window_df["close"].iloc[-1]))
        poi_high = float(max(window_df["open"].iloc[-1], window_df["close"].iloc[-1]))
        poi_src = "DISP_BODY"

    poi_low, poi_high = float(min(poi_low, poi_high)), float(max(poi_low, poi_high))
    poi_mid = 0.5 * (poi_low + poi_high)

    # --- Retest + accept (CONFIRMED) ---
    touch_i = _first_touch_after(dfx, start_i=mss_break_i, zone_low=poi_low, zone_high=poi_high)
    retest_ok = touch_i is not None
    accept_ok = False
    if retest_ok:
        after = min(len(dfx) - 1, int(touch_i) + 3)
        if bias == "MSS_LONG":
            accept_ok = float(dfx["close"].iloc[after]) >= poi_mid
        else:
            accept_ok = float(dfx["close"].iloc[after]) <= poi_mid

    # --- Actionability: ATR-distance to POI band or trigger ---
    last_price = float(dfx["close"].iloc[-1])
    atr = max(atr_last, 1e-9)

    # trigger is the MSS break level (for long, above; for short, below)
    break_trigger = float(mss_level)
    dist_to_poi = 0.0
    if last_price < poi_low:
        dist_to_poi = (poi_low - last_price)
    elif last_price > poi_high:
        dist_to_poi = (last_price - poi_high)

    dist_to_trigger = abs(last_price - break_trigger)
    actionable_gate = (min(dist_to_poi, dist_to_trigger) <= 0.75 * atr)

    stage = None
    if actionable_gate:
        stage = "PRE"
        if retest_ok and accept_ok:
            stage = "CONFIRMED"

    # --- Entries / stops (strict + practical) ---
    pullback_entry = float(poi_mid)
    pb1 = float(poi_high)
    pb2 = float(poi_low)

    raid_extreme = float(dfx["low"].iloc[raid_i]) if bias == "MSS_LONG" else float(dfx["high"].iloc[raid_i])
    strict_stop = raid_extreme - 0.05 * atr if bias == "MSS_LONG" else raid_extreme + 0.05 * atr
    practical_stop = (poi_low - 0.10 * atr) if bias == "MSS_LONG" else (poi_high + 0.10 * atr)
    stop = float(practical_stop)

    # --- Targets (monotonic, structure-first) ---
    tp0 = None
    tp1 = None
    tp2 = None

    if bias == "MSS_LONG":
        # nearest internal pivot high above last
        candidates = []
        for i in np.where(piv_high_int.values)[0]:
            if i < len(dfx) and float(dfx["high"].iloc[i]) > last_price:
                candidates.append(float(dfx["high"].iloc[i]))
        candidates = sorted(set(candidates))
        tp0 = candidates[0] if candidates else float(last_price + 1.0 * atr)

        # next external pivot high (bigger pool)
        ext_cand = []
        for i in np.where(piv_high_ext.values)[0]:
            if i < len(dfx) and float(dfx["high"].iloc[i]) > float(tp0):
                ext_cand.append(float(dfx["high"].iloc[i]))
        ext_cand = sorted(set(ext_cand))
        tp1 = ext_cand[0] if ext_cand else float(tp0 + 1.0 * atr)

        # measured move from displacement
        disp_range = float(dfx["high"].iloc[disp_i] - dfx["low"].iloc[disp_i])
        tp2 = float(max(tp1, pullback_entry + max(disp_range, 1.2 * atr)))

    else:
        candidates = []
        for i in np.where(piv_low_int.values)[0]:
            if i < len(dfx) and float(dfx["low"].iloc[i]) < last_price:
                candidates.append(float(dfx["low"].iloc[i]))
        candidates = sorted(set(candidates), reverse=True)
        tp0 = candidates[0] if candidates else float(last_price - 1.0 * atr)

        ext_cand = []
        for i in np.where(piv_low_ext.values)[0]:
            if i < len(dfx) and float(dfx["low"].iloc[i]) < float(tp0):
                ext_cand.append(float(dfx["low"].iloc[i]))
        ext_cand = sorted(set(ext_cand), reverse=True)
        tp1 = ext_cand[0] if ext_cand else float(tp0 - 1.0 * atr)

        disp_range = float(dfx["high"].iloc[disp_i] - dfx["low"].iloc[disp_i])
        tp2 = float(min(tp1, pullback_entry - max(disp_range, 1.2 * atr)))

    # ensure monotonic ordering
    if bias == "MSS_LONG":
        tp0 = float(max(tp0, last_price))
        tp1 = float(max(tp1, tp0))
        tp2 = float(max(tp2, tp1))
    else:
        tp0 = float(min(tp0, last_price))
        tp1 = float(min(tp1, tp0))
        tp2 = float(min(tp2, tp1))

    # --- Score (quality-driven) ---
    score = 0.0
    why_bits = []

    # Raid quality (size)
    try:
        if raid_side == "bull":
            raid_size = float(raid_level - lows[raid_i])
        else:
            raid_size = float(highs[raid_i] - raid_level)
        raid_q = float(np.clip(raid_size / max(atr, 1e-9), 0.0, 2.0))
    except Exception:
        raid_q = 0.0

    score += 20.0 * min(1.0, raid_q)
    why_bits.append("Raid+reclaim")

    # Displacement quality
    dq = float(np.clip((disp_ratio or 0.0) / 2.0, 0.0, 1.0))
    score += 25.0 * dq
    why_bits.append("Displacement")

    # MSS break
    score += 20.0
    why_bits.append("MSS break")

    # POI quality
    if poi_src in ("FVG", "OB", "BREAKER"):
        score += 10.0
        why_bits.append(f"POI={poi_src}")

    # Retest/accept
    if retest_ok:
        score += 10.0
        why_bits.append("Retest")
    if accept_ok:
        score += 10.0
        why_bits.append("Accept")

    # RSI exhaustion guard (prevents buying top / selling bottom)
    if rsi5 is not None and rsi14 is not None:
        try:
            r5 = float(rsi5.iloc[-1])
            r14 = float(rsi14.iloc[-1])
            if bias == "MSS_LONG" and (r5 > 88 and r14 > 72):
                score -= 12.0
                why_bits.append("RSI exhausted")
            if bias == "MSS_SHORT" and (r5 < 12 and r14 < 28):
                score -= 12.0
                why_bits.append("RSI exhausted")
            else:
                score += 5.0
        except Exception:
            pass

    score *= float(atr_score_scale)
    score_i = _cap_score(score)

    actionable = stage in ("PRE", "CONFIRMED") and bias in ("MSS_LONG", "MSS_SHORT")

    reason = " ".join(why_bits)
    if stage is None:
        reason = reason + "; Too far from POI/trigger (ATR gating)"

    # ETA TP0 using same concept as other engines
    eta_min = None
    try:
        if atr_last > 0:
            dist = abs(float(tp0) - last_price)
            # rough minutes per ATR based on liquidity phase
            pace = 7.0 if liquidity_phase == "RTH" else 11.0
            eta_min = float(max(1.0, (dist / atr_last) * pace))
    except Exception:
        eta_min = None

    return SignalResult(
        symbol=symbol,
        # Keep the MSS family bias namespace intact so app-side routing/alerting
        # can key off (MSS_LONG/MSS_SHORT) without ambiguity.
        bias=bias if actionable else "CHOP",
        setup_score=score_i,
        reason=(f"MSS {stage or 'OFF'} — {reason}"),
        last_price=last_price,
        entry=pullback_entry if actionable else None,
        stop=stop if actionable else None,
        target_1r=float(tp0) if actionable else None,
        target_2r=float(tp1) if actionable else None,
        timestamp=last_ts,
        session=session,
        extras={
            "family": "MSS",
            "stage": stage,
            "actionable": actionable,
            "poi_src": poi_src,
            "pb1": pb1,
            "pb2": pb2,
            "pullback_entry": pullback_entry,
            "break_trigger": break_trigger,
            "strict_stop": float(strict_stop),
            "tp0": float(tp0),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "eta_tp0_min": eta_min,
            "liquidity_phase": liquidity_phase,
            "raid_i": int(raid_i),
            "disp_i": int(disp_i),
            "mss_level": float(mss_level),
            "disp_ratio": float(disp_ratio) if disp_ratio is not None else None,
            "atr_pct": atr_pct,
            "baseline_atr_pct": baseline_atr_pct,
            "atr_score_scale": atr_score_scale,
            "vwap_logic": vwap_logic,
            "session_vwap_include_premarket": session_vwap_include_premarket,
        },
    )
