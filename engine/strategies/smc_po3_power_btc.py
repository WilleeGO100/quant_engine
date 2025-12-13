# engine/strategies/smc_po3_power_btc.py
#
# BTC-specific PO3 "power" strategy using:
#   - ATR% volatility filter
#   - RVOL filter
#   - Session filter
#   - Weekly position
#   - VWAP distance (PREFERRED: % distance)
#   - Trend regime (EMA-based)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import math


# ============================================================
# CONFIG
# ============================================================
@dataclass
class BTCPO3PowerConfig:
    # Volatility filter (ATR % of price)
    min_atr_pct: float = 0.0005
    max_atr_pct: float = 0.06

    # Relative volume filter
    min_rvol: float = 0.25
    max_rvol: float | None = None  # optional upper cap if you ever want it

    # Session filters
    allow_asia: bool = False
    allow_london: bool = True
    allow_ny: bool = True

    # Trend regime switch (if we have a "regime_trend_up" flag)
    use_trend_regime: bool = True

    # Weekly location constraints (0 = weekly low, 1 = weekly high)
    min_week_pos_for_longs: float = 0.10
    max_week_pos_for_shorts: float = 0.90

    # VWAP filters
    # Preferred: percent distance from VWAP (stable for tuning)
    max_vwap_abs_pct: float = 0.15  # 3% default
    # Optional: ATR-units distance from VWAP (can explode on long histories; keep as secondary)
    max_vwap_dist_atr_entry: float = 1.8

    # Liquidity sweep requirement (we keep the knob but default to OFF)
    require_sweep: bool = False
    lookback_sweep_bars: int = 5

    verbose: bool = False


# ============================================================
# STRATEGY
# ============================================================
class BTCPO3PowerStrategy:
    """
    BTC PO3-style strategy.

    on_bar(row) expects at least:
        "open", "high", "low", "close"
    and ideally:
        "atr_pct_5m", "rvol_5m", "session_type",
        "week_pos", "vwap_dist_abs_pct", "vwap_dist_atr",
        "regime_trend_up", "has_sweep"
    """

    def __init__(self, config: BTCPO3PowerConfig) -> None:
        self.config = config

    # -----------------------------
    # Helpers to safely read fields
    # -----------------------------
    def _get_float(self, row: Dict[str, Any], key: str, default: float) -> float:
        val = row.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _get_bool(self, row: Dict[str, Any], key: str, default: bool) -> bool:
        val = row.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("true", "1", "yes", "y"):
                return True
            if v in ("false", "0", "no", "n"):
                return False
        return default

    # -----------------------------
    # Filters
    # -----------------------------
    def _session_ok(self, row: Dict[str, Any]) -> bool:
        # Expect "ASIA", "LONDON", "NY"
        session = str(row.get("session_type", "NY")).upper()

        if session.startswith("ASIA"):
            return self.config.allow_asia
        if session.startswith("LON"):
            return self.config.allow_london
        if session.startswith("NY"):
            return self.config.allow_ny

        # Unknown session tag → allow
        return True

    def _vol_ok(self, atr_pct: float) -> bool:
        if not math.isfinite(atr_pct):
            return False
        return (atr_pct >= self.config.min_atr_pct) and (atr_pct <= self.config.max_atr_pct)

    def _rvol_ok(self, rvol: float) -> bool:
        if not math.isfinite(rvol):
            return False
        if rvol < self.config.min_rvol:
            return False
        if self.config.max_rvol is not None and rvol > self.config.max_rvol:
            return False
        return True

    def _trend_ok(self, row: Dict[str, Any]) -> bool:
        if not self.config.use_trend_regime:
            return True
        # presence is enough; strategy uses the value later
        _ = self._get_bool(row, "regime_trend_up", True)
        return True

    def _sweep_ok(self, row: Dict[str, Any]) -> bool:
        if not self.config.require_sweep:
            return True
        has_sweep = self._get_bool(row, "has_sweep", False)
        return has_sweep

    def _vwap_ok_atr(self, vwap_dist_atr: float) -> bool:
        # Secondary/legacy VWAP filter in ATR units
        if not math.isfinite(vwap_dist_atr):
            return True  # don't kill trades if missing
        return abs(vwap_dist_atr) <= self.config.max_vwap_dist_atr_entry

    def _vwap_ok_pct(self, vwap_abs_pct: float) -> bool:
        # Preferred VWAP filter in percent (absolute distance)
        if not math.isfinite(vwap_abs_pct):
            return True
        # max_vwap_abs_pct is always a float in config; keep guard anyway
        if self.config.max_vwap_abs_pct is None:
            return True
        return vwap_abs_pct <= self.config.max_vwap_abs_pct

    def _weekly_location_ok_for_long(self, week_pos: float) -> bool:
        if not math.isfinite(week_pos):
            return True
        return week_pos >= self.config.min_week_pos_for_longs

    def _weekly_location_ok_for_short(self, week_pos: float) -> bool:
        if not math.isfinite(week_pos):
            return True
        return week_pos <= self.config.max_week_pos_for_shorts

    # -----------------------------
    # Core signal logic
    # -----------------------------
    def on_bar(self, row: Dict[str, Any]) -> str:
        cfg = self.config

        # Price
        try:
            open_ = float(row["open"])
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            if cfg.verbose:
                print("[BTCPO3] HOLD: missing open/close")
            return "HOLD"

        atr_pct = self._get_float(row, "atr_pct_5m", 0.0)
        rvol = self._get_float(row, "rvol_5m", 1.0)
        week_pos = self._get_float(row, "week_pos", 0.5)

        # VWAP features (preferred: abs pct; secondary: ATR units)
        vwap_abs_pct = self._get_float(row, "vwap_dist_abs_pct", float("nan"))
        vwap_dist_atr = self._get_float(row, "vwap_dist_atr", float("nan"))

        regime_trend_up = self._get_bool(row, "regime_trend_up", True)

        # -----------------
        # Filters (fast → slow)
        # -----------------
        if not self._session_ok(row):
            if cfg.verbose:
                print(f"[BTCPO3] HOLD: session filter failed ({row.get('session_type')})")
            return "HOLD"

        if not self._vol_ok(atr_pct):
            if cfg.verbose:
                print(f"[BTCPO3] HOLD: ATR%% filter failed ({atr_pct:.6f})")
            return "HOLD"

        if not self._rvol_ok(rvol):
            if cfg.verbose:
                print(f"[BTCPO3] HOLD: RVOL filter failed ({rvol:.3f})")
            return "HOLD"

        # NEW: VWAP distance filter (percent) — preferred
        if not self._vwap_ok_pct(vwap_abs_pct):
            if cfg.verbose:
                print(f"[BTCPO3] HOLD: vwap abs pct filter failed ({vwap_abs_pct:.4f} > {cfg.max_vwap_abs_pct:.4f})")
            return "HOLD"

        if not self._trend_ok(row):
            if cfg.verbose:
                print("[BTCPO3] HOLD: trend regime filter failed")
            return "HOLD"

        if not self._sweep_ok(row):
            if cfg.verbose:
                print("[BTCPO3] HOLD: sweep filter failed")
            return "HOLD"

        # Optional/secondary: VWAP distance (ATR units)
        # Keep this if you still want it; comment it out if you want VWAP% only.
        if not self._vwap_ok_atr(vwap_dist_atr):
            if cfg.verbose:
                print(f"[BTCPO3] HOLD: vwap ATR distance filter failed ({vwap_dist_atr:.3f})")
            return "HOLD"

        # -----------------
        # PO3-flavored bias
        # -----------------
        long_bias = (close > open_) and self._weekly_location_ok_for_long(week_pos)
        short_bias = (close < open_) and self._weekly_location_ok_for_short(week_pos)

        if cfg.use_trend_regime:
            if regime_trend_up:
                if long_bias:
                    return "LONG"
                if short_bias and week_pos > 0.9:
                    return "SHORT"
            else:
                if short_bias:
                    return "SHORT"
                if long_bias and week_pos < 0.1:
                    return "LONG"
        else:
            if long_bias:
                return "LONG"
            if short_bias:
                return "SHORT"

        return "HOLD"
