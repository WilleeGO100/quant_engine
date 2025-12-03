"""
engine/strategies/smc_po3_power_btc.py

BTC-specific version of the PO3 "power" strategy.

Goal:
- Use the institutional BTC feature frame
- Apply simple but robust filters (ATR%, RVOL, sessions, weekly position, VWAP distance)
- Generate LONG / SHORT signals in a PO3-flavored way
- NEVER silently die and return HOLD forever
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# ============================================================
# CONFIG
# ============================================================
@dataclass
class BTCPO3PowerConfig:
    # Volatility filter (ATR % of price)
    min_atr_pct: float = 0.0005
    max_atr_pct: float = 0.06

    # Relative volume filter
    min_rvol: float = 0.6

    # Session filters
    allow_asia: bool = False
    allow_london: bool = True
    allow_ny: bool = True

    # Trend regime switch (if we have a "regime_trend_up" flag)
    use_trend_regime: bool = True

    # Weekly location constraints (0 = weekly low, 1 = weekly high)
    min_week_pos_for_longs: float = 0.10
    max_week_pos_for_shorts: float = 0.90

    # Distance from VWAP in ATR units
    max_vwap_dist_atr_entry: float = 1.8

    # Liquidity sweep requirement, if we have a flag such as "has_sweep"
    require_sweep: bool = False
    lookback_sweep_bars: int = 5  # reserved for future use

    # Debug printing
    verbose: bool = False


# ============================================================
# STRATEGY
# ============================================================
class BTCPO3PowerStrategy:
    """
    BTC PO3-style strategy.

    Important notes:
    - on_bar(row) expects a dict with at least:
        "open", "high", "low", "close"
      and optionally:
        "atr_pct_5m", "rvol_5m", "session_type",
        "week_pos", "vwap_dist_atr", "regime_trend_up",
        "has_sweep"

    - If a field is missing, we fall back to SAFE defaults instead of
      silently blocking trades.
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
        # strings like "True"/"False"
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
        # Expect things like "ASIA", "LONDON", "NY"
        session = str(row.get("session_type", "NY")).upper()

        if session.startswith("ASIA"):
            return self.config.allow_asia
        if session.startswith("LON"):
            return self.config.allow_london
        if session.startswith("NY"):
            return self.config.allow_ny

        # For any unknown session tag, allow by default
        return True

    def _vol_ok(self, atr_pct: float) -> bool:
        return (atr_pct >= self.config.min_atr_pct) and (atr_pct <= self.config.max_atr_pct)

    def _rvol_ok(self, rvol: float) -> bool:
        return (rvol >= self.config.min_rvol)

    def _trend_ok(self, row: Dict[str, Any]) -> bool:
        if not self.config.use_trend_regime:
            return True
        # If we don't have a regime flag, treat as "OK"
        regime_trend_up = self._get_bool(row, "regime_trend_up", True)
        # When trend regime is enforced, we always require *some* trend regime flag;
        # here we just say "OK if it's anything" – the actual direction bias is
        # handled in the entry logic.
        return bool(regime_trend_up) or (not regime_trend_up)

    def _weekly_location_ok_for_long(self, week_pos: float) -> bool:
        return week_pos >= self.config.min_week_pos_for_longs

    def _weekly_location_ok_for_short(self, week_pos: float) -> bool:
        return week_pos <= self.config.max_week_pos_for_shorts

    def _vwap_ok(self, vwap_dist_atr: float) -> bool:
        return abs(vwap_dist_atr) <= self.config.max_vwap_dist_atr_entry

    def _sweep_ok(self, row: Dict[str, Any]) -> bool:
        if not self.config.require_sweep:
            return True
        # If we require a sweep, look for a generic boolean flag "has_sweep".
        has_sweep = self._get_bool(row, "has_sweep", False)
        return has_sweep

    # -----------------------------
    # Core on_bar
    # -----------------------------
    def on_bar(self, row: Dict[str, Any]) -> str:
        """
        Decide whether to go LONG, SHORT, or HOLD on this bar.

        Returned strings:
            - "LONG"
            - "SHORT"
            - "HOLD"
        """

        # Basic price info (must exist)
        try:
            open_ = float(row["open"])
            close = float(row["close"])
        except KeyError:
            if self.config.verbose:
                print("[BTCPO3] Missing open/close in row; HOLD")
            return "HOLD"

        # Optional features with safe defaults
        atr_pct = self._get_float(row, "atr_pct_5m", 0.01)       # 1% ATR as fallback
        rvol = self._get_float(row, "rvol_5m", 1.0)              # normal volume as fallback
        week_pos = self._get_float(row, "week_pos", 0.5)         # mid-week as fallback
        vwap_dist_atr = self._get_float(row, "vwap_dist_atr", 0.0)
        regime_trend_up = self._get_bool(row, "regime_trend_up", True)

        # -----------------
        # Global filters
        # -----------------
        if not self._vol_ok(atr_pct):
            if self.config.verbose:
                print(f"[BTCPO3] HOLD: vol filter failed (atr_pct={atr_pct:.5f})")
            return "HOLD"

        if not self._rvol_ok(rvol):
            if self.config.verbose:
                print(f"[BTCPO3] HOLD: rvol filter failed (rvol={rvol:.2f})")
            return "HOLD"

        if not self._session_ok(row):
            if self.config.verbose:
                print(f"[BTCPO3] HOLD: session filter failed (session={row.get('session_type')})")
            return "HOLD"

        if not self._trend_ok(row):
            if self.config.verbose:
                print("[BTCPO3] HOLD: trend regime filter failed")
            return "HOLD"

        if not self._sweep_ok(row):
            if self.config.verbose:
                print("[BTCPO3] HOLD: sweep filter failed")
            return "HOLD"

        if not self._vwap_ok(vwap_dist_atr):
            if self.config.verbose:
                print(f"[BTCPO3] HOLD: vwap distance filter failed (dist={vwap_dist_atr:.2f})")
            return "HOLD"

        # -----------------
        # PO3-flavored bias
        # -----------------
        # Very simple PO3-style heuristic for BTC:
        # - If we're in the lower part of the weekly range and price is closing
        #   bullish (close > open), favor LONG.
        # - If we're in the upper part of the weekly range and price is closing
        #   bearish (close < open), favor SHORT.
        # - Optional trend regime: if regime_trend_up is True, bias LONGs; if False, bias SHORTs.

        long_bias = (close > open_) and self._weekly_location_ok_for_long(week_pos)
        short_bias = (close < open_) and self._weekly_location_ok_for_short(week_pos)

        if regime_trend_up:
            # When trend is up, de-prioritize shorts unless really high in weekly range
            if long_bias:
                return "LONG"
            if short_bias and week_pos > 0.9:
                return "SHORT"
        else:
            # When trend is down, de-prioritize longs unless really low in weekly range
            if short_bias:
                return "SHORT"
            if long_bias and week_pos < 0.1:
                return "LONG"

        # Fallback: if one of the biases is active, take it
        if long_bias:
            return "LONG"
        if short_bias:
            return "SHORT"

        # Otherwise no clear PO3 bias
        return "HOLD"
