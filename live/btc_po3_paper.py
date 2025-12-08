<<<<<<< HEAD
import sys
import os

# --------------------------------------------------------------------
# Ensure project root (C:\Python312\quant_engine) is on sys.path
# --------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
=======
# live/btc_po3_paper.py
#
# LIVE / PAPER VERSION OF BTC PO3 ENGINE
# -------------------------------------
# This script:
#   1) Pulls recent BTC/USD 5m candles from TwelveData
#   2) Builds the institutional multi-timeframe feature frame
#   3) Instantiates BTCPO3PowerStrategy (same logic as backtest)
#   4) Evaluates the latest bar and prints LONG / SHORT / FLAT decision
#
# No real trading happens here. This is a "live backtest" / signal monitor.

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Any
>>>>>>> origin/phase5-features

import numpy as np
import pandas as pd

<<<<<<< HEAD
# Strategy + config from your BTC PO3 module
from engine.strategies.smc_po3_power_btc import BTCPO3PowerStrategy, BTCPO3PowerConfig

# Use the same BTC multi-timeframe builder as the backtest
from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional,
)

Signal = Literal["LONG", "SHORT", "HOLD"]

# --------------------------------------------------------------------
# RISK PARAMETERS — mirror backtest
# --------------------------------------------------------------------
STOP_ATR_MULT = 1.2      # how many ATR below/above entry for stop
TP_ATR_MULT = 2.5        # how many ATR above/below entry for take profit
MAX_HOLD_BARS = 96       # max bars in trade (~8 hours on 5m)
ATR_PERIOD = 14          # ATR lookback used for SL/TP


# ====================================================================
# CONFIG OBJECTS
# ====================================================================

@dataclass
class PaperEngineConfig:
    """
    Configuration for the BTC PO3 paper replay engine.

    Note: feature data is loaded by build_btc_5m_multiframe_features_institutional(),
    which internally reads data/5m_btc.csv through BTC_5M_CSV_PATH.
    """
    trades_csv_path: str = "live/btc_po3_paper_trades.csv"
    bar_interval_seconds: float = 0.0  # 0 => run as fast as possible
    log_every_n_bars: int = 50
    unit_size: float = 1.0
    strategy_config: BTCPO3PowerConfig = field(default_factory=BTCPO3PowerConfig)


# ====================================================================
# HELPER: COMPUTE ATR (COPIED FROM BACKTEST)
# ====================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute classic ATR from high/low/close.
    This is ONLY for SL/TP in the paper engine, and does not change strategy logic.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()

    return atr


# ====================================================================
# PAPER BROKER
# ====================================================================

class PaperBroker:
    """
    Minimal in-memory broker for a single symbol (BTC) and single position.

    It:
      - opens LONG/SHORT trades based on signals
      - uses ATR-based stop-loss / take-profit
      - enforces a max-hold in bars
      - supports flip exits (reverse signal) the SAME WAY as backtest:
        close on FLIP, do NOT auto-open new trade on same bar.
=======
# ---------------------------------------------------------------------------
# 0. Ensure project root is on sys.path
# ---------------------------------------------------------------------------
# When you run:
#   .venv\Scripts\python.exe live\btc_po3_paper.py
# Python's sys.path[0] is C:\Python312\quant_engine\live
# We need C:\Python312\quant_engine on the path so that `import engine...` works.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG: PROJECT_ROOT on sys.path ->", PROJECT_ROOT)

# Now imports like `from engine.features...` should work
from engine.features.btc_multiframe_features import build_btc_multiframe_features
from engine.strategies.smc_po3_power_btc import BTCPO3PowerStrategy


# ---------------------------------------------------------------------------
# 1. Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# 2. TwelveData 5m BTC fetcher
# ---------------------------------------------------------------------------
def fetch_btc_5m_from_twelvedata(
    api_key: str,
    symbol: str = "BTC/USD",
    interval: str = "5min",
    outputsize: int = 500,
) -> pd.DataFrame:
    """
    Fetch latest BTC/USD 5m candles from TwelveData.

    Returns a DataFrame indexed by timestamp (ascending), with columns:
        ["open", "high", "low", "close", "volume"]

    If anything fails, raises RuntimeError with a helpful message.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "order": "ASC",  # earliest -> latest
    }

    logging.info(
        "Requesting %s %s data from TwelveData (outputsize=%d)...",
        symbol,
        interval,
        outputsize,
    )
    resp = requests.get(url, params=params, timeout=10)

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to decode TwelveData JSON: {e}")

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    if "values" not in data:
        raise RuntimeError(f"Unexpected TwelveData payload: {data}")

    values = data["values"]
    if not values:
        raise RuntimeError("TwelveData returned no candles.")

    # Convert to DataFrame
    df = pd.DataFrame(values)

    # Expected keys from TwelveData: datetime, open, high, low, close, volume
    required_cols = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns from TwelveData: {missing}")

    # Convert dtypes
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df.sort_values("datetime").set_index("datetime")

    logging.info(
        "Fetched %d BTC 5m candles (from %s to %s).",
        len(df),
        df.index[0],
        df.index[-1],
    )

    return df


# ---------------------------------------------------------------------------
# 3. Feature builder wrapper
# ---------------------------------------------------------------------------
def build_live_btc_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Wraps your institutional feature builder so the live engine
    can call it exactly like the backtest.
    """
    logging.info(
        "Building BTC multiframe features for live frame (rows=%d)...",
        len(prices),
    )

    # This function name matches what your backtest already uses:
    #   build_btc_multiframe_features(prices)
    features = build_btc_multiframe_features(prices)

    logging.info(
        "Live BTC feature frame built. Final rows: %d (index from %s to %s).",
        len(features),
        features.index[0] if len(features) > 0 else "n/a",
        features.index[-1] if len(features) > 0 else "n/a",
    )
    return features


# ---------------------------------------------------------------------------
# 4. Live evaluation helper
# ---------------------------------------------------------------------------
def evaluate_latest_bar(
    features: pd.DataFrame,
    strategy: BTCPO3PowerStrategy,
) -> Dict[str, Any]:
    """
    Take the *latest* row of the feature frame and ask the strategy
    for a decision. Returns a dict summarizing the decision and
    context for logging/printing.
>>>>>>> origin/phase5-features
    """
    if features.empty:
        raise RuntimeError("Feature frame is empty – nothing to evaluate.")

<<<<<<< HEAD
    def __init__(
        self,
        unit_size: float,
        stop_atr_mult: float = STOP_ATR_MULT,
        tp_atr_mult: float = TP_ATR_MULT,
        max_hold_bars: int = MAX_HOLD_BARS,
    ):
        self.unit_size = unit_size

        # Risk parameters (same as backtest)
        self.stop_atr_mult = stop_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.max_hold_bars = max_hold_bars

        # Position state
        self.position: Optional[str] = None  # "LONG", "SHORT", or None
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[pd.Timestamp] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.bars_in_trade: int = 0

        # Accounting
        self.realized_pnl: float = 0.0
        self.trades: list[dict] = []

    # ----------------------------- internal helpers -----------------------------

    def _open_trade(self, ts: pd.Timestamp, direction: str, price: float, atr: float):
        self.position = direction
        self.entry_price = price
        self.entry_time = ts
        self.bars_in_trade = 0

        if direction == "LONG":
            self.stop_loss = price - self.stop_atr_mult * atr
            self.take_profit = price + self.tp_atr_mult * atr
        else:  # SHORT
            self.stop_loss = price + self.stop_atr_mult * atr
            self.take_profit = price - self.tp_atr_mult * atr

    def _close_trade(self, ts: pd.Timestamp, price: float, reason: str):
        if self.position is None or self.entry_price is None or self.entry_time is None:
            return

        # PnL in "price units" * unit_size (same as backtest)
        if self.position == "LONG":
            pnl = (price - self.entry_price) * self.unit_size
        else:
            pnl = (self.entry_price - price) * self.unit_size

        self.realized_pnl += pnl

        self.trades.append(
            {
                "entry_time": self.entry_time,
                "exit_time": ts,
                "direction": self.position,
                "entry_price": self.entry_price,
                "exit_price": price,
                "pnl": pnl,
                "bars_held": self.bars_in_trade,
                "exit_reason": reason,
            }
        )

        # Reset position state
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.bars_in_trade = 0

    # Public helper to mimic backtest's final "END" close
    def close_at_end(self, ts: pd.Timestamp, price: float):
        if self.position is not None:
            self._close_trade(ts, price, "END")

    # ------------------------------- public API ---------------------------------

    def on_bar(self, ts: pd.Timestamp, row: pd.Series, signal: Signal):
        """
        Handle one new bar:
          1) manage any open position (SL/TP/time/flip)
          2) if flat and signal LONG/SHORT => open new
        """
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr_bt"])  # <- backtest-style ATR

        # 1) Manage open trade, if any
        if self.position is not None:
            self.bars_in_trade += 1
            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None

            if self.position == "LONG":
                # 1) Check STOP first
                if self.stop_loss is not None and low <= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = "SL"
                # 2) Check TP
                elif self.take_profit is not None and high >= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = "TP"
            else:  # SHORT
                if self.stop_loss is not None and high >= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = "SL"
                elif self.take_profit is not None and low <= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = "TP"

            # 3) Max bars in trade (time stop)
            if exit_price is None and self.bars_in_trade >= self.max_hold_bars:
                exit_price = close
                exit_reason = "TIME"

            # 4) Flip on hard opposite signal – MATCH BACKTEST:
            # close trade, DO NOT open a new one on the same bar.
            if (
                exit_price is None
                and signal in ("LONG", "SHORT")
                and signal != self.position
            ):
                exit_price = close
                exit_reason = "FLIP"

            if exit_price is not None and exit_reason is not None:
                self._close_trade(ts, exit_price, exit_reason)
                # IMPORTANT: no immediate re-open on FLIP here,
                # so we behave like the backtest.

        # 2) If flat, and signal is directional, open a new trade
        if self.position is None and signal in ("LONG", "SHORT"):
            self._open_trade(ts, signal, close, atr)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


# ====================================================================
# PAPER ENGINE
# ====================================================================

class BTCPO3PaperEngine:
    """
    BTC 5m PO3 paper-replay engine.

    Reuses the same institutional feature builder and strategy as the
    backtest, but feeds bars one-by-one into a PaperBroker to simulate
    a live paper session.
    """

    def __init__(self, cfg: PaperEngineConfig):
        self.cfg = cfg
        self.strategy = BTCPO3PowerStrategy(cfg.strategy_config)
        self.broker = PaperBroker(
            unit_size=cfg.unit_size,
            stop_atr_mult=STOP_ATR_MULT,
            tp_atr_mult=TP_ATR_MULT,
            max_hold_bars=MAX_HOLD_BARS,
        )
        self.logger = logging.getLogger("BTCPO3PaperEngine")

    # ---------------------------- data / features ----------------------------

    def _build_features(self) -> pd.DataFrame:
        """
        Use the same institutional feature builder as the backtest.
        That builder loads data/5m_btc.csv internally.
        """
        self.logger.info(
            "Building BTC 5m features via build_btc_5m_multiframe_features_institutional..."
        )
        df, n_rows = build_btc_5m_multiframe_features_institutional()
        self.logger.info("Feature frame built with %d rows", n_rows)

        # Sanity check
        required_cols = ["open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in feature frame: {missing}")

        # Compute backtest-style ATR for risk management
        df = df.copy()
        df["atr_bt"] = compute_atr(df, period=ATR_PERIOD)

        return df

    # ------------------------------ main loop -------------------------------

    def run_replay(self) -> pd.DataFrame:
        df = self._build_features()
        total_bars = len(df)
        self.logger.info("Starting paper replay over %d bars", total_bars)

        sleep_s = max(self.cfg.bar_interval_seconds, 0.0)
        bars_with_atr = 0

        for i, (ts, row) in enumerate(df.iterrows(), start=1):
            atr = float(row.get("atr_bt", np.nan))

            # Skip bars where ATR not ready yet (same as backtest)
            if not np.isfinite(atr) or atr <= 0:
                continue

            bars_with_atr += 1

            # Strategy sees the FULL feature row as a dict, just like in backtest
            row_dict: Dict[str, Any] = row.to_dict()
            signal_str = self.strategy.on_bar(row_dict)
            signal: Signal = signal_str  # type: ignore

            # Feed bar + signal through the broker
            self.broker.on_bar(ts, row, signal)

            # Progress logging
            if i % self.cfg.log_every_n_bars == 0 or i == total_bars:
                self.logger.info(
                    "[%d/%d] ts=%s signal=%s position=%s realized_pnl=%.2f (bars_w_atr=%d)",
                    i,
                    total_bars,
                    ts,
                    signal,
                    self.broker.position,
                    self.broker.realized_pnl,
                    bars_with_atr,
                )

            if sleep_s > 0:
                time.sleep(sleep_s)

        # Mimic backtest's final "END" close if a trade is still open
        last_ts = df.index[-1]
        last_close = float(df.iloc[-1]["close"])
        self.broker.close_at_end(last_ts, last_close)

        trades_df = self.broker.to_dataframe()
        if not trades_df.empty:
            trades_df.to_csv(self.cfg.trades_csv_path, index=False)
            self.logger.info(
                "Paper session finished. Trades: %d | Realized PnL: %.2f | Saved to %s",
                len(trades_df),
                self.broker.realized_pnl,
                self.cfg.trades_csv_path,
            )
        else:
            self.logger.info("Paper session finished. No trades were taken.")
        return trades_df


# ====================================================================
# ENTRYPOINT
# ====================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Use the SAME config as the backtest
    strat_cfg = BTCPO3PowerConfig(
        # ATR% filter: ignore ultra-dead + insane-wick zones
        min_atr_pct=0.0008,      # 0.10% – avoid totally dead chop
        max_atr_pct=0.06,        # 6% – avoid insane liquidation spikes

        # RVOL filter: require at least "normal-ish" volume
        min_rvol=1.0,            # only trade when volume is decent

        # Session filters
        allow_asia=True,         # keep Asia for now
        allow_london=True,
        allow_ny=True,

        # Trend regime
        use_trend_regime=True,

        # Weekly position
        min_week_pos_for_longs=0.20,
        max_week_pos_for_shorts=0.80,

        # VWAP distance
        max_vwap_dist_atr_entry=2.0,

        # Sweep requirement
        require_sweep=False,
        lookback_sweep_bars=5,

        verbose=False,
    )

    engine_cfg = PaperEngineConfig(
        trades_csv_path="live/btc_po3_paper_trades.csv",
        bar_interval_seconds=0.0,  # set >0 for slow-motion replay
        unit_size=1.0,
        strategy_config=strat_cfg,
    )

    engine = BTCPO3PaperEngine(engine_cfg)
    engine.run_replay()
=======
    latest_idx = features.index[-1]
    latest_row = features.iloc[-1]

    # Depending on how your BTCPO3PowerStrategy is implemented,
    # on_bar may have different signatures.
    #
    # The two common patterns we used were:
    #   on_bar(idx, row)    OR    on_bar(row)
    #
    # Right now we assume on_bar(idx, row). If your actual implementation
    # only takes the row, we will adjust this call once we see a TypeError.
    decision = strategy.on_bar(latest_idx, latest_row)

    result: Dict[str, Any] = {
        "timestamp": latest_idx,
        "raw_decision": decision,
    }

    # If decision is a simple string like "LONG", "SHORT", "FLAT":
    if isinstance(decision, str):
        result["side"] = decision
    # If decision is a dict-like object with more detail:
    elif isinstance(decision, dict):
        result["side"] = decision.get("side") or decision.get("action")
        result["context"] = {
            k: v for k, v in decision.items() if k not in ("side", "action")
        }
    else:
        # Fallback – just store repr
        result["side"] = str(decision)

    return result


# ---------------------------------------------------------------------------
# 5. Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    # 5A. Load environment and API key
    load_dotenv()
    api_key = os.getenv("TWELVEDATA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TWELVEDATA_API_KEY not found in environment. "
            "Make sure it's set in your .env file."
        )

    # 5B. Fetch recent BTC 5m candles
    prices = fetch_btc_5m_from_twelvedata(api_key=api_key, outputsize=500)

    # 5C. Build institutional feature frame (same as backtest)
    features = build_live_btc_features(prices)

    # Drop any rows that are incomplete (NaNs from rolling windows, etc.)
    features = features.dropna()
    if features.empty:
        raise RuntimeError("All feature rows are NaN after dropna – check feature builder.")

    # 5D. Instantiate the BTC PO3 strategy
    strategy = BTCPO3PowerStrategy()

    logging.info("BTCPO3PowerStrategy instantiated for live evaluation.")

    # 5E. Evaluate the latest bar
    result = evaluate_latest_bar(features, strategy)

    ts = result.get("timestamp")
    side = result.get("side")
    raw_decision = result.get("raw_decision")
    context = result.get("context", {})

    logging.info("-------------------------------------------------------")
    logging.info("[LIVE] Latest bar timestamp: %s", ts)
    logging.info("[LIVE] Strategy side       : %s", side)
    logging.info("[LIVE] Raw decision object : %r", raw_decision)
    if context:
        logging.info("[LIVE] Extra context      : %r", context)
    logging.info("-------------------------------------------------------")

    # Also print a clean, one-line human summary:
    print(
        f"[LIVE] {ts} | Decision: {side} | "
        f"Context: {context if context else 'n/a'}"
    )
>>>>>>> origin/phase5-features


if __name__ == "__main__":
    main()
