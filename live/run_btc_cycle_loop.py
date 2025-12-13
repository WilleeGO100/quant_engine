from __future__ import annotations

import os
import sys
import time
import subprocess
import logging
from datetime import datetime, timezone, timedelta

# Windows-only lock (prevents multiple instances)
try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------
# Paths (absolute, deterministic)
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")

UPDATE_SCRIPT = os.path.join(PROJECT_ROOT, "live", "update_btc_5m_csv_coinbase.py")
ENGINE_SCRIPT = os.path.join(PROJECT_ROOT, "live", "main_live_btc_csv.py")

LOCK_FILE = os.path.join(PROJECT_ROOT, "data", "btc_cycle_loop.lock")

# Timing
BAR_MINUTES = 5
BUFFER_SECONDS = int(os.getenv("BTC_LOOP_BUFFER_SECONDS", "12"))  # wait after close to avoid partials
ONCE_PER_BAR_GUARD_SECONDS = int(os.getenv("BTC_LOOP_MIN_GAP_SECONDS", "30"))  # extra safety

_last_run_utc: datetime | None = None


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)


def _acquire_lock_or_exit() -> object | None:
    """
    Prevent multiple loop instances.
    Creates/locks data/btc_cycle_loop.lock. If locked, exit.
    """
    if msvcrt is None:
        logging.warning("msvcrt not available; lock disabled. (This is unusual on Windows.)")
        return None

    _ensure_dirs()
    f = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        # lock 1 byte (non-blocking)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        f.seek(0)
        f.truncate(0)
        f.write(f"pid={os.getpid()} started_utc={datetime.now(timezone.utc).isoformat()}\n")
        f.flush()
        logging.info("Lock acquired: %s", LOCK_FILE)
        return f
    except OSError:
        logging.error("Another instance is already running (lock busy): %s", LOCK_FILE)
        try:
            f.close()
        except Exception:
            pass
        sys.exit(2)


def _run(cmd: list[str]) -> int:
    logging.info("Running: %s", " ".join(cmd))
    try:
        p = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return int(p.returncode)
    except Exception as e:
        logging.exception("Failed to run command: %s", e)
        return 1


def _next_5m_boundary_utc(now_utc: datetime) -> datetime:
    """
    Returns the next 5-minute boundary (e.g., :00, :05, :10...) in UTC.
    """
    # Normalize to minute precision
    now_utc = now_utc.replace(second=0, microsecond=0)
    minute = now_utc.minute
    # Compute minutes to add to reach next multiple of BAR_MINUTES
    add = BAR_MINUTES - (minute % BAR_MINUTES)
    if add == 0:
        add = BAR_MINUTES
    return now_utc + timedelta(minutes=add)


def _sleep_to_next_bar(buffer_seconds: int) -> datetime:
    """
    Sleep until next 5m boundary + buffer_seconds.
    Returns the target time (UTC) we aimed for.
    """
    now = datetime.now(timezone.utc)
    boundary = _next_5m_boundary_utc(now)
    target = boundary + timedelta(seconds=buffer_seconds)
    seconds = (target - now).total_seconds()

    if seconds < 0:
        seconds = 0

    logging.info(
        "Sleeping %.1fs until next %sm close + %ss (UTC target=%s)",
        seconds, BAR_MINUTES, buffer_seconds, target.isoformat()
    )
    time.sleep(seconds)
    return target


def _too_soon_guard(now_utc: datetime) -> bool:
    """
    Extra guard: don't run twice within ONCE_PER_BAR_GUARD_SECONDS.
    Helps if Task Scheduler or manual run collides.
    """
    global _last_run_utc
    if _last_run_utc is None:
        _last_run_utc = now_utc
        return False

    gap = (now_utc - _last_run_utc).total_seconds()
    if gap < ONCE_PER_BAR_GUARD_SECONDS:
        logging.warning("Guard tripped: last run was %.1fs ago (< %ss). Skipping this cycle.", gap, ONCE_PER_BAR_GUARD_SECONDS)
        return True

    _last_run_utc = now_utc
    return False


def main() -> None:
    lock_handle = _acquire_lock_or_exit()

    logging.info("BTC %sm cycle loop starting.", BAR_MINUTES)
    logging.info("Project: %s", PROJECT_ROOT)
    logging.info("Python : %s", PYTHON_EXE)
    logging.info("Update : %s", UPDATE_SCRIPT)
    logging.info("Engine : %s", ENGINE_SCRIPT)

    # First sleep aligns you to the NEXT closed bar (safe)
    _sleep_to_next_bar(buffer_seconds=BUFFER_SECONDS)

    while True:
        now_utc = datetime.now(timezone.utc)

        # Safety guard against duplicate invocations
        if _too_soon_guard(now_utc):
            _sleep_to_next_bar(buffer_seconds=BUFFER_SECONDS)
            continue

        # 1) Update CSV from Coinbase (closed candles only)
        rc_upd = _run([PYTHON_EXE, UPDATE_SCRIPT])
        if rc_upd != 0:
            logging.error("Updater failed (code=%s). Skipping engine this cycle.", rc_upd)
            _sleep_to_next_bar(buffer_seconds=BUFFER_SECONDS)
            continue

        # 2) Run CSV paper engine ONCE (this writes paper_orders/state if your code does)
        rc_eng = _run([PYTHON_EXE, ENGINE_SCRIPT])
        if rc_eng != 0:
            logging.error("CSV engine failed (code=%s). Will retry next cycle.", rc_eng)

        # Next bar
        _sleep_to_next_bar(buffer_seconds=BUFFER_SECONDS)

    # (Unreachable, but keep lock handle referenced)
    if lock_handle:
        pass


if __name__ == "__main__":
    main()
