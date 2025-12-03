# main_live_loop.py
#
# Continuous live loop for the POWER SMC PO3 engine.
#
# Run with:
#   cd C:\Python312\quant_engine
#   python main_live_loop.py
#
# Stop with CTRL+C.

from __future__ import annotations

import time
from datetime import datetime

from config.config import load_config
from main_live import run_pipeline_once, LivePerformanceTracker


def main() -> None:
    print("============================================")
    print("[PO3_LOOP] Starting continuous live PO3 loop...")
    print("============================================")

    cfg = load_config()
    interval = int(getattr(cfg, "live_loop_interval_seconds", 60))

    print(f"[PO3_LOOP] Loop interval: {interval} seconds")
    print("[PO3_LOOP] Press CTRL+C to stop the loop.\n")

    tracker = LivePerformanceTracker()
    iteration = 0

    try:
        while True:
            iteration += 1
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

            print("--------------------------------------------------------")
            print(f"[PO3_LOOP] Iteration {iteration} at {now}")
            print("--------------------------------------------------------")

            try:
                run_pipeline_once(tracker=tracker)
            except Exception as e:
                print("[PO3_LOOP] ERROR during run_pipeline_once():", e)

            print(f"[PO3_LOOP] Sleeping {interval} seconds...\n")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[PO3_LOOP] Stopped by user (CTRL+C).")
        print("[PO3_LOOP] Final forward-test summary:")
        tracker.print_summary()
        print("[PO3_LOOP] Exiting cleanly.")


if __name__ == "__main__":
    main()
