@echo off
cd /d C:\Python312\quant_engine
call .venv\Scripts\activate
python live\run_btc_cycle_loop.py >> data\btc_cycle_loop.log 2>&1
