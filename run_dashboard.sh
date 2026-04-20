#!/bin/bash
cd /home/lars/stock-screener
source .venv/bin/activate

echo "$(date): Starting daily run" >> dashboard.log

# Update fundamentals from SimFin API (incremental, ~2.5hrs first run, fast after)
python3 update_fundamentals.py >> dashboard.log 2>&1

# Generate dashboard
python3 dashboard.py --freq quarterly --top-n 20 -o dashboard.html >> dashboard.log 2>&1

# Generate live signals
python3 live_signal.py --buy-threshold 40 > signals.txt 2>&1

echo "$(date): Done" >> dashboard.log
