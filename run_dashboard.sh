#!/bin/bash
cd /home/lars/stock-screener
source .venv/bin/activate

echo "$(date): Starting daily run" >> dashboard.log

# Refresh prices via Yahoo v8 (native on .113; was Windows-only previously)
python3 fill_prices_v8.py >> dashboard.log 2>&1

# Update quarterly fundamentals from SimFin API (incremental)
python3 update_fundamentals_quarterly.py --daily >> dashboard.log 2>&1

# Generate dashboard
python3 dashboard.py --freq quarterly --top-n 20 -o dashboard.html >> dashboard.log 2>&1

# Generate live signals
python3 live_signal.py --buy-threshold 40 > signals.txt 2>&1

echo "$(date): Done" >> dashboard.log
