#!/bin/bash
cd /home/lars/stock-screener
source .venv/bin/activate

while true; do
    DONE=$(python3 -c "import json; d=json.load(open('data/simfin/fundamentals_progress.json')); print(len(d['completed']))")
    TOTAL=4414
    echo "$(date): $DONE/$TOTAL complete" >> fundamentals_update.log

    if [ "$DONE" -ge 4400 ]; then
        echo "$(date): All done!" >> fundamentals_update.log
        break
    fi

    echo "$(date): Starting run..." >> fundamentals_update.log
    python3 update_fundamentals.py >> fundamentals_update.log 2>&1
    EXIT=$?

    # Check if we made progress
    NEW_DONE=$(python3 -c "import json; d=json.load(open('data/simfin/fundamentals_progress.json')); print(len(d['completed']))")

    if [ "$NEW_DONE" -gt "$DONE" ]; then
        # Made progress, short wait
        echo "$(date): Progress $DONE -> $NEW_DONE, retrying in 60s" >> fundamentals_update.log
        sleep 60
    else
        # No progress = rate limited hard, wait longer
        echo "$(date): No progress (stuck at $DONE), waiting 600s for rate limit reset" >> fundamentals_update.log
        sleep 600
    fi
done
