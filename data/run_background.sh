#!/bin/bash

# EXACT name of your converted script
SCRIPT="tmux_getdata.py"

# Log file
LOG="background_run.log"

echo "Starting background job..."

eval "$(conda shell.bash hook)"

conda activate myenv

nohup /opt/miniconda3/envs/myenv/bin/python "$SCRIPT" > "$LOG" 2>&1 &
echo "Started with PID: $!"