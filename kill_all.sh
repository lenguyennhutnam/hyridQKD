#!/usr/bin/env bash
# Manual kill switch: stop EVERYTHING this project spawns (GUI, collector,
# qline stack) and clear pid files. Run inside WSL:  bash kill_all.sh
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pkill -f "python3 $APP_DIR/main.py" 2>/dev/null
pkill -f "collector.py --csv" 2>/dev/null
bash "$APP_DIR/stop_stack.sh"
rm -f "$APP_DIR"/logs/pids/*.pid
echo "all qkd_app processes killed."
