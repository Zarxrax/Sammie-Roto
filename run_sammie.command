#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 app.py
echo "Press any key to close this window."
read -n 1 -s
