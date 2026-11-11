#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
printf '\nOpen http://localhost:8765 in your browser.\n'
python3 server.py
