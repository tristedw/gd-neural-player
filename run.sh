#!/bin/bash
# usage: ./run.sh <level> [extra play args], defaults to learning
cd "$(dirname "$0")"
LEVEL="${1:-seven_seas}"
shift
ARGS="${@:---learn}"
.venv/bin/python -u -m gdbot.play --level "$LEVEL" $ARGS
