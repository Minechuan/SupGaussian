#!/bin/bash
# ── Interactive demo (opens a window) ──────────────────────────────────────
# Rotate: right-drag | Poke: left-drag | Materials: keys 1-9 or panel buttons
# Switch scene: B/X/M/C/T | Obstacle: O | Reset: R | Ellipsoid render: F | Quit: ESC
#
# Usage:
#   ./run_local.sh              # load the chair (real 3DGS), default material
#   ./run_local.sh chair metal  # chair as stiff metal (legs barely bend)
#   ./run_local.sh ball jelly   # procedural ball, jelly material
#
# Arg 1: object  -> a gs name in data/<name>/ (e.g. chair), OR a procedural
#                   scene (ball|box|multi|cylinder|torus). Default: chair.
# Arg 2: preset  -> jelly|rubber|putty|snow|liquid|plasticine|metal|wood|sand|foam
#
# Needs a GPU backend (mac: Vulkan/MoltenVK works out of the box).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
ENV="${PG_ENV:-py310}"          # conda env name; override: PG_ENV=myenv ./run_local.sh
OBJ="${1:-chair}"
PRESET="${2:-}"

PROC="ball box multi cylinder torus"
ARGS=()
if echo " $PROC " | grep -q " $OBJ "; then
    ARGS+=(--scene "$OBJ")
else
    ARGS+=(--gs "$OBJ" --gs-particles "${PG_PARTICLES:-8000}")
fi
[ -n "$PRESET" ] && ARGS+=(--preset "$PRESET")

echo "[run_local] env=$ENV object=$OBJ preset=${PRESET:-<from config/default>}"
exec conda run -n "$ENV" --no-capture-output python "$DIR/main.py" "${ARGS[@]}"
