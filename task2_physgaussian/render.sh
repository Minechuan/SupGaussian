#!/bin/bash
# ── Offline render to mp4 (no window) ──────────────────────────────────────
# Simulates N frames, renders each to PNG, muxes to mp4 (needs ffmpeg; if
# missing it leaves the PNGs and prints the folder). Needs a GPU backend.
#
# Usage:
#   ./render.sh                     # chair, plasticine, 360 frames @720p -> chair.mp4
#   ./render.sh chair metal         # chair as metal -> chair.mp4
#   ./render.sh ball jelly out.mp4  # procedural ball, jelly
#
# Arg 1: object (gs name in data/<name>/, or ball|box|multi|cylinder|torus). Default chair.
# Arg 2: preset (optional; omit to use the object's config material).
# Arg 3: output mp4 path (optional; default <object>.mp4).
#
# Tunables via env:
#   PG_FRAMES (360)  PG_RES (720)  PG_PARTICLES (40000)  PG_FPS (30)
#   PG_HIFI=1        -> F-deformed ellipsoids (heavier, more faithful)
#   PG_ENV (py310)   -> conda env name
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
ENV="${PG_ENV:-py310}"
OBJ="${1:-chair}"
PRESET="${2:-}"
OUT="${3:-${OBJ}.mp4}"

PROC="ball box multi cylinder torus"
ARGS=(--headless --frames "${PG_FRAMES:-360}" --fps "${PG_FPS:-30}"
      --res "${PG_RES:-720}" --out "$OUT")
[ "${PG_HIFI:-0}" = "1" ] && ARGS+=(--hifi)
if echo " $PROC " | grep -q " $OBJ "; then
    ARGS+=(--scene "$OBJ")
else
    ARGS+=(--gs "$OBJ" --gs-particles "${PG_PARTICLES:-40000}")
fi
[ -n "$PRESET" ] && ARGS+=(--preset "$PRESET")

echo "[render] env=$ENV object=$OBJ preset=${PRESET:-<from config>} -> $OUT"
exec conda run -n "$ENV" --no-capture-output python "$DIR/main.py" "${ARGS[@]}"
