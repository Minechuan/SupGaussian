#!/bin/bash
# Package the demo into a single zip to hand to teammates.
# Includes code + data/chair (the 65M .ply). Excludes caches and stray outputs.
# Usage: ./package.sh   ->  creates physgaussian_demo.zip in the parent dir.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="$(basename "$DIR")"
OUT="$DIR/../physgaussian_demo.zip"

rm -f "$OUT"
( cd "$DIR/.." && zip -r "$OUT" "$NAME" \
    -x "$NAME/__pycache__/*" \
    -x "$NAME/*.pyc" \
    -x "$NAME/imgui.ini" \
    -x "$NAME/*.mp4" \
    -x "$NAME/.DS_Store" \
    -x "$NAME/*/simulation_ply/*" \
    -x "$NAME/simulation_ply/*" \
    -x "$NAME/*_sim/*" )

echo "[package] wrote $OUT"
echo "[package] size: $(du -h "$OUT" | cut -f1)"
echo "[package] hand this zip to teammates; they unzip and follow QUICKSTART.md"
