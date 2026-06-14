#!/bin/bash
# PhysGaussian launcher (procedural scenes only — kept for back-compat).
#
# Prefer the newer scripts:
#   ./run_local.sh [object] [preset]   # interactive window; object can be a gs name (chair) or scene
#   ./render.sh    [object] [preset]   # offline render to mp4
# See QUICKSTART.md.
#
# Usage: ./run.sh [scene] [preset]
#   scene:  ball (default) | box | multi | cylinder | torus
#   preset: jelly | rubber | putty | snow | liquid | plasticine | metal | wood | sand | foam
#
# Env: PG_ARCH=cuda|vulkan|cpu  PG_NMAX=<cap>  PG_GRID=<res>  PG_ENV=<conda env>  (see README)

SCENE=${1:-ball}
PRESET=${2:-jelly}
ENV="${PG_ENV:-py310}"

echo "Launching PhysGaussian  scene=$SCENE  preset=$PRESET  arch=${PG_ARCH:-auto}"
conda run -n "$ENV" python "$(dirname "$0")/main.py" --scene "$SCENE" --preset "$PRESET"
