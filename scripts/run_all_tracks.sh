#!/bin/bash
# Run ICE-BEAM for every track in configs/tracks_filtered.txt.
# Requires the package installed:  pip install -e .
# Extra arguments are passed through, e.g.  ./scripts/run_all_tracks.sh --source sliderule
set -euo pipefail
cd "$(dirname "$0")/.."

is2retreat \
    --config configs/north_slope.toml \
    --tracks-file configs/tracks_filtered.txt \
    "$@" 2>&1 | tee "run_all_tracks_$(date +%Y%m%d_%H%M%S).log"
