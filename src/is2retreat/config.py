# ============================================================
# Configuration: run parameters (Params) and input/output paths (Paths)
# ============================================================
"""
All tunable values of the ICE-BEAM pipeline live here.

Defaults reproduce ``1ICE_BEAM_v20SRMultitracks_3.ipynb`` (version 21,
SlideRule ingestion, 5 m resolution, 180 m cluster size limit).

Paths are machine-specific, so they have no defaults: load them from a TOML
file with :func:`load_config` (see ``configs/north_slope.toml``).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class Params:
    """Scientific / processing parameters (defaults = final notebook)."""

    # ------------------------------
    # CRS
    # ------------------------------
    ORIGINAL_CRS: str = "EPSG:4326"

    # ------------------------------
    # SlideRule request (ATL06-like segments from ATL03 via atl03x)
    # ------------------------------
    SLIDERULE_URL: str = "slideruleearth.io"
    SLIDERULE_DATE_START: Optional[str] = "2019-01-01"
    SLIDERULE_DATE_END: Optional[str] = "2025-12-31"
    SLIDERULE_CYCLE: Optional[int] = None
    SLIDERULE_SEGMENT_LENGTH_M: float = 40.0
    SLIDERULE_SEGMENT_RESOLUTION_M: float = 5.0

    # ------------------------------
    # Ground-track families
    # ------------------------------
    GTX: Sequence[str] = ("gt1", "gt2", "gt3")

    # ------------------------------
    # Oriented box around each family's shoreline crossing (UTM meters)
    # ------------------------------
    HALF_ALONG_M: float = 300.0     # Half-length along the beam (i.e. across the shore)
    HALF_ACROSS_M: float = 600.0    # Half-width across the beam (i.e. along the coast)

    # ------------------------------
    # Preprocessing
    # ------------------------------
    MIN_POINTS_PCT: float = 0.9     # Beam must have >= 90% of family-typical points
    ELEV_TRASH: float = 40.0        # Drop beams with |h_li| above this (m)
    TOO_FAR_BEAM: float = 182.0     # Beams farther apart than this are isolated
    XM_PREPROCESS: float = 300.0    # Common offshore distance for the too_far check
    N_CYCLES: int = 28              # Ideal-case reference (2 beams per cycle)

    # ------------------------------
    # Lateral-growth clustering with vertical-bias control
    # ------------------------------
    X0: float = 500.0               # Offshore distance where elevation bias is evaluated
    XM: float = 500.0               # Offshore distance used for lateral beam ordering
    SIZE_LIMIT_M: Optional[float] = 180.0   # Max cluster width (m); None = unlimited
    MIN_PROFILES_PER_CLUSTER: int = 2
    BIAS_TOLERANCES: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

    # ------------------------------
    # Beam / shoreline angles
    # ------------------------------
    ANGLE_MODE_BIN_WIDTH: float = 5.0
    ANGLE_SEARCH_RADIUS: float = 10.0

    # ------------------------------
    # Bluff detection
    # ------------------------------
    BLUFF_WHICH: str = "first"
    GAP_THRESHOLD_M: float = 40.0
    CROSSING_ATOL: float = 1e-3

    # ------------------------------
    # DSAS statistics
    # ------------------------------
    CONFIDENCE: float = 0.95
    MIN_SPAN_DAYS: int = 365
    POSITIONAL_UNCERTAINTY_M: float = 4.8

    # ------------------------------
    # GIE correction
    # ------------------------------
    GIE_FALLBACK_ANGLE_COL: str = "angle_median_deg"

    @property
    def IDEAL_CASE(self) -> int:
        return self.N_CYCLES * 2

    @property
    def RES_TAG(self) -> str:
        """Resolution tag used in output/cache filenames, e.g. '5m'."""
        return f"{float(self.SLIDERULE_SEGMENT_RESOLUTION_M):g}".replace(".", "p") + "m"


@dataclass(frozen=True)
class Paths:
    """Input and output locations (machine-specific)."""

    aoi_path: Path                  # AOI polygon for the SlideRule request
    shoreline_path: Path            # Coastline with CoastType (1 = bluff)
    is2_tracks_path: Path           # RGT lines (field 'Name' = RGT number)
    sliderule_cache_dir: Path       # <dir>/<track>/ATL06_<res>.gpkg
    outdir: Path                    # DSAS / GIE CSV outputs

    def sliderule_cache_file(self, track_id: str, res_tag: str) -> Path:
        return Path(self.sliderule_cache_dir) / str(track_id) / f"ATL06_{res_tag}.gpkg"


def load_config(config_path, **param_overrides) -> tuple[Paths, Params]:
    """
    Read a TOML config with a ``[paths]`` table and an optional ``[params]``
    table. Relative paths are resolved against the config file's folder.
    Keyword overrides are applied on top of ``[params]``.
    """
    config_path = Path(config_path)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    base = config_path.parent
    raw_paths = cfg.get("paths", {})
    path_names = [f.name for f in fields(Paths)]
    missing = [name for name in path_names if name not in raw_paths]
    if missing:
        raise ValueError(f"{config_path}: [paths] is missing {missing}")

    paths = Paths(**{
        name: (base / Path(raw_paths[name]).expanduser()).resolve()
        for name in path_names
    })

    params = params_from_dict({**cfg.get("params", {}), **param_overrides})
    return paths, params


def params_from_dict(values: dict) -> Params:
    """Build Params from a dict, rejecting unknown keys (catches typos)."""
    known = {f.name for f in fields(Params)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown parameter(s): {unknown}")

    values = dict(values)
    for key in ("GTX", "BIAS_TOLERANCES"):
        if key in values:
            values[key] = tuple(values[key])

    return replace(Params(), **values)
