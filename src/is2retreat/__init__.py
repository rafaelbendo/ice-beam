"""
ICE-BEAM (ICESat-2 Bluff Erosion Assessment Method).

Quantifies Arctic coastal bluff retreat from ICESat-2 ATL06-like elevation
profiles (SlideRule), with lateral-growth beam clustering, vertical-bias
filtering, DSAS-style shoreline change metrics and a geometric (GIE)
correction for oblique beam/coast crossings.

Typical use::

    from is2retreat import load_config, run_track
    paths, params = load_config("configs/north_slope.toml")
    result = run_track("0129", paths, params)
"""
from .config import Params, Paths, load_config
from .pipeline import TrackInputs, TrackResult, prepare_track_inputs, run_track
from .utils import TrackSkipped

__version__ = "0.2.0"

__all__ = [
    "Params",
    "Paths",
    "TrackInputs",
    "TrackResult",
    "TrackSkipped",
    "load_config",
    "prepare_track_inputs",
    "run_track",
]
