from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Params:
    """
    Default parameters for the ICESat-2 shoreline framework.
    """

    # ------------------------------
    # CRS / projection
    # ------------------------------
    ORIGINAL_CRS: str = "EPSG:4326"
    UTM_EPSG: int = 32606

    # ------------------------------
    # Spatial clustering parameters
    # ------------------------------
    CLUSTER_DISTANCE_M: float = 18.0
    MIN_BEAMS: int = 2
    ANGLE_SEARCH_RADIUS: float = 10.0

    # ------------------------------
    # Vertical filtering parameters
    # ------------------------------
    VERTICAL_TOLERANCE: float = 5.0
    BIAS_TOLERANCE: float = 0.5
    BIAS_X0: float = 550.0

    # ------------------------------
    # Temporal / data sufficiency
    # ------------------------------
    MIN_PROFILES_PER_CLUSTER: int = 2
    MIN_POINTS_PCT: float = 0.8

    # ------------------------------
    # Oriented box dimensions (UTM meters)
    # ------------------------------
    HALF_ALONG_M: float = 300.0
    HALF_ACROSS_M: float = 300.0

    # ------------------------------
    # Ideal-case reference
    # ------------------------------
    N_CYCLES: int = 24

    # ------------------------------
    # Elevation filtering
    # ------------------------------
    ELEV_TRASH: float = 20.0

    # ------------------------------
    # Beam separation constraint
    # ------------------------------
    TOO_FAR_BEAM: float = 46.0

    # ------------------------------
    # Ground-track families
    # ------------------------------
    GTX: Sequence[str] = ("gt1", "gt2", "gt3")

    # ------------------------------
    # Bluff crossing parameters
    # ------------------------------
    GAP_THRESHOLD_M: float = 40.0
    CROSSING_ATOL: float = 1e-3

    # ------------------------------
    # DSAS / regression parameters
    # ------------------------------
    CONFIDENCE: float = 0.95
    MIN_SPAN_DAYS: int = 365
    
    @property
    def IDEAL_CASE(self) -> int:
        return self.N_CYCLES * 2
