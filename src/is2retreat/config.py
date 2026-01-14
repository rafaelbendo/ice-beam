from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Params:
    """
    Default parameters for the ICESat-2 shoreline framework.

    Keep your tuning here. In notebooks/scripts, create variants like:
        P = Params()
        P.BIAS_TOLERANCE = 0.75
    """

    # ------------------------------
    # Spatial clustering parameters
    # ------------------------------
    CLUSTER_DISTANCE_M: float = 18.0

    # ------------------------------
    # Vertical filtering parameters
    # ------------------------------
    VERTICAL_TOLERANCE: float = 5.0
    BIAS_TOLERANCE: float = 0.5

    # ------------------------------
    # Temporal / data sufficiency
    # ------------------------------
    MIN_PROFILES_PER_CLUSTER: int = 2
    MIN_POINTS_PCT: float = 0.8  # beam must have >= 80% of family mean points (after clipping)

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
    ELEV_TRASH: float = 20.0  # remove beams with h_li above +ELEV_TRASH or below -ELEV_TRASH

    # ------------------------------
    # Beam separation constraint
    # ------------------------------
    TOO_FAR_BEAM: float = 46.0

    # ------------------------------
    # Ground-track families
    # ------------------------------
    GTX: Sequence[str] = ("gt1", "gt2", "gt3")

    @property
    def IDEAL_CASE(self) -> int:
        # If every cycle has 2 beams
        return self.N_CYCLES * 2
