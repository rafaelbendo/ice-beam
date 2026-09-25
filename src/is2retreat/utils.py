# ============================================================
# Small shared helpers
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd


class TrackSkipped(Exception):
    """
    Raised when a track cannot produce results (no data, no shoreline
    crossing, ...). Batch runs catch it and move on to the next track,
    like the notebook's ``sys.exit(0)`` SKIP messages.
    """


def first_non_null(series, default=np.nan):
    if series is None:
        return default
    s = pd.Series(series).dropna()
    return s.iloc[0] if not s.empty else default


def cluster_member_beams(row) -> list[str]:
    """
    Member beam IDs of a cluster row, in lateral-growth order.

    Prefers ``beam_ids_ordered``; falls back to ``beam_ids`` (list of
    ``(gt_family, beam_id)`` tuples).
    """
    raw_beams = row.get("beam_ids_ordered", None)

    if raw_beams is None or not isinstance(raw_beams, (list, tuple)) or len(raw_beams) == 0:
        raw_beams = row.get("beam_ids", [])

    beam_ids = [
        (b[1] if isinstance(b, (list, tuple)) and len(b) > 1 else b)
        for b in (raw_beams if isinstance(raw_beams, (list, tuple)) else [raw_beams])
    ]

    return [str(b).strip() for b in beam_ids if pd.notna(b)]


def format_track_id(track_id) -> str:
    """'129' / 129 / '0129' -> '0129'."""
    return f"{int(track_id):04d}"
