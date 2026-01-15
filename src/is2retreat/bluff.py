# ============================================================
# Bluff crossing (reference elevation intersection)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd


def _require_param(params: object, name: str):
    if params is None:
        raise ValueError("params is required. Pass Params() from is2retreat.config.")
    if not hasattr(params, name):
        raise ValueError(f"params must define `{name}`.")
    return getattr(params, name)


def find_bluff_by_reference(
    profile,
    y_ref: float,
    params: object,
    x_col: str = "distance_from_offshore",
    y_col: str = "h_li",
    which: str = "first",
):
    """
    Locate x-position where an elevation profile crosses a given reference
    elevation (y_ref), using robust interpolation and discontinuity handling.

    Required params:
      - GAP_THRESHOLD_M (float)
      - CROSSING_ATOL   (float)

    Returns
    -------
    bx, by : float
        X-position of crossing and y_ref. (np.nan, np.nan) if not found.
    """
    gap_threshold = float(_require_param(params, "GAP_THRESHOLD_M"))
    atol = float(_require_param(params, "CROSSING_ATOL"))

    if which not in {"first", "last"}:
        raise ValueError("which must be 'first' or 'last'.")

    # ------------------------------------------------------------
    # Convert input to sorted clean arrays x[], y[]
    # ------------------------------------------------------------
    def _to_xy_arrays(prof):
        # Case 1: DataFrame
        if hasattr(prof, "loc") and hasattr(prof, "columns"):
            df = (
                prof[[x_col, y_col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .drop_duplicates(subset=[x_col, y_col])
                .sort_values(x_col)
            )
            return df[x_col].to_numpy(float), df[y_col].to_numpy(float)

        # Case 2: (x, y) tuple/list
        if isinstance(prof, (list, tuple)) and len(prof) == 2:
            x = np.asarray(prof[0], float)
            y = np.asarray(prof[1], float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size:
                idx = np.argsort(x)
                x, y = x[idx], y[idx]
            return x, y

        # Case 3: dict-like
        if isinstance(prof, dict):
            x = np.asarray(prof.get(x_col), float)
            y = np.asarray(prof.get(y_col), float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size:
                idx = np.argsort(x)
                x, y = x[idx], y[idx]
            return x, y

        return np.array([]), np.array([])

    x, y = _to_xy_arrays(profile)
    if x.size < 2:
        return np.nan, np.nan

    y_ref = float(y_ref)
    diffs = y - y_ref

    # ------------------------------------------------------------
    # 1) Exact match
    # ------------------------------------------------------------
    exact_idx = np.flatnonzero(np.isclose(diffs, 0.0, atol=atol))
    if exact_idx.size > 0:
        i = exact_idx[0] if which == "first" else exact_idx[-1]
        return float(x[i]), y_ref

    # ------------------------------------------------------------
    # helper: crossings in a segment of indices
    # ------------------------------------------------------------
    def _crossings_in_indices(idxs):
        if idxs.size < 2:
            return []

        s = np.sign(diffs[idxs])

        # handle zeros by borrowing sign from neighbors
        zeros = np.where(s == 0)[0]
        for zi in zeros:
            if 0 < zi < s.size - 1:
                s[zi] = s[zi - 1] if s[zi - 1] != 0 else s[zi + 1]
            elif zi == 0 and s.size > 1:
                s[zi] = s[1]
            elif zi == s.size - 1 and s.size > 1:
                s[zi] = s[-2]

        flips = np.where(s[:-1] * s[1:] < 0)[0]
        return (idxs[flips]).tolist()

    # ------------------------------------------------------------
    # 2) try full series
    # ------------------------------------------------------------
    primary = _crossings_in_indices(np.arange(x.size))

    # ------------------------------------------------------------
    # 3) if none, split on gaps and search segments
    # ------------------------------------------------------------
    if len(primary) == 0:
        # flat at y_ref
        if np.isclose(y.min(), y.max(), atol=atol) and np.isclose(y.min(), y_ref, atol=atol):
            return float(np.nanmedian(x)), y_ref

        finite = np.where(np.isfinite(x) & np.isfinite(y))[0]
        if finite.size >= 2:
            gaps = np.where(np.diff(x[finite]) > gap_threshold)[0]
            segments = np.split(finite, gaps + 1)

            for seg in segments:
                if seg.size >= 2:
                    found = _crossings_in_indices(seg)
                    if found:
                        primary = found
                        break

    if len(primary) == 0:
        return np.nan, np.nan

    # ------------------------------------------------------------
    # 4) interpolate crossing
    # ------------------------------------------------------------
    idx = primary[0] if which == "first" else primary[-1]

    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]

    if np.isclose(y0, y1, atol=atol):
        return np.nan, np.nan

    t = (y_ref - y0) / (y1 - y0)
    bx = x0 + t * (x1 - x0)

    return float(bx), float(y_ref)


__all__ = ["find_bluff_by_reference"]
