# ============================================================
# Bluff crossing (reference elevation intersection)
# ============================================================
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd


def _require_param(params: object, name: str):
    if params is None:
        raise ValueError("params is required. Pass Params() from is2retreat.config.")
    if not hasattr(params, name):
        raise ValueError(f"params must define `{name}`.")
    return getattr(params, name)


def _normalize_beam_ids(raw_ids) -> list[str]:
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, (list, tuple)):
        raw_ids = [raw_ids]
    out = []
    for b in raw_ids:
        if isinstance(b, (list, tuple)) and len(b) > 1:
            out.append(str(b[1]).strip())
        else:
            out.append(str(b).strip())
    return out


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

    def _to_xy_arrays(prof):
        if hasattr(prof, "loc") and hasattr(prof, "columns"):
            df = (
                prof[[x_col, y_col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .drop_duplicates(subset=[x_col, y_col])
                .sort_values(x_col)
            )
            return df[x_col].to_numpy(float), df[y_col].to_numpy(float)

        if isinstance(prof, (list, tuple)) and len(prof) == 2:
            x = np.asarray(prof[0], float)
            y = np.asarray(prof[1], float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size:
                idx = np.argsort(x)
                x, y = x[idx], y[idx]
            return x, y

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

    exact_idx = np.flatnonzero(np.isclose(diffs, 0.0, atol=atol))
    if exact_idx.size > 0:
        i = exact_idx[0] if which == "first" else exact_idx[-1]
        return float(x[i]), y_ref

    def _crossings_in_indices(idxs):
        if idxs.size < 2:
            return []

        s = np.sign(diffs[idxs])

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

    primary = _crossings_in_indices(np.arange(x.size))

    if len(primary) == 0:
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

    idx = primary[0] if which == "first" else primary[-1]

    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]

    if np.isclose(y0, y1, atol=atol):
        return np.nan, np.nan

    t = (y_ref - y0) / (y1 - y0)
    bx = x0 + t * (x1 - x0)

    return float(bx), float(y_ref)


def process_cluster_with_reference(
    filtered_profiles: gpd.GeoDataFrame,
    selected_clusters: gpd.GeoDataFrame,
    params: object,
    cluster_id: int,
    gt_family: str,
    which: str = "first",
    bias_df: Optional[pd.DataFrame] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Compute bluff positions (bluff_x, bluff_y) for each cycle in a cluster.

    Required params:
      - GAP_THRESHOLD_M (float)
      - CROSSING_ATOL   (float)

    Returns:
      bluff_df : DataFrame with columns: beam_id, acq_date, bluff_x, bluff_y, ref_line
      y_ref    : float reference elevation used, or None
    """
    if which not in {"first", "last"}:
        raise ValueError("which must be 'first' or 'last'.")

    cl = selected_clusters[
        (selected_clusters["gt_family"].astype(str) == str(gt_family)) &
        (selected_clusters["cluster_id"] == cluster_id)
    ]
    if cl.empty:
        if debug:
            print(f"⚠️ Cluster {gt_family}-{cluster_id} not found in selected_clusters.")
        return pd.DataFrame(), None

    beam_ids = _normalize_beam_ids(cl.iloc[0].get("beam_ids", []))
    if not beam_ids:
        if debug:
            print(f"⚠️ Cluster {gt_family}-{cluster_id} has no beam_ids.")
        return pd.DataFrame(), None

    fam_prof = filtered_profiles.copy()
    fam_prof["beam_id_str"] = fam_prof["beam_id"].astype(str).str.strip()
    fam_prof["gt_family"] = fam_prof["gt_family"].astype(str).str.strip()
    fam_prof["acq_date"] = pd.to_datetime(fam_prof["acq_date"], errors="coerce")

    fam_prof = fam_prof[
        (fam_prof["gt_family"] == str(gt_family)) &
        (fam_prof["beam_id_str"].isin(beam_ids)) &
        (fam_prof["cluster_id"] == cluster_id)
    ].copy()

    if fam_prof.empty:
        if debug:
            print(f"⚠️ No valid profiles found for {gt_family}-{cluster_id}.")
        return pd.DataFrame(), None

    if bias_df is not None and not bias_df.empty:
        allowed = bias_df[
            (bias_df["gt_family"].astype(str).str.strip() == str(gt_family)) &
            (bias_df["cluster_id"] == cluster_id) &
            ((bias_df["keep"] == True) | (bias_df["is_ref"] == True))
        ][["beam_id", "acq_date"]].copy()

        if not allowed.empty:
            allowed["beam_id_str"] = allowed["beam_id"].astype(str).str.strip()
            allowed["acq_date_norm"] = pd.to_datetime(allowed["acq_date"], errors="coerce").dt.normalize()

            fam_prof["acq_date_norm"] = fam_prof["acq_date"].dt.normalize()

            fam_prof = fam_prof.merge(
                allowed[["beam_id_str", "acq_date_norm"]],
                on=["beam_id_str", "acq_date_norm"],
                how="inner",
            )

    if fam_prof.empty:
        if debug:
            print(f"⚠️ All cycles removed by bias filters for {gt_family}-{cluster_id}.")
        return pd.DataFrame(), None

    oldest_idx = fam_prof["acq_date"].idxmin()
    oldest_bid = fam_prof.loc[oldest_idx, "beam_id_str"]
    oldest_dt = fam_prof.loc[oldest_idx, "acq_date"]

    ref_prof = fam_prof[
        (fam_prof["beam_id_str"] == oldest_bid) &
        (fam_prof["acq_date"] == oldest_dt)
    ].copy()

    y_min = pd.to_numeric(ref_prof["h_li"], errors="coerce").min()
    y_max = pd.to_numeric(ref_prof["h_li"], errors="coerce").max()

    if pd.isna(y_min) or pd.isna(y_max):
        if debug:
            print("⚠️ Reference profile has no valid elevations.")
        return pd.DataFrame(), None

    y_ref = float((y_min + y_max) / 2.0)

    bluff_records = []
    for (bid, dt), prof in fam_prof.groupby(["beam_id_str", "acq_date"]):
        prof = prof.sort_values("distance_from_offshore").dropna(subset=["distance_from_offshore", "h_li"])
        if len(prof) < 2:
            continue

        bx, by = find_bluff_by_reference(
            prof[["distance_from_offshore", "h_li"]],
            y_ref=y_ref,
            params=params,
            x_col="distance_from_offshore",
            y_col="h_li",
            which=which,
        )

        if np.isfinite(bx) and np.isfinite(by):
            bluff_records.append(
                {
                    "beam_id": str(bid),
                    "acq_date": dt,
                    "bluff_x": float(bx),
                    "bluff_y": float(by),
                    "ref_line": y_ref,
                }
            )

    bluff_df = pd.DataFrame(bluff_records)
    if bluff_df.empty:
        if debug:
            print(f"⚠️ No bluff positions computed for cluster {gt_family}-{cluster_id}.")
        return pd.DataFrame(), y_ref

    return bluff_df, y_ref


__all__ = ["find_bluff_by_reference", "process_cluster_with_reference"]
