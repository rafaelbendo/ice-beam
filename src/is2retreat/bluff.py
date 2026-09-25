# ============================================================
# Bluff position: where a profile crosses a fixed reference elevation
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd


def find_bluff_by_reference(
        profile,
        y_ref,
        x_col="distance_from_offshore",
        y_col="h_li",
        which="first",
        gap_threshold=40.0,
        atol=1e-3
    ):
    """
    x-position where the elevation profile crosses the reference elevation
    y_ref, with robust interpolation and discontinuity handling.

    Parameters
    ----------
    profile : DataFrame (with x_col, y_col), (x, y) tuple, or dict
    y_ref : float
    which : {"first", "last"}
        First (offshore side) or last (inland side) crossing.
    gap_threshold : float
        If Δx > gap_threshold, treat as a discontinuity and search the
        segments separately.
    atol : float
        Tolerance for flat/zero-slope and equality checks.

    Returns
    -------
    bx, by : float
        Crossing x-position and y_ref, or (nan, nan) if not found.
    """

    def _to_xy_arrays(prof):
        # DataFrame with required columns
        if hasattr(prof, "loc") and hasattr(prof, "columns"):
            df = (prof[[x_col, y_col]]
                  .replace([np.inf, -np.inf], np.nan)
                  .dropna()
                  .drop_duplicates(subset=[x_col, y_col])
                  .sort_values(x_col))
            return (
                df[x_col].to_numpy(float),
                df[y_col].to_numpy(float)
            )

        # (x, y) tuple/list
        if isinstance(prof, (list, tuple)) and len(prof) == 2:
            x = np.asarray(prof[0], float)
            y = np.asarray(prof[1], float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size:
                idx = np.argsort(x)
                x, y = x[idx], y[idx]
            return x, y

        # dict-like
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

    # 1) Exact match
    exact_idx = np.flatnonzero(np.isclose(diffs, 0.0, atol=atol))
    if exact_idx.size > 0:
        i = exact_idx[0] if which == "first" else exact_idx[-1]
        return float(x[i]), y_ref

    def _crossings_in_indices(idxs):
        if idxs.size < 2:
            return []

        s = np.sign(diffs[idxs])

        # Handle zeros by borrowing sign from neighbors
        zeros = np.where(s == 0)[0]
        for zi in zeros:
            if 0 < zi < s.size - 1:
                s[zi] = s[zi-1] if s[zi-1] != 0 else s[zi+1]
            elif zi == 0 and s.size > 1:
                s[zi] = s[1]
            elif zi == s.size - 1 and s.size > 1:
                s[zi] = s[-2]

        flips = np.where(s[:-1] * s[1:] < 0)[0]
        return (idxs[flips]).tolist()

    # 2) Quick search on the full series
    primary = _crossings_in_indices(np.arange(x.size))

    # 3) No direct crossing -> check segments split at large gaps
    if len(primary) == 0:
        if np.isclose(y.min(), y.max(), atol=atol) and np.isclose(y.min(), y_ref, atol=atol):
            # entire profile is flat at y_ref
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

    # 4) Still nothing -> no crossing
    if len(primary) == 0:
        return np.nan, np.nan

    # 5) Interpolate the crossing
    idx = primary[0] if which == "first" else primary[-1]

    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]

    if np.isclose(y0, y1, atol=atol):
        return np.nan, np.nan  # flat segment -> no interpolation

    t = (y_ref - y0) / (y1 - y0)
    bx = x0 + t * (x1 - x0)

    return float(bx), float(y_ref)


def process_cluster_with_reference(
        filtered_profiles,
        selected_clusters,
        cluster_id,
        gt_family,
        which="first",
        gap_threshold=40.0,
        atol=1e-3,
        bias_df=None,
        debug=False
    ):
    """
    Bluff positions (bluff_x, bluff_y) for each cycle in a cluster.

    - Strict filtering with bias_df (if provided)
    - Reference elevation = mid-height of the oldest cycle's profile
    - Crossing detection with find_bluff_by_reference()

    Returns
    -------
    bluff_df : DataFrame (beam_id, acq_date, bluff_x, bluff_y, ref_line)
    y_ref : float or None
    """
    # 1. Locate cluster
    cl = selected_clusters[
        (selected_clusters["gt_family"] == gt_family) &
        (selected_clusters["cluster_id"] == cluster_id)
    ]

    if cl.empty:
        if debug:
            print(f"⚠️ Cluster {gt_family}-{cluster_id} not found in selected_clusters.")
        return pd.DataFrame(), None

    raw_ids = cl.iloc[0]["beam_ids"]
    beam_ids = [
        (b[1] if isinstance(b, (list, tuple)) and len(b) > 1 else b)
        for b in (raw_ids if isinstance(raw_ids, (list, tuple)) else [raw_ids])
    ]
    beam_ids = [str(b).strip() for b in beam_ids]

    # 2. Subset filtered profiles
    df = filtered_profiles.copy()
    df["beam_id_str"] = df["beam_id"].astype(str).str.strip()

    fam_prof = df[
        (df["gt_family"] == gt_family) &
        (df["beam_id_str"].isin(beam_ids)) &
        (df["cluster_id"] == cluster_id)
    ].copy()

    if fam_prof.empty:
        if debug:
            print(f"⚠️ No valid profiles found for {gt_family}-{cluster_id}.")
        return pd.DataFrame(), None

    # 3. Optional strict filtering using bias_df
    if bias_df is not None:
        allowed = bias_df[
            (bias_df["gt_family"] == gt_family) &
            (bias_df["cluster_id"] == cluster_id) &
            ((bias_df["keep"] == True) | (bias_df["is_ref"] == True))  # noqa: E712
        ][["beam_id", "acq_date"]].copy()

        allowed["beam_id_str"] = allowed["beam_id"].astype(str).str.strip()
        allowed["acq_date_norm"] = allowed["acq_date"].dt.normalize()

        fam_prof["acq_date_norm"] = fam_prof["acq_date"].dt.normalize()

        fam_prof = fam_prof.merge(
            allowed[["beam_id_str", "acq_date_norm"]],
            on=["beam_id_str", "acq_date_norm"],
            how="inner"
        )

    if fam_prof.empty:
        if debug:
            print(f"⚠️ All cycles removed by bias filters for {gt_family}-{cluster_id}.")
        return pd.DataFrame(), None

    # 4. Reference elevation from the oldest cycle
    oldest_idx = fam_prof["acq_date"].idxmin()
    oldest_bid = fam_prof.loc[oldest_idx, "beam_id"]
    oldest_dt = fam_prof.loc[oldest_idx, "acq_date"]

    ref_prof = fam_prof[
        (fam_prof["beam_id"] == oldest_bid) &
        (fam_prof["acq_date"] == oldest_dt)
    ]

    y_min = ref_prof["h_li"].min()
    y_max = ref_prof["h_li"].max()

    if pd.isna(y_min) or pd.isna(y_max):
        if debug:
            print("⚠️ Reference profile has no valid elevations.")
        return pd.DataFrame(), None

    y_ref = float((y_min + y_max) / 2.0)

    # 5. Bluff position for every remaining cycle
    bluff_records = []

    for (bid, dt), prof in fam_prof.groupby(["beam_id", "acq_date"]):
        prof = prof.sort_values("distance_from_offshore").dropna(
            subset=["distance_from_offshore", "h_li"]
        )
        if len(prof) < 2:
            continue

        bx, by = find_bluff_by_reference(
            prof[["distance_from_offshore", "h_li"]],
            y_ref,
            which=which,
            gap_threshold=gap_threshold,
            atol=atol
        )

        if not (np.isnan(bx) or np.isnan(by)):
            bluff_records.append({
                "beam_id": str(bid),
                "acq_date": dt,
                "bluff_x": float(bx),
                "bluff_y": float(by),
                "ref_line": y_ref,
            })

    bluff_df = pd.DataFrame(bluff_records)

    if bluff_df.empty:
        if debug:
            print(f"⚠️ No bluff positions computed for cluster {cluster_id}.")
        return pd.DataFrame(), y_ref

    return bluff_df, y_ref
