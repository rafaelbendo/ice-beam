# ============================================================
# Beam-level preprocessing of the clipped points
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point


def _beam_position_at_distance(bdf, xm, dist_col="distance_from_offshore"):
    """Beam position in map space at a common offshore distance xm (no extrapolation)."""
    if dist_col not in bdf.columns:
        return None

    p = bdf.dropna(subset=[dist_col, "geometry"]).copy()
    if p.empty:
        return None

    p["gx"] = p.geometry.x
    p["gy"] = p.geometry.y
    p = p.dropna(subset=["gx", "gy"])

    if len(p) < 2:
        return None

    p[dist_col] = pd.to_numeric(p[dist_col], errors="coerce")
    p = p.dropna(subset=[dist_col]).copy()
    if len(p) < 2:
        return None

    p = p.sort_values(dist_col).copy()

    p = (
        p.groupby(dist_col, as_index=False)
         .agg(gx=("gx", "mean"), gy=("gy", "mean"))
         .sort_values(dist_col)
    )

    if len(p) < 2:
        return None

    d = p[dist_col].to_numpy(dtype=float)
    x = p["gx"].to_numpy(dtype=float)
    y = p["gy"].to_numpy(dtype=float)

    try:
        dmin = float(np.nanmin(d))
        dmax = float(np.nanmax(d))
    except Exception:
        return None

    if not (dmin <= xm <= dmax):
        return None

    try:
        return Point(float(np.interp(xm, d, x)), float(np.interp(xm, d, y)))
    except Exception:
        return None


def apply_preprocessing_to_clipped(
        dataset_clipped,
        MIN_POINTS_PCT=0.9,
        ELEV_TRASH=40.0,
        TOO_FAR_BEAM=182.0,
        XM=300.0,
        IDEAL_CASE=56,
        return_skipped=True,
        verbose=True):
    """
    Apply preprocessing constraints to already clipped beams:

      • few_points  -> beam has < MIN_POINTS_PCT of the family-typical point count
      • elev_trash  -> beam contains any |h_li| > ELEV_TRASH
      • too_far     -> beam spacing at the common offshore distance XM is
                       larger than TOO_FAR_BEAM relative to its nearest neighbor

    Priority (highest -> lowest): elev_trash > few_points > too_far > loaded

    Beams flagged only as too_far are KEPT: they may define the outer lateral
    boundary for lateral-growth clustering.

    Returns
    -------
    dataset_raw, summary_raw, flagged_df, beam_flags, midpoint_gdf
    (or dataset_raw, summary_raw when return_skipped=False)
    """

    # 0) Empty input
    if dataset_clipped is None or dataset_clipped.empty:
        empty = gpd.GeoDataFrame(columns=["geometry"],
                                 crs=getattr(dataset_clipped, "crs", None))
        return (empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty.copy()) if return_skipped else empty

    gdf = dataset_clipped.copy()

    # A) Family-typical points after clipping (mode, so sparse beams don't lower it)
    fam_typical_pts = (
        gdf.groupby(["gt_family", "beam_id"])
           .size()
           .groupby("gt_family")
           .agg(lambda x: x.mode().iloc[0])
           .to_dict()
    )

    # B) Beam-level flags (few_points, elev_trash)
    beams_info = []

    for (fam, bid), bdf in gdf.groupby(["gt_family", "beam_id"]):

        fam_typical = fam_typical_pts.get(fam, np.nan)
        few_points_flag = (
            np.isfinite(fam_typical) and fam_typical > 0 and (len(bdf) / fam_typical < MIN_POINTS_PCT)
        )

        elev_flag = (
            "h_li" in bdf.columns
            and (
                (bdf["h_li"] > ELEV_TRASH).any()
                or (bdf["h_li"] < -ELEV_TRASH).any()
            )
        )

        beams_info.append({
            "gt_family": fam,
            "beam_id": bid,
            "n_points": len(bdf),
            "fam_typical_points": fam_typical,
            "points_ratio": len(bdf) / fam_typical if np.isfinite(fam_typical) and fam_typical > 0 else np.nan,
            "few_points_threshold": MIN_POINTS_PCT,
            "few_points": few_points_flag,
            "elev_trash": elev_flag
        })

    beam_flags = pd.DataFrame(beams_info)

    # C) too_far flag via nearest-neighbor spacing at XM
    flagged_records = []
    beam_flags["nearest_dist"] = np.nan
    beam_flags["nearest_beam"] = pd.Series([None] * len(beam_flags), dtype="object")
    beam_flags["too_far"] = False
    beam_flags["spans_xm"] = False

    midpoint_records = []

    for fam, sub in gdf.groupby("gt_family"):

        beam_positions = []
        beam_ids = []

        for bid, bdf in sub.groupby("beam_id"):
            pos_pt = _beam_position_at_distance(bdf, xm=XM)

            sel = (beam_flags["gt_family"] == fam) & (beam_flags["beam_id"] == bid)
            beam_flags.loc[sel, "spans_xm"] = pos_pt is not None

            if pos_pt is None:
                continue

            beam_positions.append(pos_pt)
            beam_ids.append(bid)

        if len(beam_positions) == 0:
            continue

        pos_gdf = gpd.GeoDataFrame(
            {
                "beam_id": beam_ids,
                "geometry": beam_positions,
                "xm": XM
            },
            geometry="geometry",
            crs=gdf.crs
        )

        if len(pos_gdf) < 2:
            pos_gdf["nearest_dist"] = np.nan
            pos_gdf["nearest_idx"] = np.nan
            pos_gdf["nearest_beam"] = np.nan
            pos_gdf["too_far"] = False
            pos_gdf["gt_family"] = fam
            midpoint_records.append(pos_gdf.copy())
            continue

        coords = np.array([[p.x, p.y] for p in pos_gdf.geometry])
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        dist, idx = nbrs.kneighbors(coords)

        pos_gdf["nearest_dist"] = dist[:, 1]
        pos_gdf["nearest_idx"] = idx[:, 1]
        pos_gdf["nearest_beam"] = pos_gdf.iloc[idx[:, 1]]["beam_id"].values
        pos_gdf["too_far"] = pos_gdf["nearest_dist"] > TOO_FAR_BEAM
        pos_gdf["gt_family"] = fam

        midpoint_records.append(pos_gdf.copy())

        for _, row in pos_gdf.iterrows():
            sel = (beam_flags["gt_family"] == fam) & (beam_flags["beam_id"] == row["beam_id"])
            beam_flags.loc[sel, "nearest_dist"] = float(row["nearest_dist"])
            beam_flags.loc[sel, "nearest_beam"] = row["nearest_beam"]
            beam_flags.loc[sel, "too_far"] = bool(row["too_far"])

            if row["too_far"]:
                flagged_records.append({
                    "gt_family": fam,
                    "beam_id": row["beam_id"],
                    "nearest_beam": row["nearest_beam"],
                    "nearest_dist": float(row["nearest_dist"]),
                    "xm": XM
                })

    flagged_df = pd.DataFrame(flagged_records)

    if len(midpoint_records) > 0:
        midpoint_gdf = gpd.GeoDataFrame(
            pd.concat(midpoint_records, ignore_index=True),
            geometry="geometry",
            crs=gdf.crs
        )
    else:
        midpoint_gdf = gpd.GeoDataFrame(
            columns=["gt_family", "beam_id", "nearest_beam", "nearest_dist", "too_far", "xm", "geometry"],
            geometry="geometry",
            crs=gdf.crs
        )

    # D) Final beam status: elev_trash > few_points > too_far > loaded
    def classify(row):
        if row.elev_trash:
            return "elev_trash"
        if row.few_points:
            return "few_points"
        if row.too_far:
            return "too_far"
        return "loaded"

    beam_flags["status"] = beam_flags.apply(classify, axis=1)

    kept_pairs = beam_flags.loc[
        ~beam_flags["status"].isin(["elev_trash", "few_points"]),
        ["gt_family", "beam_id"]
    ].drop_duplicates()

    dataset_raw = gdf.merge(
        kept_pairs.assign(_keep_preprocess=True),
        on=["gt_family", "beam_id"],
        how="left"
    )
    dataset_raw = dataset_raw[dataset_raw["_keep_preprocess"].fillna(False)].copy()
    dataset_raw = dataset_raw.drop(columns="_keep_preprocess")

    # E) Summary table
    summary = (
        beam_flags.groupby(["gt_family", "status"])
                  .size()
                  .unstack(fill_value=0)
    )

    summary["files_found"] = beam_flags.groupby("gt_family")["beam_id"].nunique()
    summary["ideal_case"] = IDEAL_CASE
    summary["missing"] = summary["ideal_case"] - summary["files_found"]

    col_order = [
        "ideal_case", "files_found", "missing",
        "elev_trash", "few_points", "too_far", "loaded"
    ]
    for col in col_order:
        if col not in summary:
            summary[col] = 0

    summary_raw = summary[col_order].copy()

    if verbose:
        print(summary_raw.to_string())

    if return_skipped:
        return dataset_raw, summary_raw, flagged_df, beam_flags, midpoint_gdf
    else:
        return dataset_raw, summary_raw
