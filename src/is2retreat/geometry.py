# ============================================================
# Oriented shoreline-crossing boxes and offshore distances
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon


def _beam_lines_sorted_by_northing(fam_df):
    """One LineString per beam, points ordered by northing."""
    lines = {}
    for beam_id, g in fam_df.groupby("beam_id"):
        if len(g) < 2:
            continue
        g_sorted = (
            g.assign(_y=g.geometry.y)
             .sort_values("_y")
             .drop(columns="_y")
        )
        lines[beam_id] = LineString(g_sorted.geometry.tolist())
    return lines


def get_middle_crossing(dataset_raw, shoreline_utm, utm_epsg, gt_family):
    """
    For one ground-track family (gt1, gt2 or gt3):

        1. Reproject family points to UTM
        2. Compute the centroid of all beam points
        3. Build a LineString for each beam
        4. Choose the beam whose centroid is closest to the overall centroid
        5. Intersect it with the shoreline (already in UTM)

    Returns
    -------
    center_pt, cross_pt (None if no crossing), nearest_beam, nearest_line
    """
    fam_df = dataset_raw[dataset_raw["gt_family"] == gt_family].copy()
    if fam_df.empty:
        return None, None, None, None

    fam_df = fam_df.to_crs(utm_epsg)
    center_pt = fam_df.geometry.union_all().centroid

    lines = _beam_lines_sorted_by_northing(fam_df)
    if not lines:
        return center_pt, None, None, None

    nearest_beam, nearest_line = min(
        lines.items(),
        key=lambda kv: kv[1].centroid.distance(center_pt)
    )

    shore_union = shoreline_utm.geometry.union_all()
    inter = nearest_line.intersection(shore_union)

    if inter.is_empty:
        cross_pt = None
    elif inter.geom_type == "Point":
        cross_pt = inter
    elif inter.geom_type == "MultiPoint":
        # Choose the point closest to the family centroid
        cross_pt = min(inter.geoms, key=lambda p: p.distance(center_pt))
    else:
        # LineString or mixed geometries
        cross_pt = inter.centroid

    return center_pt, cross_pt, nearest_beam, nearest_line


def build_boxes_for_families(dataset_raw, shoreline_gdf,
                             utm_epsg,
                             half_along=300.0,
                             half_across=600.0,
                             gtx=("gt1", "gt2", "gt3"),
                             verbose=True):
    """
    Build an oriented extraction box around the shoreline crossing of each
    ground-track family and clip the family's points to it.

    Returns
    -------
    dict:
        fam: {"box", "clipped", "cross", "center", "nearest_beam"}
    """
    dataset_clean = {}

    shoreline_utm = shoreline_gdf.to_crs(utm_epsg)

    for fam in gtx:

        fam_df = dataset_raw[dataset_raw["gt_family"] == fam].copy()
        if fam_df.empty:
            if verbose:
                print(f"[INFO] {fam}: no points found.")
            continue

        fam_df = fam_df.to_crs(utm_epsg)

        # -------- 1) Shoreline crossing point ---------------------
        center_pt, cross_pt, nearest_beam, nearest_line = \
            get_middle_crossing(dataset_raw, shoreline_utm, utm_epsg, fam)

        if cross_pt is None:
            if verbose:
                print(f"[WARN] {fam}: no shoreline crossing found.")
            continue

        # -------- 2) Orientation from beam tangent ----------------
        p0 = nearest_line.interpolate(0.4, normalized=True)
        p1 = nearest_line.interpolate(0.6, normalized=True)

        vx, vy = p1.x - p0.x, p1.y - p0.y
        norm = np.hypot(vx, vy)
        if norm == 0:
            if verbose:
                print(f"[WARN] {fam}: invalid tangent vector.")
            continue

        t_hat = np.array([vx, vy]) / norm
        n_hat = np.array([-t_hat[1], t_hat[0]])

        # -------- 3) Oriented box centered on the crossing --------
        C = np.array([cross_pt.x, cross_pt.y])
        corners = [
            C + t_hat * half_along + n_hat * half_across,
            C - t_hat * half_along + n_hat * half_across,
            C - t_hat * half_along - n_hat * half_across,
            C + t_hat * half_along - n_hat * half_across,
        ]
        box_poly = Polygon(corners)
        box_gdf = gpd.GeoDataFrame({"geometry": [box_poly]},
                                   crs=f"EPSG:{utm_epsg}")

        # -------- 4) Clip beams inside the box --------------------
        mask = fam_df.geometry.within(box_poly) | fam_df.geometry.touches(box_poly)
        clipped = fam_df.loc[mask].copy()

        if clipped.empty:
            if verbose:
                print(f"[WARN] {fam}: no clipped points inside box.")
            continue

        # -------- 5) Euclidean distance from the crossing ---------
        cx, cy = cross_pt.x, cross_pt.y
        clipped["distance_from_offshore"] = clipped.geometry.apply(
            lambda p: np.hypot(p.x - cx, p.y - cy)
        )

        dataset_clean[fam] = {
            "box": box_gdf,
            "clipped": clipped,
            "cross": cross_pt,
            "center": center_pt,
            "nearest_beam": nearest_beam
        }

        if verbose:
            print(f"[OK] {fam}: box built with {len(clipped)} clipped points.")

    return dataset_clean


def compute_distances_from_nearest_beam(dataset_clean, utm_epsg, track_id=None):
    """
    Along-track distance of every clipped point, projected on the nearest
    beam's direction, measured from the box's offshore edge (~0-600 m).

    Also adds beam-level QA columns that flag beams missing offshore points.
    """
    rows = []

    KEEP_COLS = [
        "gt_family", "beam_id", "track_id", "acq_date",
        "point_id",
        "distance_from_offshore", "alongtrack_distance",
        "beam_start_offset_m", "beam_end_distance_m",
        "median_point_spacing_m",
        "estimated_missing_start_points",
        "missing_start_flag",
        "box_length_m",
        "h_li", "geometry"
    ]

    for fam, content in dataset_clean.items():

        clipped = content.get("clipped")
        box_gdf = content.get("box")
        nearest_beam = content.get("nearest_beam")

        if clipped is None or clipped.empty:
            continue
        if box_gdf is None or box_gdf.empty:
            continue
        if nearest_beam is None:
            print(f"SKIP family {fam}: nearest_beam is missing.")
            continue

        gdf = clipped.to_crs(utm_epsg).copy()
        box_poly = box_gdf.to_crs(utm_epsg).geometry.iloc[0]

        # 1. Rebuild nearest_line from nearest_beam
        nearest_df = gdf[gdf["beam_id"] == nearest_beam].copy()

        if len(nearest_df) < 2:
            print(f"SKIP family {fam}: nearest_beam has < 2 points.")
            continue

        nearest_df = (
            nearest_df.assign(_y=nearest_df.geometry.y)
            .sort_values("_y")
            .drop(columns="_y")
        )
        nearest_line = LineString(nearest_df.geometry.tolist())

        # 2. Along-track direction (t_hat)
        p0 = nearest_line.interpolate(0.4, normalized=True)
        p1 = nearest_line.interpolate(0.6, normalized=True)

        vx, vy = p1.x - p0.x, p1.y - p0.y
        norm = np.hypot(vx, vy)

        if norm == 0:
            print(f"SKIP family {fam}: invalid beam direction.")
            continue

        t_hat = np.array([vx, vy]) / norm

        # Ensure offshore -> inland direction
        if t_hat[1] > 0:
            t_hat = -t_hat

        # 3. Offshore origin from the box projection
        box_coords = np.array(box_poly.exterior.coords[:-1])
        box_proj = box_coords @ t_hat

        offshore_origin_proj = box_proj.min()
        inland_end_proj = box_proj.max()
        box_length = inland_end_proj - offshore_origin_proj

        # 4. Project all points onto the along-track axis
        xy = np.array([(p.x, p.y) for p in gdf.geometry])
        proj = xy @ t_hat

        gdf["alongtrack_distance"] = proj - offshore_origin_proj
        gdf["distance_from_offshore"] = gdf["alongtrack_distance"]

        gdf.loc[gdf["alongtrack_distance"].abs() < 1e-6, "alongtrack_distance"] = 0.0

        # 5. Beam-level QA (detect missing offshore points)
        for beam_id, bdf in gdf.groupby("beam_id"):

            bdf = bdf.sort_values("alongtrack_distance").copy()

            if len(bdf) >= 2:
                diffs = np.diff(bdf["alongtrack_distance"].values)
                diffs = diffs[diffs > 0]
                spacing = np.median(diffs) if len(diffs) > 0 else np.nan
            else:
                spacing = np.nan

            start_offset = bdf["alongtrack_distance"].min()
            end_distance = bdf["alongtrack_distance"].max()

            if np.isfinite(spacing) and spacing > 0:
                estimated_missing = int(round(start_offset / spacing))
                missing_flag = start_offset > 1.5 * spacing
            else:
                estimated_missing = np.nan
                missing_flag = False

            bdf["point_id"] = np.arange(len(bdf))
            bdf["box_length_m"] = box_length
            bdf["beam_start_offset_m"] = start_offset
            bdf["beam_end_distance_m"] = end_distance
            bdf["median_point_spacing_m"] = spacing
            bdf["estimated_missing_start_points"] = estimated_missing
            bdf["missing_start_flag"] = missing_flag

            if "acq_date" in bdf.columns:
                bdf["acq_date"] = pd.to_datetime(bdf["acq_date"], errors="coerce")

            rows.append(bdf[[c for c in KEEP_COLS if c in bdf.columns]])

    if not rows:
        # No family produced a beam that survived clipping/QA. Return an empty
        # GeoDataFrame with the same schema as the non-empty path.
        print(
            f"⚠ No beam data survived distance filtering for TRACK_ID={track_id} "
            "— track will produce empty/degenerate output."
        )
        empty_df = pd.DataFrame(columns=[c for c in KEEP_COLS if c != "geometry"])
        empty_df["geometry"] = gpd.GeoSeries([], dtype="geometry")
        return gpd.GeoDataFrame(empty_df, geometry="geometry", crs=f"EPSG:{utm_epsg}")

    return gpd.GeoDataFrame(
        pd.concat(rows, ignore_index=True),
        crs=f"EPSG:{utm_epsg}"
    )
