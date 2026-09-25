# ============================================================
# Lateral-growth clustering with vertical-bias control
# ============================================================
"""
For each ``gt_family``, beam positions are evaluated at a common offshore
distance ``xm`` to define lateral order. Clustering starts at one outer
lateral boundary and grows inward using the first beam as the reference,
then restarts from the opposite boundary.

A candidate beam is added only when it brackets ``x0``, has a new acquisition
date in the current cluster, and satisfies
``abs(elev_candidate(x0) - elev_reference(x0)) <= bias_tolerance``.
Same-date candidates are skipped without stopping growth; a candidate that
fails the bias tolerance (or exceeds ``size_limit`` meters of lateral width)
becomes the breakpoint and starts the next cluster. Clusters smaller than
``min_profiles`` are discarded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .utils import cluster_member_beams


def _as_utm_gdf(gdf, utm_epsg):
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{utm_epsg}")
    out = gpd.GeoDataFrame(gdf.copy(), geometry="geometry", crs=gdf.crs)
    target = f"EPSG:{utm_epsg}"
    if out.crs is None:
        out = out.set_crs(target, allow_override=True)
    elif str(out.crs) != target:
        out = out.to_crs(target)
    return out


def beam_position_at_xm(bdf, xm, dist_col="distance_from_offshore"):
    """Interpolated beam map position at xm; no extrapolation."""
    if bdf is None or bdf.empty or dist_col not in bdf.columns:
        return None

    p = bdf.dropna(subset=[dist_col, "geometry"]).copy()
    if len(p) < 2:
        return None

    p[dist_col] = pd.to_numeric(p[dist_col], errors="coerce")
    p["gx"] = p.geometry.x
    p["gy"] = p.geometry.y
    p = p.dropna(subset=[dist_col, "gx", "gy"])
    if len(p) < 2:
        return None

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

    if not (np.nanmin(d) <= xm <= np.nanmax(d)):
        return None

    return Point(float(np.interp(xm, d, x)), float(np.interp(xm, d, y)))


def interpolate_elevation_at_x0(prof, x0,
                                x_col="distance_from_offshore",
                                y_col="h_li"):
    """Interpolate h_li at x0 only when the profile brackets x0."""
    if prof is None or prof.empty or x_col not in prof.columns or y_col not in prof.columns:
        return np.nan

    p = prof.dropna(subset=[x_col, y_col]).copy()
    if len(p) < 2:
        return np.nan

    p[x_col] = pd.to_numeric(p[x_col], errors="coerce")
    p[y_col] = pd.to_numeric(p[y_col], errors="coerce")
    p = p.dropna(subset=[x_col, y_col])
    if len(p) < 2:
        return np.nan

    p = (
        p.groupby(x_col, as_index=False)[y_col]
         .mean()
         .sort_values(x_col)
    )
    if len(p) < 2:
        return np.nan

    xx = p[x_col].to_numpy(dtype=float)
    yy = p[y_col].to_numpy(dtype=float)
    if not (np.nanmin(xx) <= x0 <= np.nanmax(xx)):
        return np.nan

    return float(np.interp(x0, xx, yy))


def _beam_acq_date(bdf):
    if bdf is None or bdf.empty or "acq_date" not in bdf.columns:
        return pd.NaT
    vals = pd.to_datetime(bdf["acq_date"], errors="coerce").dropna()
    return vals.iloc[0].normalize() if len(vals) else pd.NaT


def _beam_line(bdf):
    p = bdf.dropna(subset=["geometry"]).copy()
    if len(p) < 2:
        return None
    sort_col = "distance_from_offshore" if "distance_from_offshore" in p.columns else None
    if sort_col is not None:
        p[sort_col] = pd.to_numeric(p[sort_col], errors="coerce")
        p = p.dropna(subset=[sort_col]).sort_values(sort_col)
    else:
        p = p.assign(_y=p.geometry.y).sort_values("_y")
    coords = [(geom.x, geom.y) for geom in p.geometry if geom is not None and not geom.is_empty]
    if len(coords) < 2:
        return None
    line = LineString(coords)
    return line if not line.is_empty and line.length > 0 else None


def lateral_order_beams(fam_pts, xm, x0):
    """Lateral ordering of beams from their positions at a common offshore distance."""
    records = []
    for beam_id, bdf in fam_pts.groupby("beam_id"):
        pos = beam_position_at_xm(bdf, xm=xm)
        if pos is None:
            continue
        records.append({
            "beam_id": str(beam_id).strip(),
            "geometry": pos,
            "x_xm": pos.x,
            "y_xm": pos.y,
            "acq_date": _beam_acq_date(bdf),
            "elev_x0": interpolate_elevation_at_x0(bdf, x0=x0),
            "n_points": len(bdf)
        })

    if not records:
        return pd.DataFrame(columns=["beam_id", "cross_track_pos"])

    order_df = pd.DataFrame(records)
    coords = order_df[["x_xm", "y_xm"]].to_numpy(dtype=float)

    if len(order_df) == 1:
        order_df["cross_track_pos"] = 0.0
    else:
        centered = coords - coords.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        order_df["cross_track_pos"] = centered @ axis

    return order_df.sort_values("cross_track_pos").reset_index(drop=True)


def grow_clusters_from_side(order_df,
                            fam_pts,
                            bias_tolerance,
                            side="left",
                            min_profiles=2,
                            size_limit=None):
    """
    Grow clusters inward from one lateral boundary.

    Same-date candidates are skipped and do not stop growth. A candidate that
    fails the vertical-bias test becomes the breakpoint/reference for the next
    cluster.

    If size_limit is not None, it is the maximum cluster width in meters
    along the lateral cross-track axis.
    """
    if order_df is None or order_df.empty:
        return []

    ordered = order_df.sort_values("cross_track_pos").copy()
    if side in ("right", "max", "opposite"):
        ordered = ordered.iloc[::-1].reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    profiles = {str(bid).strip(): g.copy() for bid, g in fam_pts.groupby("beam_id")}

    clusters = []
    current = []
    current_dates = set()
    ref_beam = None
    ref_elev = np.nan
    ref_pos = np.nan

    def finalize_current(reason):
        if len(current) >= min_profiles:
            clusters.append({
                "growth_side": side,
                "reference_beam": ref_beam,
                "beam_ids_ordered": current.copy(),
                "num_beams": len(current),
                "break_reason": reason,
                "reference_elev_x0": ref_elev
            })

    for _, row in ordered.iterrows():
        bid = str(row["beam_id"]).strip()
        prof = profiles.get(bid)
        elev = row.get("elev_x0", np.nan)
        acq_date = row.get("acq_date", pd.NaT)
        cross_pos = row.get("cross_track_pos", np.nan)

        if prof is None or prof.empty:
            continue

        if not np.isfinite(elev):
            # Bias cannot be evaluated without bracketing x0. This is not a
            # breakpoint because there is no valid bias failure to anchor on.
            continue

        if not current:
            ref_beam = bid
            ref_elev = elev
            ref_pos = cross_pos
            current = [bid]
            current_dates = set()
            if pd.notna(acq_date):
                current_dates.add(acq_date)
            continue

        if pd.notna(acq_date) and acq_date in current_dates:
            continue

        bias = elev - ref_elev
        if abs(bias) <= bias_tolerance:
            candidate_width = abs(cross_pos - ref_pos)

            if size_limit is not None and candidate_width > size_limit:
                finalize_current("size_limit")
                ref_beam = bid
                ref_elev = elev
                ref_pos = cross_pos
                current = [bid]
                current_dates = set()
                if pd.notna(acq_date):
                    current_dates.add(acq_date)
            else:
                current.append(bid)
                if pd.notna(acq_date):
                    current_dates.add(acq_date)
        else:
            finalize_current("bias_breakpoint")
            ref_beam = bid
            ref_elev = elev
            ref_pos = cross_pos
            current = [bid]
            current_dates = set()
            if pd.notna(acq_date):
                current_dates.add(acq_date)

    finalize_current("end_of_family")
    return clusters


def _cluster_geometry(member_lines):
    member_lines = [geom for geom in member_lines if geom is not None and not geom.is_empty]
    if not member_lines:
        return None
    merged = unary_union(member_lines)
    hull = merged.convex_hull
    if hull.geom_type in ("LineString", "MultiLineString"):
        return hull.buffer(1.0)
    if hull.geom_type == "Point":
        return hull.buffer(1.0)
    return hull


def make_clusters(pts_gdf,
                  utm_epsg,
                  bias_tolerance,
                  xm,
                  x0,
                  track_id=None,
                  min_beams=2,
                  size_limit=None,
                  cluster_distance_m=None):
    """
    Create clusters by lateral growth plus vertical-bias control.

    cluster_distance_m is legacy metadata only (the old buffered clustering
    used it); it is written to the output but does not affect clustering.

    Returns
    -------
    clusters_gdf, beam_gdf
    """
    target_crs = f"EPSG:{utm_epsg}"
    if cluster_distance_m is None:
        cluster_distance_m = np.nan
    source = _as_utm_gdf(pts_gdf, utm_epsg=utm_epsg)

    rows = []
    beam_line_rows = []
    cluster_id_counter = 1

    fams = sorted(source["gt_family"].dropna().astype(str).unique()) if "gt_family" in source.columns else []

    for fam in fams:
        fam_pts = source[source["gt_family"].astype(str) == fam].copy()
        if fam_pts.empty:
            continue

        fam_pts["beam_id"] = fam_pts["beam_id"].astype(str).str.strip()
        if "acq_date" in fam_pts.columns:
            fam_pts["acq_date"] = pd.to_datetime(fam_pts["acq_date"], errors="coerce").dt.normalize()

        order_df = lateral_order_beams(fam_pts, xm=xm, x0=x0)
        if order_df.empty:
            continue

        line_lookup = {}
        for beam_id, bdf in fam_pts.groupby("beam_id"):
            line = _beam_line(bdf)
            if line is None:
                continue
            bid = str(beam_id).strip()
            line_lookup[bid] = line
            beam_line_rows.append({
                "fam_beam": (fam, bid),
                "gt_family": fam,
                "beam_id": bid,
                "geometry": line
            })

        for side in ["left", "right"]:
            grown = grow_clusters_from_side(
                order_df=order_df,
                fam_pts=fam_pts,
                bias_tolerance=bias_tolerance,
                side=side,
                min_profiles=min_beams,
                size_limit=size_limit
            )

            for cl in grown:
                beam_ids = [str(b).strip() for b in cl["beam_ids_ordered"]]
                member_lines = [line_lookup.get(b) for b in beam_ids]
                geom = _cluster_geometry(member_lines)
                if geom is None:
                    continue

                ref_beam = str(cl["reference_beam"]).strip()
                member_points = fam_pts[fam_pts["beam_id"].isin(beam_ids)]
                center = geom.centroid
                rows.append({
                    "track_id": track_id,
                    "gt_family": fam,
                    "beam_id": ref_beam,
                    "reference_beam": ref_beam,
                    "beam_ids": [(fam, b) for b in beam_ids],
                    "beam_ids_ordered": beam_ids,
                    "num_beams": len(beam_ids),
                    "num_points": int(len(member_points)),
                    "cluster_distance_m": cluster_distance_m,
                    "cluster_id": cluster_id_counter,
                    "growth_side": cl["growth_side"],
                    "break_reason": cl["break_reason"],
                    "reference_elev_x0": cl["reference_elev_x0"],
                    "cluster_center": center,
                    "geometry": geom
                })
                cluster_id_counter += 1

    if rows:
        clusters_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)

        # GeoPandas can lose CRS context on an object column of centroid Points,
        # so convert the centroid column as an explicit GeoSeries.
        centers_ll = gpd.GeoSeries(
            clusters_gdf["cluster_center"],
            crs=target_crs
        ).to_crs(4326)

        clusters_gdf["center_lon"] = centers_ll.x.values
        clusters_gdf["center_lat"] = centers_ll.y.values
        clusters_gdf = clusters_gdf.set_geometry("geometry")
    else:
        clusters_gdf = gpd.GeoDataFrame(
            columns=[
                "track_id", "gt_family", "beam_id", "reference_beam", "beam_ids",
                "beam_ids_ordered", "num_beams", "num_points", "cluster_distance_m",
                "cluster_id", "growth_side", "break_reason", "reference_elev_x0",
                "cluster_center", "center_lon", "center_lat", "geometry"
            ],
            geometry="geometry",
            crs=target_crs
        )

    if beam_line_rows:
        beam_gdf = gpd.GeoDataFrame(beam_line_rows, geometry="geometry", crs=target_crs)
    else:
        beam_gdf = gpd.GeoDataFrame(
            columns=["fam_beam", "gt_family", "beam_id", "geometry"],
            geometry="geometry",
            crs=target_crs
        )

    return clusters_gdf, beam_gdf


# ======================================================================
#  Selected-cluster construction
# ======================================================================
# Growing from both sides can produce the same beam set twice; keep one.

def _cluster_beam_signature(row):
    return "|".join(sorted(cluster_member_beams(row)))


def select_clusters_per_family(clusters_gdf, min_profiles=2, track_id=None):
    """
    Drop clusters below min_profiles and exact duplicate beam sets within
    each gt_family (preferring more beams, then left-side growth).

    Returns
    -------
    selected_clusters, skipped_dict, summary_df
    """
    empty_summary_cols = [
        "gt_family", "track_id", "total_clusters", "selected_clusters",
        "skipped_clusters", "too_far", "cluster_id", "angle_deg"
    ]

    if clusters_gdf is None or clusters_gdf.empty:
        empty_gdf = gpd.GeoDataFrame(
            columns=["cluster_id", "gt_family", "geometry"],
            crs=getattr(clusters_gdf, "crs", None),
            geometry="geometry"
        )
        return empty_gdf, {}, pd.DataFrame(columns=empty_summary_cols)

    sc = clusters_gdf.copy()

    if "num_beams" in sc.columns:
        sc = sc.loc[sc["num_beams"] >= min_profiles].copy()

    if sc.empty:
        empty_gdf = gpd.GeoDataFrame(sc, geometry="geometry", crs=clusters_gdf.crs)
        return empty_gdf, {}, pd.DataFrame(columns=empty_summary_cols)

    if "member_union_geom" not in sc.columns:
        sc["member_union_geom"] = sc.geometry

    sc["beam_signature"] = sc.apply(_cluster_beam_signature, axis=1)

    if "growth_side" not in sc.columns:
        sc["growth_side"] = None

    sc["_side_rank"] = sc["growth_side"].map({"left": 0, "right": 1}).fillna(2)

    sc = (
        sc.sort_values(
            ["gt_family", "beam_signature", "num_beams", "_side_rank", "cluster_id"],
            ascending=[True, True, False, True, True]
        )
        .drop_duplicates(subset=["gt_family", "beam_signature"], keep="first")
        .drop(columns=["_side_rank"])
        .reset_index(drop=True)
    )

    skipped_dict = {}
    summary_rows = []

    for fam, fam_clusters in clusters_gdf.groupby("gt_family"):
        fam_selected = sc[sc["gt_family"] == fam].copy()

        track_val = (
            fam_clusters["track_id"].dropna().iloc[0]
            if "track_id" in fam_clusters.columns and fam_clusters["track_id"].notna().any()
            else track_id
        )

        selected_ids = (
            fam_selected["cluster_id"].tolist()
            if "cluster_id" in fam_selected.columns
            else []
        )

        all_ids = (
            fam_clusters["cluster_id"].tolist()
            if "cluster_id" in fam_clusters.columns
            else []
        )

        skipped_ids = sorted(set(all_ids) - set(selected_ids))
        skipped_dict[fam] = skipped_ids

        rep_cluster_id = selected_ids[0] if selected_ids else None

        rep_angle = np.nan
        if rep_cluster_id is not None and "angle_deg" in fam_selected.columns:
            vals = fam_selected.loc[
                fam_selected["cluster_id"] == rep_cluster_id,
                "angle_deg"
            ].dropna()
            rep_angle = vals.iloc[0] if len(vals) else np.nan

        summary_rows.append({
            "gt_family": fam,
            "track_id": track_val,
            "total_clusters": int(len(fam_clusters)),
            "selected_clusters": int(len(fam_selected)),
            "skipped_clusters": int(len(skipped_ids)),
            "too_far": 0,
            "cluster_id": rep_cluster_id,
            "angle_deg": rep_angle
        })

    summary_df = pd.DataFrame(summary_rows)

    return sc.reset_index(drop=True), skipped_dict, summary_df


def add_cluster_width_m(selected_clusters, dataset_raw, xm, utm_epsg):
    """
    Add cluster_width_m: distance in meters between the outermost member
    beams, evaluated at the common offshore distance xm.
    """
    if selected_clusters is None or selected_clusters.empty:
        return selected_clusters

    sc = selected_clusters.copy()

    df = dataset_raw.copy()
    df["gt_family"] = df["gt_family"].astype(str)
    df["beam_id"] = df["beam_id"].astype(str).str.strip()

    target_crs = f"EPSG:{utm_epsg}"
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=dataset_raw.crs)

    if df.crs is None:
        df = df.set_crs(target_crs, allow_override=True)
    elif str(df.crs) != target_crs:
        df = df.to_crs(target_crs)

    widths = []

    for _, cl in sc.iterrows():
        fam = str(cl["gt_family"])
        beam_ids = cluster_member_beams(cl)

        pts = []

        for bid in beam_ids:
            bdf = df[
                (df["gt_family"] == fam) &
                (df["beam_id"] == bid)
            ].copy()

            if bdf.empty:
                continue

            pt = beam_position_at_xm(bdf, xm=xm)

            if pt is not None:
                pts.append(pt)

        if len(pts) < 2:
            widths.append(0.0 if len(pts) == 1 else np.nan)
            continue

        coords = np.array([[p.x, p.y] for p in pts], dtype=float)

        if len(coords) == 2:
            width_m = float(np.linalg.norm(coords[1] - coords[0]))
        else:
            centered = coords - coords.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            lateral_pos = centered @ axis
            width_m = float(lateral_pos.max() - lateral_pos.min())

        widths.append(width_m)

    sc["cluster_width_m"] = widths

    return sc
