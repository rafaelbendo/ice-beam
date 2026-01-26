# src/is2retreat/clustering.py
from __future__ import annotations

"""
Clustering (buffered centerlines) + cluster/shoreline angle + selection.

IMPORTANT (legacy contract)
--------------------------
Clusters store physical beam identifiers as (gt_family, beam_id) tuples in `beam_ids`.
Downstream code depends on this. Do NOT change that contract without updating all consumers.

NEW (your intent)
-----------------
You want MIN_BEAMS to mean minimum number of "files" (acquisitions), where a file is:
  (gt_family, beam_id, cycle_date)

We store those in:
  - acq_ids : list[(gt_family, beam_id, cycle_date)]
  - num_acq : len(acq_ids)

Selection
---------
We implement your original greedy selection logic, but applied to acquisitions (acq_ids),
not physical beams (beam_ids). This avoids the "1 cluster per family" collapse.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import nearest_points


# ============================================================
# Clustering (buffered centerlines)
# ============================================================
def make_clusters(
    dataset_clean: Dict[str, Dict[str, object]],
    cluster_distance_m: Optional[float] = None,
    pts_gdf: Optional[gpd.GeoDataFrame] = None,
    min_beams: Optional[int] = None,   # interpreted as min acquisitions/files
    utm_epsg: Optional[int] = None,
    params: Optional[object] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create clusters per (gt_family, beam_id, acq_date) by buffering a per-cycle centerline.

    Outputs clusters_gdf with:
      - beam_ids (legacy): list[(gt_family, beam_id)]
      - acq_ids          : list[(gt_family, beam_id, cycle_date)]   <-- "files"
      - num_beams        : len(beam_ids)
      - num_acq          : len(acq_ids)

    Filtering:
      - min_beams is treated as min acquisitions/files => keep clusters where num_acq >= min_beams
    """
    # Params defaults
    if params is not None:
        if cluster_distance_m is None:
            cluster_distance_m = getattr(params, "CLUSTER_DISTANCE_M", None)
        if min_beams is None:
            min_beams = getattr(params, "MIN_BEAMS", 2)
        if utm_epsg is None:
            utm_epsg = getattr(params, "UTM_EPSG", 32606)

    if cluster_distance_m is None:
        raise ValueError("cluster_distance_m must be provided (or via params.CLUSTER_DISTANCE_M).")
    if min_beams is None:
        min_beams = 2
    if utm_epsg is None:
        utm_epsg = 32606

    cluster_distance_m = float(cluster_distance_m)
    min_beams = int(min_beams)
    utm_epsg = int(utm_epsg)

    target_crs = f"EPSG:{utm_epsg}"
    clusters: List[gpd.GeoDataFrame] = []

    # per-cycle centerlines
    beam_lines: Dict[Tuple[str, str, pd.Timestamp], LineString] = {}
    cluster_id_counter = 1

    for fam, content in dataset_clean.items():
        if not content or content.get("box") is None:
            continue

        fam = str(fam).strip()

        # family box
        box_gdf = content["box"]
        box_utm = box_gdf.to_crs(utm_epsg) if str(box_gdf.crs) != target_crs else box_gdf
        box_geom = box_utm.geometry.iloc[0]

        # points source
        if pts_gdf is not None:
            fam_pts = pts_gdf.loc[pts_gdf["gt_family"].astype(str).str.strip() == fam].copy()
        else:
            fam_pts = content.get("clipped", gpd.GeoDataFrame()).copy()

        if fam_pts is None or fam_pts.empty:
            continue

        # CRS harmonization
        if fam_pts.crs is None or str(fam_pts.crs) != target_crs:
            fam_pts = fam_pts.to_crs(utm_epsg)

        # required columns
        if "beam_id" not in fam_pts.columns:
            raise ValueError("Points GeoDataFrame must have a 'beam_id' column.")
        fam_pts["beam_id"] = fam_pts["beam_id"].astype(str).str.strip()

        if "acq_date" in fam_pts.columns:
            fam_pts["acq_date"] = pd.to_datetime(fam_pts["acq_date"], errors="coerce")
        else:
            fam_pts["acq_date"] = pd.NaT

        # within/touch box
        fam_pts = fam_pts[fam_pts.geometry.within(box_geom) | fam_pts.geometry.touches(box_geom)]
        if fam_pts.empty:
            continue

        # Build per (beam_id, acq_date) centerlines + buffered polygons
        for (beam_id, acq_date), g in fam_pts.groupby(["beam_id", "acq_date"]):
            if len(g) < 2:
                continue

            cycle_date = pd.to_datetime(acq_date, errors="coerce")
            if pd.notna(cycle_date):
                cycle_date = cycle_date.normalize()

            g_sorted = (
                g.assign(_y=g.geometry.y)
                 .sort_values("_y")
                 .drop(columns="_y")
            )

            line = LineString(g_sorted.geometry.values)
            if line.length == 0:
                continue

            key = (fam, str(beam_id), cycle_date)
            beam_lines[key] = line

            poly = line.buffer(cluster_distance_m / 2.0)

            clusters.append(
                gpd.GeoDataFrame(
                    {
                        "gt_family": [fam],
                        "beam_id": [str(beam_id)],
                        "cycle_date": [cycle_date],
                        "num_points": [int(len(g))],
                        "cluster_distance_m": [cluster_distance_m],
                        "cluster_id": [int(cluster_id_counter)],
                    },
                    geometry=[poly],
                    crs=target_crs,
                )
            )
            cluster_id_counter += 1

    if not clusters:
        empty_clusters = gpd.GeoDataFrame(columns=["cluster_id"], crs=target_crs)
        empty_beams = gpd.GeoDataFrame(columns=["gt_family", "beam_id", "geometry"], crs=target_crs)
        return empty_clusters, empty_beams

    clusters_gdf = gpd.GeoDataFrame(pd.concat(clusters, ignore_index=True), crs=target_crs)

    # Beam centerlines GeoDataFrame (per-cycle)
    if beam_lines:
        beam_gdf = gpd.GeoDataFrame(
            {
                "fam_beam_cycle": list(beam_lines.keys()),  # (gt_family, beam_id, cycle_date)
                "gt_family": [k[0] for k in beam_lines.keys()],
                "beam_id": [k[1] for k in beam_lines.keys()],
                "cycle_date": [k[2] for k in beam_lines.keys()],
                "fam_beam": [(k[0], k[1]) for k in beam_lines.keys()],  # legacy
            },
            geometry=list(beam_lines.values()),
            crs=target_crs,
        )
    else:
        beam_gdf = gpd.GeoDataFrame(
            columns=["fam_beam_cycle", "gt_family", "beam_id", "cycle_date", "fam_beam", "geometry"],
            crs=target_crs,
        )

    # Intersections: acquisitions + legacy beams
    beam_ids_legacy: List[list] = []
    acq_ids: List[list] = []

    for _, row in clusters_gdf.iterrows():
        hit_acq = beam_gdf.loc[beam_gdf.intersects(row.geometry), "fam_beam_cycle"].tolist()

        # unique acquisitions (stable order)
        seen_a = set()
        uniq_acq = []
        for a in hit_acq:
            if a not in seen_a:
                seen_a.add(a)
                uniq_acq.append(a)
        acq_ids.append(uniq_acq)

        # derive unique legacy beams from acquisitions
        seen_b = set()
        uniq_beams = []
        for fam0, bid0, _cyc0 in uniq_acq:
            key = (fam0, bid0)
            if key not in seen_b:
                seen_b.add(key)
                uniq_beams.append(key)
        beam_ids_legacy.append(uniq_beams)

    clusters_gdf["beam_ids"] = beam_ids_legacy
    clusters_gdf["acq_ids"] = acq_ids
    clusters_gdf["num_beams"] = clusters_gdf["beam_ids"].apply(len)
    clusters_gdf["num_acq"] = clusters_gdf["acq_ids"].apply(len)

    # Filter by minimum acquisitions/files (your meaning of MIN_BEAMS)
    if min_beams > 1:
        clusters_gdf = clusters_gdf.loc[clusters_gdf["num_acq"] >= min_beams].copy()

    # Cluster center lat/lon
    clusters_gdf["cluster_center"] = clusters_gdf.geometry.centroid
    centers_ll = clusters_gdf.set_geometry("cluster_center").to_crs(4326)
    clusters_gdf["center_lon"] = centers_ll.geometry.x
    clusters_gdf["center_lat"] = centers_ll.geometry.y
    clusters_gdf = clusters_gdf.set_geometry("geometry")

    return clusters_gdf, beam_gdf


# ============================================================
# Cluster/shoreline angle (PCA axis vs shoreline tangent)
# ============================================================
def _union_all(geoseries):
    if hasattr(geoseries, "union_all"):
        try:
            return geoseries.union_all()
        except Exception:
            pass
    return geoseries.unary_union


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan
    cosang = np.clip(dot / denom, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cosang)))
    return ang if ang <= 90.0 else 180.0 - ang


def extract_shoreline_tangent(shoreline_geom, pt, search_radius: float = 10.0) -> np.ndarray:
    proj_dist = shoreline_geom.project(pt)
    d1 = max(float(proj_dist) - float(search_radius), 0.0)
    d2 = min(float(proj_dist) + float(search_radius), float(shoreline_geom.length))
    p1 = shoreline_geom.interpolate(d1)
    p2 = shoreline_geom.interpolate(d2)
    return np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)


def polygon_principal_axis(poly) -> Optional[np.ndarray]:
    if poly is None or poly.is_empty:
        return None
    coords = np.array(poly.exterior.coords, dtype=float)
    if coords.shape[0] < 3:
        return None
    coords_centered = coords - coords.mean(axis=0)
    C = np.cov(coords_centered.T)
    eigenvals, eigenvecs = np.linalg.eig(C)
    idx = int(np.argmax(eigenvals))
    return eigenvecs[:, idx]


def compute_cluster_angles(
    clusters_gdf: gpd.GeoDataFrame,
    shoreline: Union[gpd.GeoDataFrame, str],
    search_radius: Optional[float] = None,
    params: Optional[object] = None,
) -> gpd.GeoDataFrame:
    if params is not None and search_radius is None:
        search_radius = getattr(params, "ANGLE_SEARCH_RADIUS", None)
    if search_radius is None:
        search_radius = 10.0

    if clusters_gdf is None or clusters_gdf.empty:
        out = clusters_gdf.copy()
        out["angle_deg"] = []
        return out

    if clusters_gdf.crs is None:
        raise ValueError("clusters_gdf has no CRS. Set a CRS before computing angles.")

    if isinstance(shoreline, str):
        shoreline = gpd.read_file(shoreline)

    shoreline = shoreline.to_crs(clusters_gdf.crs)
    shoreline_geom = _union_all(shoreline.geometry)

    angle_vals: List[float] = []
    for _, row in clusters_gdf.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            angle_vals.append(np.nan)
            continue

        v_cluster = polygon_principal_axis(poly)
        if v_cluster is None or np.linalg.norm(v_cluster) == 0:
            angle_vals.append(np.nan)
            continue

        nearest = nearest_points(poly, shoreline_geom)[1]
        v_shore = extract_shoreline_tangent(shoreline_geom, nearest, search_radius=float(search_radius))
        angle_vals.append(angle_between(v_cluster, v_shore))

    out = clusters_gdf.copy()
    out["angle_deg"] = np.round(angle_vals, 2)
    return out


# ============================================================
# Cluster selection (greedy acquisition coverage per family)
# ============================================================
def select_clusters_per_family(
    clusters_gdf: gpd.GeoDataFrame,
    min_beams: Optional[int] = None,      # interpreted as min acquisitions/files
    track_id: Optional[str] = None,
    params: Optional[object] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, List[int]], pd.DataFrame]:
    """
    Greedy selection:
      - Eligible if num_acq >= min_beams (your "files")
      - Prefer clusters with more acquisitions (desc num_acq)
      - Select ONLY if it adds at least one new acquisition not covered yet

    Optional:
      - if params.ANGLE_CUTOFF_DEG (or ANGLE_MAX_DEG) is defined and angle_deg exists,
        we filter out clusters with angle_deg > cutoff BEFORE greedy selection.
    """
    if params is not None and min_beams is None:
        min_beams = getattr(params, "MIN_BEAMS", None)
    if min_beams is None:
        min_beams = 2
    min_beams = int(min_beams)

    angle_cutoff = None
    if params is not None:
        angle_cutoff = getattr(params, "ANGLE_CUTOFF_DEG", None)
        if angle_cutoff is None:
            angle_cutoff = getattr(params, "ANGLE_MAX_DEG", None)

    if clusters_gdf is None or clusters_gdf.empty:
        empty_gdf = gpd.GeoDataFrame(
            columns=["cluster_id", "gt_family", "geometry"],
            crs=getattr(clusters_gdf, "crs", None),
            geometry="geometry",
        )
        summary_df = pd.DataFrame(
            columns=[
                "gt_family",
                "track_id",
                "total_clusters",
                "selected_clusters",
                "skipped_clusters",
                "too_few_beams",
                "representative_cluster_id",
            ]
        )
        return empty_gdf, {}, summary_df

    gdf = clusters_gdf.copy()

    if "acq_ids" not in gdf.columns:
        raise ValueError("clusters_gdf must include 'acq_ids'. Re-run make_clusters with updated code.")
    if "num_acq" not in gdf.columns:
        gdf["num_acq"] = gdf["acq_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Optional angle pre-filter
    if angle_cutoff is not None and "angle_deg" in gdf.columns:
        gdf = gdf.loc[(gdf["angle_deg"].isna()) | (gdf["angle_deg"] <= float(angle_cutoff))].copy()

    selected_list: List[gpd.GeoDataFrame] = []
    skipped_dict: Dict[str, List[int]] = {}
    summary_rows: List[dict] = []

    for fam, fam_clusters in gdf.groupby("gt_family", sort=True):
        fam_clusters = fam_clusters.copy()
        total = len(fam_clusters)

        fam_track = None
        if "track_id" in fam_clusters.columns and fam_clusters["track_id"].notna().any():
            modes = fam_clusters["track_id"].mode()
            fam_track = modes.iat[0] if not modes.empty else fam_clusters["track_id"].dropna().iloc[0]
        elif track_id is not None:
            fam_track = track_id

        too_few_ids = fam_clusters.loc[fam_clusters["num_acq"] < min_beams, "cluster_id"].astype(int).tolist()
        fam_core = fam_clusters.loc[fam_clusters["num_acq"] >= min_beams].copy()

        if fam_core.empty:
            summary_rows.append(
                {
                    "gt_family": fam,
                    "track_id": fam_track,
                    "total_clusters": total,
                    "selected_clusters": 0,
                    "skipped_clusters": 0,
                    "too_few_beams": len(too_few_ids),  # keep column name for compatibility
                    "representative_cluster_id": None,
                }
            )
            skipped_dict[str(fam)] = []
            continue

        # Sort: more acquisitions first, then smaller angle if available, then cluster_id
        sort_cols = ["num_acq"]
        asc = [False]
        if "angle_deg" in fam_core.columns:
            sort_cols.append("angle_deg")
            asc.append(True)
        sort_cols.append("cluster_id")
        asc.append(True)
        fam_core = fam_core.sort_values(sort_cols, ascending=asc)

        covered_acq = set()
        selected_ids: List[int] = []
        skipped_ids: List[int] = []

        for _, row in fam_core.iterrows():
            acqs = set(row["acq_ids"]) if isinstance(row.get("acq_ids"), list) else set()
            if acqs - covered_acq:
                selected_ids.append(int(row["cluster_id"]))
                covered_acq |= acqs
            else:
                skipped_ids.append(int(row["cluster_id"]))

        fam_selected = fam_core.loc[fam_core["cluster_id"].isin(selected_ids)].copy()
        if fam_track is not None:
            fam_selected["track_id"] = fam_track

        selected_list.append(fam_selected)
        skipped_dict[str(fam)] = skipped_ids

        summary_rows.append(
            {
                "gt_family": fam,
                "track_id": fam_track,
                "total_clusters": total,
                "selected_clusters": len(selected_ids),
                "skipped_clusters": len(skipped_ids),
                "too_few_beams": len(too_few_ids),
                "representative_cluster_id": selected_ids[0] if selected_ids else None,
            }
        )

    selected_clusters = (
        gpd.GeoDataFrame(pd.concat(selected_list, ignore_index=True), crs=gdf.crs)
        if selected_list
        else gpd.GeoDataFrame(columns=["cluster_id", "gt_family", "geometry"], crs=gdf.crs, geometry="geometry")
    )

    summary_df = pd.DataFrame(summary_rows)
    return selected_clusters, skipped_dict, summary_df


__all__ = [
    "make_clusters",
    "compute_cluster_angles",
    "select_clusters_per_family",
    "angle_between",
    "extract_shoreline_tangent",
    "polygon_principal_axis",
]
