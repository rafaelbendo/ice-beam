# src/is2retreat/clustering.py
from __future__ import annotations

"""
Clustering (buffered centerlines) + cluster/shoreline angle + greedy selection.

IMPORTANT
---------
Clusters store beam identifiers as (gt_family, beam_id) tuples in the `beam_ids`
column. This is legacy but intentional.

Several downstream steps (bias filtering, bluff extraction) depend on this exact
format and commonly do things like: [b[1] for b in beam_ids].

Do NOT refactor `beam_ids` into plain strings unless you update ALL consumers.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import nearest_points


def _require_param(params: object, name: str):
    if params is None:
        raise ValueError("params is required. Pass Params() from is2retreat.config.")
    if not hasattr(params, name):
        raise ValueError(f"params must define `{name}`.")
    return getattr(params, name)


# ============================================================
# Clustering (buffered centerlines)
# ============================================================
def make_clusters(
    dataset_clean: Dict[str, Dict[str, object]],
    cluster_distance_m: Optional[float] = None,
    pts_gdf: Optional[gpd.GeoDataFrame] = None,
    min_beams: Optional[int] = None,
    utm_epsg: Optional[int] = None,
    params: Optional[object] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create clusters of beams for each gt_family using buffered centerlines.
    Each beam becomes a polygon buffer around a centerline, and clusters are
    defined by intersecting polygons.

    Adds cluster center coordinates (lat/lon).

    Notes
    -----
    `beam_ids` are stored as (gt_family, beam_id) tuples. Keep this.
    """

    # Params defaults (only if values not explicitly provided)
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

    target_crs = f"EPSG:{int(utm_epsg)}"

    clusters: List[gpd.GeoDataFrame] = []
    beam_lines: Dict[Tuple[str, str], LineString] = {}
    cluster_id_counter = 1

    for fam, content in dataset_clean.items():
        if not content or content.get("box") is None:
            continue

        # Oriented box for this family
        box_gdf = content["box"]
        box_utm = box_gdf.to_crs(int(utm_epsg)) if str(box_gdf.crs) != target_crs else box_gdf
        box_geom = box_utm.geometry.iloc[0]

        # Points source
        if pts_gdf is not None:
            fam_pts = pts_gdf.loc[pts_gdf["gt_family"] == fam].copy()
        else:
            fam_pts = content.get("clipped", gpd.GeoDataFrame()).copy()

        if fam_pts.empty:
            continue

        # CRS harmonization
        if fam_pts.crs is None or str(fam_pts.crs) != target_crs:
            fam_pts = fam_pts.to_crs(int(utm_epsg))

        # Keep only points inside/touching box
        fam_pts = fam_pts[fam_pts.geometry.within(box_geom) | fam_pts.geometry.touches(box_geom)]
        if fam_pts.empty:
            continue

        # Build centerlines per beam + buffer polygons
        for beam_id, g in fam_pts.groupby("beam_id"):
            if len(g) < 2:
                continue

            g_sorted = (
                g.assign(_y=g.geometry.y)
                .sort_values("_y")
                .drop(columns="_y")
            )

            line = LineString(g_sorted.geometry.values)
            if line.length == 0:
                continue

            key = (str(fam), str(beam_id))
            beam_lines[key] = line

            poly = line.buffer(float(cluster_distance_m) / 2.0)

            clusters.append(
                gpd.GeoDataFrame(
                    {
                        "gt_family": [str(fam)],
                        "beam_id": [str(beam_id)],
                        "num_points": [int(len(g))],
                        "cluster_distance_m": [float(cluster_distance_m)],
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

    # Beam centerlines
    if beam_lines:
        beam_gdf = gpd.GeoDataFrame(
            {
                "fam_beam": list(beam_lines.keys()),  # (gt_family, beam_id)
                "gt_family": [fb[0] for fb in beam_lines.keys()],
                "beam_id": [fb[1] for fb in beam_lines.keys()],
            },
            geometry=list(beam_lines.values()),
            crs=target_crs,
        )
    else:
        beam_gdf = gpd.GeoDataFrame(
            columns=["fam_beam", "gt_family", "beam_id", "geometry"],
            crs=target_crs,
        )

    # Beams intersecting each buffer polygon
    # NOTE: `beam_ids` are stored as (gt_family, beam_id) tuples on purpose.
    clusters_gdf["beam_ids"] = [
        beam_gdf.loc[beam_gdf.intersects(row.geometry), "fam_beam"].tolist()
        for _, row in clusters_gdf.iterrows()
    ]
    clusters_gdf["num_beams"] = clusters_gdf["beam_ids"].apply(len)

    # Optional filter by required minimum beams
    if isinstance(min_beams, (int, np.integer)) and int(min_beams) > 1:
        clusters_gdf = clusters_gdf.loc[clusters_gdf["num_beams"] >= int(min_beams)].copy()

    # Cluster center lat/lon (centroid computed in UTM, then projected to 4326)
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
    """Angle between vectors (acute, 0–90°)."""
    dot = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan

    cosang = np.clip(dot / denom, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cosang)))
    return ang if ang <= 90.0 else 180.0 - ang


def extract_shoreline_tangent(shoreline_geom, pt, search_radius: float = 10.0) -> np.ndarray:
    """Approximate shoreline tangent vector near the closest point."""
    proj_dist = shoreline_geom.project(pt)

    d1 = max(float(proj_dist) - float(search_radius), 0.0)
    d2 = min(float(proj_dist) + float(search_radius), float(shoreline_geom.length))

    p1 = shoreline_geom.interpolate(d1)
    p2 = shoreline_geom.interpolate(d2)

    return np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)


def polygon_principal_axis(poly) -> Optional[np.ndarray]:
    """Compute dominant direction of a polygon using PCA on its exterior coords."""
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
    """
    Compute angle between cluster dominant axis (PCA) and local shoreline tangent.

    Adds column:
      - angle_deg
    """
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
# Cluster selection (greedy coverage per family)
# ============================================================
def select_clusters_per_family(
    clusters_gdf: gpd.GeoDataFrame,
    min_beams: Optional[int] = None,
    track_id: Optional[str] = None,
    params: Optional[object] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, List[int]], pd.DataFrame]:
    """
    Greedy cluster selection per gt_family.

    Rule:
      - select a cluster only if it adds at least one new beam
        not already covered by previously selected clusters.
    """
    if params is not None and min_beams is None:
        min_beams = getattr(params, "MIN_BEAMS", None)
    if min_beams is None:
        min_beams = 2

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

    selected_list: List[gpd.GeoDataFrame] = []
    skipped_dict: Dict[str, List[int]] = {}
    summary_rows: List[dict] = []

    for fam, fam_clusters in clusters_gdf.groupby("gt_family"):
        fam_clusters = fam_clusters.copy()
        total = len(fam_clusters)

        # Track id priority: column -> arg -> None
        fam_track = None
        if "track_id" in fam_clusters.columns and fam_clusters["track_id"].notna().any():
            modes = fam_clusters["track_id"].mode()
            fam_track = modes.iat[0] if not modes.empty else fam_clusters["track_id"].dropna().iloc[0]
        elif track_id is not None:
            fam_track = track_id

        # Clusters that fail min_beams
        too_few_ids: List[int] = []
        if "num_beams" in fam_clusters.columns:
            too_few_ids = fam_clusters.loc[fam_clusters["num_beams"] < int(min_beams), "cluster_id"].astype(int).tolist()

        # Eligible clusters
        if "num_beams" in fam_clusters.columns:
            fam_core = fam_clusters.loc[fam_clusters["num_beams"] >= int(min_beams)].copy()
        else:
            fam_core = fam_clusters.copy()

        if fam_core.empty:
            summary_rows.append(
                {
                    "gt_family": fam,
                    "track_id": fam_track,
                    "total_clusters": total,
                    "selected_clusters": 0,
                    "skipped_clusters": 0,
                    "too_few_beams": len(too_few_ids),
                    "representative_cluster_id": None,
                }
            )
            skipped_dict[fam] = []
            continue

        covered_beams = set()
        selected_ids: List[int] = []
        skipped_ids: List[int] = []

        fam_core = fam_core.sort_values("num_beams", ascending=False) if "num_beams" in fam_core.columns else fam_core

        for _, row in fam_core.iterrows():
            beams = set(row["beam_ids"]) if isinstance(row.get("beam_ids"), list) else set()
            if beams - covered_beams:
                selected_ids.append(int(row["cluster_id"]))
                covered_beams |= beams
            else:
                skipped_ids.append(int(row["cluster_id"]))

        fam_selected = fam_core[fam_core["cluster_id"].isin(selected_ids)].copy()

        if fam_track is not None:
            fam_selected["track_id"] = fam_track

        selected_list.append(fam_selected)
        skipped_dict[fam] = skipped_ids

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
        gpd.GeoDataFrame(pd.concat(selected_list, ignore_index=True), crs=clusters_gdf.crs)
        if selected_list
        else gpd.GeoDataFrame(
            columns=["cluster_id", "gt_family", "geometry"],
            crs=clusters_gdf.crs,
            geometry="geometry",
        )
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
