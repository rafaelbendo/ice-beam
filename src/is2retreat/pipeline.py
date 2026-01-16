# src/is2retreat/pipeline.py
from __future__ import annotations

import pandas as pd
import geopandas as gpd

from .preprocessing import build_boxes_for_families
from .clustering import make_clusters, compute_cluster_angles, select_clusters_per_family
from .bias import apply_bias_filter_clusters


def _require_param(params: object, name: str):
    if params is None:
        raise ValueError("params is required. Pass Params() from is2retreat.config.")
    if not hasattr(params, name):
        raise ValueError(f"params must define `{name}`.")
    return getattr(params, name)


def run_workflow(
    track_id: str | int,
    dataset_raw: gpd.GeoDataFrame,
    shoreline_gdf: gpd.GeoDataFrame,
    params: object,
    verbose: bool = False,
):
    """
    Full ICESat-2 clustered-shoreline workflow for ONE parameter set.

    Required params (minimum):
      - UTM_EPSG
      - HALF_ALONG_M
      - HALF_ACROSS_M
      - CLUSTER_DISTANCE_M
      - MIN_BEAMS
      - BIAS_TOLERANCE
      - BIAS_X0
      - ANGLE_SEARCH_RADIUS
    """
    utm_epsg = int(_require_param(params, "UTM_EPSG"))
    half_along = float(_require_param(params, "HALF_ALONG_M"))
    half_across = float(_require_param(params, "HALF_ACROSS_M"))

    cluster_distance_m = float(_require_param(params, "CLUSTER_DISTANCE_M"))
    min_beams = int(_require_param(params, "MIN_BEAMS"))

    bias_tolerance = float(_require_param(params, "BIAS_TOLERANCE"))
    x0 = float(_require_param(params, "BIAS_X0"))

    angle_search_radius = float(_require_param(params, "ANGLE_SEARCH_RADIUS"))

    # Normalize track_id
    try:
        track_id_str = f"{int(track_id):04d}"
    except Exception:
        track_id_str = str(track_id)

    empty_gdf = gpd.GeoDataFrame()
    empty_df = pd.DataFrame()

    # ------------------------------------------------------------
    # STEP 1 — Oriented boxes + clipped profiles (DICT)
    # ------------------------------------------------------------
    dataset_clean = build_boxes_for_families(
        dataset_raw=dataset_raw,
        shoreline_gdf=shoreline_gdf,
        utm_epsg=utm_epsg,
        half_along=half_along,
        half_across=half_across,
        verbose=verbose,
    )

    if not isinstance(dataset_clean, dict) or len(dataset_clean) == 0:
        if verbose:
            print("⚠ dataset_clean is empty or invalid.")
        return None, None, dataset_clean, empty_gdf, empty_gdf, empty_gdf, empty_df, empty_df

    # ------------------------------------------------------------
    # STEP 2 — Clusters (Option A: explicit arguments)
    # ------------------------------------------------------------
    clusters_gdf, beam_gdf = make_clusters(
        dataset_clean=dataset_clean,
        cluster_distance_m=cluster_distance_m,
        pts_gdf=None,
        min_beams=min_beams,
        utm_epsg=utm_epsg,
        params=None,  # explicit style: do not rely on params inside clustering
    )

    if clusters_gdf is None or clusters_gdf.empty:
        if verbose:
            print(f"⚠ No clusters created at cluster size {cluster_distance_m}")
        return None, None, dataset_clean, empty_gdf, empty_gdf, empty_gdf, empty_df, empty_df

    # ------------------------------------------------------------
    # STEP 3 — Angles
    # ------------------------------------------------------------
    clusters_gdf = compute_cluster_angles(
        clusters_gdf,
        shoreline_gdf,
        search_radius=angle_search_radius,
    )

    # ------------------------------------------------------------
    # STEP 4 — Greedy selection
    # ------------------------------------------------------------
    selected_clusters, skipped_dict, summary_fam = select_clusters_per_family(
        clusters_gdf,
        min_beams=min_beams,
        track_id=track_id_str,
    )

    if selected_clusters is None or selected_clusters.empty:
        if verbose:
            print("⚠ All clusters were skipped by greedy selection.")
        return summary_fam, summary_fam, dataset_clean, clusters_gdf, empty_gdf, empty_gdf, empty_df, empty_df

    selected_clusters = selected_clusters.copy()
    selected_clusters["track_id"] = track_id_str

    # propagate angle_deg safely (join on gt_family + cluster_id)
    if "angle_deg" in clusters_gdf.columns and "angle_deg" not in selected_clusters.columns:
        selected_clusters = selected_clusters.merge(
            clusters_gdf[["gt_family", "cluster_id", "angle_deg"]],
            on=["gt_family", "cluster_id"],
            how="left",
        )

    # ------------------------------------------------------------
    # STEP 5 — Bias filter
    # (bias.py still reads tolerance from params; Option A here means
    #  we keep pipeline explicit where possible, but bias.py can still
    #  require params for its internal settings)
    # ------------------------------------------------------------
    filtered_profiles, bias_summary, bias_df = apply_bias_filter_clusters(
        dataset_raw=dataset_raw,
        selected_clusters=selected_clusters,
        params=params,
        bias_tolerance=bias_tolerance,
        x0=x0,
        verbose=verbose,
    )

    if filtered_profiles is None or filtered_profiles.empty:
        if verbose:
            print("⚠ Vertical bias filter removed all profiles.")
        return (
            summary_fam,
            summary_fam,
            dataset_clean,
            clusters_gdf,
            selected_clusters,
            gpd.GeoDataFrame(),
            empty_df,
            empty_df,
        )

    summary_clust = summary_fam.copy()

    return (
        summary_fam,
        summary_clust,
        dataset_clean,
        clusters_gdf,
        selected_clusters,
        filtered_profiles,
        bias_summary,
        bias_df,
    )


__all__ = ["run_workflow"]
