# ============================================================
# Clustering workflow for ONE bias tolerance
# ============================================================
from __future__ import annotations

from typing import NamedTuple, Optional

import pandas as pd
import geopandas as gpd

from .angles import compute_cluster_angles
from .bias import apply_bias_filter_clusters
from .clustering import add_cluster_width_m, make_clusters, select_clusters_per_family
from .config import Params
from .geometry import build_boxes_for_families
from .utils import format_track_id


class WorkflowResult(NamedTuple):
    summary_fam: Optional[pd.DataFrame]
    summary_clust: Optional[pd.DataFrame]
    dataset_clean: dict
    clusters_gdf: gpd.GeoDataFrame
    selected_clusters: gpd.GeoDataFrame
    filtered_profiles: gpd.GeoDataFrame
    bias_summary: pd.DataFrame
    bias_df: pd.DataFrame
    beam_angle_table: pd.DataFrame


def run_workflow(
        track_id,
        dataset_raw,
        shoreline_gdf,
        bias_tolerance,
        params: Params,
        utm_epsg: int,
        verbose=False
    ) -> WorkflowResult:
    """
    Clustered-shoreline workflow for one bias tolerance:

        1) Build oriented shoreline boxes per gt_family
        2) Build lateral-growth clusters (XM ordering, X0 bias control)
        3) Per-beam shoreline angle summaries for each cluster
        4) Remove duplicate clusters with identical beam sets
        5) Add cluster_width_m, apply the final bias/profile filter
    """
    track_id = format_track_id(track_id)
    empty_gdf = gpd.GeoDataFrame()
    empty_df = pd.DataFrame()

    dataset_clean = build_boxes_for_families(
        dataset_raw=dataset_raw,
        shoreline_gdf=shoreline_gdf,
        utm_epsg=utm_epsg,
        half_along=params.HALF_ALONG_M,
        half_across=params.HALF_ACROSS_M,
        gtx=params.GTX,
        verbose=verbose
    )

    if dataset_clean is None or not isinstance(dataset_clean, dict) or len(dataset_clean) == 0:
        if verbose:
            print("No valid dataset_clean produced.")
        return WorkflowResult(None, None, dataset_clean, empty_gdf, empty_gdf, empty_gdf, empty_df, empty_df, empty_df)

    clusters_gdf, _beam_gdf = make_clusters(
        pts_gdf=dataset_raw,
        utm_epsg=utm_epsg,
        bias_tolerance=bias_tolerance,
        xm=params.XM,
        x0=params.X0,
        track_id=track_id,
        min_beams=params.MIN_PROFILES_PER_CLUSTER,
        size_limit=params.SIZE_LIMIT_M,
    )

    if clusters_gdf is None or clusters_gdf.empty:
        if verbose:
            print("No lateral-growth clusters created.")
        return WorkflowResult(None, None, dataset_clean, empty_gdf, empty_gdf, empty_gdf, empty_df, empty_df, empty_df)

    clusters_gdf, beam_angle_table = compute_cluster_angles(
        clusters_gdf=clusters_gdf,
        shoreline_gdf=shoreline_gdf,
        profiles_gdf=dataset_raw,
        track_id=track_id,
        bias_tolerance=bias_tolerance,
        mode_bin_width=params.ANGLE_MODE_BIN_WIDTH,
        shoreline_search_radius=params.ANGLE_SEARCH_RADIUS,
    )

    selected_clusters, _skipped, summary_fam = select_clusters_per_family(
        clusters_gdf,
        min_profiles=params.MIN_PROFILES_PER_CLUSTER,
        track_id=track_id,
    )

    if selected_clusters is None or selected_clusters.empty:
        if verbose:
            print("All clusters were skipped during selected-cluster cleanup.")
        return WorkflowResult(summary_fam, summary_fam, dataset_clean, clusters_gdf, empty_gdf, empty_gdf, empty_df, empty_df, beam_angle_table)

    selected_clusters = selected_clusters.copy()
    selected_clusters["track_id"] = track_id

    selected_clusters = add_cluster_width_m(
        selected_clusters,
        dataset_raw,
        xm=params.XM,
        utm_epsg=utm_epsg
    )

    filtered_profiles, bias_summary, bias_df = apply_bias_filter_clusters(
        dataset_raw=dataset_raw,
        selected_clusters=selected_clusters,
        bias_tolerance=bias_tolerance,
        x0=params.X0,
        verbose=verbose
    )

    if bias_summary is not None and not bias_summary.empty and "elev_avg" in bias_summary.columns:
        tmp = bias_summary.rename(columns={"reference_beam": "beam_id"})
        selected_clusters = selected_clusters.merge(
            tmp[["gt_family", "cluster_id", "beam_id", "elev_avg"]],
            on=["gt_family", "cluster_id", "beam_id"],
            how="left"
        )

    if filtered_profiles is None or filtered_profiles.empty:
        if verbose:
            print("Vertical bias filter removed all profiles.")
        return WorkflowResult(
            summary_fam, summary_fam, dataset_clean, clusters_gdf, selected_clusters,
            gpd.GeoDataFrame(), empty_df, empty_df, beam_angle_table
        )

    return WorkflowResult(
        summary_fam,
        summary_fam.copy(),
        dataset_clean,
        clusters_gdf,
        selected_clusters,
        filtered_profiles,
        bias_summary,
        bias_df,
        beam_angle_table
    )
