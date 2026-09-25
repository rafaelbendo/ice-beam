# ============================================================
# End-to-end processing of one ICESat-2 track
# ============================================================
"""
    inputs      UTM zone, bluff shoreline, SlideRule beams (cached or fresh)
    geometry    oriented boxes around each family's shoreline crossing,
                along-track distance from the box's offshore edge
    preprocess  drop elev_trash / few_points beams
    dsas        clustering + bias filter + bluff + DSAS metrics, for every
                bias tolerance
    gie         geometric correction of the same clusters
    outputs     locked merge into the shared CSV tables
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import geopandas as gpd

from .config import Params, Paths
from .dsas import run_dsas
from .geometry import build_boxes_for_families, compute_distances_from_nearest_beam
from .gie import (
    GIEDynamicRunResult,
    _normalize_track_id,
    get_available_bias_values_for_track,
    run_gie_dynamic_for_bias_values,
)
from .inputs import load_bluff_shoreline, load_track_beams, resolve_utm_epsg
from .outputs import (
    KEY_DSAS_BEAM_ANGLE,
    KEY_DSAS_INTERVAL,
    KEY_DSAS_SUMMARY,
    KEY_GIE_BEAMS,
    KEY_GIE_INTERVAL,
    KEY_GIE_SUMMARY,
    OutputFiles,
    norm_key_columns,
    read_csv_locked,
    save_dsas_table,
    save_gie_table,
)
from .preprocessing import apply_preprocessing_to_clipped
from .utils import TrackSkipped, format_track_id


@dataclass
class TrackInputs:
    track_id: str
    utm_epsg: int
    shoreline_gdf: gpd.GeoDataFrame      # bluff shoreline (CoastType == 1)
    beams_gdf: gpd.GeoDataFrame          # standardized SlideRule points, before clipping
    dataset_clean: dict                  # per-family boxes / clipped points
    dataset_raw: gpd.GeoDataFrame        # clipped + preprocessed points used downstream
    summary_raw: pd.DataFrame            # beam status counts per family
    beam_flags: pd.DataFrame             # per-beam preprocessing flags


@dataclass
class TrackResult:
    inputs: TrackInputs
    dsas_summary: pd.DataFrame
    dsas_interval: pd.DataFrame
    dsas_beam_angle: pd.DataFrame
    gie: Optional[GIEDynamicRunResult]
    gie_skip_reason: Optional[str] = None


def prepare_track_inputs(track_id, paths: Paths, params: Params,
                         source="auto", save_cache=True, verbose=True) -> TrackInputs:
    """Load, clip and preprocess one track. Raises TrackSkipped when unusable."""
    track_id = format_track_id(track_id)

    utm_epsg = resolve_utm_epsg(track_id, paths, params, verbose=verbose)
    shoreline_gdf = load_bluff_shoreline(paths.shoreline_path)

    beams_gdf = load_track_beams(
        track_id, paths, params, utm_epsg,
        source=source, save_cache=save_cache, verbose=verbose,
    )

    dataset_clean = build_boxes_for_families(
        beams_gdf,
        shoreline_gdf,
        utm_epsg=utm_epsg,
        half_along=params.HALF_ALONG_M,
        half_across=params.HALF_ACROSS_M,
        gtx=params.GTX,
        verbose=False
    )

    dataset_clipped = compute_distances_from_nearest_beam(
        dataset_clean, utm_epsg=utm_epsg, track_id=track_id
    )
    if dataset_clipped is None or dataset_clipped.empty:
        raise TrackSkipped(
            f"no valid clipped beam rows after distance computation for TRACK_ID={track_id}."
        )

    dataset_raw, summary_raw, _flagged, beam_flags, _midpoints = apply_preprocessing_to_clipped(
        dataset_clipped,
        MIN_POINTS_PCT=params.MIN_POINTS_PCT,
        ELEV_TRASH=params.ELEV_TRASH,
        TOO_FAR_BEAM=params.TOO_FAR_BEAM,
        XM=params.XM_PREPROCESS,
        IDEAL_CASE=params.IDEAL_CASE,
        return_skipped=True,
        verbose=False
    )

    return TrackInputs(
        track_id=track_id,
        utm_epsg=utm_epsg,
        shoreline_gdf=shoreline_gdf,
        beams_gdf=beams_gdf,
        dataset_clean=dataset_clean,
        dataset_raw=dataset_raw,
        summary_raw=summary_raw,
        beam_flags=beam_flags,
    )


def _gie_cluster_source(files: OutputFiles, dsas_summary_all, write_outputs):
    """DSAS summary rows the GIE step reads its bias tolerances from."""
    sources = []
    if write_outputs:
        existing = read_csv_locked(files.dsas_summary)
        if not existing.empty or files.dsas_summary.exists():
            sources.append(existing)
    if dsas_summary_all is not None and not dsas_summary_all.empty:
        sources.append(dsas_summary_all.copy())

    if not sources:
        return None

    source = norm_key_columns(pd.concat(sources, ignore_index=True, sort=False))
    missing = [col for col in KEY_DSAS_SUMMARY if col not in source.columns]
    if missing:
        raise ValueError(f"DSAS cluster/source table is missing required columns: {missing}")

    return source.drop_duplicates(subset=KEY_GIE_SUMMARY, keep="last")


def run_gie_for_track(inputs: TrackInputs, params: Params, cluster_summary_source):
    """
    GIE correction for one track. Returns (GIEDynamicRunResult or None, skip reason or None).
    """
    track_id = inputs.track_id

    if cluster_summary_source is None:
        return None, "no DSAS summary table was found for GIE post-processing."

    if inputs.dataset_raw is None or inputs.dataset_raw.empty:
        return None, f"dataset_raw is empty before GIE post-processing for TRACK_ID={track_id}."

    summary_tracks = set(cluster_summary_source["track_id"].dropna().map(_normalize_track_id))
    if track_id not in summary_tracks:
        return None, f"no DSAS summary rows for TRACK_ID={track_id}."

    dataset_raw_track = inputs.dataset_raw
    if "track_id" in dataset_raw_track.columns:
        dataset_raw_track = dataset_raw_track[
            dataset_raw_track["track_id"].map(_normalize_track_id) == track_id
        ].copy()
    if dataset_raw_track.empty:
        return None, f"no rows found in dataset_raw for TRACK_ID={track_id}."

    bias_table = get_available_bias_values_for_track(cluster_summary_source, track_id)
    if bias_table.empty:
        return None, f"no bias-tolerance values found for TRACK_ID={track_id}."

    gie_result = run_gie_dynamic_for_bias_values(
        track_id=track_id,
        dataset_raw=dataset_raw_track,
        shoreline_gdf=inputs.shoreline_gdf,
        params=params,
        utm_epsg=inputs.utm_epsg,
        cluster_summary_df=cluster_summary_source,
    )

    if gie_result.summary_df.empty:
        return gie_result, "the dynamic GIE runner produced no summary rows."

    return gie_result, None


def run_track(track_id, paths: Paths, params: Params = Params(),
              source="auto", save_cache=True, write_outputs=True,
              verbose=True) -> TrackResult:
    """
    Full pipeline for one track.

    write_outputs=True merges results into the shared CSV tables in
    paths.outdir (locked). Raises TrackSkipped when the track has no usable
    data; a track that yields DSAS rows but no GIE rows is not skipped
    (see TrackResult.gie_skip_reason).
    """
    inputs = prepare_track_inputs(track_id, paths, params, source=source,
                                  save_cache=save_cache, verbose=verbose)

    dsas_summary, dsas_interval, dsas_beam_angle = run_dsas(
        inputs.track_id, inputs.dataset_raw, inputs.shoreline_gdf,
        params=params, utm_epsg=inputs.utm_epsg, verbose=verbose,
    )

    files = OutputFiles.in_dir(paths.outdir, params.RES_TAG)
    dsas_summary_all = dsas_summary

    if write_outputs:
        dsas_summary_all = save_dsas_table(files.dsas_summary, dsas_summary, KEY_DSAS_SUMMARY)
        save_dsas_table(files.dsas_interval, dsas_interval, KEY_DSAS_INTERVAL)
        save_dsas_table(files.dsas_beam_angle, dsas_beam_angle, KEY_DSAS_BEAM_ANGLE)
        if verbose:
            print(f"DSAS tables updated -> {paths.outdir}")

    cluster_source = _gie_cluster_source(files, dsas_summary_all, write_outputs)
    gie_result, gie_skip = run_gie_for_track(inputs, params, cluster_source)

    if gie_skip is None and write_outputs:
        save_gie_table(files.gie_summary, gie_result.summary_df, KEY_GIE_SUMMARY)
        save_gie_table(files.gie_interval, gie_result.interval_df, KEY_GIE_INTERVAL)
        save_gie_table(files.gie_beams, gie_result.beam_df, KEY_GIE_BEAMS)
        if verbose:
            print(f"GIE tables updated -> {paths.outdir}")
    elif gie_skip is not None and verbose:
        print(f"GIE skipped: {gie_skip}")

    return TrackResult(
        inputs=inputs,
        dsas_summary=dsas_summary,
        dsas_interval=dsas_interval,
        dsas_beam_angle=dsas_beam_angle,
        gie=gie_result,
        gie_skip_reason=gie_skip,
    )
