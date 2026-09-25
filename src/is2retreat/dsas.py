# ============================================================
# DSAS runner: bias-tolerance sensitivity loop for one track
# ============================================================
"""
- Clusters are created dynamically from lateral growth + bias control; the
  sensitivity loop varies only the bias tolerance.
- Meter-scale cluster size is reported as ``cluster_width_m``; the number of
  member beams as ``n_beams``.
- Angle outputs are per-beam shoreline-angle summaries
  (angle_min/max/mean/median/mode_deg). No ``angle_deg`` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .angles import ANGLE_SUMMARY_COLS, BEAM_ANGLE_COLS
from .bluff import process_cluster_with_reference
from .config import Params
from .metrics import compute_cluster_intervals, compute_cluster_statistics
from .utils import cluster_member_beams, first_non_null, format_track_id
from .workflow import run_workflow


def get_angle_summary_values(cluster_rows):
    vals = {}
    for col in ANGLE_SUMMARY_COLS:
        vals[col] = (
            float(first_non_null(cluster_rows[col]))
            if col in cluster_rows.columns and pd.notna(first_non_null(cluster_rows[col]))
            else np.nan
        )
    return vals


def filter_beam_angle_table_to_selected(beam_angle_table, selected_clusters):
    if beam_angle_table is None or beam_angle_table.empty:
        return pd.DataFrame(columns=BEAM_ANGLE_COLS)

    out = beam_angle_table.copy()

    if selected_clusters is None or selected_clusters.empty:
        return out.iloc[0:0].copy()

    selected_keys = selected_clusters[["gt_family", "cluster_id"]].drop_duplicates().copy()
    selected_keys["gt_family"] = selected_keys["gt_family"].astype(str).str.strip()
    selected_keys["cluster_id"] = pd.to_numeric(selected_keys["cluster_id"], errors="coerce")

    out["gt_family"] = out["gt_family"].astype(str).str.strip()
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")

    return out.merge(selected_keys, on=["gt_family", "cluster_id"], how="inner")


def _first_float(cluster_rows, col):
    if col in cluster_rows.columns and pd.notna(first_non_null(cluster_rows[col])):
        return float(first_non_null(cluster_rows[col]))
    return np.nan


def run_dsas_for_bias_tolerance(
        track_id,
        dataset_raw,
        shoreline_gdf,
        bias_tolerance,
        params: Params,
        utm_epsg: int,
    ):
    """
    DSAS summary, interval and beam-angle rows for one bias tolerance.

    Returns
    -------
    summary_rows, interval_rows, beam_angle_rows : list[dict]
    """
    track_str = format_track_id(track_id)

    result = run_workflow(
        track_id=track_str,
        dataset_raw=dataset_raw,
        shoreline_gdf=shoreline_gdf,
        bias_tolerance=bias_tolerance,
        params=params,
        utm_epsg=utm_epsg,
        verbose=False
    )
    selected_clusters = result.selected_clusters
    filtered_profiles = result.filtered_profiles
    bias_df = result.bias_df

    summary_results = []
    interval_results = []
    beam_angle_results = []

    if selected_clusters is None or selected_clusters.empty:
        return summary_results, interval_results, beam_angle_results

    beam_angle_selected = filter_beam_angle_table_to_selected(
        beam_angle_table=result.beam_angle_table,
        selected_clusters=selected_clusters
    )
    if not beam_angle_selected.empty:
        beam_angle_results = beam_angle_selected.to_dict("records")

    for fam in selected_clusters["gt_family"].unique():
        fam_clusters = selected_clusters.query("gt_family == @fam").copy()

        initial_cycles = (
            dataset_raw.query("gt_family == @fam")["beam_id"].nunique()
            if dataset_raw is not None and not dataset_raw.empty
            else 0
        )

        used_cycles = (
            filtered_profiles.query("gt_family == @fam")["beam_id"].nunique()
            if filtered_profiles is not None and not filtered_profiles.empty
            else 0
        )

        for cid in fam_clusters["cluster_id"].unique():
            cluster_rows = fam_clusters.loc[fam_clusters["cluster_id"] == cid].copy()

            if cluster_rows.empty:
                continue

            bluff_df, _y_ref = process_cluster_with_reference(
                filtered_profiles=filtered_profiles,
                selected_clusters=selected_clusters,
                cluster_id=cid,
                gt_family=fam,
                which=params.BLUFF_WHICH,
                gap_threshold=params.GAP_THRESHOLD_M,
                atol=params.CROSSING_ATOL,
                bias_df=bias_df,
                debug=False
            )

            if bluff_df is None or bluff_df.empty:
                continue

            stats = compute_cluster_statistics(
                bluff_df,
                confidence=params.CONFIDENCE,
                min_span_days=params.MIN_SPAN_DAYS,
                positional_uncertainty_m=params.POSITIONAL_UNCERTAINTY_M,
            )

            cluster_years = (
                round(stats["TemporalSpan_days"] / 365.25, 2)
                if pd.notna(stats["TemporalSpan_days"])
                else np.nan
            )

            bluff_df = bluff_df.copy()
            bluff_df["acq_date_norm"] = pd.to_datetime(bluff_df["acq_date"]).dt.normalize()

            first_dt = bluff_df["acq_date_norm"].min()
            last_dt = bluff_df["acq_date_norm"].max()

            angle_summary_vals = get_angle_summary_values(cluster_rows)

            center_lat = _first_float(cluster_rows, "center_lat")
            center_lon = _first_float(cluster_rows, "center_lon")
            elev_avg = _first_float(cluster_rows, "elev_avg")
            cluster_width_m = _first_float(cluster_rows, "cluster_width_m")

            if "num_beams" in cluster_rows.columns and pd.notna(first_non_null(cluster_rows["num_beams"])):
                n_beams = int(first_non_null(cluster_rows["num_beams"]))
            else:
                n_beams = len(cluster_member_beams(cluster_rows.iloc[0]))

            used_cycles_cluster = (
                filtered_profiles.query("gt_family == @fam and cluster_id == @cid")["beam_id"].nunique()
                if filtered_profiles is not None
                and not filtered_profiles.empty
                and "cluster_id" in filtered_profiles.columns
                else np.nan
            )

            summary_results.append({
                "track_id": track_str,
                "bias_tolerance": float(bias_tolerance),
                "gt_family": fam,
                "cluster_id": int(cid),

                "cluster_width_m": round(cluster_width_m, 2) if np.isfinite(cluster_width_m) else np.nan,
                "n_beams": int(n_beams) if pd.notna(n_beams) else np.nan,

                "NSM": stats["NSM"],
                "SCE": stats["SCE"],
                "EPR": stats["EPR"],
                "LRR": stats["LRR"],
                "LR2": stats["LR2"],
                "LSE": stats["LSE"],
                "LCI": stats["LCI"],
                "TemporalSpan_days": stats["TemporalSpan_days"],
                "ClusterTemporalSpanYears": cluster_years,
                "ValidRegression": stats["ValidRegression"],

                "U_position_m": stats["U_position_m"],
                "U_NSM_m": stats["U_NSM_m"],
                "U_EPR_myr": stats["U_EPR_myr"],

                **angle_summary_vals,

                "center_lat": center_lat,
                "center_lon": center_lon,
                "elev_avg": elev_avg,

                "first_date": first_dt,
                "last_date": last_dt,

                "initial_cycles": initial_cycles,
                "used_cycles": used_cycles,
                "used_cycles_cluster": used_cycles_cluster,
            })

            interval_rows = compute_cluster_intervals(
                bluff_df=bluff_df,
                gt_family=fam,
                cluster_id=cid,
                track_id=track_str,
                bias_tolerance=bias_tolerance,
                cluster_width_m=cluster_width_m,
                n_beams=n_beams
            )

            for r in interval_rows:
                r.update(angle_summary_vals)
                r["center_lat"] = center_lat
                r["center_lon"] = center_lon
                r["elev_avg"] = elev_avg
                r["cluster_first_date"] = first_dt
                r["cluster_last_date"] = last_dt
                r["cluster_temporal_span_days"] = stats["TemporalSpan_days"]
                r["cluster_temporal_span_years"] = cluster_years
                r["cluster_NSM"] = stats["NSM"]
                r["cluster_SCE"] = stats["SCE"]
                r["cluster_EPR"] = stats["EPR"]
                r["cluster_LRR"] = stats["LRR"]

            interval_results.extend(interval_rows)

    return summary_results, interval_results, beam_angle_results


def run_dsas(track_id, dataset_raw, shoreline_gdf, params: Params, utm_epsg: int, verbose=True):
    """
    Run the DSAS analysis for every bias tolerance in params.BIAS_TOLERANCES.

    Returns
    -------
    summary_df, interval_df, beam_angle_df : DataFrame
    """
    all_summary_rows = []
    all_interval_rows = []
    all_beam_angle_rows = []

    for tol in params.BIAS_TOLERANCES:
        summary_rows, interval_rows, beam_angle_rows = run_dsas_for_bias_tolerance(
            track_id=track_id,
            dataset_raw=dataset_raw,
            shoreline_gdf=shoreline_gdf,
            bias_tolerance=tol,
            params=params,
            utm_epsg=utm_epsg,
        )

        if verbose:
            print(
                f"BiasTol={tol:.2f} m -> {len(summary_rows)} clusters, "
                f"{len(interval_rows)} intervals, {len(beam_angle_rows)} beam angles"
            )

        all_summary_rows.extend(summary_rows)
        all_interval_rows.extend(interval_rows)
        all_beam_angle_rows.extend(beam_angle_rows)

    return (
        pd.DataFrame(all_summary_rows),
        pd.DataFrame(all_interval_rows),
        pd.DataFrame(all_beam_angle_rows),
    )
