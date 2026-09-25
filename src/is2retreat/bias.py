# ============================================================
# Vertical-bias filter for lateral-growth clusters
# ============================================================
"""
Builds the final per-cluster profiles used for bluff detection.

- Prefers ``beam_ids_ordered`` (keeps the lateral-growth member order).
- No polygon clipping: once a beam belongs to a selected cluster, all of its
  points are used.
- Each non-reference (beam, date) profile is compared at x0 with the
  reference beam's most recent profile that spans x0; it is kept when
  ``abs(bias) <= bias_tolerance``. No extrapolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from .utils import cluster_member_beams


def apply_bias_filter_clusters(
        dataset_raw,
        selected_clusters,
        bias_tolerance,
        x0=500.0,
        verbose=False
    ):
    """
    Returns
    -------
    filtered_profiles : GeoDataFrame
        Accepted points, tagged with cluster_id.
    bias_summary : DataFrame
        Per-cluster bias statistics (incl. elev_avg).
    bias_df : DataFrame
        One row per (cluster, beam, date) with bias and keep flag.
    """
    bias_col = f"bias_{int(round(x0))}"
    elev_col = f"elev_{int(round(x0))}"

    def _interp_at(prof):
        if prof.empty:
            return np.nan

        p = prof.dropna(subset=["distance_from_offshore", "h_li"]).copy()
        if len(p) < 2:
            return np.nan

        p["distance_from_offshore"] = pd.to_numeric(p["distance_from_offshore"], errors="coerce")
        p["h_li"] = pd.to_numeric(p["h_li"], errors="coerce")
        p = p.dropna(subset=["distance_from_offshore", "h_li"])

        if len(p) < 2:
            return np.nan

        p = (
            p.groupby("distance_from_offshore", as_index=False)["h_li"]
             .mean()
             .sort_values("distance_from_offshore")
        )

        x = p["distance_from_offshore"].to_numpy(dtype=float)
        y = p["h_li"].to_numpy(dtype=float)

        if not (np.nanmin(x) <= x0 <= np.nanmax(x)):
            return np.nan

        return float(np.interp(x0, x, y))

    if dataset_raw is None or dataset_raw.empty or selected_clusters is None or selected_clusters.empty:
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(dataset_raw, "crs", None))
        return empty, pd.DataFrame(), pd.DataFrame()

    df = dataset_raw.copy()
    df["beam_id"] = df["beam_id"].astype(str).str.strip()
    df["gt_family"] = df["gt_family"].astype(str)
    df["acq_date"] = pd.to_datetime(df.get("acq_date"), errors="coerce").dt.normalize()
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=dataset_raw.crs)

    sc = selected_clusters.copy()
    sc["gt_family"] = sc["gt_family"].astype(str)
    sc["beam_id"] = sc["beam_id"].astype(str).str.strip()

    if sc.crs and df.crs and sc.crs != df.crs:
        sc = sc.to_crs(df.crs)

    records = []
    accepted_blocks = []

    for (fam, cid), cl in sc.groupby(["gt_family", "cluster_id"]):
        row0 = cl.iloc[0]

        ref_beam = str(row0.get("reference_beam", row0.get("beam_id"))).strip()
        beam_ids = cluster_member_beams(row0)

        if not beam_ids:
            if verbose:
                print(f"[WARN] {fam}-{cid}: no member beams found.")
            continue

        fam_data = df[
            (df["gt_family"] == fam) &
            (df["beam_id"].isin(beam_ids))
        ].copy()

        if fam_data.empty:
            if verbose:
                print(f"[WARN] {fam}-{cid}: no raw data found for member beams.")
            continue

        base = fam_data[fam_data["beam_id"] == ref_beam].copy()
        if base.empty:
            if verbose:
                print(f"[WARN] {fam}-{cid}: reference beam {ref_beam} not found.")
            continue

        base_dates = sorted(base["acq_date"].dropna().unique())
        if not base_dates:
            continue

        baseline_y = {
            dt: _interp_at(base[base["acq_date"] == dt])
            for dt in base_dates
        }

        finite_dt = [dt for dt in base_dates if np.isfinite(baseline_y[dt])]
        if not finite_dt:
            if verbose:
                print(f"[WARN] {fam}-{cid}: no valid reference profile spans x0={x0}.")
            continue

        baseline_date = max(finite_dt)
        y_ref_x0 = baseline_y[baseline_date]

        for dt in base_dates:
            prof_dt = base[base["acq_date"] == dt].copy()
            if prof_dt.empty:
                continue

            prof_dt["cluster_id"] = cid
            accepted_blocks.append(prof_dt)

            records.append({
                "gt_family": fam,
                "cluster_id": cid,
                "reference_beam": ref_beam,
                "beam_id": ref_beam,
                "acq_date": dt,
                elev_col: baseline_y.get(dt, np.nan),
                bias_col: 0.0,
                "keep": True,
                "is_ref": True
            })

        others = fam_data[fam_data["beam_id"] != ref_beam].copy()

        for bid, g in others.groupby("beam_id"):
            dates = sorted(g["acq_date"].dropna().unique())

            for dt in dates:
                prof_dt = g[g["acq_date"] == dt].copy()
                if prof_dt.empty:
                    continue

                y_test = _interp_at(prof_dt)

                if np.isfinite(y_test):
                    bval = y_test - y_ref_x0
                    keep = abs(bval) <= bias_tolerance
                else:
                    bval = np.nan
                    keep = False

                if keep:
                    prof_dt["cluster_id"] = cid
                    accepted_blocks.append(prof_dt)

                records.append({
                    "gt_family": fam,
                    "cluster_id": cid,
                    "reference_beam": ref_beam,
                    "beam_id": bid,
                    "acq_date": dt,
                    elev_col: y_test,
                    bias_col: bval,
                    "keep": keep,
                    "is_ref": False
                })

    bias_df = pd.DataFrame.from_records(records)

    if bias_df.empty:
        return (
            gpd.GeoDataFrame(columns=df.columns, geometry="geometry", crs=df.crs),
            pd.DataFrame(),
            pd.DataFrame()
        )

    bias_df["sort"] = (bias_df["beam_id"] != bias_df["reference_beam"]).astype(int)
    bias_df = (
        bias_df
        .sort_values(["gt_family", "cluster_id", "sort", "beam_id", "acq_date"])
        .drop(columns="sort")
        .reset_index(drop=True)
    )

    keys = ["gt_family", "cluster_id", "reference_beam"]
    nonref = bias_df[bias_df["is_ref"] == False].copy()  # noqa: E712

    summary_all = (
        nonref.groupby(keys, as_index=False)
              .agg(
                  n_beams=("beam_id", "nunique"),
                  n_kept_nonref=("keep", "sum"),
                  bias_min=(bias_col, "min"),
                  bias_max=(bias_col, "max"),
                  bias_mean=(bias_col, "mean"),
                  bias_std=(bias_col, "std"),
              )
    )

    kept_only = nonref[nonref["keep"]].copy()
    summary_kept = (
        kept_only.groupby(keys, as_index=False)
                 .agg(
                     kept_bias_min=(bias_col, "min"),
                     kept_bias_max=(bias_col, "max"),
                     kept_bias_mean=(bias_col, "mean"),
                     kept_bias_std=(bias_col, "std"),
                 )
    )

    kept_incl_ref = (
        bias_df.groupby(keys)["keep"]
               .sum()
               .reset_index()
               .rename(columns={"keep": "n_kept"})
    )

    sc2 = sc.copy()
    sc2["n_beams_total_calc"] = sc2.apply(lambda row: len(cluster_member_beams(row)), axis=1)

    counts = (
        sc2.groupby(["gt_family", "cluster_id"], as_index=False)
           .agg(
               n_beams_total=("n_beams_total_calc", "first"),
               reference_beam=("beam_id", "first")
           )
    )

    elev_avg_df = (
        bias_df.loc[bias_df["keep"] & bias_df[elev_col].notna()]
               .groupby(keys, as_index=False)[elev_col]
               .mean()
               .rename(columns={elev_col: "elev_avg"})
    )

    bias_summary = (
        summary_all
        .merge(summary_kept, on=keys, how="left")
        .merge(kept_incl_ref, on=keys, how="left")
        .merge(counts, on=["gt_family", "cluster_id", "reference_beam"], how="left")
        .merge(elev_avg_df, on=keys, how="left")
    )

    if accepted_blocks:
        filtered = pd.concat(accepted_blocks, ignore_index=True)
        filtered["accepted_Bias"] = True

        filtered_profiles = gpd.GeoDataFrame(filtered, geometry="geometry", crs=df.crs)

        filtered_profiles = filtered_profiles.merge(
            bias_df[["gt_family", "cluster_id", "beam_id", "acq_date", elev_col, bias_col, "keep"]],
            on=["gt_family", "cluster_id", "beam_id", "acq_date"],
            how="left"
        )
    else:
        filtered_profiles = gpd.GeoDataFrame(columns=df.columns, geometry="geometry", crs=df.crs)

    return filtered_profiles, bias_summary, bias_df
