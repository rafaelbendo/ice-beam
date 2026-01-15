"""
bias.py

Vertical bias utilities for IS2Retreat framework.

Rule in this repo:
- params is REQUIRED for all public functions in this file.
- No hidden numeric defaults (except purely “technical” ones like column names).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# Helpers
# ============================================================

def _normalize_beam_ids(beam_ids) -> List[str]:
    """
    Accepts beam_ids stored as:
      - list of (fam, beam_id) tuples
      - list of beam_id strings

    Returns:
      list of beam_id strings (stripped)
    """
    out: List[str] = []
    if not isinstance(beam_ids, list):
        return out
    for b in beam_ids:
        if isinstance(b, (tuple, list)) and len(b) >= 2:
            out.append(str(b[1]).strip())
        else:
            out.append(str(b).strip())
    return out


def _require_param(params: object, name: str):
    if params is None:
        raise ValueError("params is required. Pass Params() from is2retreat.config.")
    if not hasattr(params, name):
        raise ValueError(f"params must define `{name}`.")
    return getattr(params, name)


def _interp_at_x0(
    prof: pd.DataFrame,
    x0: float,
    x_col: str,
    y_col: str,
) -> float:
    """
    Interpolate y_col at x0 from a profile dataframe.
    Requires at least 2 finite points.
    """
    if prof is None or prof.empty:
        return np.nan
    if x_col not in prof.columns or y_col not in prof.columns:
        return np.nan

    xx = pd.to_numeric(prof[x_col], errors="coerce").to_numpy()
    yy = pd.to_numeric(prof[y_col], errors="coerce").to_numpy()

    m = np.isfinite(xx) & np.isfinite(yy)
    if m.sum() < 2:
        return np.nan

    xx = xx[m]
    yy = yy[m]
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]

    # np.interp extrapolates flat outside bounds (ok for your use here)
    return float(np.interp(float(x0), xx, yy))


# ============================================================
# Public API
# ============================================================

def compute_cluster_bias(
    dataset_raw: gpd.GeoDataFrame,
    selected_clusters: gpd.GeoDataFrame,
    params: object,
    gt_family: str | None = None,
    x_col: str = "distance_from_offshore",
    y_col: str = "h_li",
    include_reference: bool = True,
) -> pd.DataFrame:
    """
    Compute vertical bias between beams in each selected cluster at x0.

    Required params:
      - BIAS_X0  (float)

    Bias definition:
      bias = y_test(x0) - y_ref(x0)

    Reference beam:
      selected_clusters["beam_id"] (cluster seed)
    Compared beams:
      selected_clusters["beam_ids"] (typically list of (fam, beam_id) tuples)

    Returns DataFrame with columns:
      gt_family, cluster_id, reference_beam, beam_id, acq_date, bias_<x0>
    """
    x0 = float(_require_param(params, "BIAS_X0"))
    bias_col = f"bias_{int(round(x0))}"
    cols = ["gt_family", "cluster_id", "reference_beam", "beam_id", "acq_date", bias_col]

    if dataset_raw is None or dataset_raw.empty or selected_clusters is None or selected_clusters.empty:
        return pd.DataFrame(columns=cols)

    fam_clusters = selected_clusters if gt_family is None else selected_clusters[selected_clusters["gt_family"] == gt_family]
    if fam_clusters.empty:
        return pd.DataFrame(columns=cols)

    data = dataset_raw.copy()
    if "acq_date" in data.columns:
        data["acq_date"] = pd.to_datetime(data["acq_date"], errors="coerce")
    else:
        data["acq_date"] = pd.NaT

    out = []

    for _, cl in fam_clusters.iterrows():
        fam = cl["gt_family"]
        cid = cl["cluster_id"]

        refb = str(cl["beam_id"]).strip()
        beams = _normalize_beam_ids(cl.get("beam_ids", []))
        if not beams:
            continue

        fam_data = data[(data["gt_family"] == fam) & (data["beam_id"].isin(beams))].copy()
        if fam_data.empty:
            continue

        ref_prof = fam_data[fam_data["beam_id"] == refb]
        if ref_prof.empty:
            continue

        y_ref_x0 = _interp_at_x0(ref_prof, x0=x0, x_col=x_col, y_col=y_col)
        if not np.isfinite(y_ref_x0):
            continue

        if include_reference:
            acqd = ref_prof["acq_date"].iloc[0] if "acq_date" in ref_prof.columns else pd.NaT
            out.append(
                {
                    "gt_family": fam,
                    "cluster_id": cid,
                    "reference_beam": refb,
                    "beam_id": refb,
                    "acq_date": acqd,
                    bias_col: 0.0,
                }
            )

        for bid in beams:
            bid = str(bid).strip()
            if bid == refb:
                continue

            test_prof = fam_data[fam_data["beam_id"] == bid]
            if test_prof.empty:
                continue

            y_test = _interp_at_x0(test_prof, x0=x0, x_col=x_col, y_col=y_col)
            bias_x0 = (y_test - y_ref_x0) if (np.isfinite(y_test) and np.isfinite(y_ref_x0)) else np.nan
            acqd = test_prof["acq_date"].iloc[0] if "acq_date" in test_prof.columns else pd.NaT

            out.append(
                {
                    "gt_family": fam,
                    "cluster_id": cid,
                    "reference_beam": refb,
                    "beam_id": bid,
                    "acq_date": acqd,
                    bias_col: bias_x0,
                }
            )

    bias_df = pd.DataFrame(out, columns=cols)
    if not bias_df.empty:
        bias_df = bias_df.sort_values(["gt_family", "cluster_id", "acq_date"]).reset_index(drop=True)

    return bias_df


def apply_bias_filter_clusters(
    dataset_raw: gpd.GeoDataFrame,
    selected_clusters: gpd.GeoDataFrame,
    params: object,
    x_col: str = "distance_from_offshore",
    y_col: str = "h_li",
    verbose: bool = False,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply bias filtering, producing accepted profiles and bias summary.

    Required params:
      - BIAS_TOLERANCE (float)
      - BIAS_X0        (float)

    Workflow:
      - reference beam = selected_clusters["beam_id"] (cluster seed)
      - compute y_ref(x0) from the most recent baseline cycle with finite y_ref
      - always keep ALL baseline (reference) cycles
      - for non-reference beams, keep cycles where |bias| <= BIAS_TOLERANCE
      - attach cluster_id to accepted profiles during loop (no spatial join needed)

    Returns:
      filtered_profiles : GeoDataFrame (accepted only) with:
          - cluster_id
          - accepted_Bias=True
          - bias_<x0> and keep
      bias_summary      : DataFrame summary per (gt_family, cluster_id, reference_beam)
      bias_df           : DataFrame record for every evaluated beam/date with keep flag
    """
    bias_tolerance = float(_require_param(params, "BIAS_TOLERANCE"))
    x0 = float(_require_param(params, "BIAS_X0"))
    bias_col = f"bias_{int(round(x0))}"

    if dataset_raw is None or dataset_raw.empty or selected_clusters is None or selected_clusters.empty:
        empty = gpd.GeoDataFrame(columns=getattr(dataset_raw, "columns", []), geometry="geometry", crs=getattr(dataset_raw, "crs", None))
        return empty, pd.DataFrame(), pd.DataFrame()

    # --------------------------
    # Prepare dataset_raw
    # --------------------------
    df = dataset_raw.copy()
    df["beam_id"] = df["beam_id"].astype(str).str.strip()
    df["gt_family"] = df["gt_family"].astype(str).str.strip()

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    else:
        df["acq_date"] = pd.NaT

    df = gpd.GeoDataFrame(df, geometry="geometry", crs=dataset_raw.crs)

    # --------------------------
    # Align CRS of clusters
    # --------------------------
    sc = selected_clusters.copy()
    sc["gt_family"] = sc["gt_family"].astype(str).str.strip()
    sc["beam_id"] = sc["beam_id"].astype(str).str.strip()

    if sc.crs and df.crs and sc.crs != df.crs:
        sc = sc.to_crs(df.crs)

    # --------------------------
    # Main loop
    # --------------------------
    records = []
    accepted_blocks = []

    for (fam, cid), cl in sc.groupby(["gt_family", "cluster_id"]):
        row0 = cl.iloc[0]
        poly = row0.geometry
        if poly is None or poly.is_empty:
            continue

        ref_beam = str(row0["beam_id"]).strip()
        beam_ids = _normalize_beam_ids(row0.get("beam_ids", []))
        if not beam_ids:
            continue

        fam_data = df[(df["gt_family"] == fam) & (df["beam_id"].isin(beam_ids))].copy()
        if fam_data.empty:
            continue

        # points inside cluster polygon
        cp = fam_data[fam_data.geometry.intersects(poly)].copy()
        if cp.empty:
            continue

        # baseline (reference beam) profiles
        base = cp[cp["beam_id"] == ref_beam].copy()
        if base.empty:
            continue

        base_dates = sorted(base["acq_date"].dropna().unique())
        if not base_dates:
            continue

        # choose most recent date with finite y_ref(x0)
        y_ref_by_dt = {dt: _interp_at_x0(base[base["acq_date"] == dt], x0, x_col, y_col) for dt in base_dates}
        finite_dt = [dt for dt in base_dates if np.isfinite(y_ref_by_dt[dt])]
        if not finite_dt:
            continue

        baseline_date = max(finite_dt)
        y_ref_x0 = float(y_ref_by_dt[baseline_date])

        # Always keep ALL baseline cycles
        for dt in base_dates:
            prof_dt = base[base["acq_date"] == dt].copy()
            if prof_dt.empty:
                continue

            prof_dt["cluster_id"] = cid
            accepted_blocks.append(prof_dt)

            records.append(
                {
                    "gt_family": fam,
                    "cluster_id": cid,
                    "reference_beam": ref_beam,
                    "beam_id": ref_beam,
                    "acq_date": dt,
                    bias_col: 0.0,
                    "keep": True,
                    "is_ref": True,
                }
            )

        # Non-reference beams
        others = cp[cp["beam_id"] != ref_beam].copy()

        for bid, g in others.groupby("beam_id"):
            dates = sorted(g["acq_date"].dropna().unique())
            for dt in dates:
                prof_dt = g[g["acq_date"] == dt].copy()
                if prof_dt.empty:
                    continue

                y_test = _interp_at_x0(prof_dt, x0, x_col, y_col)

                if np.isfinite(y_test):
                    bval = float(y_test - y_ref_x0)
                    keep = abs(bval) <= bias_tolerance
                else:
                    bval = np.nan
                    keep = False

                if keep:
                    prof_dt["cluster_id"] = cid
                    accepted_blocks.append(prof_dt)

                records.append(
                    {
                        "gt_family": fam,
                        "cluster_id": cid,
                        "reference_beam": ref_beam,
                        "beam_id": str(bid).strip(),
                        "acq_date": dt,
                        bias_col: bval,
                        "keep": bool(keep),
                        "is_ref": False,
                    }
                )

    # --------------------------
    # Build bias_df
    # --------------------------
    bias_df = pd.DataFrame.from_records(records)
    if bias_df.empty:
        empty = gpd.GeoDataFrame(columns=df.columns, geometry="geometry", crs=df.crs)
        return empty, pd.DataFrame(), pd.DataFrame()

    bias_df["sort"] = (bias_df["beam_id"] != bias_df["reference_beam"]).astype(int)
    bias_df = (
        bias_df.sort_values(["gt_family", "cluster_id", "sort", "beam_id", "acq_date"])
              .drop(columns="sort")
              .reset_index(drop=True)
    )

    # --------------------------
    # Build bias_summary
    # --------------------------
    keys = ["gt_family", "cluster_id", "reference_beam"]
    nonref = bias_df[bias_df["is_ref"] == False].copy()

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

    # total beams from selected_clusters (beam_ids length)
    counts = (
        sc.groupby(["gt_family", "cluster_id"], as_index=False)
          .agg(
              n_beams_total=("beam_ids", lambda s: len(s.iloc[0]) if isinstance(s.iloc[0], list) else np.nan),
              reference_beam=("beam_id", "first"),
          )
    )

    bias_summary = (
        summary_all
        .merge(summary_kept, on=keys, how="left")
        .merge(kept_incl_ref, on=keys, how="left")
        .merge(counts, on=["gt_family", "cluster_id", "reference_beam"], how="left")
    )

    # --------------------------
    # Build filtered_profiles (accepted only)
    # --------------------------
    if accepted_blocks:
        filtered = pd.concat(accepted_blocks, ignore_index=True)
        filtered["accepted_Bias"] = True

        filtered_profiles = gpd.GeoDataFrame(filtered, geometry="geometry", crs=df.crs)

        filtered_profiles = filtered_profiles.merge(
            bias_df[["gt_family", "cluster_id", "beam_id", "acq_date", bias_col, "keep"]],
            on=["gt_family", "cluster_id", "beam_id", "acq_date"],
            how="left",
        )
    else:
        filtered_profiles = gpd.GeoDataFrame(columns=df.columns, geometry="geometry", crs=df.crs)

    if verbose:
        n_nonref_kept = int(bias_df.loc[bias_df["is_ref"] == False, "keep"].sum())
        print(f"✅ Bias filtering: kept {n_nonref_kept} non-reference cycles (+ all baseline cycles).")

    return filtered_profiles, bias_summary, bias_df


__all__ = [
    "compute_cluster_bias",
    "apply_bias_filter_clusters",
]
