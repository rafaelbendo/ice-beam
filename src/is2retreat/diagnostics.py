# ============================================================
# Sanity checks for interactive inspection (not used by the outputs)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors

from .clustering import interpolate_elevation_at_x0
from .utils import cluster_member_beams


def compute_beam_distances(dataset_clean, utm_epsg, pts_gdf=None):
    """
    Beam spacing inside each family's box, from beam midpoints.

    pts_gdf=None uses the clipped points; pass dataset_raw for kept beams.

    Returns
    -------
    beam_midpoints : GeoDataFrame (one midpoint per beam + nearest neighbor)
    dist_summary : DataFrame (per-family spacing summary)
    """
    target_crs = f"EPSG:{utm_epsg}"
    midpoints = []

    for fam, content in dataset_clean.items():

        if not content or content.get("box") is None:
            continue

        box_gdf = content["box"]
        box = box_gdf.to_crs(utm_epsg) if str(box_gdf.crs) != target_crs else box_gdf
        box_geom = box.geometry.iloc[0]

        fam_df = content.get("clipped") if pts_gdf is None else pts_gdf.loc[pts_gdf["gt_family"] == fam]

        if fam_df is None or fam_df.empty:
            continue

        if fam_df.crs is None or str(fam_df.crs) != target_crs:
            fam_df = fam_df.to_crs(utm_epsg)

        fam_df = fam_df[
            fam_df.geometry.within(box_geom) |
            fam_df.geometry.touches(box_geom)
        ]

        if fam_df.empty:
            continue

        for bid, g in fam_df.groupby("beam_id"):

            if len(g) < 2:
                continue

            order_col = next(
                (c for c in ["distance_from_offshore", "alongtrack_distance", "x_atc", "along_track_dist"]
                 if c in g.columns),
                None,
            )

            if order_col is not None:
                g_sorted = g.sort_values(order_col).copy()
            else:
                g_sorted = g.assign(_y=g.geometry.y).sort_values("_y").drop(columns="_y")

            coords = [(geom.x, geom.y) for geom in g_sorted.geometry if geom is not None]

            clean_coords = []
            for xy in coords:
                if not clean_coords or xy != clean_coords[-1]:
                    clean_coords.append(xy)

            if len(clean_coords) < 2:
                continue

            line = LineString(clean_coords)
            if line.length == 0:
                continue

            midpoints.append({
                "gt_family": fam,
                "beam_id": bid,
                "geometry": line.interpolate(0.5, normalized=True)
            })

    if not midpoints:
        return (
            gpd.GeoDataFrame(columns=["gt_family", "beam_id", "geometry"], crs=target_crs),
            pd.DataFrame(columns=[
                "gt_family", "n_beams",
                "max_dist", "avg_dist",
                "nn_max_dist", "nn_avg_dist",
                "west_beam", "east_beam"
            ])
        )

    beam_midpoints = gpd.GeoDataFrame(midpoints, crs=target_crs)
    beam_midpoints["nearest_dist"] = np.nan
    beam_midpoints["nearest_beam"] = None

    for fam, fam_df in beam_midpoints.groupby("gt_family"):
        if fam_df.shape[0] < 2:
            continue

        coords = np.array([[p.x, p.y] for p in fam_df.geometry])
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        dist, idx = nbrs.kneighbors(coords)

        fam_idx = fam_df.index.to_numpy()
        beam_midpoints.loc[fam_idx, "nearest_dist"] = dist[:, 1]
        beam_midpoints.loc[fam_idx, "nearest_beam"] = fam_df.iloc[idx[:, 1]]["beam_id"].values

    rows = []

    for fam, fam_df in beam_midpoints.groupby("gt_family"):

        if fam_df.shape[0] < 2:
            continue

        fam_df = fam_df.assign(x=fam_df.geometry.x).sort_values("x")
        xvals = fam_df["x"].to_numpy()
        adj = np.diff(xvals)

        rows.append({
            "gt_family": fam,
            "n_beams": len(fam_df),
            "max_dist": float(xvals[-1] - xvals[0]),
            "avg_dist": float(adj.mean()) if len(adj) else np.nan,
            "nn_max_dist": float(fam_df["nearest_dist"].max()),
            "nn_avg_dist": float(fam_df["nearest_dist"].mean()),
            "west_beam": fam_df.iloc[0]["beam_id"],
            "east_beam": fam_df.iloc[-1]["beam_id"]
        })

    return beam_midpoints, pd.DataFrame(rows)


def build_cluster_beam_elevation_check(
        dataset_raw,
        selected_clusters,
        x0=500.0,
        x_col="distance_from_offshore",
        y_col="h_li"
    ):
    """Beam list, cluster_id and elevation at x0 for each selected cluster."""

    elev_col = f"elev_{int(round(x0))}"
    cols = [
        "gt_family", "cluster_id", "growth_side", "reference_beam",
        "beam_id", "beam_order", "acq_date", elev_col, "is_ref"
    ]

    if dataset_raw is None or dataset_raw.empty or selected_clusters is None or selected_clusters.empty:
        return pd.DataFrame(columns=cols)

    df = dataset_raw.copy()
    df["gt_family"] = df["gt_family"].astype(str)
    df["beam_id"] = df["beam_id"].astype(str).str.strip()
    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    else:
        df["acq_date"] = pd.NaT

    records = []

    for _, cl in selected_clusters.iterrows():
        fam = str(cl["gt_family"])
        cid = cl["cluster_id"]
        ref_beam = str(cl.get("reference_beam", cl.get("beam_id", ""))).strip()
        growth_side = cl.get("growth_side", None)

        for beam_order, bid in enumerate(cluster_member_beams(cl), start=1):
            sub = df[(df["gt_family"] == fam) & (df["beam_id"] == bid)].copy()
            base = {
                "gt_family": fam,
                "cluster_id": cid,
                "growth_side": growth_side,
                "reference_beam": ref_beam,
                "beam_id": bid,
                "beam_order": beam_order,
                "is_ref": bid == ref_beam,
            }
            if sub.empty:
                records.append({**base, "acq_date": pd.NaT, elev_col: np.nan})
                continue

            for acq_date, prof in sub.groupby("acq_date", dropna=False):
                elev = interpolate_elevation_at_x0(prof, x0=x0, x_col=x_col, y_col=y_col)
                records.append({**base, "acq_date": acq_date, elev_col: elev})

    check_df = pd.DataFrame(records, columns=cols)
    if not check_df.empty:
        check_df = check_df.sort_values(
            ["gt_family", "cluster_id", "beam_order", "acq_date"]
        ).reset_index(drop=True)

    return check_df


def build_cluster_bias_table(
        dataset_raw,
        selected_clusters,
        bias_tolerance,
        x0=500.0,
        x_col="distance_from_offshore",
        y_col="h_li",
    ):
    """
    Bias of every member beam against the cluster's reference beam at x0
    (whole-beam profiles, not per date), plus per-cluster statistics.

    Returns
    -------
    bias_table, cluster_stats : DataFrame
    """
    bias_col = f"bias_{int(round(x0))}"
    cols = [
        "gt_family", "cluster_id", "reference_beam", "beam_id",
        "acq_date", bias_col, "keep", "is_ref", "bias_tolerance"
    ]

    if selected_clusters is None or selected_clusters.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame()

    if dataset_raw is None or dataset_raw.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame()

    df = dataset_raw.copy()
    df["acq_date"] = pd.to_datetime(df.get("acq_date"), errors="coerce")

    def interp_at(prof):
        if prof.empty:
            return np.nan

        xx = prof[x_col].astype(float).values
        yy = prof[y_col].astype(float).values
        mask = np.isfinite(xx) & np.isfinite(yy)

        if mask.sum() < 2:
            return np.nan

        xx = xx[mask]
        yy = yy[mask]

        order = np.argsort(xx)
        xx = xx[order]
        yy = yy[order]

        if not (np.nanmin(xx) <= x0 <= np.nanmax(xx)):
            return np.nan

        return np.interp(x0, xx, yy)

    records = []

    for (fam, cid), grp in selected_clusters.groupby(["gt_family", "cluster_id"]):

        cl = grp.iloc[0]
        ref_beam = cl["beam_id"]
        beams = [b[1] for b in cl["beam_ids"]]

        fam_prof = df[(df["gt_family"] == fam) & (df["beam_id"].isin(beams))]
        if fam_prof.empty:
            continue

        ref_prof = fam_prof[fam_prof["beam_id"] == ref_beam]
        if ref_prof.empty:
            continue

        y_ref = interp_at(ref_prof)

        for bid in beams:
            sub = fam_prof[fam_prof["beam_id"] == bid]
            if sub.empty:
                continue

            if bid == ref_beam:
                bias_val = 0.0
                keep = True
                is_ref = True
            else:
                y_test = interp_at(sub)
                if np.isfinite(y_test) and np.isfinite(y_ref):
                    bias_val = y_test - y_ref
                    keep = abs(bias_val) <= bias_tolerance
                else:
                    bias_val = np.nan
                    keep = False
                is_ref = False

            records.append({
                "gt_family": fam,
                "cluster_id": cid,
                "reference_beam": ref_beam,
                "beam_id": bid,
                "acq_date": sub["acq_date"].iloc[0] if "acq_date" in sub.columns else pd.NaT,
                bias_col: round(bias_val, 2),
                "keep": keep,
                "is_ref": is_ref,
                "bias_tolerance": bias_tolerance
            })

    bias_table = pd.DataFrame(records)

    if bias_table.empty:
        return bias_table, pd.DataFrame()

    bias_table["sort_key"] = (bias_table["beam_id"] != bias_table["reference_beam"]).astype(int)
    bias_table = (
        bias_table.sort_values(["gt_family", "cluster_id", "sort_key", "beam_id"])
                  .drop(columns="sort_key")
                  .reset_index(drop=True)
    )

    keys = ["gt_family", "cluster_id", "reference_beam"]
    nonref = bias_table[~bias_table["is_ref"]]

    summary_all = (
        nonref.groupby(keys, as_index=False)
              .agg(
                  n_beams=("beam_id", "count"),
                  n_kept_nonref=("keep", "sum"),
                  bias_min=(bias_col, "min"),
                  bias_max=(bias_col, "max"),
                  bias_mean=(bias_col, "mean"),
                  bias_std=(bias_col, "std")
              )
    )

    kept_only = nonref[nonref["keep"]]
    summary_kept = (
        kept_only.groupby(keys, as_index=False)
                 .agg(
                     kept_bias_min=(bias_col, "min"),
                     kept_bias_max=(bias_col, "max"),
                     kept_bias_mean=(bias_col, "mean"),
                     kept_bias_std=(bias_col, "std")
                 )
    )

    kept_with_ref = (
        bias_table.groupby(keys)["keep"]
                  .sum()
                  .reset_index()
                  .rename(columns={"keep": "n_kept"})
    )

    cluster_stats = (
        summary_all
        .merge(summary_kept, on=keys, how="left")
        .merge(kept_with_ref, on=keys, how="left")
    )
    cluster_stats["n_beams_total"] = cluster_stats["n_beams"] + 1

    return bias_table, cluster_stats
