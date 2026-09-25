# ============================================================
# GIE (geometric) correction for dynamic lateral-growth clusters
# ============================================================
"""
ICESat-2 beams rarely cross the coast perpendicular to it, and repeat beams
of a cluster are offset laterally from each other. Measuring bluff position
change along each beam therefore mixes true shoreline movement with a
geometric term. For each observation:

    NSM_geo       = coast_slope_sign * d_signed * tan(90° - angle_used_deg)
    NSM_corrected = NSM_measured - NSM_geo
    bluff_x_gie   = bluff_x + NSM_geo

where ``d_signed`` is the beam's signed offset from the reference beam along
a transect perpendicular to the reference beam, ``angle_used_deg`` the
beam/shoreline crossing angle (per beam, falling back to the cluster's
median angle), and ``coast_slope_sign`` the local coastline orientation in
the reference beam's frame.

Uses bias_tolerance only (no ClusterSize).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, GeometryCollection

from .bluff import process_cluster_with_reference
from .config import Params
from .utils import first_non_null
from .workflow import run_workflow


@dataclass
class GIEDynamicRunResult:
    summary_df: pd.DataFrame
    interval_df: pd.DataFrame
    beam_df: pd.DataFrame
    bias_values_df: pd.DataFrame


def _normalize_track_id(value) -> Optional[str]:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    return f"{int(num):04d}"


def unpack_cluster_beam_ids(raw_ids) -> list[str]:
    """Sorted, unique beam IDs from beam_ids_ordered / beam_ids."""
    if isinstance(raw_ids, (list, tuple, set)):
        values = raw_ids
    else:
        values = [raw_ids]

    beam_ids = []
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) > 1:
            beam_id = value[1]
        else:
            beam_id = value

        if pd.notna(beam_id):
            beam_ids.append(str(beam_id).strip())

    return sorted(set(beam_ids))


def get_available_bias_values_for_track(
    df_clusters: pd.DataFrame,
    track_id,
    *,
    bias_values: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Bias tolerances that produced DSAS clusters for this track."""

    if df_clusters is None or df_clusters.empty:
        raise ValueError("df_clusters is empty.")

    required = ["track_id", "bias_tolerance"]
    missing = [c for c in required if c not in df_clusters.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    track_str = _normalize_track_id(track_id)

    df = df_clusters.copy()
    df["track_id_norm"] = pd.to_numeric(df["track_id"], errors="coerce").map(
        lambda v: None if pd.isna(v) else f"{int(v):04d}"
    )
    df["bias_tolerance"] = pd.to_numeric(df["bias_tolerance"], errors="coerce")

    if "gt_family" in df.columns:
        df["gt_family"] = df["gt_family"].astype(str).str.strip().str.lower()
    else:
        df["gt_family"] = np.nan

    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    else:
        df["cluster_id"] = np.nan

    df = df[df["track_id_norm"] == track_str].dropna(subset=["bias_tolerance"]).copy()

    if bias_values is not None:
        allowed = pd.Series(list(bias_values)).dropna().astype(float).tolist()
        keep = pd.Series(False, index=df.index)
        for val in allowed:
            keep = keep | np.isclose(df["bias_tolerance"], val)
        df = df[keep].copy()

    if df.empty:
        return pd.DataFrame(columns=["bias_tolerance", "n_clusters", "gt_families"])

    df["cluster_key"] = (
        df["gt_family"].fillna("unknown")
        + ":"
        + df["cluster_id"].astype("Int64").astype(str)
    )

    return (
        df.groupby("bias_tolerance", as_index=False)
        .agg(
            n_clusters=("cluster_key", "nunique"),
            gt_families=("gt_family", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        .sort_values("bias_tolerance")
        .reset_index(drop=True)
    )


def _iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            yield from _iter_lines(part)
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            yield from _iter_lines(part)


def _collect_line_coords(geom) -> np.ndarray:
    coords = []
    for line in _iter_lines(geom):
        coords.extend((float(x), float(y)) for x, y in line.coords)
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 2))


def _unit_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Zero-length vector.")
    return arr / norm


def _project_xy_to_local(xy, origin_xy, axis_u, axis_v):
    rel = np.asarray(xy, dtype=float) - origin_xy
    return np.column_stack([rel @ axis_u, rel @ axis_v])


def get_local_coastline_slope_sign(
    ref_line: LineString,
    coastline_gdf: gpd.GeoDataFrame,
    search_distance: Optional[float] = None,
) -> int:
    """
    Sign of the geometric (GIE) correction for a cluster.

    Anchored to the REFERENCE BEAM's own tangent/normal (the same frame
    build_reference_cross_section_positions() uses to compute d_signed).

    An earlier version derived the frame from PCA on the cluster polygon.
    That frame has no physical connection to the beam geometry d_signed is
    measured in, and PCA's sign is arbitrary: the same coastline and cluster
    location with 5 different cluster shapes gave signs -1, +1, -1, +1, -1.
    Anchoring to ref_line removes cluster shape as an input.

    Validated against independently hand-derived cases:
      - Track 586 GT3 cluster 4/5 (ref beam ATL06_0586_gt3l_20201101): -1
      - Track 129 GT2 cluster 6   (ref beam ATL06_0129_gt2l_20210101): -1

    Parameters
    ----------
    ref_line : LineString
        Reference beam line (same one used for d_signed), in coastline_gdf's CRS.
    coastline_gdf : GeoDataFrame
        Coastline geometry (UTM).
    search_distance : float, optional
        Search radius around ref_line to collect nearby coastline points.
        Defaults to max(25.0, 1.5 * ref_line's bounding-box diagonal).
    """

    if coastline_gdf is None or coastline_gdf.empty:
        return 1

    ref_mid = ref_line.interpolate(0.5, normalized=True)
    origin_xy = np.array([ref_mid.x, ref_mid.y])

    # axis_u = beam tangent, axis_v = beam normal -- SAME frame as d_signed
    axis_u, axis_v = _line_tangent(ref_line)

    minx, miny, maxx, maxy = ref_line.bounds
    diag = float(np.hypot(maxx - minx, maxy - miny))

    if search_distance is None:
        search_distance = max(25.0, 1.5 * diag)

    search_geom = ref_line.buffer(search_distance)
    subset = coastline_gdf.loc[coastline_gdf.intersects(search_geom)].copy()

    if subset.empty:
        distances = coastline_gdf.distance(ref_mid)
        nearest_idx = distances.idxmin()
        coastal_geom = coastline_gdf.loc[[nearest_idx]].geometry.union_all()
    else:
        coastal_geom = subset.geometry.union_all().intersection(search_geom)
        if coastal_geom.is_empty:
            coastal_geom = subset.geometry.union_all()

    coords_xy = _collect_line_coords(coastal_geom)

    if coords_xy.shape[0] < 2:
        return 1

    coords_local = _project_xy_to_local(coords_xy, origin_xy, axis_u, axis_v)
    coords_local = coords_local[np.isfinite(coords_local).all(axis=1)]

    if coords_local.shape[0] < 2:
        return 1

    grouped = (
        pd.DataFrame({"x": np.round(coords_local[:, 0], 6), "y": coords_local[:, 1]})
        .groupby("x", as_index=False)
        .mean()
        .sort_values("x")
    )

    if len(grouped) >= 2 and np.ptp(grouped["x"]) > 0:
        slope = np.polyfit(grouped["x"], grouped["y"], 1)[0]
        if np.isfinite(slope) and slope != 0:
            # NOTE: rule is INVERTED relative to the original PCA-based
            # version -- validated empirically against both hand-derived
            # cases above. Do not "simplify" this back to
            # `1 if slope > 0 else -1` without re-validating.
            return -1 if slope > 0 else 1

    return 1


def build_cluster_beam_lines(
    filtered_profiles: gpd.GeoDataFrame,
    gt_family: str,
    cluster_id,
    beam_ids: Optional[Iterable[str]] = None,
) -> dict[str, LineString]:

    if filtered_profiles is None or filtered_profiles.empty:
        return {}

    df = filtered_profiles.copy()
    df["beam_id"] = df["beam_id"].astype(str).str.strip()
    df["gt_family"] = df["gt_family"].astype(str).str.strip().str.lower()

    subset = df[
        (df["gt_family"] == str(gt_family).strip().lower())
        & (df["cluster_id"] == cluster_id)
    ].copy()

    if beam_ids is not None:
        beam_ids_norm = {str(b).strip() for b in beam_ids}
        subset = subset[subset["beam_id"].isin(beam_ids_norm)].copy()

    subset = subset.dropna(subset=["geometry"]).copy()

    if subset.empty:
        return {}

    beam_lines = {}

    for beam_id, beam_df in subset.groupby("beam_id"):
        order_col = None
        for candidate in [
            "distance_from_offshore",
            "alongtrack_distance",
            "x_atc",
            "along_track_dist",
        ]:
            if candidate in beam_df.columns:
                order_col = candidate
                break

        beam_sorted = beam_df.sort_values(order_col) if order_col else beam_df.copy()

        coords = []
        for geom in beam_sorted.geometry:
            if geom is not None and not geom.is_empty:
                coords.append((float(geom.x), float(geom.y)))

        clean = []
        for xy in coords:
            if not clean or xy != clean[-1]:
                clean.append(xy)

        if len(clean) >= 2:
            line = LineString(clean)
            if line.length > 0:
                beam_lines[str(beam_id).strip()] = line

    return beam_lines


def _line_tangent(line: LineString, frac=0.5, step_frac=0.02):
    length = float(line.length)
    step = max(length * step_frac, 0.5)
    center = length * frac

    p1 = line.interpolate(max(center - step, 0.0))
    p2 = line.interpolate(min(center + step, length))

    vec = np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)
    tangent = _unit_vector(vec)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)

    return tangent, normal


def _extract_intersection_points(geom) -> list[Point]:
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Point":
        return [geom]

    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        pts = []
        for part in geom.geoms:
            pts.extend(_extract_intersection_points(part))
        return pts

    if geom.geom_type in {"LineString", "MultiLineString"}:
        c = geom.centroid
        return [c] if c is not None and not c.is_empty else []

    return []


def build_reference_cross_section_positions(
    filtered_profiles: gpd.GeoDataFrame,
    gt_family: str,
    cluster_id,
    ref_beam_id: str,
    beam_ids: Optional[Iterable[str]] = None,
    transect_half_length: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Signed distance (d_signed) of each cluster beam from the reference beam,
    along a transect perpendicular to the reference beam at its midpoint.
    """

    beam_lines = build_cluster_beam_lines(
        filtered_profiles=filtered_profiles,
        gt_family=gt_family,
        cluster_id=cluster_id,
        beam_ids=beam_ids,
    )

    ref_beam_id = str(ref_beam_id).strip()

    if ref_beam_id not in beam_lines:
        raise ValueError(f"Reference beam {ref_beam_id} not found.")

    ref_line = beam_lines[ref_beam_id]
    ref_mid = ref_line.interpolate(0.5, normalized=True)
    _tangent, normal = _line_tangent(ref_line)

    bounds = np.asarray([line.bounds for line in beam_lines.values()], dtype=float)
    diag = float(
        np.hypot(
            bounds[:, 2].max() - bounds[:, 0].min(),
            bounds[:, 3].max() - bounds[:, 1].min(),
        )
    )

    if transect_half_length is None:
        transect_half_length = max(diag * 0.75, 50.0)

    p_start = Point(
        ref_mid.x - normal[0] * transect_half_length,
        ref_mid.y - normal[1] * transect_half_length,
    )
    p_end = Point(
        ref_mid.x + normal[0] * transect_half_length,
        ref_mid.y + normal[1] * transect_half_length,
    )

    transect = LineString([p_start, p_end])
    ref_proj = float(transect.project(ref_mid))

    records = []

    for beam_id, line in beam_lines.items():
        points = _extract_intersection_points(line.intersection(transect))

        if not points:
            continue

        point = min(points, key=lambda p: p.distance(ref_mid))
        signed_distance = float(transect.project(point) - ref_proj)

        records.append(
            {
                "beam_id": str(beam_id).strip(),
                "geometry": point,
                "d_signed": signed_distance,
                "d_abs": abs(signed_distance),
            }
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=getattr(filtered_profiles, "crs", None),
    )


def prepare_beam_angle_table(beam_angle_table: pd.DataFrame) -> pd.DataFrame:

    if beam_angle_table is None or beam_angle_table.empty:
        return pd.DataFrame()

    df = beam_angle_table.copy()

    rename_map = {}
    if "Acq_date" in df.columns:
        rename_map["Acq_date"] = "acq_date"
    if "beam_angle" in df.columns:
        rename_map["beam_angle"] = "beam_angle_deg"

    df = df.rename(columns=rename_map)

    required = ["gt_family", "cluster_id", "beam_id", "acq_date", "beam_angle_deg"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"beam_angle_table missing columns: {missing}")

    df["gt_family"] = df["gt_family"].astype(str).str.strip().str.lower()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df["beam_id"] = df["beam_id"].astype(str).str.strip()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    df["beam_angle_deg"] = pd.to_numeric(df["beam_angle_deg"], errors="coerce")

    df = df.dropna(
        subset=["gt_family", "cluster_id", "beam_id", "acq_date", "beam_angle_deg"]
    ).copy()

    return df.drop_duplicates(
        subset=["gt_family", "cluster_id", "beam_id", "acq_date"],
        keep="last",
    )


def apply_dynamic_gie_to_cluster_bluff_df(
    *,
    bluff_df: pd.DataFrame,
    cluster_rows: gpd.GeoDataFrame,
    filtered_profiles: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    beam_angle_table: pd.DataFrame,
    fallback_angle_col: str = "angle_median_deg",
) -> pd.DataFrame:
    """Add d_signed, angle_used_deg, NSM_geo, NSM_measured/corrected and bluff_x_gie."""

    if bluff_df is None or bluff_df.empty:
        return pd.DataFrame()

    if cluster_rows is None or cluster_rows.empty:
        raise ValueError("cluster_rows is empty.")

    cluster_row = cluster_rows.iloc[0]
    gt_family = str(cluster_row["gt_family"]).strip().lower()
    cluster_id = cluster_row["cluster_id"]

    out = bluff_df.copy()
    out["beam_id"] = out["beam_id"].astype(str).str.strip()
    out["gt_family"] = gt_family
    out["cluster_id"] = pd.to_numeric(cluster_id, errors="coerce")
    out["acq_date"] = pd.to_datetime(out["acq_date"], errors="coerce")
    out["acq_date_norm"] = out["acq_date"].dt.normalize()
    out["bluff_x"] = pd.to_numeric(out["bluff_x"], errors="coerce")

    out = out.dropna(subset=["beam_id", "acq_date_norm", "bluff_x"]).copy()

    if out.empty:
        return out

    out = out.sort_values(["acq_date_norm", "beam_id"]).reset_index(drop=True)

    ref_row = out.iloc[0]
    ref_beam_id = str(ref_row["beam_id"]).strip()
    ref_date = pd.to_datetime(ref_row["acq_date_norm"]).normalize()
    ref_bluff_x = float(ref_row["bluff_x"])

    beam_ids = unpack_cluster_beam_ids(
        cluster_row.get("beam_ids_ordered", cluster_row.get("beam_ids", []))
    )

    # Build the beam lines once and reuse ref_line for BOTH d_signed and
    # coast_slope_sign, so both come from the same reference-beam frame.
    beam_lines_for_sign = build_cluster_beam_lines(
        filtered_profiles=filtered_profiles,
        gt_family=gt_family,
        cluster_id=cluster_id,
        beam_ids=beam_ids,
    )

    if ref_beam_id not in beam_lines_for_sign:
        raise ValueError(f"Reference beam {ref_beam_id} not found for cluster {gt_family}-{cluster_id}.")

    ref_line_for_sign = beam_lines_for_sign[ref_beam_id]

    cross_positions = build_reference_cross_section_positions(
        filtered_profiles=filtered_profiles,
        gt_family=gt_family,
        cluster_id=cluster_id,
        ref_beam_id=ref_beam_id,
        beam_ids=beam_ids,
    )

    if cross_positions.empty:
        raise ValueError(f"No cross-section positions for {gt_family}-{cluster_id}.")

    out = out.merge(
        cross_positions[["beam_id", "d_signed", "d_abs"]],
        on="beam_id",
        how="left",
    )

    angle_df = prepare_beam_angle_table(beam_angle_table)

    if not angle_df.empty:
        out = out.merge(
            angle_df[
                ["gt_family", "cluster_id", "beam_id", "acq_date", "beam_angle_deg"]
            ],
            left_on=["gt_family", "cluster_id", "beam_id", "acq_date_norm"],
            right_on=["gt_family", "cluster_id", "beam_id", "acq_date"],
            how="left",
            suffixes=("", "_angle"),
        )
        out = out.drop(columns=["acq_date_angle"], errors="ignore")
    else:
        out["beam_angle_deg"] = np.nan

    fallback_angle = (
        float(first_non_null(cluster_rows.get(fallback_angle_col)))
        if fallback_angle_col in cluster_rows.columns
        else np.nan
    )

    out["angle_used_deg"] = out["beam_angle_deg"].fillna(fallback_angle)
    out["angle_source"] = np.where(
        out["beam_angle_deg"].notna(),
        "per_beam",
        f"fallback_{fallback_angle_col}",
    )

    out["angle_used_deg"] = pd.to_numeric(out["angle_used_deg"], errors="coerce")
    out = out.dropna(subset=["d_signed", "angle_used_deg"]).copy()

    if out.empty:
        return out

    coast_slope_sign = get_local_coastline_slope_sign(ref_line_for_sign, coastline_gdf)

    out["reference_beam_id"] = ref_beam_id
    out["reference_date"] = ref_date
    out["reference_bluff_x"] = ref_bluff_x
    out["coast_slope_sign"] = int(coast_slope_sign)

    out["angle_term"] = np.tan(np.deg2rad(90.0 - out["angle_used_deg"]))
    out["NSM_geo"] = out["coast_slope_sign"] * out["d_signed"] * out["angle_term"]

    out["NSM_measured"] = ref_bluff_x - out["bluff_x"]
    out["NSM_corrected"] = out["NSM_measured"] - out["NSM_geo"]
    out["bluff_x_gie"] = out["bluff_x"] + out["NSM_geo"]

    out["distance_method"] = "signed_reference_perpendicular_cross_section"
    out["gie_formula"] = "coast_slope_sign * d_signed * tan(90 - angle_used_deg)"

    return out


def compute_cluster_statistics_from_x(
    bluff_df: pd.DataFrame,
    x_col: str = "bluff_x",
    confidence: float = 0.95,
    min_span_days: int = 365,
    positional_uncertainty_m: float = 4.8,
) -> dict:
    """
    DSAS metrics from any position column (bluff_x or bluff_x_gie).

    Same conventions as metrics.compute_cluster_statistics, but dates are
    normalized to days before averaging.
    """
    from scipy.stats import t as student_t
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    base = {
        "NSM": np.nan,
        "SCE": np.nan,
        "EPR": np.nan,
        "LRR": np.nan,
        "LR2": np.nan,
        "LSE": np.nan,
        "LCI": np.nan,
        "TemporalSpan_days": np.nan,
        "ClusterTemporalSpanYears": np.nan,
        "ValidRegression": False,
        "U_position_m": positional_uncertainty_m,
        "U_NSM_m": np.nan,
        "U_EPR_myr": np.nan,
    }

    if bluff_df is None or bluff_df.empty or x_col not in bluff_df.columns:
        return base

    df = bluff_df[["acq_date", x_col]].copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=["acq_date", x_col]).copy()

    if df.empty:
        return base

    df = (
        df.groupby("acq_date", as_index=False)
        .agg(position=(x_col, "mean"))
        .sort_values("acq_date")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        return base

    x_first = float(df["position"].iloc[0])
    x_last = float(df["position"].iloc[-1])

    nsm = x_first - x_last
    sce = df["position"].max() - df["position"].min()

    span_days = int((df["acq_date"].iloc[-1] - df["acq_date"].iloc[0]).days)
    time_years = span_days / 365.25 if span_days > 0 else np.nan

    epr = (
        nsm / time_years
        if span_days >= min_span_days and np.isfinite(time_years) and time_years > 0
        else np.nan
    )

    lrr = lr2 = lse = lci = np.nan

    if len(df) >= 3 and span_days >= min_span_days:
        years = df["acq_date"].map(lambda d: d.year + d.dayofyear / 365.25).to_numpy()
        xvals = df["position"].to_numpy(dtype=float)

        model = LinearRegression().fit(years.reshape(-1, 1), xvals)
        pred = model.predict(years.reshape(-1, 1))

        slope = float(model.coef_[0])
        lr2 = float(r2_score(xvals, pred))

        resid = xvals - pred
        dof = len(xvals) - 2
        s_yx = np.sqrt(np.sum(resid**2) / dof) if dof > 0 else np.nan
        sxx = np.sum((years - years.mean()) ** 2)
        se_slope = s_yx / np.sqrt(sxx) if sxx > 0 else np.nan

        if np.isfinite(se_slope) and dof > 0:
            tcrit = student_t.ppf(1 - (1 - confidence) / 2, df=dof)
            lci = float(tcrit * se_slope)

        lrr = round(-slope, 2)
        lr2 = round(lr2, 2)
        lse = round(s_yx, 2) if np.isfinite(s_yx) else np.nan
        lci = round(lci, 2) if np.isfinite(lci) else np.nan

    u_nsm = np.sqrt(positional_uncertainty_m**2 + positional_uncertainty_m**2)
    u_epr = (
        u_nsm / time_years
        if np.isfinite(epr) and np.isfinite(time_years) and time_years >= 1
        else np.nan
    )

    return {
        "NSM": int(round(nsm)) if np.isfinite(nsm) else np.nan,
        "SCE": int(round(sce)) if np.isfinite(sce) else np.nan,
        "EPR": round(float(epr), 2) if np.isfinite(epr) else np.nan,
        "LRR": lrr,
        "LR2": lr2,
        "LSE": lse,
        "LCI": lci,
        "TemporalSpan_days": span_days,
        "ClusterTemporalSpanYears": round(time_years, 2) if np.isfinite(time_years) else np.nan,
        "ValidRegression": bool(np.isfinite(lrr) and len(df) >= 3 and span_days >= min_span_days),
        "U_position_m": positional_uncertainty_m,
        "U_NSM_m": round(float(u_nsm), 2),
        "U_EPR_myr": round(float(u_epr), 2) if np.isfinite(u_epr) else np.nan,
    }


def build_daily_bluff_series(
    bluff_df: pd.DataFrame,
    measured_col: str = "bluff_x",
    corrected_col: str = "bluff_x_gie",
) -> pd.DataFrame:

    if bluff_df is None or bluff_df.empty:
        return pd.DataFrame()

    keep_cols = ["beam_id", "acq_date", measured_col, corrected_col]

    for col in ["bluff_y", "NSM_geo"]:
        if col in bluff_df.columns:
            keep_cols.append(col)

    df = bluff_df[keep_cols].copy()
    df["beam_id"] = df["beam_id"].astype(str)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    df[measured_col] = pd.to_numeric(df[measured_col], errors="coerce")
    df[corrected_col] = pd.to_numeric(df[corrected_col], errors="coerce")

    if "bluff_y" not in df.columns:
        df["bluff_y"] = np.nan

    if "NSM_geo" not in df.columns:
        df["NSM_geo"] = np.nan

    df = df.dropna(subset=["acq_date", measured_col, corrected_col]).copy()

    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby("acq_date", as_index=False)
        .agg(
            beam_id=("beam_id", lambda s: "|".join(sorted(set(map(str, s))))),
            bluff_x=(measured_col, "mean"),
            bluff_x_gie=(corrected_col, "mean"),
            bluff_y=("bluff_y", "mean"),
            NSM_geo=("NSM_geo", "mean"),
            n_obs=("beam_id", "size"),
        )
        .sort_values("acq_date")
        .reset_index(drop=True)
    )


def compute_dynamic_dual_intervals(
    *,
    bluff_df: pd.DataFrame,
    gt_family: str,
    cluster_id,
    track_id: str,
    bias_tolerance: float,
    cluster_width_m=np.nan,
    n_beams=np.nan,
) -> list[dict]:
    """Inter-date intervals with both measured and GIE-corrected positions."""

    day_df = build_daily_bluff_series(bluff_df)

    if day_df.empty or len(day_df) < 2:
        return []

    rows = []

    for i in range(len(day_df) - 1):
        a = day_df.iloc[i]
        b = day_df.iloc[i + 1]

        date_from = pd.to_datetime(a["acq_date"])
        date_to = pd.to_datetime(b["acq_date"])

        x_meas_from = float(a["bluff_x"])
        x_meas_to = float(b["bluff_x"])
        x_corr_from = float(a["bluff_x_gie"])
        x_corr_to = float(b["bluff_x_gie"])

        interval_nsm_measured = x_meas_from - x_meas_to
        interval_nsm_corrected = x_corr_from - x_corr_to

        rows.append(
            {
                "track_id": track_id,
                "bias_tolerance": float(bias_tolerance),
                "gt_family": gt_family,
                "cluster_id": int(cluster_id),
                "cluster_width_m": round(cluster_width_m, 2) if np.isfinite(cluster_width_m) else np.nan,
                "n_beams": int(n_beams) if pd.notna(n_beams) else np.nan,
                "interval_order": int(i + 1),
                "beam_from": a["beam_id"],
                "beam_to": b["beam_id"],
                "date_from": date_from.normalize(),
                "date_to": date_to.normalize(),
                "delta_days": int((date_to - date_from).days),
                "n_obs_from": int(a["n_obs"]),
                "n_obs_to": int(b["n_obs"]),
                "bluff_x_from_measured": int(round(x_meas_from)),
                "bluff_x_to_measured": int(round(x_meas_to)),
                "interval_NSM_measured": int(round(interval_nsm_measured)),
                "bluff_x_from_corrected": int(round(x_corr_from)),
                "bluff_x_to_corrected": int(round(x_corr_to)),
                "interval_NSM_corrected": int(round(interval_nsm_corrected)),
                "mean_interval_geo": (
                    round(float(b["NSM_geo"] - a["NSM_geo"]), 3)
                    if pd.notna(a["NSM_geo"]) and pd.notna(b["NSM_geo"])
                    else np.nan
                ),
                "direction_measured": (
                    "retreat" if interval_nsm_measured < 0
                    else "advance" if interval_nsm_measured > 0
                    else "stable"
                ),
                "direction_corrected": (
                    "retreat" if interval_nsm_corrected < 0
                    else "advance" if interval_nsm_corrected > 0
                    else "stable"
                ),
            }
        )

    return rows


def run_gie_dynamic_for_bias_values(
    *,
    track_id,
    dataset_raw,
    shoreline_gdf: gpd.GeoDataFrame,
    params: Params,
    utm_epsg: int,
    cluster_summary_df: Optional[pd.DataFrame] = None,
    bias_values: Optional[Iterable[float]] = None,
) -> GIEDynamicRunResult:
    """
    Re-run the clustering workflow for each bias tolerance and apply the GIE
    correction to every selected cluster.

    Bias tolerances come from ``bias_values``, or, when None, from the DSAS
    summary rows of this track in ``cluster_summary_df``.
    """
    track_str = _normalize_track_id(track_id)

    if bias_values is None:
        if cluster_summary_df is None or cluster_summary_df.empty:
            raise ValueError("cluster_summary_df required when bias_values is None.")

        bias_values_df = get_available_bias_values_for_track(
            cluster_summary_df,
            track_id,
        )
        bias_values_list = bias_values_df["bias_tolerance"].dropna().tolist()
    else:
        bias_values_list = sorted(pd.Series(list(bias_values)).dropna().astype(float).unique())
        bias_values_df = pd.DataFrame({"bias_tolerance": bias_values_list})

    summary_rows = []
    interval_rows = []
    beam_tables = []

    for bias_tolerance in bias_values_list:

        workflow_result = run_workflow(
            track_id=track_id,
            dataset_raw=dataset_raw,
            shoreline_gdf=shoreline_gdf,
            bias_tolerance=float(bias_tolerance),
            params=params,
            utm_epsg=utm_epsg,
            verbose=False,
        )

        selected_clusters = workflow_result.selected_clusters
        filtered_profiles = workflow_result.filtered_profiles
        bias_df = workflow_result.bias_df
        beam_angle_table = workflow_result.beam_angle_table

        if selected_clusters is None or selected_clusters.empty:
            continue

        selected_clusters = selected_clusters.copy()
        selected_clusters["gt_family"] = selected_clusters["gt_family"].astype(str).str.strip().str.lower()
        selected_clusters["cluster_id"] = pd.to_numeric(selected_clusters["cluster_id"], errors="coerce")

        coastline_local = shoreline_gdf

        if selected_clusters.crs is not None and shoreline_gdf.crs != selected_clusters.crs:
            coastline_local = shoreline_gdf.to_crs(selected_clusters.crs)

        if filtered_profiles is not None and not filtered_profiles.empty:
            filtered_profiles = filtered_profiles.copy()
            filtered_profiles["gt_family"] = filtered_profiles["gt_family"].astype(str).str.strip().str.lower()

        for gt_family in sorted(selected_clusters["gt_family"].dropna().unique()):
            fam_clusters = selected_clusters[selected_clusters["gt_family"] == gt_family].copy()

            initial_cycles = (
                dataset_raw.query("gt_family == @gt_family")["beam_id"].nunique()
                if dataset_raw is not None and not dataset_raw.empty and "gt_family" in dataset_raw.columns
                else 0
            )

            used_cycles = (
                filtered_profiles.query("gt_family == @gt_family")["beam_id"].nunique()
                if filtered_profiles is not None and not filtered_profiles.empty
                else 0
            )

            for cluster_id in sorted(fam_clusters["cluster_id"].dropna().unique()):
                cluster_rows = fam_clusters[fam_clusters["cluster_id"] == cluster_id].copy()

                if cluster_rows.empty:
                    continue

                bluff_df, _y_ref = process_cluster_with_reference(
                    filtered_profiles=filtered_profiles,
                    selected_clusters=selected_clusters,
                    cluster_id=cluster_id,
                    gt_family=gt_family,
                    which=params.BLUFF_WHICH,
                    gap_threshold=params.GAP_THRESHOLD_M,
                    atol=params.CROSSING_ATOL,
                    bias_df=bias_df,
                    debug=False,
                )

                if bluff_df is None or bluff_df.empty:
                    continue

                gie_df = apply_dynamic_gie_to_cluster_bluff_df(
                    bluff_df=bluff_df,
                    cluster_rows=cluster_rows,
                    filtered_profiles=filtered_profiles,
                    coastline_gdf=coastline_local,
                    beam_angle_table=beam_angle_table,
                    fallback_angle_col=params.GIE_FALLBACK_ANGLE_COL,
                )

                if gie_df.empty:
                    continue

                stats_kwargs = dict(
                    confidence=params.CONFIDENCE,
                    min_span_days=params.MIN_SPAN_DAYS,
                    positional_uncertainty_m=params.POSITIONAL_UNCERTAINTY_M,
                )
                measured_stats = compute_cluster_statistics_from_x(gie_df, x_col="bluff_x", **stats_kwargs)
                corrected_stats = compute_cluster_statistics_from_x(gie_df, x_col="bluff_x_gie", **stats_kwargs)

                daily_df = build_daily_bluff_series(gie_df)
                first_dt = daily_df["acq_date"].min() if not daily_df.empty else pd.NaT
                last_dt = daily_df["acq_date"].max() if not daily_df.empty else pd.NaT

                cluster_width_m = (
                    float(first_non_null(cluster_rows.get("cluster_width_m")))
                    if "cluster_width_m" in cluster_rows.columns
                    else np.nan
                )

                if "num_beams" in cluster_rows.columns:
                    n_beams = first_non_null(cluster_rows.get("num_beams"))
                elif "n_beams" in cluster_rows.columns:
                    n_beams = first_non_null(cluster_rows.get("n_beams"))
                else:
                    n_beams = len(
                        unpack_cluster_beam_ids(
                            cluster_rows.iloc[0].get(
                                "beam_ids_ordered",
                                cluster_rows.iloc[0].get("beam_ids", []),
                            )
                        )
                    )

                def _first_float(col):
                    return float(first_non_null(cluster_rows.get(col))) if col in cluster_rows.columns else np.nan

                used_cycles_cluster = (
                    filtered_profiles.query("gt_family == @gt_family and cluster_id == @cluster_id")["beam_id"].nunique()
                    if filtered_profiles is not None and not filtered_profiles.empty and "cluster_id" in filtered_profiles.columns
                    else np.nan
                )

                summary_rows.append(
                    {
                        "track_id": track_str,
                        "bias_tolerance": float(bias_tolerance),
                        "gt_family": gt_family,
                        "cluster_id": int(cluster_id),
                        "cluster_width_m": round(cluster_width_m, 2) if np.isfinite(cluster_width_m) else np.nan,
                        "n_beams": int(n_beams) if pd.notna(n_beams) else np.nan,
                        "reference_beam_id": gie_df["reference_beam_id"].iloc[0],
                        "reference_date": gie_df["reference_date"].iloc[0],
                        "coast_slope_sign": int(gie_df["coast_slope_sign"].iloc[0]),
                        "center_lat": _first_float("center_lat"),
                        "center_lon": _first_float("center_lon"),
                        "elev_avg": _first_float("elev_avg"),
                        "first_date": first_dt,
                        "last_date": last_dt,
                        "TemporalSpan_days": measured_stats["TemporalSpan_days"],
                        "ClusterTemporalSpanYears": measured_stats["ClusterTemporalSpanYears"],
                        "initial_cycles": initial_cycles,
                        "used_cycles": used_cycles,
                        "used_cycles_cluster": used_cycles_cluster,
                        "angle_mean_deg": _first_float("angle_mean_deg"),
                        "angle_median_deg": _first_float("angle_median_deg"),
                        "angle_mode_deg": _first_float("angle_mode_deg"),
                        "angle_used_mean_deg": round(float(gie_df["angle_used_deg"].mean()), 3),
                        "angle_used_median_deg": round(float(gie_df["angle_used_deg"].median()), 3),
                        "max_abs_gie": round(float(gie_df["NSM_geo"].abs().max()), 3),
                        "mean_abs_gie": round(float(gie_df["NSM_geo"].abs().mean()), 3),
                        "U_position_m": measured_stats["U_position_m"],
                        "U_NSM_measured_m": measured_stats["U_NSM_m"],
                        "U_EPR_measured_myr": measured_stats["U_EPR_myr"],
                        "U_NSM_corrected_m": corrected_stats["U_NSM_m"],
                        "U_EPR_corrected_myr": corrected_stats["U_EPR_myr"],
                        "NSM_measured": measured_stats["NSM"],
                        "SCE_measured": measured_stats["SCE"],
                        "EPR_measured": measured_stats["EPR"],
                        "LRR_measured": measured_stats["LRR"],
                        "LR2_measured": measured_stats["LR2"],
                        "LSE_measured": measured_stats["LSE"],
                        "LCI_measured": measured_stats["LCI"],
                        "ValidRegression_measured": measured_stats["ValidRegression"],
                        "NSM_corrected": corrected_stats["NSM"],
                        "SCE_corrected": corrected_stats["SCE"],
                        "EPR_corrected": corrected_stats["EPR"],
                        "LRR_corrected": corrected_stats["LRR"],
                        "LR2_corrected": corrected_stats["LR2"],
                        "LSE_corrected": corrected_stats["LSE"],
                        "LCI_corrected": corrected_stats["LCI"],
                        "ValidRegression_corrected": corrected_stats["ValidRegression"],
                    }
                )

                interval_rows.extend(
                    compute_dynamic_dual_intervals(
                        bluff_df=gie_df,
                        gt_family=gt_family,
                        cluster_id=cluster_id,
                        track_id=track_str,
                        bias_tolerance=bias_tolerance,
                        cluster_width_m=cluster_width_m,
                        n_beams=n_beams,
                    )
                )

                beam_tables.append(
                    gie_df.assign(
                        track_id=track_str,
                        bias_tolerance=float(bias_tolerance),
                        gt_family=gt_family,
                        cluster_id=int(cluster_id),
                        cluster_width_m=cluster_width_m,
                        n_beams=n_beams,
                    )
                )

    return GIEDynamicRunResult(
        summary_df=pd.DataFrame(summary_rows),
        interval_df=pd.DataFrame(interval_rows),
        beam_df=pd.concat(beam_tables, ignore_index=True) if beam_tables else pd.DataFrame(),
        bias_values_df=bias_values_df,
    )
