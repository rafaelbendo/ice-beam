# ============================================================
# Per-beam crossing angle between each beam and the local shoreline
# ============================================================
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import nearest_points

ANGLE_SUMMARY_COLS = [
    "angle_min_deg",
    "angle_max_deg",
    "angle_mean_deg",
    "angle_median_deg",
    "angle_mode_deg",
]

BEAM_ANGLE_COLS = [
    "track_id", "bias_tolerance", "gt_family", "cluster_id",
    "Acq_date", "beam_id", "beam_angle"
]


def _parse_beam_list(value):
    """Parse beam_ids or beam_ids_ordered stored as list/tuple/string/CSV string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        txt = value.strip()
        if not txt or txt.lower() in {"nan", "none", "null"}:
            return []
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)
            else:
                items = [parsed]
        except Exception:
            items = [part.strip() for part in txt.replace("|", ",").split(",")]
    else:
        items = [value]

    beams = []
    for item in items:
        if isinstance(item, np.ndarray):
            item = item.tolist()

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            beam = item[1]
        else:
            beam = item

        if beam is not None and not (isinstance(beam, float) and pd.isna(beam)):
            beam = str(beam).strip().strip("'\"")
            if beam:
                beams.append(beam)

    return beams


def _angle_between_vectors_acute_deg(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0 or not np.isfinite(denom):
        return np.nan

    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return float(min(angle, 180.0 - angle))


def _shoreline_union_in_crs(shoreline_gdf, target_crs):
    shoreline = shoreline_gdf.copy()
    if target_crs is not None:
        if shoreline.crs is None:
            shoreline = shoreline.set_crs(target_crs, allow_override=True)
        elif shoreline.crs != target_crs:
            shoreline = shoreline.to_crs(target_crs)

    return shoreline.geometry.union_all()


def _local_shoreline_tangent(shoreline_geom, point, search_radius=10.0):
    """Shoreline tangent vector around the nearest shoreline location."""
    if shoreline_geom is None or shoreline_geom.is_empty or point is None or point.is_empty:
        return None

    try:
        nearest_on_shore = nearest_points(point, shoreline_geom)[1]
        proj = shoreline_geom.project(nearest_on_shore)
    except Exception:
        return None

    for radius in [search_radius, search_radius * 2, search_radius * 5, search_radius * 10]:
        d1 = max(proj - radius, 0.0)
        d2 = min(proj + radius, shoreline_geom.length)

        if d2 <= d1:
            continue

        p1 = shoreline_geom.interpolate(d1)
        p2 = shoreline_geom.interpolate(d2)
        tangent = np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)

        if np.linalg.norm(tangent) > 0:
            return tangent

    return None


def _beam_profile_line_and_direction(bdf, dist_col="distance_from_offshore"):
    """Beam direction from first to last point, sorted by distance when available."""
    if bdf is None or bdf.empty or "geometry" not in bdf.columns:
        return None, None

    p = bdf.dropna(subset=["geometry"]).copy()
    if len(p) < 2:
        return None, None

    if dist_col in p.columns:
        p[dist_col] = pd.to_numeric(p[dist_col], errors="coerce")
        p = p.dropna(subset=[dist_col]).sort_values(dist_col)
    else:
        p = p.sort_index()

    coords = [(geom.x, geom.y) for geom in p.geometry if geom is not None and not geom.is_empty]
    if len(coords) < 2:
        return None, None

    direction = np.array(
        [coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]],
        dtype=float
    )

    if np.linalg.norm(direction) == 0:
        return None, None

    return LineString(coords), direction


def _histogram_mode_angle(values, bin_width=5.0):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[(vals >= 0.0) & (vals <= 90.0)]

    if len(vals) == 0:
        return np.nan

    edges = np.arange(0.0, 90.0 + bin_width, bin_width)
    if edges[-1] < 90.0:
        edges = np.append(edges, 90.0)

    counts, edges = np.histogram(vals, bins=edges)
    if counts.sum() == 0:
        return np.nan

    mode_idx = int(np.argmax(counts))
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)


def compute_cluster_angles(
        clusters_gdf,
        shoreline_gdf,
        profiles_gdf,
        track_id=None,
        bias_tolerance=None,
        mode_bin_width=5.0,
        shoreline_search_radius=10.0,
    ):
    """
    Angle between each ICESat-2 beam and the local shoreline tangent.

    Returns
    -------
    clusters_out : GeoDataFrame
        Clusters with angle_min/max/mean/median/mode_deg columns.
    beam_angle_table : DataFrame
        One row per beam per cluster.
    """
    clusters_out = clusters_gdf.copy()

    if "angle_deg" in clusters_out.columns:
        clusters_out = clusters_out.drop(columns=["angle_deg"])

    for col in ANGLE_SUMMARY_COLS:
        clusters_out[col] = np.nan

    if clusters_out is None or clusters_out.empty:
        return clusters_out, pd.DataFrame(columns=BEAM_ANGLE_COLS)

    profiles = profiles_gdf.copy()
    if clusters_out.crs is not None:
        if profiles.crs is None:
            profiles = profiles.set_crs(clusters_out.crs, allow_override=True)
        elif profiles.crs != clusters_out.crs:
            profiles = profiles.to_crs(clusters_out.crs)

    shoreline_geom = _shoreline_union_in_crs(shoreline_gdf, clusters_out.crs)

    date_col = "Acq_date" if "Acq_date" in profiles.columns else "acq_date" if "acq_date" in profiles.columns else None
    beam_rows = []

    for idx, cluster_row in clusters_out.iterrows():
        gt_family = str(cluster_row.get("gt_family", "")).strip()
        cluster_id = cluster_row.get("cluster_id", np.nan)

        beam_ids = _parse_beam_list(cluster_row.get("beam_ids_ordered", None))
        if not beam_ids:
            beam_ids = _parse_beam_list(cluster_row.get("beam_ids", None))

        cluster_angles = []

        for beam_id in beam_ids:
            mask = profiles["beam_id"].astype(str).str.strip().eq(str(beam_id).strip())

            if gt_family and "gt_family" in profiles.columns:
                mask = mask & profiles["gt_family"].astype(str).str.strip().eq(gt_family)

            bdf = profiles.loc[mask].copy()
            line, beam_vec = _beam_profile_line_and_direction(bdf)

            beam_angle = np.nan
            acq_date = pd.NaT

            if date_col is not None and not bdf.empty:
                dates = pd.to_datetime(bdf[date_col], errors="coerce").dropna()
                if len(dates):
                    acq_date = dates.iloc[0].normalize()

            if line is not None and beam_vec is not None:
                try:
                    local_point = nearest_points(line, shoreline_geom)[1]
                except Exception:
                    local_point = line.interpolate(0.5, normalized=True)

                shore_vec = _local_shoreline_tangent(
                    shoreline_geom,
                    local_point,
                    search_radius=shoreline_search_radius
                )

                if shore_vec is not None:
                    beam_angle = _angle_between_vectors_acute_deg(beam_vec, shore_vec)
                    if pd.notna(beam_angle):
                        beam_angle = float(np.clip(beam_angle, 0.0, 90.0))
                        cluster_angles.append(beam_angle)

            beam_rows.append({
                "track_id": (
                    f"{int(track_id):04d}"
                    if track_id is not None and pd.notna(track_id)
                    else cluster_row.get("track_id", np.nan)
                ),
                "bias_tolerance": bias_tolerance,
                "gt_family": gt_family,
                "cluster_id": cluster_id,
                "Acq_date": acq_date,
                "beam_id": str(beam_id).strip(),
                "beam_angle": round(beam_angle, 2) if pd.notna(beam_angle) else np.nan,
            })

        vals = pd.to_numeric(pd.Series(cluster_angles), errors="coerce").dropna()

        if not vals.empty:
            clusters_out.loc[idx, "angle_min_deg"] = round(float(vals.min()), 2)
            clusters_out.loc[idx, "angle_max_deg"] = round(float(vals.max()), 2)
            clusters_out.loc[idx, "angle_mean_deg"] = round(float(vals.mean()), 2)
            clusters_out.loc[idx, "angle_median_deg"] = round(float(vals.median()), 2)
            clusters_out.loc[idx, "angle_mode_deg"] = round(
                _histogram_mode_angle(vals, bin_width=mode_bin_width),
                2
            )

    return clusters_out, pd.DataFrame(beam_rows)
