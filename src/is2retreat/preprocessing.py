"""
preprocessing.py

One-file (for now) module that contains:
- filename metadata helpers
- I/O helpers (shoreline + beams)
- geometry helpers (middle beam, shoreline crossing, oriented boxes)
- preprocessing filters applied to clipped data
- distance along beam (offshore -> inland)

IMPORTANT FIX
-------------
Your old workflow assumes `beam_id` is the PHYSICAL beam name:
    gt1l, gt1r, gt2l, gt2r, gt3l, gt3r

A previous version set:
    beam_id = f.stem   (unique per file/cycle)

That breaks "middle beam" logic and shoreline crossing stability.

This file restores:
    beam_id = extract_beam_id(filename)  # gt1l/gt1r/...

And stores provenance separately:
    file_stem, file_path
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Polygon, Point
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Internal helpers
# ============================================================

def _union_all(geoseries: gpd.GeoSeries):
    """
    Compatibility helper for Shapely/GeoPandas union API differences.
    """
    if hasattr(geoseries, "union_all"):
        try:
            return geoseries.union_all()
        except Exception:
            pass
    return geoseries.unary_union


def _build_beam_lines(fam_df_utm: gpd.GeoDataFrame) -> Dict[str, LineString]:
    """
    Build an ordered LineString for each physical beam_id in a (UTM-projected) family GeoDataFrame.
    beam_id must be one of: gt1l/gt1r/gt2l/gt2r/gt3l/gt3r
    """
    lines: Dict[str, LineString] = {}
    if fam_df_utm is None or fam_df_utm.empty:
        return lines

    for beam_id, g in fam_df_utm.groupby("beam_id"):
        if len(g) < 2:
            continue
        g_sorted = (
            g.assign(_y=g.geometry.y)
             .sort_values("_y")
             .drop(columns="_y")
        )
        line = LineString(g_sorted.geometry.tolist())
        if line.is_empty or line.length == 0:
            continue
        lines[str(beam_id)] = line

    return lines


def _pick_cross_point(intersection_geom, center_pt: Point) -> Optional[Point]:
    """
    Given an intersection geometry, return a single Point that represents the crossing.
    """
    if intersection_geom is None or intersection_geom.is_empty:
        return None

    gt = intersection_geom.geom_type
    if gt == "Point":
        return intersection_geom
    if gt == "MultiPoint":
        return min(intersection_geom.geoms, key=lambda p: p.distance(center_pt))

    # LineString / GeometryCollection / etc.
    return intersection_geom.centroid


def _local_tangent(nearest_line: LineString, cross_pt: Point, ds: float = 50.0) -> Optional[Tuple[float, float]]:
    """
    Compute a unit tangent vector near the shoreline crossing.
    ds is the half-window length (meters) along the line used to compute direction.
    """
    if nearest_line is None or nearest_line.is_empty or nearest_line.length == 0:
        return None
    if cross_pt is None or cross_pt.is_empty:
        return None

    s = float(nearest_line.project(cross_pt))
    s0 = max(0.0, s - float(ds))
    s1 = min(float(nearest_line.length), s + float(ds))

    p0 = nearest_line.interpolate(s0)
    p1 = nearest_line.interpolate(s1)

    vx, vy = (p1.x - p0.x), (p1.y - p0.y)
    n = float(np.hypot(vx, vy))
    if n == 0:
        return None
    return (vx / n, vy / n)


# ============================================================
# Metadata helpers (filename parsing)
# ============================================================

def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract acquisition date (YYYYMMDD) from an ATL06-like filename.

    Example:
        ATL06_0129_gt1l_20190105.shp  →  datetime(2019, 1, 5)
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    candidates = [p for p in parts if len(p) == 8 and p.isdigit()]
    if not candidates:
        return None

    try:
        return datetime.strptime(candidates[-1], "%Y%m%d")
    except Exception as e:
        warnings.warn(f"Date parse failed for {filename}: {e}")
        return None


def extract_gt_family(filename: str) -> Optional[str]:
    """
    Extract ICESat-2 beam family: 'gt1', 'gt2', or 'gt3'
    from tokens like gt1l/gt1r/gt2l/gt2r/gt3l/gt3r.
    """
    tokens = Path(filename).stem.lower().split("_")
    for t in tokens:
        if t.startswith("gt1"):
            return "gt1"
        if t.startswith("gt2"):
            return "gt2"
        if t.startswith("gt3"):
            return "gt3"
    return None


def extract_beam_id(filename: str) -> Optional[str]:
    """
    Extract the physical beam id: gt1l/gt1r/gt2l/gt2r/gt3l/gt3r.
    """
    tokens = Path(filename).stem.lower().split("_")
    valid = {"gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"}

    for t in tokens:
        if t in valid:
            return t

    # fallback: substring scan
    for t in tokens:
        for b in valid:
            if b in t:
                return b

    return None


def extract_track_number(filename: str) -> Optional[str]:
    """
    Extract the 4-digit track number from a filename.

    Examples:
        ATL06_0129_gt1l_20190105.shp → '0129'
        ATL06_20190105_01290203_007  → '0129'
    """
    stem = Path(filename).stem

    for token in stem.split("_"):
        if token.isdigit() and len(token) == 4:
            return token

    digits = "".join(c if c.isdigit() else " " for c in stem).split()
    for seq in digits:
        if len(seq) >= 4:
            return seq[:4]

    return None


# ============================================================
# I/O helpers (shoreline + beams)
# ============================================================

def load_shoreline(
    shoreline_fp: Union[str, Path],
    default_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Load a shoreline file (shp/gpkg/geojson) and ensure it has a CRS.
    """
    shoreline_fp = Path(shoreline_fp)
    if not shoreline_fp.exists():
        raise FileNotFoundError(f"Shoreline file not found:\n{shoreline_fp}")

    gdf = gpd.read_file(shoreline_fp)
    if gdf.crs is None:
        warnings.warn(f"Shoreline file has no CRS. Assigning {default_crs}.")
        gdf = gdf.set_crs(default_crs)

    return gdf


def load_beams(
    input_folder: Union[str, Path],
    original_crs: str = "EPSG:4326",
    utm_epsg: int = 32606,
    pattern: str = "*.shp",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Read every file matching `pattern` in input_folder and attach metadata:
        - gt_family (gt1, gt2, gt3)
        - track_id  (4-digit)
        - acq_date  (datetime)
        - year
        - beam_id   (PHYSICAL beam: gt1l/gt1r/...)
        - file_stem (per-file id for provenance/debug)
        - file_path

    Returns:
        GeoDataFrame projected to UTM (EPSG:utm_epsg)
    """
    input_folder = Path(input_folder)
    files = sorted(input_folder.glob(pattern))

    if verbose:
        print(f"📁 Found {len(files)} files in {input_folder}")

    rows: List[gpd.GeoDataFrame] = []

    for f in files:
        try:
            gdf = gpd.read_file(f)
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to read {f.name}: {e}")
            continue

        if gdf.empty or not {lat_col, lon_col}.issubset(gdf.columns):
            if verbose:
                print(f"[INFO] Skipped (empty or missing {lat_col}/{lon_col}): {f.name}")
            continue

        # ensure geometry built from lat/lon
        gdf = gdf.set_geometry(
            gpd.points_from_xy(gdf[lon_col], gdf[lat_col], crs=original_crs)
        )
        gdf = gdf.to_crs(utm_epsg)

        # metadata
        date_val  = extract_date_from_filename(f.name)
        fam       = extract_gt_family(f.name)
        track_val = extract_track_number(f.name)
        beam_phys = extract_beam_id(f.name)

        # skip files with missing metadata (keeps geometry logic clean downstream)
        if beam_phys is None or fam is None or track_val is None or date_val is None:
            if verbose:
                print(f"[WARN] Missing metadata → skipped: {f.name}")
            continue

        gdf["gt_family"] = fam
        gdf["track_id"]  = track_val
        gdf["acq_date"]  = date_val
        gdf["year"]      = date_val.year

        # IMPORTANT: physical beam name (gt1l/gt1r/...)
        gdf["beam_id"] = beam_phys

        # provenance/debug
        gdf["file_stem"] = f.stem
        gdf["file_path"] = str(f)

        rows.append(gdf)

    if not rows:
        if verbose:
            print("⚠️ No valid files were loaded.")
        return gpd.GeoDataFrame(columns=["geometry"], crs=f"EPSG:{utm_epsg}")

    dataset = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), crs=f"EPSG:{utm_epsg}")

    if verbose:
        nb_phys = dataset["beam_id"].nunique()
        nb_files = dataset["file_stem"].nunique() if "file_stem" in dataset.columns else np.nan
        print(f"[RAW] Loaded beams: {nb_phys} physical beams")
        print(f"      Files (cycles): {nb_files}")
        print(f"      Total points: {len(dataset):,}")
        print(f"      Physical beams present: {sorted(dataset['beam_id'].dropna().unique().tolist())}")

    return dataset


# ============================================================
# Geometry helpers (middle beam, shoreline crossing, oriented boxes)
# ============================================================

def get_middle_crossing(
    dataset_raw: gpd.GeoDataFrame,
    shoreline_utm: gpd.GeoDataFrame,
    utm_epsg: int,
    gt_family: str,
) -> Tuple[Optional[Point], Optional[Point], Optional[str], Optional[LineString]]:
    """
    For a given gt_family:
        - project family points to UTM
        - compute overall centroid
        - build LineString per PHYSICAL beam (gtXl/gtXr)
        - pick beam line closest to centroid (the "middle beam")
        - intersect with shoreline (already in UTM)

    Returns:
        center_pt, cross_pt, nearest_beam, nearest_line
    """
    if dataset_raw is None or dataset_raw.empty:
        return None, None, None, None

    fam_df = dataset_raw[dataset_raw["gt_family"] == gt_family].copy()
    if fam_df.empty:
        return None, None, None, None

    # Defensive CRS check for shoreline
    if shoreline_utm is None or shoreline_utm.empty:
        raise ValueError("shoreline_utm is empty.")
    if shoreline_utm.crs is None:
        raise ValueError("shoreline_utm has no CRS.")
    epsg = shoreline_utm.crs.to_epsg()
    if epsg is None or int(epsg) != int(utm_epsg):
        raise ValueError(f"shoreline_utm must be in EPSG:{utm_epsg} (got {shoreline_utm.crs}).")

    fam_df = fam_df.to_crs(utm_epsg)

    center_pt = _union_all(fam_df.geometry).centroid
    lines = _build_beam_lines(fam_df)

    if not lines:
        return center_pt, None, None, None

    nearest_beam, nearest_line = min(
        lines.items(),
        key=lambda kv: kv[1].centroid.distance(center_pt)
    )

    shore_union = _union_all(shoreline_utm.geometry)
    inter = nearest_line.intersection(shore_union)
    cross_pt = _pick_cross_point(inter, center_pt)

    return center_pt, cross_pt, nearest_beam, nearest_line


def build_box_from_centroid(
    dataset_raw: gpd.GeoDataFrame,
    shoreline_gdf: gpd.GeoDataFrame,
    utm_epsg: int,
    gt_family: str,
    half_along: float,
    half_across: float,
    verbose: bool = False,
) -> Tuple[Optional[Point], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[Point], Optional[str]]:
    """
    Build an oriented extraction box around the shoreline crossing point for a SINGLE gt_family.

    Returns:
        (center_pt, box_gdf, clipped_points, cross_pt, nearest_beam)
    """
    fam_df = dataset_raw[dataset_raw["gt_family"] == gt_family].copy()
    if fam_df.empty:
        if verbose:
            print(f"[INFO] {gt_family}: no points found.")
        return None, None, None, None, None

    try:
        fam_df = fam_df.to_crs(utm_epsg)
        shoreline_utm = shoreline_gdf.to_crs(utm_epsg)

        center_pt = _union_all(fam_df.geometry).centroid
        lines = _build_beam_lines(fam_df)
        if not lines:
            if verbose:
                print(f"[INFO] {gt_family}: no valid beam lines.")
            return center_pt, None, None, None, None

        nearest_beam, nearest_line = min(
            lines.items(),
            key=lambda kv: kv[1].centroid.distance(center_pt)
        )

        shore_union = _union_all(shoreline_utm.geometry)
        inter = nearest_line.intersection(shore_union)
        cross_pt = _pick_cross_point(inter, center_pt)

        if cross_pt is None:
            if verbose:
                print(f"[WARN] {gt_family}: no shoreline crossing found.")
            return center_pt, None, None, None, nearest_beam

        # local tangent near crossing
        t = _local_tangent(nearest_line, cross_pt, ds=50.0)
        if t is None:
            if verbose:
                print(f"[WARN] {gt_family}: invalid tangent vector.")
            return center_pt, None, None, cross_pt, nearest_beam

        vx, vy = t
        t_hat = np.array([vx, vy])
        n_hat = np.array([-t_hat[1], t_hat[0]])

        c = np.array([cross_pt.x, cross_pt.y])
        corners = [
            c + t_hat * half_along + n_hat * half_across,
            c - t_hat * half_along + n_hat * half_across,
            c - t_hat * half_along - n_hat * half_across,
            c + t_hat * half_along - n_hat * half_across,
        ]

        box_poly = Polygon(corners)
        box_gdf = gpd.GeoDataFrame({"geometry": [box_poly]}, crs=f"EPSG:{utm_epsg}")

        mask = fam_df.geometry.within(box_poly) | fam_df.geometry.touches(box_poly)
        clipped_points = fam_df.loc[mask].copy()

        return center_pt, box_gdf, clipped_points, cross_pt, nearest_beam

    except Exception as e:
        if verbose:
            print(f"[ERROR] {gt_family}: box-build failed → {e}")
        return None, None, None, None, None


def build_boxes_for_families(
    dataset_raw: gpd.GeoDataFrame,
    shoreline_gdf: gpd.GeoDataFrame,
    utm_epsg: int,
    half_along: Optional[float] = None,
    half_across: Optional[float] = None,
    families: Optional[Sequence[str]] = None,
    params: Optional[object] = None,
    verbose: bool = True,
):
    """
    Build oriented extraction boxes for ALL families.

    Returns dict:
        fam: {
            "box": GeoDataFrame,
            "clipped": GeoDataFrame,
            "cross": Point,
            "center": Point,
            "nearest_beam": str
        }
    """
    if params is not None:
        half_along  = getattr(params, "HALF_ALONG_M", half_along)
        half_across = getattr(params, "HALF_ACROSS_M", half_across)
        families    = getattr(params, "GTX", families)

    if half_along is None or half_across is None:
        raise ValueError("half_along and half_across must be provided (or via params).")

    if families is None:
        families = ("gt1", "gt2", "gt3")

    dataset_clean: Dict[str, Dict[str, object]] = {}

    shoreline_utm = shoreline_gdf.to_crs(utm_epsg)

    for fam in families:
        fam_df = dataset_raw[dataset_raw["gt_family"] == fam].copy()
        if fam_df.empty:
            if verbose:
                print(f"[INFO] {fam}: no points found.")
            continue

        fam_df = fam_df.to_crs(utm_epsg)

        center_pt, cross_pt, nearest_beam, nearest_line = get_middle_crossing(
            dataset_raw=dataset_raw,
            shoreline_utm=shoreline_utm,
            utm_epsg=utm_epsg,
            gt_family=fam,
        )

        if cross_pt is None or nearest_line is None:
            if verbose:
                print(f"[WARN] {fam}: no shoreline crossing found.")
            continue

        # local tangent near crossing (more stable than normalized 0.4/0.6)
        t = _local_tangent(nearest_line, cross_pt, ds=50.0)
        if t is None:
            if verbose:
                print(f"[WARN] {fam}: invalid tangent vector.")
            continue

        vx, vy = t
        t_hat = np.array([vx, vy])
        n_hat = np.array([-t_hat[1], t_hat[0]])

        C = np.array([cross_pt.x, cross_pt.y])
        corners = [
            C + t_hat * half_along + n_hat * half_across,
            C - t_hat * half_along + n_hat * half_across,
            C - t_hat * half_along - n_hat * half_across,
            C + t_hat * half_along - n_hat * half_across,
        ]
        box_poly = Polygon(corners)
        box_gdf = gpd.GeoDataFrame({"geometry": [box_poly]}, crs=f"EPSG:{utm_epsg}")

        mask = fam_df.geometry.within(box_poly) | fam_df.geometry.touches(box_poly)
        clipped = fam_df.loc[mask].copy()

        if clipped.empty:
            if verbose:
                print(f"[WARN] {fam}: no clipped points inside box.")
            continue

        # NOTE: radial distance to crossing point (legacy behavior).
        cx, cy = cross_pt.x, cross_pt.y
        clipped["distance_from_offshore"] = clipped.geometry.apply(
            lambda p: float(np.hypot(p.x - cx, p.y - cy))
        )

        dataset_clean[fam] = {
            "box": box_gdf,
            "clipped": clipped,
            "cross": cross_pt,
            "center": center_pt,
            "nearest_beam": nearest_beam,
        }

        if verbose:
            print(f"[OK] {fam}: box built with {len(clipped)} clipped points. nearest_beam={nearest_beam}")

    return dataset_clean


# ============================================================
# Preprocessing constraints on clipped data
# ============================================================

def apply_preprocessing_to_clipped(
    dataset_clipped: gpd.GeoDataFrame,
    params: Optional[object] = None,
    min_points_pct: float = 0.8,
    elev_trash: float = 20.0,
    too_far_beam: float = 46.0,
    ideal_case: Optional[int] = None,
    return_skipped: bool = True,
    verbose: bool = False,
):
    """
    Apply preprocessing constraints to *already clipped* beams.

    Flags:
      • few_points  → beam has < min_points_pct of family mean points (after clipping)
      • elev_trash  → beam contains any h_li > elev_trash OR h_li < -elev_trash
      • too_far     → beam's median point too far from neighbor (> too_far_beam)

    Priority (highest → lowest):
        elev_trash > few_points > too_far > loaded

    Returns
    -------
    dataset_raw : GeoDataFrame
        Only beams that pass all constraints.

    summary_raw : DataFrame
        Per-family counts: ideal_case, files_found, missing,
        elev_trash, few_points, too_far, loaded.

    flagged_df : DataFrame
        Records for beams flagged 'too_far', including nearest_dist.

    If `params` is provided, values are read from:
        params.MIN_POINTS_PCT
        params.ELEV_TRASH
        params.TOO_FAR_BEAM
        params.IDEAL_CASE
    """
    if params is not None:
        min_points_pct = getattr(params, "MIN_POINTS_PCT", min_points_pct)
        elev_trash     = getattr(params, "ELEV_TRASH", elev_trash)
        too_far_beam   = getattr(params, "TOO_FAR_BEAM", too_far_beam)
        ideal_case     = getattr(params, "IDEAL_CASE", ideal_case)

    if dataset_clipped is None or dataset_clipped.empty:
        empty = gpd.GeoDataFrame(columns=["geometry"], crs=getattr(dataset_clipped, "crs", None))
        if return_skipped:
            return empty, pd.DataFrame(), pd.DataFrame()
        return empty

    gdf = dataset_clipped.copy()

    fam_mean_pts = (
        gdf.groupby(["gt_family", "beam_id"])
           .size()
           .groupby("gt_family")
           .mean()
           .to_dict()
    )

    beams_info: List[dict] = []
    for (fam, bid), bdf in gdf.groupby(["gt_family", "beam_id"]):
        fam_avg = fam_mean_pts.get(fam, np.nan)
        few_points_flag = (
            np.isfinite(fam_avg) and fam_avg > 0 and (len(bdf) / fam_avg < min_points_pct)
        )

        elev_flag = (
            "h_li" in bdf.columns
            and (
                (bdf["h_li"] > elev_trash).any()
                or (bdf["h_li"] < -elev_trash).any()
            )
        )

        beams_info.append(
            {
                "gt_family": fam,
                "beam_id": bid,
                "few_points": bool(few_points_flag),
                "elev_trash": bool(elev_flag),
            }
        )

    beam_flags = pd.DataFrame(beams_info)

    flagged_records: List[dict] = []
    beam_flags["nearest_dist"] = np.nan
    beam_flags["too_far"] = False

    for fam, sub in gdf.groupby("gt_family"):
        med_xy = sub.groupby("beam_id")["geometry"].apply(
            lambda s: Point(float(np.median(s.x)), float(np.median(s.y)))
        )

        if med_xy.shape[0] < 2:
            continue

        coords = np.array([[p.x, p.y] for p in med_xy])
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        dist, _ = nbrs.kneighbors(coords)

        nearest_dist = dist[:, 1]  # second neighbor is the nearest other beam
        too_far_mask = nearest_dist > too_far_beam

        for (beam_id, d, tf) in zip(med_xy.index, nearest_dist, too_far_mask):
            sel = (beam_flags["gt_family"] == fam) & (beam_flags["beam_id"] == beam_id)
            beam_flags.loc[sel, "nearest_dist"] = float(d)
            beam_flags.loc[sel, "too_far"] = bool(tf)

            if tf:
                flagged_records.append(
                    {"gt_family": fam, "beam_id": beam_id, "nearest_dist": float(d)}
                )

    flagged_df = pd.DataFrame(flagged_records)

    def classify(row) -> str:
        if bool(row.get("elev_trash")):
            return "elev_trash"
        if bool(row.get("few_points")):
            return "few_points"
        if bool(row.get("too_far")):
            return "too_far"
        return "loaded"

    beam_flags["status"] = beam_flags.apply(classify, axis=1)

    kept = set(beam_flags.loc[beam_flags["status"] == "loaded", "beam_id"])
    dataset_raw = gdf[gdf["beam_id"].isin(kept)].copy()

    summary = (
        beam_flags.groupby(["gt_family", "status"])
                  .size()
                  .unstack(fill_value=0)
    )

    summary["files_found"] = beam_flags.groupby("gt_family")["beam_id"].nunique()

    if ideal_case is not None:
        summary["ideal_case"] = int(ideal_case)
        summary["missing"] = summary["ideal_case"] - summary["files_found"]
    else:
        summary["ideal_case"] = np.nan
        summary["missing"] = np.nan

    col_order = ["ideal_case", "files_found", "missing", "elev_trash", "few_points", "too_far", "loaded"]
    for col in col_order:
        if col not in summary.columns:
            summary[col] = 0

    summary_raw = summary[col_order].copy()

    if verbose:
        print(f"✅ Kept {len(kept)} beams after preprocessing constraints")
        print(summary_raw)

    return (dataset_raw, summary_raw, flagged_df) if return_skipped else (dataset_raw, summary_raw)


# ============================================================
# Distance along beam (offshore -> inland)
# ============================================================

def compute_distances(
    dataset_clean: Dict[str, Dict[str, object]],
    utm_epsg: int = 32606,
    sort_descending_y: bool = True,
) -> gpd.GeoDataFrame:
    """
    Compute cumulative along-track distance for each beam in each gt_family.

    Expects dataset_clean in the format returned by build_boxes_for_families():
        dataset_clean[fam]["clipped"] -> GeoDataFrame of points

    Behavior:
    - projects to UTM
    - sorts each beam by northing to enforce offshore -> inland ordering (your convention)
    - computes cumulative distance along the beam polyline

    Outputs:
      - point_id
      - distance_from_offshore (cumulative)
      - alongtrack_distance (same)
    """
    rows: List[gpd.GeoDataFrame] = []

    for fam, content in dataset_clean.items():
        clipped = content.get("clipped")
        if clipped is None or getattr(clipped, "empty", True):
            continue

        gdf = clipped.to_crs(utm_epsg).copy()

        for bid, g in gdf.groupby("beam_id"):
            if len(g) < 2:
                continue

            # offshore -> inland sorting (northernmost first) if sort_descending_y=True
            g = (
                g.assign(_y=g.geometry.y)
                 .sort_values("_y", ascending=not sort_descending_y)
                 .drop(columns="_y")
            )

            xy = np.array([(p.x, p.y) for p in g.geometry])
            seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
            dist = np.concatenate(([0.0], np.cumsum(seg)))

            out = g.copy()
            out["point_id"] = np.arange(len(out), dtype=int)

            out["distance_from_offshore"] = dist
            out["alongtrack_distance"] = dist

            if "acq_date" in out.columns:
                out["acq_date"] = pd.to_datetime(out["acq_date"], errors="coerce")

            keep = [
                "gt_family",
                "beam_id",
                "track_id",
                "acq_date",
                "point_id",
                "distance_from_offshore",
                "alongtrack_distance",
                "h_li",
                "geometry",
            ]
            out = out[[c for c in keep if c in out.columns]]

            rows.append(out)

    if not rows:
        return gpd.GeoDataFrame(crs=f"EPSG:{utm_epsg}")

    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), crs=f"EPSG:{utm_epsg}")


__all__ = [
    "load_shoreline",
    "load_beams",
    "extract_date_from_filename",
    "extract_gt_family",
    "extract_beam_id",
    "extract_track_number",
    "get_middle_crossing",
    "build_box_from_centroid",
    "build_boxes_for_families",
    "apply_preprocessing_to_clipped",
    "compute_distances",
]