# ============================================================
# Per-track inputs: UTM zone, bluff shoreline, ATL06-like beams
# ============================================================
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from .config import Params, Paths
from .sliderule_io import load_sliderule_atl06like, standardize_sliderule_gdf
from .utils import TrackSkipped


def resolve_utm_epsg(track_id: str, paths: Paths, params: Params, verbose=True) -> int:
    """
    UTM zone for a track, from the centroid of its RGT line inside the AOI.

    The North Slope spans UTM zones 3N-7N, so the zone is derived per track
    instead of hardcoded. Raises TrackSkipped if it can't be resolved.
    """
    original_crs = params.ORIGINAL_CRS

    is2_tracks_gdf = gpd.read_file(paths.is2_tracks_path)
    if is2_tracks_gdf.crs is None:
        raise TrackSkipped(f"TRACK_ID={track_id}: IS2 tracks file has no CRS ({paths.is2_tracks_path}).")
    if str(is2_tracks_gdf.crs) != original_crs:
        is2_tracks_gdf = is2_tracks_gdf.to_crs(original_crs)

    track_rgt_rows = is2_tracks_gdf.loc[is2_tracks_gdf["Name"].astype(int) == int(track_id)]
    if track_rgt_rows.empty:
        raise TrackSkipped(
            f"TRACK_ID={track_id}: RGT not found in IS2 tracks file (field 'Name'): {paths.is2_tracks_path}"
        )

    aoi_gdf = gpd.read_file(paths.aoi_path)
    if aoi_gdf.crs is None:
        raise TrackSkipped(f"TRACK_ID={track_id}: AOI file has no CRS ({paths.aoi_path}).")
    if str(aoi_gdf.crs) != original_crs:
        aoi_gdf = aoi_gdf.to_crs(original_crs)

    # RGT lines run the full orbital pass, so intersect with the AOI first to
    # isolate just the segment over the study area before estimating the zone.
    track_rgt_segment = gpd.overlay(
        track_rgt_rows[["geometry"]], aoi_gdf[["geometry"]], how="intersection"
    )
    if track_rgt_segment.empty:
        raise TrackSkipped(
            f"TRACK_ID={track_id}: RGT line does not intersect the AOI polygon.\n"
            f"      RGT source: {paths.is2_tracks_path}\n"
            f"      AOI source: {paths.aoi_path}"
        )

    centroid = track_rgt_segment.union_all().centroid
    centroid_gdf = gpd.GeoDataFrame(geometry=[centroid], crs=original_crs)
    utm_epsg = int(centroid_gdf.estimate_utm_crs().to_epsg())

    if verbose:
        print(
            f"TRACK_ID={track_id}: centroid (lon, lat) = "
            f"({centroid.x:.6f}, {centroid.y:.6f}) -> UTM_EPSG={utm_epsg}"
        )

    return utm_epsg


def load_bluff_shoreline(shoreline_path) -> gpd.GeoDataFrame:
    """Shoreline features with CoastType == 1 (bluff coast), in their own CRS."""
    shoreline_path = Path(shoreline_path)
    if not shoreline_path.exists():
        raise FileNotFoundError(f"Shoreline file not found:\n{shoreline_path}")

    shoreline_gdf = gpd.read_file(shoreline_path)

    if "CoastType" not in shoreline_gdf.columns:
        raise ValueError("Shoreline file is missing required field: CoastType")

    shoreline_gdf = shoreline_gdf.loc[shoreline_gdf["CoastType"] == 1].copy()

    if shoreline_gdf.crs is None:
        print("Warning: shoreline file has no CRS. Assigning WGS84 (EPSG:4326).")
        shoreline_gdf = shoreline_gdf.set_crs("EPSG:4326")

    return shoreline_gdf


def load_track_beams(
        track_id: str,
        paths: Paths,
        params: Params,
        utm_epsg: int,
        source: str = "auto",
        save_cache: bool = True,
        verbose: bool = True,
    ) -> gpd.GeoDataFrame:
    """
    Standardized ATL06-like points for one track, in UTM.

    source:
        "auto"      use the cached GeoPackage if it exists, otherwise SlideRule
        "cache"     cached GeoPackage only
        "sliderule" always request from SlideRule (the notebook's behavior)

    Fresh SlideRule pulls are written to the cache when save_cache is True.
    Raises TrackSkipped when the track has no usable beam pairs.
    """
    if source not in {"auto", "cache", "sliderule"}:
        raise ValueError(f"source must be 'auto', 'cache' or 'sliderule', not {source!r}")

    cache_file = paths.sliderule_cache_file(track_id, params.RES_TAG)
    layer = f"sliderule_ATL06_{params.RES_TAG}"
    from_cache = source == "cache" or (source == "auto" and cache_file.exists())

    if from_cache:
        if not cache_file.exists():
            raise TrackSkipped(f"TRACK_ID={track_id}: no cached SlideRule file at {cache_file}.")
        beams = gpd.read_file(cache_file, layer=layer)
        if beams.crs is not None and beams.crs.to_epsg() != int(utm_epsg):
            beams = beams.to_crs(utm_epsg)
        if verbose:
            print(f"[SR] Loaded cached SlideRule beams: {cache_file}")
    else:
        sliderule_raw = load_sliderule_atl06like(
            aoi_path=paths.aoi_path,
            rgt=track_id,
            date_start=params.SLIDERULE_DATE_START,
            date_end=params.SLIDERULE_DATE_END,
            cycle=params.SLIDERULE_CYCLE,
            segment_length_m=params.SLIDERULE_SEGMENT_LENGTH_M,
            segment_resolution_m=params.SLIDERULE_SEGMENT_RESOLUTION_M,
            sliderule_url=params.SLIDERULE_URL,
            original_crs=params.ORIGINAL_CRS,
            verbose=verbose,
        )
        beams = standardize_sliderule_gdf(
            sliderule_gdf=sliderule_raw,
            track_id=track_id,
            original_crs=params.ORIGINAL_CRS,
            utm_epsg=utm_epsg,
            aoi_path=paths.aoi_path,
            rgt_filter=track_id,
            date_start=params.SLIDERULE_DATE_START,
            date_end=params.SLIDERULE_DATE_END,
            cycle_filter=params.SLIDERULE_CYCLE,
            verbose=verbose,
        )

    if beams.empty:
        raise TrackSkipped(f"SlideRule returned no usable beams for TRACK_ID={track_id}.")

    # Stop unless at least one gt_family has a pair of beams
    beam_counts_by_family = beams.groupby("gt_family")["beam_id"].nunique()
    if not (beam_counts_by_family >= 2).any():
        raise TrackSkipped(
            "no gt_family has at least two unique beams in the SlideRule result.\n"
            f"Beam counts by gt_family:\n{beam_counts_by_family.to_string()}"
        )

    if not from_cache and save_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        beams.to_file(cache_file, layer=layer, driver="GPKG")
        if verbose:
            print(f"[SR] Saved standardized SlideRule output -> {cache_file}")

    return beams
