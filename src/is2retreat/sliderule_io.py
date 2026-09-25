# ============================================================
# SlideRule ingestion: request ATL06-like segments and standardize them
# to the columns ICE-BEAM expects.
# ============================================================
from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

try:
    from sliderule import sliderule, icesat2
except ImportError:
    sliderule = None
    icesat2 = None


def _normalize_gt_name(value):
    """Return gt1l/gt1r/... when available, otherwise None."""
    if pd.isna(value):
        return None

    gt_map = {
        10: "gt1l", 20: "gt1r",
        30: "gt2l", 40: "gt2r",
        50: "gt3l", 60: "gt3r",
    }

    if isinstance(value, str):
        v = value.strip().lower()
        if v.startswith("gt"):
            return v
        try:
            value = int(float(v))
        except Exception:
            return None

    try:
        return gt_map.get(int(value))
    except Exception:
        return None


def _gt_to_family(value):
    gt_name = _normalize_gt_name(value)
    if gt_name is None:
        return None
    return gt_name[:3]


def _pick_first_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_sliderule_time(values, field_name=None):
    """Coerce SlideRule time fields into timezone-naive pandas datetimes."""
    ser = pd.Series(values)
    num = pd.to_numeric(ser, errors="coerce")

    def _strip_tz(dt):
        if getattr(dt.dt, "tz", None) is not None:
            return dt.dt.tz_convert(None)
        return dt

    def _plausible(dt):
        years = dt.dropna().dt.year
        if years.empty:
            return False
        return (years.between(2018, 2035)).mean() >= 0.5

    if num.notna().sum() > 0:
        magnitude = float(num.dropna().abs().median())

        if field_name == "time_ns" or magnitude > 1e17:
            dt = pd.to_datetime(num, unit="ns", origin="unix", errors="coerce", utc=True)
            return _strip_tz(dt)

        if field_name == "delta_time":
            dt = pd.to_datetime(num, unit="s", origin=pd.Timestamp("2018-01-01"), errors="coerce", utc=True)
            return _strip_tz(dt)

        dt_unix = pd.to_datetime(num, unit="s", origin="unix", errors="coerce", utc=True)
        if _plausible(dt_unix):
            return _strip_tz(dt_unix)

        dt_gps = pd.to_datetime(num, unit="s", origin=pd.Timestamp("1980-01-06"), errors="coerce", utc=True)
        if _plausible(dt_gps):
            return _strip_tz(dt_gps)

        return _strip_tz(dt_unix)

    dt = pd.to_datetime(ser, errors="coerce", utc=True)
    return _strip_tz(dt)


def aoi_to_sliderule_region(aoi_path, verbose=True):
    """
    Load the AOI polygon, reproject it to EPSG:4326, and convert the
    exterior ring to SlideRule's required [{"lon": ..., "lat": ...}, ...] format.
    """
    aoi_path = Path(aoi_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found:\n{aoi_path}")

    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.empty:
        raise ValueError(f"AOI file is empty:\n{aoi_path}")
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file has no CRS:\n{aoi_path}")

    aoi_ll = aoi_gdf.to_crs(4326)
    geom = unary_union(aoi_ll.geometry)

    # Handle AOIs that arrive as MultiPolygon or GeometryCollection and
    # keep the largest polygon part for the SlideRule request polygon.
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
        if not polys:
            raise ValueError("AOI geometry collection does not contain any polygons.")
        geom = unary_union(polys)

    if geom.geom_type == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        raise ValueError(f"AOI must resolve to a polygon, not {geom.geom_type}.")

    geom = orient(geom, sign=1.0)
    coords = list(geom.exterior.coords)

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Some shapefiles store Z coordinates, so read only lon/lat for SlideRule.
    region = [{"lon": float(coord[0]), "lat": float(coord[1])} for coord in coords]

    if verbose:
        print(f"[AOI] Loaded {aoi_path.name} with {len(region)} exterior vertices for SlideRule")

    return region, aoi_ll


def load_sliderule_atl06like(
        aoi_path,
        rgt=None,
        date_start=None,
        date_end=None,
        cycle=None,
        segment_length_m=40.0,
        segment_resolution_m=20.0,
        sliderule_url="slideruleearth.io",
        original_crs="EPSG:4326",
        verbose=True):
    """
    Request ATL06-like elevation segments from ATL03 using SlideRule.
    """
    if sliderule is None or icesat2 is None:
        raise ImportError(
            "SlideRule is not installed in this environment. Install the 'sliderule' Python package."
        )

    region, _ = aoi_to_sliderule_region(aoi_path, verbose=verbose)

    parms = {
        "poly": region,
        "srt": icesat2.SRT_LAND,
        "cnf": icesat2.CNF_SURFACE_HIGH,
        "ats": 10.0,
        "cnt": 10,
        "len": float(segment_length_m),
        "res": float(segment_resolution_m),
        "fit": {}
    }

    if rgt is not None:
        parms["rgt"] = int(rgt)
    if cycle is not None:
        parms["cycle"] = int(cycle)
    if date_start:
        parms["t0"] = pd.Timestamp(date_start).strftime("%Y-%m-%dT00:00:00Z")
    if date_end:
        parms["t1"] = pd.Timestamp(date_end).strftime("%Y-%m-%dT23:59:59Z")

    if verbose:
        print("[SR] Initializing SlideRule client...")
    icesat2.init(sliderule_url, verbose=False)

    if verbose:
        print("[SR] Requesting ATL06-like segments from ATL03 via atl03x...")
        print(f"     len={parms['len']} m | res={parms['res']} m | rgt={parms.get('rgt', 'all')} | cycle={parms.get('cycle', 'all')}")
        print(f"     t0={parms.get('t0', 'all')} | t1={parms.get('t1', 'all')}")

    gdf = sliderule.run("atl03x", parms)

    if gdf is None or len(gdf) == 0:
        if verbose:
            print("SKIP: SlideRule returned no ATL06-like segments for this request.")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=original_crs)

    sr_gdf = gpd.GeoDataFrame(gdf.copy(), geometry="geometry", crs=getattr(gdf, "crs", original_crs))

    if verbose:
        print(f"[SR] Received {len(sr_gdf):,} ATL06-like segment rows")

    return sr_gdf


def standardize_sliderule_gdf(
        sliderule_gdf,
        track_id,
        original_crs,
        utm_epsg,
        aoi_path,
        rgt_filter=None,
        date_start=None,
        date_end=None,
        cycle_filter=None,
        verbose=True):
    """
    Rename/create the columns ICE-BEAM expects from the ATL06 shapefile workflow:
        gt_family, track_id, acq_date, year, beam_id, file_path, geometry (UTM), h_li
    """
    if sliderule_gdf is None or len(sliderule_gdf) == 0:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{utm_epsg}")

    if "geometry" in sliderule_gdf.columns:
        gdf = gpd.GeoDataFrame(
            sliderule_gdf.copy(),
            geometry="geometry",
            crs=getattr(sliderule_gdf, "crs", original_crs)
        )
    else:
        lon_col = _pick_first_column(sliderule_gdf, ["longitude", "lon", "x"])
        lat_col = _pick_first_column(sliderule_gdf, ["latitude", "lat", "y"])
        if lon_col is None or lat_col is None:
            raise ValueError("SlideRule result must include geometry or longitude/latitude columns.")
        gdf = gpd.GeoDataFrame(
            sliderule_gdf.copy(),
            geometry=gpd.points_from_xy(sliderule_gdf[lon_col], sliderule_gdf[lat_col], crs=original_crs),
            crs=original_crs
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(original_crs, allow_override=True)
    elif str(gdf.crs) != original_crs:
        gdf = gdf.to_crs(original_crs)

    gdf["longitude"] = gdf.geometry.x
    gdf["latitude"] = gdf.geometry.y

    elev_col = _pick_first_column(gdf, ["h_li", "h_mean", "h_te_median", "height", "elevation", "z"])
    if elev_col is None:
        raise ValueError(
            "SlideRule result is missing an elevation field. Expected one of: "
            "h_li, h_mean, h_te_median, height, elevation, z."
        )
    gdf["h_li"] = pd.to_numeric(gdf[elev_col], errors="coerce")

    time_col = _pick_first_column(gdf, ["acq_date", "time_ns", "time", "datetime", "delta_time"])
    if time_col is not None:
        gdf["acq_date"] = _coerce_sliderule_time(gdf[time_col], field_name=time_col)
    elif getattr(gdf.index, "name", None) in {"time_ns", "time"}:
        gdf["acq_date"] = _coerce_sliderule_time(gdf.index.to_series(), field_name=gdf.index.name)
    else:
        gdf["acq_date"] = pd.NaT

    gt_col = _pick_first_column(gdf, ["gt", "beam", "beam_name"])
    if gt_col is not None:
        gdf["gt_name"] = gdf[gt_col].apply(_normalize_gt_name)
    else:
        gdf["gt_name"] = None

    gdf["gt_family"] = gdf["gt_name"].apply(_gt_to_family)
    if gdf["gt_family"].isna().all():
        raise ValueError(
            "SlideRule result could not be mapped to ICE-BEAM gt_family values (gt1/gt2/gt3)."
        )

    if "rgt" in gdf.columns:
        gdf["rgt"] = pd.to_numeric(gdf["rgt"], errors="coerce")
    else:
        gdf["rgt"] = pd.to_numeric(pd.Series([rgt_filter] * len(gdf)), errors="coerce")

    if "cycle" in gdf.columns:
        gdf["cycle"] = pd.to_numeric(gdf["cycle"], errors="coerce")
    else:
        gdf["cycle"] = pd.to_numeric(pd.Series([cycle_filter] * len(gdf)), errors="coerce")

    if rgt_filter is not None and "rgt" in gdf.columns:
        gdf = gdf[gdf["rgt"] == int(rgt_filter)].copy()

    if date_start:
        gdf = gdf[gdf["acq_date"] >= pd.Timestamp(date_start)].copy()
    if date_end:
        gdf = gdf[gdf["acq_date"] <= pd.Timestamp(date_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)].copy()

    if cycle_filter is not None:
        if "cycle" in gdf.columns:
            gdf = gdf[gdf["cycle"] == int(cycle_filter)].copy()
        elif verbose:
            print("[SR] Cycle filter requested, but no cycle column was returned by SlideRule.")

    track_fallback = f"{int(track_id):04d}" if str(track_id).isdigit() else str(track_id)
    gdf["track_id"] = gdf["rgt"].apply(
        lambda v: f"{int(v):04d}" if pd.notna(v) else track_fallback
    )

    gdf["year"] = pd.to_datetime(gdf["acq_date"], errors="coerce").dt.year

    def make_beam_id(row):
        pieces = ["sliderule", str(row["track_id"])]

        if pd.notna(row.get("gt_name", None)):
            pieces.append(str(row["gt_name"]))
        elif pd.notna(row.get("spot", None)):
            pieces.append(f"spot{int(row['spot'])}")
        else:
            pieces.append(str(row["gt_family"]))

        if pd.notna(row.get("cycle", None)):
            pieces.append(f"cyc{int(row['cycle']):02d}")

        acq = row.get("acq_date", pd.NaT)
        if pd.notna(acq):
            pieces.append(pd.Timestamp(acq).strftime("%Y%m%d"))

        return "_".join(pieces)

    gdf["beam_id"] = gdf.apply(make_beam_id, axis=1)
    gdf["file_path"] = f"sliderule:atl03x:{Path(aoi_path).name}"

    # Keep lon/lat columns in WGS84 like the shapefile workflow, but use UTM geometry downstream.
    gdf = gdf.to_crs(utm_epsg)

    standardized = gpd.GeoDataFrame(gdf, geometry="geometry", crs=f"EPSG:{utm_epsg}")

    if verbose:
        nb = standardized["beam_id"].nunique()
        print(f"[RAW] Standardized SlideRule beams: {nb} unique beam profiles")
        print(f"     Total points: {len(standardized):,}")

    return standardized
