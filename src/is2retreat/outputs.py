# ============================================================
# Shared output tables, written safely by many track runs
# ============================================================
"""
All tracks append to the same CSV tables. Each save:

    1. takes a file lock (<table>.csv.lock) so parallel runs can't
       overwrite each other,
    2. re-reads the table from disk inside the lock,
    3. appends this run's rows,
    4. drops duplicates on the table's key, keeping the newest row
       (re-running a track replaces its old rows),
    5. drops legacy columns (ClusterSize, angle_deg) and writes the table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

KEY_DSAS_SUMMARY = ["track_id", "bias_tolerance", "gt_family", "cluster_id"]
KEY_DSAS_INTERVAL = ["track_id", "bias_tolerance", "gt_family", "cluster_id", "interval_order"]
KEY_DSAS_BEAM_ANGLE = ["track_id", "bias_tolerance", "gt_family", "cluster_id", "Acq_date", "beam_id"]

KEY_GIE_SUMMARY = ["track_id", "bias_tolerance", "gt_family", "cluster_id"]
KEY_GIE_INTERVAL = ["track_id", "bias_tolerance", "gt_family", "cluster_id", "interval_order"]
KEY_GIE_BEAMS = ["track_id", "bias_tolerance", "gt_family", "cluster_id", "beam_id", "acq_date"]


@dataclass(frozen=True)
class OutputFiles:
    dsas_summary: Path
    dsas_interval: Path
    dsas_beam_angle: Path
    gie_summary: Path
    gie_interval: Path
    gie_beams: Path

    @classmethod
    def in_dir(cls, outdir, res_tag: str) -> "OutputFiles":
        outdir = Path(outdir)
        return cls(
            dsas_summary=outdir / f"DSAS_{res_tag}.csv",
            dsas_interval=outdir / f"DSAS_Intervals_{res_tag}.csv",
            dsas_beam_angle=outdir / f"DSAS_BeamAngles_{res_tag}.csv",
            gie_summary=outdir / f"DSAS_GIE_AllBiasTol_{res_tag}.csv",
            gie_interval=outdir / f"DSAS_GIE_AllBiasTol_Intervals_{res_tag}.csv",
            gie_beams=outdir / f"DSAS_GIE_AllBiasTol_BeamDetails_{res_tag}.csv",
        )


def _lock(path) -> FileLock:
    return FileLock(str(path) + ".lock")


def read_csv_locked(path) -> pd.DataFrame:
    path = Path(path)
    with _lock(path):
        return pd.read_csv(path, dtype={"track_id": str}) if path.exists() else pd.DataFrame()


# ======================================================================
# DSAS tables
# ======================================================================
def _drop_dsas_legacy_columns(df):
    if df is None:
        return pd.DataFrame()

    df = df.copy()
    return df.drop(
        columns=[
            c for c in df.columns
            if c.lower() == "clustersize" or c == "angle_deg"
        ],
        errors="ignore",
    )


def _read_dsas_existing(path, needed_cols):
    if path.exists():
        df = pd.read_csv(path, dtype={"track_id": str})
    else:
        df = pd.DataFrame()

    df = _drop_dsas_legacy_columns(df)

    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


def _parse_date_keys(df, keys):
    """
    Dates read back from CSV are text while new rows hold Timestamps; parse
    date key columns so drop_duplicates can match them. (The notebook skipped
    this, so re-running a track duplicated its DSAS_BeamAngles rows.)
    """
    df = df.copy()
    for col in keys:
        if "date" in col.lower() and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed").dt.normalize()
    return df


def save_dsas_table(path, new_df, keys):
    """Merge new DSAS rows into the table at path (locked). Returns the full table."""
    path = Path(path)

    with _lock(path):
        existing = _parse_date_keys(_read_dsas_existing(path, keys), keys)
        new_df = _parse_date_keys(_drop_dsas_legacy_columns(new_df), keys)

        if existing.empty:
            combined = new_df.copy()
        elif new_df.empty:
            combined = existing.copy()
        else:
            combined = pd.concat([existing, new_df], ignore_index=True, sort=False)

        for col in keys:
            if col not in combined.columns:
                combined[col] = np.nan

        combined = combined.drop_duplicates(subset=keys, keep="last")
        combined = _drop_dsas_legacy_columns(combined)

        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)

    return combined


# ======================================================================
# GIE tables (keys are normalized before de-duplication)
# ======================================================================
def _norm_track_id(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    try:
        text = str(int(float(text)))
    except Exception:
        pass

    return text.zfill(4)


def norm_key_columns(df):
    """Normalize key columns so rows from CSV and from memory compare equal."""
    df = df.copy()

    if df.empty:
        return df

    if "Acq_date" in df.columns and "acq_date" not in df.columns:
        df = df.rename(columns={"Acq_date": "acq_date"})
    elif "Acq_date" in df.columns and "acq_date" in df.columns:
        df["acq_date"] = df["acq_date"].combine_first(df["Acq_date"])
        df = df.drop(columns=["Acq_date"])

    for col in list(df.columns):
        if col.lower() == "clustersize" or col == "angle_deg":
            df = df.drop(columns=[col])

    if "track_id" in df.columns:
        df["track_id"] = df["track_id"].map(_norm_track_id)

    if "bias_tolerance" in df.columns:
        df["bias_tolerance"] = pd.to_numeric(df["bias_tolerance"], errors="coerce").round(6)

    if "gt_family" in df.columns:
        df["gt_family"] = df["gt_family"].astype(str).str.strip().str.lower()

    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")

    if "beam_id" in df.columns:
        df["beam_id"] = df["beam_id"].astype(str).str.strip()

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()

    if "interval_order" in df.columns:
        df["interval_order"] = pd.to_numeric(df["interval_order"], errors="coerce")

    return df


def _read_existing_normalized(path, needed_cols):
    if path.exists():
        df = pd.read_csv(path, dtype={"track_id": str})
        df = norm_key_columns(df)
    else:
        df = pd.DataFrame()

    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


def save_gie_table(path, new_df, keys):
    """Merge new GIE rows into the table at path (locked). Returns the full table."""
    path = Path(path)

    with _lock(path):
        fresh_existing = _read_existing_normalized(path, keys)
        parts = []

        if fresh_existing is not None and not fresh_existing.empty:
            parts.append(norm_key_columns(fresh_existing))

        if new_df is not None and not new_df.empty:
            parts.append(norm_key_columns(new_df))

        if parts:
            out = pd.concat(parts, ignore_index=True, sort=False)
        else:
            out = pd.DataFrame(columns=keys)

        for col in keys:
            if col not in out.columns:
                out[col] = np.nan

        out = out.drop_duplicates(subset=keys, keep="last")

        out = out.drop(
            columns=[
                c for c in out.columns
                if c.lower() == "clustersize" or c == "angle_deg"
            ],
            errors="ignore",
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)

    return out
