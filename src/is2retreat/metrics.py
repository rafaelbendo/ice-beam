# ============================================================
# DSAS-style shoreline change metrics
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compute_cluster_statistics(
    bluff_df,
    confidence=0.95,
    min_span_days=365,
    positional_uncertainty_m=4.8,
):
    """
    DSAS-style shoreline change metrics from bluff positions.

    DSAS conventions:
        • x increases landward (retreat = negative)
        • NSM = first_x - last_x
        • SCE = max_x - min_x
        • EPR = NSM / Δt_years   (requires span >= min_span_days)
        • Regression metrics require >= 3 unique dates AND span >= min_span_days

    Returns
    -------
    dict with NSM, SCE, EPR, LRR, LR2, LSE, LCI, TemporalSpan_days,
    ClusterTemporalSpanYears, ValidRegression, U_position_m, U_NSM_m, U_EPR_myr
    """

    def _empty_stats():
        return {
            "NSM": np.nan, "SCE": np.nan, "EPR": np.nan,
            "LRR": np.nan, "LR2": np.nan, "LSE": np.nan, "LCI": np.nan,
            "TemporalSpan_days": np.nan,
            "ClusterTemporalSpanYears": np.nan,
            "ValidRegression": False,
            "U_position_m": positional_uncertainty_m,
            "U_NSM_m": np.nan,
            "U_EPR_myr": np.nan,
        }

    def _add_uncertainty(stats):
        span_days = stats.get("TemporalSpan_days", np.nan)
        span_years = span_days / 365.25 if pd.notna(span_days) and span_days > 0 else np.nan

        u_nsm = np.sqrt(positional_uncertainty_m**2 + positional_uncertainty_m**2)
        u_epr = (
            u_nsm / span_years
            if pd.notna(stats.get("EPR", np.nan)) and pd.notna(span_years) and span_years >= 1
            else np.nan
        )

        stats["ClusterTemporalSpanYears"] = (
            round(span_years, 2) if pd.notna(span_years) else np.nan
        )
        stats["U_position_m"] = positional_uncertainty_m
        stats["U_NSM_m"] = round(float(u_nsm), 2)
        stats["U_EPR_myr"] = round(float(u_epr), 2) if np.isfinite(u_epr) else np.nan

        return stats

    if bluff_df is None or bluff_df.empty:
        return _empty_stats()

    df = (
        bluff_df[["acq_date", "bluff_x"]]
        .dropna(subset=["acq_date", "bluff_x"])
        .copy()
    )

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["bluff_x"] = pd.to_numeric(df["bluff_x"], errors="coerce")
    df = df.dropna()

    if df.empty:
        return _empty_stats()

    df = df.groupby("acq_date", as_index=False).agg(bluff_x=("bluff_x", "mean"))
    df = df.sort_values("acq_date")

    if len(df) < 2:
        return _empty_stats()

    x_first = df["bluff_x"].iloc[0]
    x_last = df["bluff_x"].iloc[-1]

    NSM = float(x_first - x_last)
    SCE = float(df["bluff_x"].max() - df["bluff_x"].min())

    span_days = int((df["acq_date"].iloc[-1] - df["acq_date"].iloc[0]).days)
    t_years = span_days / 365.25 if span_days > 0 else np.nan

    EPR = NSM / t_years if span_days >= min_span_days and np.isfinite(t_years) and t_years > 0 else np.nan

    LRR = LR2 = LSE = LCI = np.nan

    if len(df) >= 3 and span_days >= min_span_days:
        years = df["acq_date"].map(lambda d: d.year + d.dayofyear / 365.25).to_numpy()
        xvals = df["bluff_x"].to_numpy()

        model = LinearRegression().fit(years.reshape(-1, 1), xvals)
        pred = model.predict(years.reshape(-1, 1))

        slope = float(model.coef_[0])
        R2 = float(r2_score(xvals, pred))

        resid = xvals - pred
        dof = len(xvals) - 2
        S_yx = np.sqrt(np.sum(resid**2) / dof) if dof > 0 else np.nan

        Sxx = np.sum((years - years.mean())**2)
        se_slope = S_yx / np.sqrt(Sxx) if Sxx > 0 else np.nan

        if np.isfinite(se_slope) and dof > 0:
            tcrit = student_t.ppf(1 - (1 - confidence) / 2, df=dof)
            LCI = float(tcrit * se_slope)

        LRR = round(-slope, 2)
        LR2 = round(R2, 2)
        LSE = round(S_yx, 2)
        LCI = round(LCI, 2) if np.isfinite(LCI) else np.nan

    result = {
        "NSM": int(round(NSM)) if np.isfinite(NSM) else np.nan,
        "SCE": int(round(SCE)) if np.isfinite(SCE) else np.nan,
        "EPR": float(round(EPR, 2)) if np.isfinite(EPR) else np.nan,
        "LRR": LRR,
        "LR2": LR2,
        "LSE": LSE,
        "LCI": LCI,
        "TemporalSpan_days": span_days,
        "ValidRegression": np.isfinite(LRR) and len(df) >= 3 and span_days >= min_span_days,
    }

    return _add_uncertainty(result)


def compute_cluster_intervals(
        bluff_df,
        gt_family,
        cluster_id,
        track_id,
        bias_tolerance,
        cluster_width_m=np.nan,
        n_beams=np.nan
    ):
    """
    Inter-date intervals of bluff position for one cluster (same-day
    observations are averaged first). One dict per consecutive date pair.
    """
    if bluff_df is None or bluff_df.empty:
        return []

    df = bluff_df.copy()

    keep_cols = [
        c for c in ["beam_id", "acq_date", "bluff_x", "bluff_y", "ref_line"]
        if c in df.columns
    ]
    df = df[keep_cols].copy()

    df["beam_id"] = df["beam_id"].astype(str)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["acq_date_norm"] = df["acq_date"].dt.normalize()
    df["bluff_x"] = pd.to_numeric(df["bluff_x"], errors="coerce")

    if "bluff_y" in df.columns:
        df["bluff_y"] = pd.to_numeric(df["bluff_y"], errors="coerce")
    else:
        df["bluff_y"] = np.nan

    if "ref_line" in df.columns:
        df["ref_line"] = pd.to_numeric(df["ref_line"], errors="coerce")

    df = df.dropna(subset=["acq_date_norm", "bluff_x"]).copy()
    if df.empty:
        return []

    agg_dict = {
        "bluff_x": ("bluff_x", "mean"),
        "bluff_y": ("bluff_y", "mean"),
        "beam_id": ("beam_id", lambda s: "|".join(sorted(set(map(str, s)))))
    }

    if "ref_line" in df.columns:
        agg_dict["ref_line"] = ("ref_line", "mean")

    df_day = (
        df.groupby("acq_date_norm", as_index=False)
          .agg(**agg_dict)
          .rename(columns={"acq_date_norm": "acq_date"})
          .sort_values("acq_date")
          .reset_index(drop=True)
    )

    if len(df_day) < 2:
        return []

    rows = []
    cumulative_days = 0
    cumulative_abs_retreat = 0.0

    for i in range(len(df_day) - 1):
        row_a = df_day.iloc[i]
        row_b = df_day.iloc[i + 1]

        date_from = pd.to_datetime(row_a["acq_date"])
        date_to = pd.to_datetime(row_b["acq_date"])

        x_from = float(row_a["bluff_x"])
        x_to = float(row_b["bluff_x"])

        delta_days = int((date_to - date_from).days)
        delta_x = float(x_to - x_from)

        interval_retreat_signed = float(-delta_x)
        interval_retreat_abs = float(abs(interval_retreat_signed))

        rows.append({
            "track_id": track_id,
            "bias_tolerance": float(bias_tolerance),
            "gt_family": gt_family,
            "cluster_id": int(cluster_id),

            "cluster_width_m": round(cluster_width_m, 2) if np.isfinite(cluster_width_m) else np.nan,
            "n_beams": int(n_beams) if pd.notna(n_beams) else np.nan,

            "interval_order": int(i + 1),
            "beam_from": row_a["beam_id"],
            "beam_to": row_b["beam_id"],

            "date_from": date_from.normalize(),
            "date_to": date_to.normalize(),

            "bluff_x_from": int(round(x_from)) if np.isfinite(x_from) else np.nan,
            "bluff_x_to": int(round(x_to)) if np.isfinite(x_to) else np.nan,
            "bluff_y_from": int(round(row_a["bluff_y"])) if pd.notna(row_a["bluff_y"]) else np.nan,
            "bluff_y_to": int(round(row_b["bluff_y"])) if pd.notna(row_b["bluff_y"]) else np.nan,

            "delta_days": int(delta_days),
            "delta_x": int(round(delta_x)),

            "interval_retreat_signed": int(round(interval_retreat_signed)),
            "interval_retreat_abs": int(round(interval_retreat_abs)),

            "start_days": int(cumulative_days),
            "width_days": int(delta_days),
            "start_retreat": int(round(cumulative_abs_retreat)),
            "width_retreat": int(round(interval_retreat_abs)),

            "direction": (
                "retreat" if interval_retreat_signed < 0
                else "advance" if interval_retreat_signed > 0
                else "stable"
            )
        })

        cumulative_days += delta_days
        cumulative_abs_retreat += interval_retreat_abs

    return rows
