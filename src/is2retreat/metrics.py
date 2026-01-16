from __future__ import annotations

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _require_param(params: object, name: str):
    if params is None:
        raise ValueError("params is required. Pass Params() from is2retreat.config.")
    if not hasattr(params, name):
        raise ValueError(f"params must define `{name}`.")
    return getattr(params, name)


def compute_cluster_statistics(
    bluff_df: pd.DataFrame,
    params: object,
) -> dict:
    """
    Compute DSAS-style change metrics from bluff positions.

    Conventions used here:
      - x increases landward
      - retreat is negative (DSAS-like sign)
      - NSM = first_x - last_x  (negative if x moved landward over time)

    Required params:
      - CONFIDENCE    (float) e.g., 0.95
      - MIN_SPAN_DAYS (int)   e.g., 365

    Returns dict with:
      NSM, SCE, EPR, LRR, LR2, LSE, LCI,
      TemporalSpan_days, ValidRegression
    """
    confidence = float(_require_param(params, "CONFIDENCE"))
    min_span_days = int(_require_param(params, "MIN_SPAN_DAYS"))

    empty_out = {
        "NSM": np.nan, "SCE": np.nan, "EPR": np.nan,
        "LRR": np.nan, "LR2": np.nan, "LSE": np.nan, "LCI": np.nan,
        "TemporalSpan_days": np.nan, "ValidRegression": False
    }

    if bluff_df is None or len(bluff_df) == 0:
        return empty_out

    df = bluff_df[["acq_date", "bluff_x"]].copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["bluff_x"] = pd.to_numeric(df["bluff_x"], errors="coerce")
    df = df.dropna(subset=["acq_date", "bluff_x"])

    if df.empty:
        return empty_out

    # Collapse duplicates within same timestamp (or day if your acq_date is normalized)
    df = df.groupby("acq_date", as_index=False).agg(bluff_x=("bluff_x", "mean"))
    df = df.sort_values("acq_date")

    if len(df) < 2:
        return empty_out

    # Endpoints + envelope
    x_first = float(df["bluff_x"].iloc[0])
    x_last = float(df["bluff_x"].iloc[-1])

    NSM = x_first - x_last
    SCE = float(df["bluff_x"].min() - df["bluff_x"].max())

    span_days = int((df["acq_date"].iloc[-1] - df["acq_date"].iloc[0]).days)
    t_years = span_days / 365.25 if span_days > 0 else np.nan

    # EPR only if span is long enough
    if span_days >= min_span_days and np.isfinite(t_years) and t_years > 0:
        EPR = NSM / t_years
    else:
        EPR = np.nan

    # Regression metrics
    LRR = LR2 = LSE = LCI = np.nan
    valid_reg = False

    if _HAS_SKLEARN and len(df) >= 3 and span_days >= min_span_days:
        years = df["acq_date"].map(lambda d: d.year + d.dayofyear / 365.25).to_numpy()
        xvals = df["bluff_x"].to_numpy()

        model = LinearRegression().fit(years.reshape(-1, 1), xvals)
        pred = model.predict(years.reshape(-1, 1))

        slope = float(model.coef_[0])  # + means landward movement per year
        R2 = float(r2_score(xvals, pred))

        resid = xvals - pred
        dof = len(xvals) - 2
        S_yx = np.sqrt(np.sum(resid ** 2) / dof) if dof > 0 else np.nan

        Sxx = np.sum((years - years.mean()) ** 2)
        se_slope = (S_yx / np.sqrt(Sxx)) if (Sxx > 0 and np.isfinite(S_yx)) else np.nan

        if _HAS_SCIPY and np.isfinite(se_slope) and dof > 0:
            tcrit = student_t.ppf(1 - (1 - confidence) / 2, df=dof)
            LCI = float(tcrit * se_slope)

        # DSAS-like sign convention: retreat negative
        LRR = -slope
        LR2 = R2
        LSE = S_yx

        valid_reg = np.isfinite(LRR)

    return {
        "NSM": float(round(NSM, 2)) if np.isfinite(NSM) else np.nan,
        "SCE": float(round(SCE, 2)) if np.isfinite(SCE) else np.nan,
        "EPR": float(round(EPR, 2)) if np.isfinite(EPR) else np.nan,
        "LRR": float(round(LRR, 2)) if np.isfinite(LRR) else np.nan,
        "LR2": float(round(LR2, 2)) if np.isfinite(LR2) else np.nan,
        "LSE": float(round(LSE, 2)) if np.isfinite(LSE) else np.nan,
        "LCI": float(round(LCI, 2)) if np.isfinite(LCI) else np.nan,
        "TemporalSpan_days": span_days,
        "ValidRegression": bool(valid_reg and len(df) >= 3 and span_days >= min_span_days),
    }


__all__ = ["compute_cluster_statistics"]
