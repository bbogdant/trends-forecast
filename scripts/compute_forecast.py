"""
Compute a Prophet-based demand forecast + predicted spike windows for one
Google Trends keyword (Apify Google Trends Scraper export, "Past 5 years" weekly window).

This replaces the earlier hand-rolled "seasonal-ratio" forecast with Prophet,
which backtested at ~7.7% mean abs error on peak prediction vs ~15.7% for the
seasonal-ratio heuristic (3-fold backtest on ATI TEAS January peaks, 2024-2026).

Usage:
    python3 compute_forecast.py <apify_export.json> <output.json> [app_label]

When you have multiple apps, just run this once per app's Apify export and
add each output as a new key in the dashboard's APPS object (see
trend_dashboard.html). Once you have a full multi-app dataset, the residuals
this script exposes (see NOTE at the bottom) are what you'd pool into a
cross-keyword XGBoost correction layer later.
"""
import sys, json, statistics, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet

FORECAST_WEEKS = 52
GT_SCALE_MAX = 100          # Google Trends index is always 0-100
CAP_MULTIPLIER = 1.15       # logistic growth cap = min(100, historical_max * 1.15)
ANOMALY_Z = 2.0             # trailing-12-week z-score threshold for historical anomalies
ANOMALY_WINDOW = 12

# spike-window detection (local-maxima based, not a broad threshold plateau)
PEAK_RADIUS_WEEKS = 3       # point must be >= all points within +-3 weeks to count as a local peak
MIN_PEAK_RATIO = 1.10       # peak must be >=10% above the recent baseline to matter
BAND = 0.96                 # window = weeks where value >= 92% of that peak's value
MAX_HALF_WIDTH_WEEKS = 3    # cap window at +-4 weeks around the peak
IGNORE_WEEKS_FROM_NOW = 3   # skip peaks that are just the tail of an already-active surge


def load_apify_export(path):
    with open(path) as f:
        raw = json.load(f)
    entry = raw[0] if isinstance(raw, list) else raw
    points = []
    for p in entry["interestOverTime"]:
        dt = datetime.fromtimestamp(int(p["time"]), tz=None)
        points.append({"date": dt.date().isoformat(), "value": p["value"], "partial": p.get("isPartial", False)})
    breakouts = [r["query"] for r in entry.get("relatedSearches", {}).get("rising", [])
                 if r.get("formattedValue") == "Breakout"]
    return entry["keyword"], points, breakouts


def load_google_trends_csv(path):
    """Parse a direct Google Trends UI export (Trends > Download > CSV).
    Format: a 'Category: ...' line, a blank line, then 'Week,<keyword>: (<geo>)'
    header, then weekly date,value rows. No partial-week flag and no rising
    queries in this format, so we infer 'partial' from whether the most recent
    week has fully elapsed yet, and breakouts come back empty."""
    with open(path, newline="") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    header_idx = next(i for i, ln in enumerate(lines) if ln.lower().startswith("week,"))
    header = lines[header_idx].split(",")
    raw_keyword = header[1]
    keyword = raw_keyword.split(":")[0].strip()

    points = []
    for ln in lines[header_idx + 1:]:
        parts = ln.split(",")
        date_str, value_str = parts[0], parts[1]
        value = 0 if value_str in ("", "<1") else int(float(value_str))
        points.append({"date": date_str, "value": value, "partial": False})

    if points:
        last_week_start = datetime.fromisoformat(points[-1]["date"]).date()
        if (last_week_start + timedelta(days=6)) >= datetime.now().date():
            points[-1]["partial"] = True

    return keyword, points, []  # no breakout-query signal available from this export format


def load_trends_export(path):
    if path.lower().endswith(".csv"):
        return load_google_trends_csv(path)
    return load_apify_export(path)


def historical_anomalies(complete):
    vals = [p["value"] for p in complete]
    out = []
    for i in range(ANOMALY_WINDOW, len(complete)):
        window = vals[i - ANOMALY_WINDOW:i]
        mean_w, std_w = statistics.mean(window), (statistics.pstdev(window) or 1e-6)
        z = (vals[i] - mean_w) / std_w
        if z >= ANOMALY_Z:
            out.append({"date": complete[i]["date"], "value": vals[i], "z": round(z, 2)})
    return out


def fit_prophet_forecast(complete):
    dates = [p["date"] for p in complete]
    vals = [p["value"] for p in complete]
    hist_max = max(vals)
    cap = min(GT_SCALE_MAX, hist_max * CAP_MULTIPLIER)

    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": vals})
    df["cap"] = cap
    df["floor"] = 0

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                     changepoint_prior_scale=0.1, interval_width=0.8, growth="logistic")
    model.fit(df)

    future = model.make_future_dataframe(periods=FORECAST_WEEKS, freq="7D")
    future["cap"] = cap
    future["floor"] = 0
    fc = model.predict(future)
    fc_future = fc.iloc[len(dates):].reset_index(drop=True)

    # residuals on the training window (kept for a future pooled-model / accuracy dashboard)
    fitted_train = fc["yhat"].values[:len(dates)]
    residuals = [round(float(v - f), 2) for v, f in zip(vals, fitted_train)]

    forecast = []
    for _, row in fc_future.iterrows():
        yhat = float(np.clip(row["yhat"], 0, GT_SCALE_MAX))
        lower = float(np.clip(row["yhat_lower"], 0, GT_SCALE_MAX))
        upper = float(np.clip(row["yhat_upper"], 0, GT_SCALE_MAX))
        iwr = (upper - lower) / yhat if yhat > 0 else 1.0
        forecast.append({
            "date": row["ds"].date().isoformat(),
            "value": round(yhat, 1),
            "yhat_lower": round(lower, 1),
            "yhat_upper": round(upper, 1),
            "interval_width_ratio": round(iwr, 3),
        })
    return forecast, residuals, cap


def detect_spike_windows(forecast, recent_baseline, last_date):
    values = [f["value"] for f in forecast]
    n = len(values)
    ignore_before = (last_date + timedelta(weeks=IGNORE_WEEKS_FROM_NOW)).date().isoformat()

    peak_idx = []
    for i in range(n):
        lo, hi = max(0, i - PEAK_RADIUS_WEEKS), min(n, i + PEAK_RADIUS_WEEKS + 1)
        if values[i] == max(values[lo:hi]) and values[i] >= MIN_PEAK_RATIO * recent_baseline:
            if forecast[i]["date"] >= ignore_before:
                peak_idx.append(i)

    # de-duplicate adjacent indices that belong to the same flat-topped peak
    deduped = []
    for i in peak_idx:
        if deduped and i - deduped[-1] <= PEAK_RADIUS_WEEKS:
            if values[i] > values[deduped[-1]]:
                deduped[-1] = i
        else:
            deduped.append(i)

    windows = []
    for i in deduped:
        peak_val = values[i]
        lo = hi = i
        while lo > 0 and values[lo - 1] >= BAND * peak_val and (i - (lo - 1)) <= MAX_HALF_WIDTH_WEEKS:
            lo -= 1
        while hi < n - 1 and values[hi + 1] >= BAND * peak_val and ((hi + 1) - i) <= MAX_HALF_WIDTH_WEEKS:
            hi += 1
        avg_iwr = statistics.mean(f["interval_width_ratio"] for f in forecast[lo:hi + 1])
        confidence = "high" if avg_iwr < 0.25 else ("medium" if avg_iwr < 0.45 else "low")
        windows.append({
            "start": forecast[lo]["date"], "end": forecast[hi]["date"],
            "peak_date": forecast[i]["date"], "peak_value": peak_val,
            "confidence": confidence,
        })

    # merge windows that overlap OR sit within a short gap of each other (avoids
    # fragmenting one smooth seasonal hump into several near-adjacent windows)
    from datetime import date as _date
    MERGE_GAP_WEEKS = 2

    def _d(s):
        return _date.fromisoformat(s)

    merged = []
    for w in sorted(windows, key=lambda x: x["start"]):
        if merged and (_d(w["start"]) - _d(merged[-1]["end"])).days <= MERGE_GAP_WEEKS * 7:
            if w["peak_value"] > merged[-1]["peak_value"]:
                merged[-1]["peak_value"] = w["peak_value"]
                merged[-1]["peak_date"] = w["peak_date"]
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(w)
    return merged


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compute_forecast.py <apify_export.json> <output.json> [app_label]")
        sys.exit(1)
    src, out_path = sys.argv[1], sys.argv[2]
    app_label = sys.argv[3] if len(sys.argv) > 3 else None

    keyword, points, breakouts = load_trends_export(src)
    complete = [p for p in points if not p["partial"]]
    last_partial = points[-1] if points[-1]["partial"] else None

    recent_baseline = statistics.mean(p["value"] for p in complete[-8:])
    last_date = datetime.fromisoformat(complete[-1]["date"])
    one_year_ago = last_date - timedelta(days=365)
    prior_window = [p["value"] for p in complete if abs((datetime.fromisoformat(p["date"]) - one_year_ago).days) <= 28]
    prior_mean = statistics.mean(prior_window) if prior_window else 0.0
    yoy_growth = ((recent_baseline - prior_mean) / prior_mean) if prior_mean > 0 else 0.0

    anomalies = historical_anomalies(complete)
    forecast, residuals, cap = fit_prophet_forecast(complete)
    spike_windows = detect_spike_windows(forecast, recent_baseline, last_date)

    output = {
        "keyword": keyword,
        "app_label": app_label or keyword.title(),
        "last_actual_date": complete[-1]["date"],
        "recent_baseline": round(recent_baseline, 1),
        "yoy_growth_pct": round(yoy_growth * 100, 1),
        "forecast_engine": "prophet_logistic_v1",
        "logistic_cap": round(cap, 1),
        "history": [{"date": p["date"], "value": p["value"]} for p in complete],
        "active_surge": ({
            "date": last_partial["date"], "value": last_partial["value"],
            "note": f"Partial current week — breakout queries: {', '.join(breakouts[:3])}" if breakouts else "Partial current week."
        } if last_partial else None),
        "anomalies": anomalies,
        "forecast": forecast,
        "spike_windows": spike_windows,
        "breakout_queries": breakouts,
        # kept for a future pooled cross-keyword residual model, not used by the dashboard yet
        "_train_residuals": residuals,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}: {len(complete)} history points, {len(forecast)} forecast points, "
          f"{len(spike_windows)} spike window(s), {len(anomalies)} historical anomalies.")
    for w in spike_windows:
        print("  spike window:", w)


if __name__ == "__main__":
    main()
