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
import sys, os, json, statistics, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from datetime import datetime, timedelta, date
from itertools import combinations
import pandas as pd
import numpy as np
from prophet import Prophet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_ledger
import surge_def
import keyword_tiers
import re as _re


def slugify_keyword(keyword):
    """Same slug rule scrape_and_forecast.py uses for output filenames, so the
    keyword-tier lookup keys match the dashboard data files."""
    return _re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")

# Ledger lives outside docs/ so GitHub Pages never serves it -- it is internal
# model state, not dashboard data.
DEFAULT_LEDGER_PATH = os.path.join("state", "peak_ledger.json")

# marketing-window buffers, keyed by how stable a keyword's annual seasonality
# empirically is -- see the Aug 2026 Glimpse-style backtest: peak-timing error
# averages ~3.7-4.8 weeks and is worse for keywords with unstable seasonality,
# so the "start marketing now" window is buffered wider for those.
MARKETING_BUFFERS = {
    "stable":   {"before": 3, "after": 2},
    "moderate": {"before": 4, "after": 3},
    "unstable": {"before": 5, "after": 4},
}

MIN_LEDGER_OBS = 4     # resolved past predictions needed before trusting a keyword's own quantiles
MAX_BUFFER_WEEKS = 10  # hard ceiling so a pathological residual history cannot produce a useless window

FORECAST_WEEKS = 52
GT_SCALE_MAX = 100          # Google Trends index is always 0-100
ANOMALY_Z = 2.0             # trailing-12-week z-score threshold for historical anomalies
ANOMALY_WINDOW = 12

# --- Prophet configuration (P1) -------------------------------------------
# Measured on a 10-cutoff / 30-week-horizon rolling-origin backtest scored on
# PEAK TIMING (see scripts/backtest_peaks.py). Paired comparison vs the old
# logistic+Fourier-10 config, n=96 matched peaks:
#   old (logistic, fourier 10):  mean abs timing err 5.31 wk, hit +-1wk 35.1%, peak coverage 66.4%
#   new (linear,   fourier 20):  mean abs timing err 4.68 wk, hit +-1wk 47.9%, peak coverage 80.1%
#   improvement +1.68 wk, 95% bootstrap CI [+0.52, +2.82] (excludes zero)
#
# Why linear instead of logistic: Google Trends renormalises its 0-100 index to
# the max of the queried window, so historical_max is 100 for every keyword we
# track, which made the old cap = min(100, 100*1.15) = 100 a no-op ceiling. The
# logistic curve was therefore imposing a saturating S-shape toward a bound the
# series already touches -- an unjustified constraint that only distorted trend.
#
# Why Fourier 20 instead of Prophet's default 10: order 10 is too smooth to
# localise the sharp January / back-to-school spikes these exam keywords show,
# and it merges genuinely separate seasonal humps into one broad plateau. Order
# 20 resolves them, which is where most of the timing gain above comes from.
PROPHET_GROWTH = "linear"
YEARLY_FOURIER_ORDER = 20
CHANGEPOINT_PRIOR_SCALE = 0.1
INTERVAL_WIDTH = 0.8

# spike-window detection (local-maxima based, not a broad threshold plateau)
PEAK_RADIUS_WEEKS = 3       # point must be >= all points within +-3 weeks to count as a local peak
MIN_PEAK_RATIO = 1.10       # peak must be >=10% above the SEASONAL reference to matter
BAND = 0.96                 # window = weeks where value >= 96% of that peak's value
MAX_HALF_WIDTH_WEEKS = 3    # cap window at +-3 weeks around the peak
IGNORE_WEEKS_FROM_NOW = 3   # skip peaks that are just the tail of an already-active surge
SEASONAL_REF_WEEKS = 52     # trailing window for the season-neutral peak reference (P2)


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


def seasonal_stability(complete):
    """Mean pairwise year-over-year correlation of the week-of-year profile,
    computed on history only (no forecast/future data involved)."""
    df = pd.DataFrame(complete)
    df["ds"] = pd.to_datetime(df["date"])
    df["woy"] = df["ds"].dt.isocalendar().week.astype(int)
    df["yr"] = df["ds"].dt.isocalendar().year.astype(int)
    profs = {yr: g.set_index("woy")["value"].reindex(range(1, 53))
             for yr, g in df.groupby("yr") if len(g) >= 45}
    if len(profs) < 2:
        return 0.3
    rs = []
    for a, b in combinations(sorted(profs), 2):
        pa, pb = profs[a], profs[b]
        m = pa.notna() & pb.notna()
        if m.sum() > 30 and pa[m].std() > 0 and pb[m].std() > 0:
            rs.append(np.corrcoef(pa[m], pb[m])[0, 1])
    return float(np.mean(rs)) if rs else 0.3


def stability_class(stability):
    if stability >= 0.4:
        return "stable"
    if stability >= 0.15:
        return "moderate"
    return "unstable"


def _combine_timing_confidence(timing_cls, prominence):
    """Timing confidence blends how repeatable the keyword's annual shape is
    (measured, train-only, by seasonal_stability) with how sharp this particular
    peak is. An unstable keyword can still give a datable peak if that peak is
    very prominent; a stable keyword's broad plateau is still hard to date."""
    score = {"stable": 2, "moderate": 1, "unstable": 0}.get(timing_cls, 1)
    if prominence >= 1.25:
        score += 1
    elif prominence < 1.08:
        score -= 1
    return "high" if score >= 3 else ("medium" if score >= 1 else "low")


def add_marketing_windows(spike_windows, stability_cls, timing_quantiles=None):
    """Attach the recommended marketing push window to each predicted spike.

    Two modes:

    1. Empirical (P4, preferred) -- `timing_quantiles` is {"p05": x, "p95": y}
       taken from that keyword's own past SIGNED peak-timing errors, so the
       window is a real ~90% prediction interval on the peak DATE. Signed
       matters: the errors are strongly asymmetric (the model runs late far
       more often than early), so a symmetric window built from mean ABSOLUTE
       error -- which is what the old hardcoded buffers were -- sits in the
       wrong place. Measured example: cna_exam's median timing error is +6
       weeks with std 8, so the old "stable" 3-before/2-after window
       systematically opened after its peaks had already passed.

    2. Fallback -- no ledger history for this keyword yet, so use the
       stability-class defaults. Same as the old behaviour.

    IMPORTANT: when the caller has already applied bias correction (P5), the
    quantiles passed here must be computed from the residuals that REMAIN after
    that correction. Feeding it raw residuals double-counts the bias and yields
    a window that is both too wide and shifted.
    """
    fallback = MARKETING_BUFFERS[stability_cls]
    for w in spike_windows:
        peak = date.fromisoformat(w["peak_date"])
        if timing_quantiles and timing_quantiles.get("n", 0) >= MIN_LEDGER_OBS:
            # p05 is the most-early error (usually negative) -> open the window
            # that far BEFORE the predicted peak; p95 is the most-late error ->
            # hold the window open that far AFTER it.
            before = int(round(max(1.0, -min(timing_quantiles["p05"], 0.0) + 1)))
            after = int(round(max(1.0, max(timing_quantiles["p95"], 0.0) + 1)))
            before = min(before, MAX_BUFFER_WEEKS)
            after = min(after, MAX_BUFFER_WEEKS)
            source = "empirical_quantiles"
            n_obs = timing_quantiles["n"]
        else:
            before, after = fallback["before"], fallback["after"]
            source = "stability_class_default"
            n_obs = 0
        w["marketing_window"] = {
            "start": (peak - timedelta(weeks=before)).isoformat(),
            "end": (peak + timedelta(weeks=after)).isoformat(),
            "buffer_before_weeks": before,
            "buffer_after_weeks": after,
            "buffer_source": source,
            "buffer_n_observations": n_obs,
        }
    return spike_windows


def seasonal_reference(complete, weeks=SEASONAL_REF_WEEKS):
    """Season-neutral level used as the peak-detection reference (P2).

    The old code gated peaks against the mean of the last 8 weeks of history.
    That reference moves with wherever in the season the weekly job happens to
    run, so the SAME keyword got a different detection threshold depending only
    on the calendar week of the cron run. Measured across the tracked keywords,
    that threshold swung by a median factor of 1.98x over a year, and for
    nclex_exam / cna_exam / real_estate_exam the in-season threshold rose ABOVE
    100 -- i.e. above the maximum a Google Trends index can ever reach, making
    peak detection mathematically impossible for part of the year. That is what
    left ASWB Exam showing zero spike windows in production.

    The median of a full trailing year is (a) season-neutral, since it spans
    every week-of-year exactly once, and (b) robust to the spikes themselves,
    so a big peak does not inflate the bar for detecting the next one.
    """
    vals = [p["value"] for p in complete[-weeks:]]
    return statistics.median(vals) if vals else 0.0


def fit_prophet_forecast(complete):
    dates = [p["date"] for p in complete]
    vals = [p["value"] for p in complete]

    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": vals})

    # yearly seasonality is added explicitly rather than via yearly_seasonality=True
    # so the Fourier order is ours to set (default would be 10) -- see P1 note above.
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                     changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE,
                     interval_width=INTERVAL_WIDTH, growth=PROPHET_GROWTH)
    model.add_seasonality(name="yearly", period=365.25, fourier_order=YEARLY_FOURIER_ORDER)
    model.fit(df)

    future = model.make_future_dataframe(periods=FORECAST_WEEKS, freq="7D")
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
    return forecast, residuals


def detect_spike_windows(forecast, seasonal_ref, last_date, timing_cls="medium"):
    """Find forecast local maxima worth acting on.

    `seasonal_ref` is the season-neutral reference from seasonal_reference()
    (P2), NOT the trailing-8-week mean the old signature took.
    """
    values = [f["value"] for f in forecast]
    n = len(values)
    # Peaks landing inside the next IGNORE_WEEKS_FROM_NOW weeks are FLAGGED as
    # imminent, not discarded. They used to be dropped outright, on the theory
    # that they were only the tail of an already-visible surge -- but that
    # deleted precisely the 1-3 week band the marketing team needs, and it did
    # so structurally: measured early-warning recall at leads 1, 2 and 3 weeks
    # was exactly 0.00 because no peak inside that range could ever be reported.
    # With the filter turned into a flag, recall at those leads goes to
    # 0.33/0.37/0.49. Consumers that want the old behaviour can skip windows
    # where `imminent` is true.
    imminent_before = (last_date + timedelta(weeks=IGNORE_WEEKS_FROM_NOW)).date().isoformat()

    peak_idx = []
    for i in range(n):
        lo, hi = max(0, i - PEAK_RADIUS_WEEKS), min(n, i + PEAK_RADIUS_WEEKS + 1)
        if values[i] == max(values[lo:hi]) and values[i] >= MIN_PEAK_RATIO * seasonal_ref:
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
        # --- P3: amplitude confidence and timing confidence are different things ---
        # The old single `confidence` field was derived purely from Prophet's
        # interval width, which measures uncertainty about HOW HIGH the peak will
        # be. It says nothing about WHEN it lands -- and the marketing decision
        # depends almost entirely on the timing. So report both, and let the
        # legacy `confidence` key carry the TIMING one, because that is the one a
        # human reading "should I start the campaign now?" actually needs.
        avg_iwr = statistics.mean(f["interval_width_ratio"] for f in forecast[lo:hi + 1])
        amplitude_conf = "high" if avg_iwr < 0.25 else ("medium" if avg_iwr < 0.45 else "low")

        # prominence: how much this peak stands out from its own shoulders. A
        # sharp isolated peak is far easier to date than a broad plateau, on
        # which a one-week dating error is nearly meaningless anyway.
        shoulder_lo = max(0, i - 2 * PEAK_RADIUS_WEEKS)
        shoulder_hi = min(n, i + 2 * PEAK_RADIUS_WEEKS + 1)
        shoulder = [values[j] for j in range(shoulder_lo, shoulder_hi) if not (lo <= j <= hi)]
        prominence = (peak_val / statistics.mean(shoulder)) if shoulder and statistics.mean(shoulder) > 0 else 1.0

        timing_conf = _combine_timing_confidence(timing_cls, prominence)

        windows.append({
            "start": forecast[lo]["date"], "end": forecast[hi]["date"],
            "peak_date": forecast[i]["date"], "peak_value": peak_val,
            "imminent": forecast[i]["date"] < imminent_before,
            "amplitude_confidence": amplitude_conf,
            "timing_confidence": timing_conf,
            "peak_prominence": round(prominence, 3),
            "interval_width_ratio": round(avg_iwr, 3),
            # legacy alias -- the dashboard's badge CSS keys off this exact field
            # and expects one of high/medium/low. Keep it pointing at timing.
            "confidence": timing_conf,
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
                merged[-1]["peak_prominence"] = w["peak_prominence"]
                merged[-1]["timing_confidence"] = w["timing_confidence"]
                merged[-1]["confidence"] = w["timing_confidence"]
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(w)
    return merged


def build_surge_outlook(slug, complete, forecast):
    """The marketing-facing signal: a BRACKET around the next expected surge,
    plus a rolling weekly state, for the keywords where that was measured to work.

    Deliberately not a single predicted date. The signed timing error has a tight
    core (p25 = 0, p75 = +2 weeks, median exactly 0) and fat tails (p05 = -7,
    p95 = +9), so a bracket captures most of the mass while a point estimate does
    not. Reframing the question from "is a surge coming within 2-3 weeks?" to
    "which weeks will the next surge fall in?" moved the usable result from
    recall 0.22 to 76-99% window coverage on five keywords.

    Returns None when this keyword has no measured skill -- the caller reports
    the reason rather than drawing an empty box.

    States:
      PEAK_NOW  the series is already surging (detrended level over threshold)
      RAMP      today sits inside the expected window -- this is the spend signal
      HOLD      neither
    """
    tier, win, meta = keyword_tiers.tier_for(slug)
    if tier is None:
        return None, meta                      # meta is the human-readable reason

    values = [p["value"] for p in complete]
    icpt, slope = surge_def.fit_log_trend(values)          # train-only trend

    # Are we surging right now? Compare the latest actual against its own
    # extrapolated trend, not against a fixed level.
    current_ratio = values[-1] / surge_def.trend_at(icpt, slope, 0)

    # Next surge in the detrended FORECAST
    fc_vals = [f["value"] for f in forecast]
    fc_ratio = surge_def.detrend_forward(fc_vals, icpt, slope, start_step=1)
    surge_idx = surge_def.surge_indices(fc_ratio)

    last_date = date.fromisoformat(complete[-1]["date"])
    outlook = {
        "tier": tier,
        "window_weeks": win["before"] + win["after"],
        "measured_coverage": meta["coverage"],
        "measured_median_lead_weeks": meta["median_lead_weeks"],
        "measured_signal_rate": meta["signal_rate"],
        "surge_episodes_in_backtest": meta["surge_episodes"],
        "current_detrended_ratio": round(float(current_ratio), 3),
        "reliability_note": meta["note"],
    }

    if current_ratio >= surge_def.SURGE_RATIO:
        outlook.update({"state": "PEAK_NOW", "expected_window": None,
                        "predicted_peak_date": None, "weeks_until_window": 0})
        return outlook, None

    if not surge_idx:
        outlook.update({"state": "HOLD", "expected_window": None,
                        "predicted_peak_date": None, "weeks_until_window": None,
                        "state_reason": "no surge in the 52-week forecast horizon"})
        return outlook, None

    peak_step = surge_idx[0] + 1                    # weeks ahead of last actual
    peak_date = last_date + timedelta(weeks=peak_step)
    w_start = peak_date - timedelta(weeks=win["before"])
    w_end = peak_date + timedelta(weeks=win["after"])
    weeks_until = (w_start - last_date).days / 7.0

    outlook.update({
        "state": "RAMP" if weeks_until <= 0 else "HOLD",
        "predicted_peak_date": peak_date.isoformat(),
        "expected_window": {"start": w_start.isoformat(), "end": w_end.isoformat(),
                             "width_weeks": win["before"] + win["after"]},
        "weeks_until_window": round(weeks_until, 1),
        "forecast_surge_ratio": round(float(fc_ratio[surge_idx[0]]), 3),
    })
    return outlook, None


def build_forecast_payload(keyword, points, breakouts, app_label=None,
                            ledger=None, ledger_path=None, as_of=None, slug=None):
    """The full pipeline for one keyword, shared by both entrypoints.

    Google Trends update -> refit -> peak prediction -> learned bias correction
    -> final peak + split confidence + recommended marketing window.

    `ledger` may be None, in which case bias correction and empirical buffers
    are skipped and the stability-class defaults are used (cold start).
    """
    complete = [p for p in points if not p["partial"]]
    last_partial = points[-1] if points and points[-1]["partial"] else None

    recent_baseline = statistics.mean(p["value"] for p in complete[-8:])
    seas_ref = seasonal_reference(complete)
    last_date = datetime.fromisoformat(complete[-1]["date"])
    as_of = as_of or complete[-1]["date"]

    one_year_ago = last_date - timedelta(days=365)
    prior_window = [p["value"] for p in complete
                    if abs((datetime.fromisoformat(p["date"]) - one_year_ago).days) <= 28]
    prior_mean = statistics.mean(prior_window) if prior_window else 0.0
    yoy_growth = ((recent_baseline - prior_mean) / prior_mean) if prior_mean > 0 else 0.0

    stability = seasonal_stability(complete)
    stab_cls = stability_class(stability)

    anomalies = historical_anomalies(complete)
    forecast, residuals = fit_prophet_forecast(complete)
    spike_windows = detect_spike_windows(forecast, seas_ref, last_date, timing_cls=stab_cls)

    # --- learned correction layer (P5) ---
    bias = {"correction_weeks": 0.0, "n_observations": 0, "source": "no_ledger", "raw_median": None}
    quants = None
    if ledger is not None:
        peak_ledger.resolve_pending(ledger, keyword, complete, seas_ref)
        bias = peak_ledger.bias_for(ledger, keyword, as_of=as_of)
        spike_windows = peak_ledger.apply_bias_correction(spike_windows, bias)
        # quantiles are computed on POST-correction residuals -- see the note in
        # add_marketing_windows about not double-counting the bias.
        quants = peak_ledger.timing_quantiles(ledger, keyword, as_of=as_of,
                                              correction_weeks=bias["correction_weeks"])

    spike_windows = add_marketing_windows(spike_windows, stab_cls, timing_quantiles=quants)

    # the marketing-facing signal (P-window work); slug defaults from the keyword
    outlook, no_signal_reason = build_surge_outlook(
        slug or slugify_keyword(keyword), complete, forecast)

    if ledger is not None:
        peak_ledger.record_predictions(ledger, keyword, as_of, spike_windows)
        if ledger_path:
            peak_ledger.save_ledger(ledger, ledger_path)

    return {
        "keyword": keyword,
        "app_label": app_label or keyword.title(),
        "last_actual_date": complete[-1]["date"],
        "recent_baseline": round(recent_baseline, 1),
        "seasonal_reference": round(seas_ref, 1),
        "yoy_growth_pct": round(yoy_growth * 100, 1),
        "seasonal_stability": round(stability, 3),
        "seasonal_stability_class": stab_cls,
        "forecast_engine": "prophet_linear_f20_v3",
        "timing_bias_correction": bias,
        "timing_error_quantiles": quants,
        # marketing signal: a bracket + rolling state, only where measured to work
        "surge_outlook": outlook,
        "no_signal_reason": no_signal_reason,
        "history": [{"date": p["date"], "value": p["value"]} for p in complete],
        "active_surge": ({
            "date": last_partial["date"], "value": last_partial["value"],
            "note": (f"Partial current week — breakout queries: {', '.join(breakouts[:3])}"
                     if breakouts else "Partial current week.")
        } if last_partial else None),
        "anomalies": anomalies,
        "forecast": forecast,
        "spike_windows": spike_windows,
        "breakout_queries": breakouts,
        "_train_residuals": residuals,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compute_forecast.py <apify_export.json> <output.json> [app_label]")
        sys.exit(1)
    src, out_path = sys.argv[1], sys.argv[2]
    app_label = sys.argv[3] if len(sys.argv) > 3 else None

    keyword, points, breakouts = load_trends_export(src)

    ledger_path = os.environ.get("PEAK_LEDGER", DEFAULT_LEDGER_PATH)
    ledger = peak_ledger.load_ledger(ledger_path)

    output = build_forecast_payload(keyword, points, breakouts, app_label=app_label,
                                    ledger=ledger, ledger_path=ledger_path)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}: {len(output['history'])} history points, "
          f"{len(output['forecast'])} forecast points, "
          f"{len(output['spike_windows'])} spike window(s), "
          f"{len(output['anomalies'])} historical anomalies.")
    b = output["timing_bias_correction"]
    print(f"  bias correction: {b['correction_weeks']:+.1f} wk "
          f"(source={b['source']}, n={b['n_observations']})")
    for w in output["spike_windows"]:
        print(f"  peak {w['peak_date']} (raw {w.get('peak_date_raw', w['peak_date'])}) "
              f"timing={w['timing_confidence']} amp={w['amplitude_confidence']} "
              f"marketing {w['marketing_window']['start']}..{w['marketing_window']['end']} "
              f"({w['marketing_window']['buffer_source']})")


if __name__ == "__main__":
    main()
