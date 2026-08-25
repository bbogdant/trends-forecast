"""
Rolling-origin backtest scored on PEAK TIMING and EARLY WARNING (P6).

This is the measurement harness for the thing the system is actually for:
being right about WHEN the next surge lands, early enough to matter. Forecast
MAE is reported but is explicitly not the objective -- a model can win on MAE
by hugging the mean and never committing to a peak at all.

Two jobs:

  1. Report the metrics that decide whether a change is an improvement:
     signed and absolute peak-timing error, hit rate within +-1/2/3 weeks,
     peak coverage (how often the model commits to a peak at all -- the old
     config silently emitted none in a third of cases), and precision / recall
     / false-alarm rate by lead time.

  2. Emit a ledger seed. Without this the P5 correction layer would sit inert
     for its first ~6 months waiting for weekly runs to accumulate resolved
     predictions. The backtest already produces exactly those resolved
     (prediction, outcome) pairs, so it can hand the correction layer a warm
     start on day one.

Usage:
    python3 scripts/backtest_peaks.py panel.csv                 # metrics only
    python3 scripts/backtest_peaks.py panel.csv --seed-ledger   # + write state/peak_ledger.json

`panel.csv` is long-format: unique_id, ds, y (one row per keyword per week).
Build it from the dashboard data files with --from-dashboard-data.

Every fit is TRAIN-ONLY per cutoff. Nothing downstream of a cutoff is visible
to the model that predicts past it.
"""
import sys, os, json, argparse, statistics, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
from datetime import date, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_forecast import (
    seasonal_reference, seasonal_stability, stability_class,
    fit_prophet_forecast, detect_spike_windows,
    PEAK_RADIUS_WEEKS, MIN_PEAK_RATIO,
)
import peak_ledger

HORIZON_WEEKS = 30      # long enough to contain a real seasonal peak (the old 12 was not)
CUTOFF_STEP_WEEKS = 4
MIN_TRAIN_WEEKS = 156   # 3 years, so yearly seasonality is identifiable
LEAD_TIMES = [2, 4, 6, 8, 10, 12]


def build_cutoffs(panel):
    """Cutoffs every CUTOFF_STEP_WEEKS, needing MIN_TRAIN_WEEKS of train and a
    full HORIZON_WEEKS of held-out truth after each one."""
    lo = panel.ds.min() + pd.Timedelta(weeks=MIN_TRAIN_WEEKS)
    hi = panel.ds.max() - pd.Timedelta(weeks=HORIZON_WEEKS)
    out, c = [], lo
    while c <= hi:
        out.append(c)
        c += pd.Timedelta(weeks=CUTOFF_STEP_WEEKS)
    return out


def observed_peaks(vals, dates, seas_ref):
    """Local maxima in truth, by the same rule the forecast side uses."""
    n = len(vals)
    raw = [i for i in range(n)
           if vals[i] == max(vals[max(0, i - PEAK_RADIUS_WEEKS):min(n, i + PEAK_RADIUS_WEEKS + 1)])
           and vals[i] >= MIN_PEAK_RATIO * seas_ref]
    ded = []
    for i in raw:
        if ded and i - ded[-1] <= PEAK_RADIUS_WEEKS:
            if vals[i] > vals[ded[-1]]:
                ded[-1] = i
        else:
            ded.append(i)
    return [dates[i] for i in ded]


def run(panel, seed_ledger_path=None):
    cutoffs = build_cutoffs(panel)
    print(f"{len(cutoffs)} cutoffs, {panel.unique_id.nunique()} keywords, "
          f"horizon {HORIZON_WEEKS}wk\n")

    rows, ledger = [], {"version": peak_ledger.LEDGER_VERSION, "entries": []}

    for uid, g in panel.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)
        for cut in cutoffs:
            tr = g[g.ds <= cut]
            te = g[(g.ds > cut) & (g.ds <= cut + pd.Timedelta(weeks=HORIZON_WEEKS))]
            if len(tr) < MIN_TRAIN_WEEKS or len(te) < HORIZON_WEEKS - 2:
                continue

            hist = [{"date": d.date().isoformat(), "value": float(v)}
                    for d, v in zip(tr.ds, tr.y)]
            seas_ref = seasonal_reference(hist)
            stab_cls = stability_class(seasonal_stability(hist))

            try:
                forecast, _ = fit_prophet_forecast(hist)
            except Exception as exc:
                print(f"  fit failed {uid} @ {cut.date()}: {exc}")
                continue

            windows = detect_spike_windows(forecast, seas_ref, cut.to_pydatetime(),
                                            timing_cls=stab_cls)
            # only peaks inside the evaluated horizon are scoreable
            horizon_end = (cut + pd.Timedelta(weeks=HORIZON_WEEKS)).date().isoformat()
            windows = [w for w in windows if w["peak_date"] <= horizon_end]

            # Actual peaks the model was ALLOWED to call: detect_spike_windows
            # skips anything within IGNORE_WEEKS_FROM_NOW of the cutoff as the
            # tail of an already-visible surge, so scoring against such a peak
            # would charge the model for a call it was never permitted to make.
            all_actual = observed_peaks(te.y.values, [d.date().isoformat() for d in te.ds], seas_ref)
            earliest_callable = (cut + pd.Timedelta(weeks=peak_ledger.IGNORE_LEAD_WEEKS)).date().isoformat()
            actual = [a for a in all_actual if a >= earliest_callable]

            pred_date = windows[0]["peak_date"] if windows else None
            act_date = actual[0] if actual else None

            err = None
            if pred_date and act_date:
                err = (date.fromisoformat(pred_date) - date.fromisoformat(act_date)).days / 7.0

            yhat = np.array([f["value"] for f in forecast[:len(te)]])
            mae = float(np.mean(np.abs(yhat - te.y.values[:len(yhat)]))) if len(yhat) else np.nan

            rows.append({
                "unique_id": uid, "cutoff": cut.date().isoformat(),
                "stability_class": stab_cls, "seasonal_ref": round(seas_ref, 1),
                "predicted_peak": pred_date, "actual_peak": act_date,
                "signed_error_weeks": err, "n_windows": len(windows),
                "timing_confidence": windows[0]["timing_confidence"] if windows else None,
                "mae": mae,
                "all_predicted": ";".join(w["peak_date"] for w in windows),
                "all_actual": ";".join(actual),
            })

            if seed_ledger_path and pred_date:
                # One entry per cutoff: the SOONEST announced peak vs the first
                # actual peak that followed -- identical to what
                # peak_ledger.record_predictions/resolve_pending do in
                # production, so the seed and the live rows are the same
                # quantity. (Seeding one row per window and nearest-matching
                # them, as a first attempt did, silently zeroes out the bias --
                # see the note in record_predictions.)
                pk = date.fromisoformat(pred_date)
                # resolved_on is dated when the outcome BECAME OBSERVABLE, not at
                # the cutoff. bias_for() filters on resolved_on, so dating these
                # honestly is what keeps a replay from seeing its own future.
                resolved_on = (pk + timedelta(weeks=peak_ledger.RESOLVE_MARGIN_WEEKS)).isoformat()
                ledger["entries"].append({
                    "keyword": uid,
                    "made_on": cut.date().isoformat(),
                    "predicted_peak_date": pred_date,
                    "predicted_peak_value": windows[0].get("peak_value"),
                    "lead_weeks": round((pk - cut.date()).days / 7.0, 1),
                    "timing_confidence": windows[0].get("timing_confidence"),
                    "signed_error_weeks": err if err is not None else None,
                    "actual_peak_date": act_date,
                    "outcome": "matched" if err is not None else "no_peak_occurred",
                    "resolved_on": resolved_on,
                    "source": "backtest_seed",
                })
        print(f"  done {uid}", flush=True)

    df = pd.DataFrame(rows)
    if seed_ledger_path:
        peak_ledger.save_ledger(ledger, seed_ledger_path)
        n_matched = sum(1 for e in ledger["entries"] if e["outcome"] == "matched")
        print(f"\nSeeded {seed_ledger_path}: {len(ledger['entries'])} entries "
              f"({n_matched} matched, {len(ledger['entries']) - n_matched} false positives)")
    return df


def report(df):
    m = df.dropna(subset=["signed_error_weeks"])
    scoreable = df.dropna(subset=["actual_peak"])
    print("\n" + "=" * 68)
    print("PEAK TIMING")
    print("=" * 68)
    print(f"  scoreable cutoffs (a real peak occurred): {len(scoreable)}")
    print(f"  of those, model committed to a peak:      {len(m)} "
          f"({100 * len(m) / max(len(scoreable), 1):.1f}% coverage)")
    if len(m):
        e = m.signed_error_weeks
        print(f"  mean abs error   {e.abs().mean():.2f} wk")
        print(f"  median abs error {e.abs().median():.2f} wk")
        print(f"  signed mean      {e.mean():+.2f} wk   (positive = predicts LATE)")
        print(f"  signed median    {e.median():+.2f} wk")
        print(f"  hit +-1wk {100 * (e.abs() <= 1).mean():.1f}%   "
              f"+-2wk {100 * (e.abs() <= 2).mean():.1f}%   "
              f"+-3wk {100 * (e.abs() <= 3).mean():.1f}%")
        print(f"  forecast MAE (not the objective) {df.mae.mean():.2f}")

    print("\n" + "=" * 68)
    print("PER KEYWORD (signed error -- drives the P5 correction)")
    print("=" * 68)
    if len(m):
        t = m.groupby("unique_id").signed_error_weeks.agg(["count", "mean", "median", "std"])
        t = t.join(df.groupby("unique_id").stability_class.first())
        print(t.round(2).sort_values("median").to_string())

    print("\n" + "=" * 68)
    print("EARLY WARNING by lead time")
    print("=" * 68)
    print(f"  {'lead':>5} {'precision':>10} {'recall':>8} {'false alarm':>12}   tp/fp/fn")
    for L in LEAD_TIMES:
        tp = fp = fn = 0
        for _, r in df.iterrows():
            cut = date.fromisoformat(r.cutoff)
            horizon = cut + timedelta(weeks=L)
            pred_soon = bool(r.predicted_peak and date.fromisoformat(r.predicted_peak) <= horizon)
            act_soon = bool(r.actual_peak and date.fromisoformat(r.actual_peak) <= horizon)
            if pred_soon and act_soon:
                tp += 1
            elif pred_soon:
                fp += 1
            elif act_soon:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        far = fp / len(df) if len(df) else 0.0
        print(f"  {L:>5} {prec:>10.2f} {rec:>8.2f} {far:>12.3f}   {tp}/{fp}/{fn}")


def panel_from_dashboard_data(data_dir):
    rows = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith("_dashboard_data.json"):
            continue
        with open(os.path.join(data_dir, fn)) as f:
            d = json.load(f)
        uid = fn.replace("_dashboard_data.json", "")
        for p in d["history"]:
            rows.append({"unique_id": uid, "ds": p["date"], "y": p["value"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="panel.csv")
    ap.add_argument("--from-dashboard-data", metavar="DIR",
                    help="build the panel from docs/data/*.json instead of a csv")
    ap.add_argument("--seed-ledger", nargs="?", const=os.path.join("state", "peak_ledger.json"),
                    help="also write a warm-start ledger for the P5 correction layer")
    ap.add_argument("--out", default="backtest_peaks_raw.csv")
    args = ap.parse_args()

    if args.from_dashboard_data:
        panel = panel_from_dashboard_data(args.from_dashboard_data)
    else:
        panel = pd.read_csv(args.panel)
    panel["ds"] = pd.to_datetime(panel["ds"])

    df = run(panel, seed_ledger_path=args.seed_ledger)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
