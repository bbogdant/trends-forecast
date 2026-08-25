"""
Append-only ledger of past peak predictions, and the bias correction learned
from it (P5).

STATUS: the ledger runs; the CORRECTION IS OFF BY DEFAULT
--------------------------------------------------------
The ledger records and resolves every prediction, because knowing how wrong the
system has been is worth having regardless. The bias-correction step that reads
it is disabled (ENABLE_BIAS_CORRECTION = False) because, measured honestly, it
does not help. Do not turn it on without re-running the check described below.

HOW THAT CONCLUSION WAS REACHED -- and two wrong turns worth not repeating
-------------------------------------------------------------------------
The layer was motivated by an apparent +5.4 week systematic lateness. Two
measurement bugs were inflating that figure, and fixing both mostly dissolved it:

  1. Nearest-neighbour matching. The first ledger design recorded every
     predicted window and later paired each with its nearest actual peak within
     12 weeks. Peaks here recur every ~20-26 weeks, so a systematically late
     prediction simply got paired with a later actual peak. Under match radii of
     4/6/8/12 weeks the signed median came out as exactly 0.00 every time: the
     matching was absorbing the very bias the layer existed to learn. Fixed by
     scoring the operational claim instead -- the soonest announced peak against
     the first actual peak that followed.

  2. Scoring against peaks the model is forbidden to call. detect_spike_windows
     deliberately skips peaks within IGNORE_WEEKS_FROM_NOW of the run date,
     treating them as an already-visible surge. Resolution was not applying the
     same blackout, so the forecaster was charged for "missing" peaks it was
     never allowed to predict. Fixing this alone moved mean absolute error from
     6.02 to 4.07 weeks and signed mean from +5.36 to +1.50.

After both fixes the pooled signed median is 0.00 and the pooled mean +1.50, with
per-keyword medians scattered between -4 and +5 on n~16 and per-keyword std of
4-9 weeks. Those tilts are mostly noise. Replayed with strict no-leakage,
applying the shrunk correction unconditionally made timing WORSE by 0.18 weeks
(95% CI [-0.37, -0.00]), helped only 31% of cases, and degraded 11 of 15
keywords. Adding the evidence gate below reduced it to a wash (-0.05 weeks,
95% CI [-0.16, +0.06]) by letting almost nothing through -- harmless, but not
useful either. Hence: off.

WHY THE CODE IS KEPT
--------------------
The ledger keeps accumulating truth at one row per keyword per week. Two things
could make this layer earn its place later, and both arrive on their own:
  * a real bias developing (a Google Trends methodology change, a portfolio
    shift), which the gate would then let through automatically; or
  * enough data to estimate bias PER LEAD TIME, where it is far more likely to
    be real -- a peak called 40 weeks out is not the same estimation problem as
    one called 8 weeks out. `lead_weeks` is recorded on every entry for exactly
    this, though nothing reads it yet.

Design notes, should it be switched on: median rather than mean throughout
(with std that large a single outlier moves a mean uselessly far); shrinkage
toward the pooled median with a minimum-n gate and a hard cap; and deliberately
not an ML model -- one shrunk number per keyword is one parameter, and anything
richer would overfit this much data.

NO LEAKAGE, STRUCTURALLY
------------------------
Entries are written when a prediction is MADE, with no error attached. The
error is filled in only once enough real history exists to observe what actually
happened. `bias_for()` reads only entries that already carry a resolved error,
so a prediction can never be corrected using knowledge of its own outcome, or
of anything that happened after it. This is a property of the data layout, not
a rule someone has to remember to follow.
"""
import json, os, statistics
from datetime import date, datetime, timedelta

LEDGER_VERSION = 1

# Master switch for the correction layer. OFF because it does not currently help
# -- see the module docstring for the measurements. The ledger still records and
# resolves predictions when this is False; only the date-shifting is skipped.
# Before flipping this, re-run: python3 scripts/backtest_peaks.py panel.csv
# --seed-ledger, then replay and confirm the correction shows a positive
# improvement whose confidence interval excludes zero.
ENABLE_BIAS_CORRECTION = False

SHRINK_PRIOR_WEIGHT = 4.0   # k0: pseudo-observations pulling a thin per-keyword bias toward the pooled median
MIN_OBS_FOR_BIAS = 8        # resolved observations needed before a bias is even considered
MIN_BIAS_WEEKS = 2.0        # a median tilt smaller than this is not worth moving a date over
BIAS_SIGN_CONSISTENCY = 0.7  # fraction of observations that must share the median's sign
MAX_ABS_CORRECTION_WK = 6   # never shift a peak date by more than this
RESOLVE_MARGIN_WEEKS = 6    # weeks of real history needed past a predicted peak before scoring it
IGNORE_LEAD_WEEKS = 3       # mirrors compute_forecast.IGNORE_WEEKS_FROM_NOW: peaks nearer than
                            # this are treated as an active surge and are not predictable calls


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def load_ledger(path):
    if not os.path.exists(path):
        return {"version": LEDGER_VERSION, "entries": []}
    with open(path) as f:
        led = json.load(f)
    led.setdefault("version", LEDGER_VERSION)
    led.setdefault("entries", [])
    return led


def save_ledger(led, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(led, f, indent=2, sort_keys=True)


def record_predictions(led, keyword, made_on, spike_windows):
    """Append this run's ANNOUNCEMENT as one unresolved entry (no error yet).

    Deliberately records only the SOONEST predicted peak, not every window.

    That is the operational claim the system makes: the dashboard tells the
    marketing team "your next peak is on date X", and the error that matters is
    how wrong that specific announcement turned out to be.

    Recording every window and later matching each to its nearest actual peak --
    which is what an earlier version of this file did -- destroys the signal.
    Peaks in these series recur every ~20-26 weeks, so with a generous match
    radius almost any prediction finds some actual peak to pair with, and a
    systematically late prediction simply gets paired with a later actual peak.
    Measured: under nearest-match radii of 4/6/8/12 weeks the signed median came
    out as exactly 0.00 in every case, while the same predictions scored
    first-announced vs first-actual retained a clear same-signed tilt. The
    matching was absorbing the bias. (That tilt then shrank again once
    resolution stopped scoring against peaks inside the forecaster's blackout
    window -- see wrong turn #2 in the module docstring. Both fixes were needed
    to see the true, much smaller, bias.)

    Idempotent per (keyword, made_on).
    """
    if not spike_windows:
        return 0
    if any(e["keyword"] == keyword and e["made_on"] == made_on for e in led["entries"]):
        return 0

    first = min(spike_windows, key=lambda w: w["peak_date"])
    led["entries"].append({
        "keyword": keyword,
        "made_on": made_on,
        "predicted_peak_date": first["peak_date"],
        "predicted_peak_value": first.get("peak_value"),
        "lead_weeks": _weeks_between(made_on, first["peak_date"]),
        "timing_confidence": first.get("timing_confidence"),
        "signed_error_weeks": None,   # filled in by resolve_pending()
        "actual_peak_date": None,
        "resolved_on": None,
    })
    return 1


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------
def observed_peaks(history, seasonal_ref, radius_weeks=3, min_ratio=1.10):
    """Local maxima in ACTUAL history, using the same rule the forecast side
    uses so predicted and observed peaks are defined identically."""
    vals = [p["value"] for p in history]
    dates = [p["date"] for p in history]
    n = len(vals)
    raw = []
    for i in range(n):
        lo, hi = max(0, i - radius_weeks), min(n, i + radius_weeks + 1)
        if vals[i] == max(vals[lo:hi]) and vals[i] >= min_ratio * seasonal_ref:
            raw.append(i)
    ded = []
    for i in raw:
        if ded and i - ded[-1] <= radius_weeks:
            if vals[i] > vals[ded[-1]]:
                ded[-1] = i
        else:
            ded.append(i)
    return [dates[i] for i in ded]


def resolve_pending(led, keyword, history, seasonal_ref, today=None):
    """Score any unresolved prediction whose outcome is now observable.

    A prediction becomes resolvable once we hold RESOLVE_MARGIN_WEEKS of real
    history past its predicted peak date -- enough for a local maximum around
    that date to be identifiable rather than still being the leading edge of
    the series.
    """
    today = today or date.today()
    last_actual = date.fromisoformat(history[-1]["date"])
    actuals = observed_peaks(history, seasonal_ref)
    resolved = 0

    for e in led["entries"]:
        if e["keyword"] != keyword or e["signed_error_weeks"] is not None:
            continue
        if e.get("outcome") == "no_peak_occurred":
            continue
        pred = date.fromisoformat(e["predicted_peak_date"])
        made = date.fromisoformat(e["made_on"])
        if last_actual < pred + timedelta(weeks=RESOLVE_MARGIN_WEEKS):
            continue  # outcome not yet observable -- leave unresolved

        # The FIRST actual peak the model was ALLOWED to predict, i.e. one
        # landing at least IGNORE_LEAD_WEEKS after the announcement. The
        # forecaster deliberately skips nearer peaks as the tail of an already
        # visible surge, so scoring it against one of those would charge it for
        # a call it was never permitted to make.
        horizon_start = made + timedelta(weeks=IGNORE_LEAD_WEEKS)
        after = [date.fromisoformat(a) for a in actuals
                 if date.fromisoformat(a) >= horizon_start]
        if not after:
            e["outcome"] = "no_peak_occurred"
            e["resolved_on"] = today.isoformat()
        else:
            actual = after[0]
            e["signed_error_weeks"] = round((pred - actual).days / 7.0, 1)
            e["actual_peak_date"] = actual.isoformat()
            e["outcome"] = "matched"
            e["resolved_on"] = today.isoformat()
        resolved += 1
    return resolved


# --------------------------------------------------------------------------
# what the forecast actually consumes
# --------------------------------------------------------------------------
def _resolved_errors(led, keyword, as_of=None):
    """Signed errors known to be available at `as_of`. Filtering on resolved_on
    (not on the prediction date) is what makes a backtest replay honest."""
    out = []
    for e in led["entries"]:
        if e["keyword"] != keyword or e.get("signed_error_weeks") is None:
            continue
        if as_of and e.get("resolved_on") and e["resolved_on"] > as_of:
            continue
        out.append(e["signed_error_weeks"])
    return out


def pooled_bias(led, keyword=None, as_of=None):
    """Median signed error across ALL keywords -- the empirical-Bayes prior that
    a thin per-keyword estimate is shrunk toward.

    Optionally excludes `keyword` (leave-one-out), so a keyword with many
    observations cannot dominate the prior it is then shrunk toward.
    """
    errs = []
    for e in led["entries"]:
        if e.get("signed_error_weeks") is None:
            continue
        if keyword and e["keyword"] == keyword:
            continue
        if as_of and e.get("resolved_on") and e["resolved_on"] > as_of:
            continue
        errs.append(e["signed_error_weeks"])
    return statistics.median(errs) if errs else 0.0


def bias_for(led, keyword, as_of=None):
    """Shrunk per-keyword timing bias, in weeks. Positive = predicts too late,
    so the correction subtracts it.

        shrunk = (n * median_keyword + k0 * pooled_median) / (n + k0)

    then gated: the shift is only returned if the keyword's error history is
    actually lopsided enough to justify moving a date. See EVIDENCE GATE below
    for why that gate currently blocks almost everything -- that is the correct
    behaviour on today's data, not a bug.

    EVIDENCE GATE
    -------------
    Measured, replayed with strict no-leakage on the P1/P2 backtest: applying
    the shrunk correction unconditionally makes timing WORSE by 0.18 weeks
    (95% CI [-0.37, -0.00]), helps only 31% of individual cases, and degrades
    11 of 15 keywords. Signed mean barely moves (+1.50 -> +1.49).

    The reason is that there is not much systematic bias left to remove. An
    earlier version of this file was built on a measured +5.4 week lateness, but
    that number came from scoring the forecaster against actual peaks falling
    inside its own IGNORE_WEEKS_FROM_NOW blackout -- peaks it is deliberately
    forbidden to call. Once scoring only counted peaks the model was allowed to
    predict, the pooled signed mean fell to +1.50 and the pooled median to
    exactly 0.00, with per-keyword medians scattered between -4 and +5 and
    per-keyword std of 4-9 weeks at n~16. Those per-keyword tilts are mostly
    noise, and correcting by them injects more error than it removes.

    So the gate demands the bias look real before it is acted on:
      * enough observations (MIN_OBS_FOR_BIAS),
      * a median at least MIN_BIAS_WEEKS from zero -- smaller than that is not
        worth shifting a date over,
      * and a consistent sign in at least BIAS_SIGN_CONSISTENCY of the
        observations, so one big outlier cannot drag the median.

    Today that leaves the layer essentially inert, which is the honest outcome.
    It is kept rather than deleted because the ledger keeps accumulating: if a
    genuine bias develops -- or if enough data arrives to estimate it per lead
    time, where it is much more likely to be real -- the gate opens on its own
    with no code change. `bias_gate_passed` in the output makes the state
    visible rather than silent.
    """
    errs = _resolved_errors(led, keyword, as_of)
    n = len(errs)
    prior = pooled_bias(led, keyword=keyword, as_of=as_of)

    base = {"n_observations": n, "pooled_prior": round(prior, 1)}

    if not ENABLE_BIAS_CORRECTION:
        # Still report what the estimate WOULD be, so the ledger's view of the
        # model's accuracy stays visible on the dashboard even while the
        # correction itself is disabled.
        med = statistics.median(errs) if errs else None
        return {**base, "correction_weeks": 0.0, "source": "disabled",
                "raw_median": round(med, 1) if med is not None else None,
                "bias_gate_passed": False}

    if n < MIN_OBS_FOR_BIAS:
        return {**base, "correction_weeks": 0.0, "source": "insufficient_history",
                "raw_median": None, "bias_gate_passed": False}

    med = statistics.median(errs)
    shrunk = (n * med + SHRINK_PRIOR_WEIGHT * prior) / (n + SHRINK_PRIOR_WEIGHT)

    same_sign = sum(1 for e in errs if (e > 0) == (med > 0) and e != 0)
    consistency = same_sign / n if n else 0.0

    gate_passed = (abs(med) >= MIN_BIAS_WEEKS and consistency >= BIAS_SIGN_CONSISTENCY)

    if not gate_passed:
        return {**base, "correction_weeks": 0.0, "source": "bias_not_established",
                "raw_median": round(med, 1), "sign_consistency": round(consistency, 2),
                "would_have_shifted": round(shrunk, 1), "bias_gate_passed": False}

    capped = max(-MAX_ABS_CORRECTION_WK, min(MAX_ABS_CORRECTION_WK, shrunk))
    return {**base, "correction_weeks": round(capped, 1),
            "source": "shrunk_per_keyword", "raw_median": round(med, 1),
            "sign_consistency": round(consistency, 2),
            "shrinkage_factor": round(n / (n + SHRINK_PRIOR_WEIGHT), 3),
            "bias_gate_passed": True}


def timing_quantiles(led, keyword, as_of=None, correction_weeks=0.0):
    """Empirical p05/p95 of the signed timing error, for the P4 marketing window.

    `correction_weeks` is subtracted first so the quantiles describe what is
    left AFTER bias correction -- otherwise the bias gets counted twice, once
    in the shifted peak date and again in the window width.
    """
    errs = [e - correction_weeks for e in _resolved_errors(led, keyword, as_of)]
    n = len(errs)
    if n < MIN_OBS_FOR_BIAS:
        return {"n": n}
    errs.sort()

    def q(p):
        if n == 1:
            return errs[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return errs[lo] + (errs[hi] - errs[lo]) * (pos - lo)

    return {"n": n, "p05": round(q(0.05), 1), "p50": round(q(0.50), 1),
            "p95": round(q(0.95), 1)}


def apply_bias_correction(spike_windows, bias):
    """Shift each predicted peak (and its window) by the learned bias.

    Both the raw and corrected dates are kept in the output so the layer's
    effect is always auditable rather than silently baked in.
    """
    shift = bias["correction_weeks"]
    for w in spike_windows:
        w["peak_date_raw"] = w["peak_date"]
        if shift:
            for field in ("peak_date", "start", "end"):
                d = date.fromisoformat(w[field])
                w[field] = (d - timedelta(weeks=shift)).isoformat()
        w["bias_correction_weeks"] = shift
        w["bias_correction_source"] = bias["source"]
        w["bias_correction_n"] = bias["n_observations"]
    return spike_windows


def _weeks_between(a, b):
    da = datetime.fromisoformat(a).date() if isinstance(a, str) else a
    db = datetime.fromisoformat(b).date() if isinstance(b, str) else b
    return round((db - da).days / 7.0, 1)
