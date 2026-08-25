"""
Trend-robust definition of a "real seasonal surge".

THE PROBLEM BEING FIXED
-----------------------
Defining a surge as "X% above the trailing annual median" conflates two very
different things: a keyword whose whole level is drifting upward, and a keyword
that genuinely spikes above its own current baseline. Measured on the 15 live
keywords, the number of events found by that definition correlates r=0.79 with
how fast the keyword is growing. Consequences ran both ways: fast-growing
keywords (ccrn_exam, emt_certification, aapc_cpc) were credited with many
"surges" that a plain linear trend predicts trivially, which is why teas_test
scored a suspicious 1.00/1.00; flat keywords (cna_test) produced ZERO events in
five years despite having visible seasonality.

THE FIX
-------
Estimate the slow trend explicitly and measure the surge against it:

  1. Fit a log-linear trend to the last TREND_WINDOW weeks (train only). Log
     because growth here is multiplicative, and the amplitude of seasonal peaks
     scales with the level. Two full years of window means seasonality averages
     out of the slope estimate instead of being absorbed into it.
  2. Extrapolate that trend forward across the short label horizon. Because the
     horizon is only a few weeks, extrapolation is safe -- and crucially the
     surge cannot inflate its own baseline, which is what a trailing median
     lets it do.
  3. Detrended ratio = y / trend_hat. A surge is a local maximum where that
     ratio clears SURGE_RATIO and holds for SUSTAIN weeks.

Both the actual series and the forecast are detrended by the SAME train-only
trend, so the label and the signal are measured on the same footing.

A DELIBERATE CHOICE ABOUT RECURRENCE
------------------------------------
The event is defined purely on trend-robust magnitude -- NOT on whether it
recurs annually. Defining the event as "the surge that repeats every year" and
then asking whether a seasonal model can predict it would partly bake in the
answer and inflate apparent skill. Recurrence is instead reported separately as
an explanation of WHY some keywords turn out predictable.
"""
import numpy as np

TREND_WINDOW = 104      # weeks of history for the log-linear trend fit (2 full years)
SURGE_RATIO = 1.20      # detrended level that counts as a surge
SUSTAIN = 2             # weeks it must hold
PEAK_RADIUS = 3         # local-max radius
EPS = 1.0               # guards log() on zero-valued weeks
GT_SCALE_MAX = 100.0    # Google Trends index ceiling -- the trend cannot exceed it
MAX_EXTRAP_WEEKS = 26   # freeze trend drift past this horizon (see trend_at)


def fit_log_trend(v, window=TREND_WINDOW):
    """Robust log-linear trend on the last `window` observations. TRAIN ONLY.

    Returns (intercept, slope) in log space, indexed so x=0 is the LAST training
    point, which makes forward extrapolation just x=1,2,3...

    Two robustness details that turned out to matter a lot:

    * Zero and near-zero weeks are EXCLUDED from the fit. log(0) has to be
      clamped to log(EPS), which lands enormously far below the rest of the
      series, and ordinary least squares chases it. On ccrn_exam (11.9% zero
      weeks) that dragged the fitted trend so far under the real level that the
      MEDIAN week scored a detrended ratio of 1.24 -- above the surge threshold
      -- so 53% of all weeks registered as surges.

    * The slope is estimated by Theil-Sen (median of pairwise slopes) rather
      than least squares, so a handful of extreme weeks cannot tilt it.

    Sanity check for any change here: the median detrended ratio must come out
    near 1.0 for EVERY keyword. If it does not, the trend fit is wrong for that
    keyword and every surge count built on it is meaningless.
    """
    v = np.asarray(v, dtype=float)
    seg = v[-window:] if len(v) >= window else v
    n_all = len(seg)
    if n_all < 26:
        pos = seg[seg > 0]
        return np.log(max(np.median(pos) if len(pos) else EPS, EPS)), 0.0

    x_all = np.arange(n_all) - (n_all - 1)      # last point at x=0
    keep = seg > 0
    if keep.sum() < 20:                          # too sparse to fit a trend at all
        pos = seg[keep]
        return np.log(max(np.median(pos) if len(pos) else EPS, EPS)), 0.0

    x = x_all[keep].astype(float)
    y = np.log(seg[keep])

    # Theil-Sen slope: median of pairwise slopes, subsampled for speed when long
    m = len(x)
    if m <= 60:
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    else:
        step = max(1, m // 60)
        idx = list(range(0, m, step))
        pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i, j in pairs if x[j] != x[i]]
    slope = float(np.median(slopes)) if slopes else 0.0

    # intercept anchored so the fit passes through the median residual level
    intercept = float(np.median(y - slope * x))
    return intercept, slope


def trend_at(intercept, slope, steps_ahead):
    """Extrapolated trend level `steps_ahead` weeks past the last training point.
    steps_ahead=0 reproduces the fitted level at the training edge.

    Two bounds on the extrapolation, both of which matter in production where
    the forecast horizon is a full year:

    * Hard cap at GT_SCALE_MAX. A Google Trends index cannot exceed 100, so a
      trend line above it is not a possible level -- it is an artifact of
      projecting a slope past the top of the scale. Left uncapped, nclex_exam's
      trend reached 106.5 by week 52, which crushed its detrended ratio to 0.82
      and made every surge in the second half of the forecast undetectable. Same
      for cna_test (100.1) and ancc_certification (94.5, close enough to bite).

    * Drift frozen after MAX_EXTRAP_WEEKS. Beyond about two quarters a growth
      extrapolation on this index is not credible, partly because Trends
      renormalises the whole series to 100 at its running maximum: a keyword
      that just set an all-time high has all its earlier values rescaled
      downward, which manufactures apparent growth that will not continue. This
      one is a judgement call rather than an arithmetic necessity -- it is the
      difference between asking "is this high for where the level will be in a
      year" (which decays the test with horizon, backwards for our purpose) and
      "is this high for roughly where the level is now".
    """
    steps = min(steps_ahead, MAX_EXTRAP_WEEKS)
    return float(min(np.exp(intercept + slope * steps), GT_SCALE_MAX))


def detrend_forward(values, intercept, slope, start_step=1):
    """Divide a forward series by the extrapolated trend, week by week."""
    return np.array([v / trend_at(intercept, slope, start_step + i)
                     for i, v in enumerate(values)], dtype=float)


def surge_indices(ratio, surge_ratio=SURGE_RATIO, sustain=SUSTAIN,
                  radius=PEAK_RADIUS):
    """Local maxima in a DETRENDED ratio series that clear the surge threshold
    and hold it for `sustain` weeks."""
    r = np.asarray(ratio, dtype=float)
    n = len(r)
    cand = [i for i in range(n)
            if r[i] == max(r[max(0, i-radius):min(n, i+radius+1)])
            and r[i] >= surge_ratio]
    if sustain > 1:
        cand = [i for i in cand
                if sum(1 for j in range(max(0, i-sustain+1), min(n, i+sustain))
                       if r[j] >= surge_ratio) >= sustain]
    ded = []
    for i in cand:
        if ded and i - ded[-1] <= radius:
            if r[i] > r[ded[-1]]:
                ded[-1] = i
        else:
            ded.append(i)
    return ded


def seasonal_recurrence(hist_dates, hist_values, woy_target, tol=3):
    """Fraction of prior years in which `woy_target` was elevated after
    detrending. Diagnostic only -- never part of the event definition."""
    import pandas as pd
    s = pd.Series(hist_values, index=pd.to_datetime(hist_dates))
    ic = s.index.isocalendar()
    df = pd.DataFrame({"y": s.values, "woy": ic.week.values, "yr": ic.year.values})
    hits = tot = 0
    for yr, g in df.groupby("yr"):
        if len(g) < 40:
            continue
        tot += 1
        inw = g[(g.woy >= woy_target - tol) & (g.woy <= woy_target + tol)]
        rest = g[(g.woy < woy_target - tol) | (g.woy > woy_target + tol)]
        if len(inw) and len(rest) and inw.y.mean() >= 1.15 * rest.y.median():
            hits += 1
    return (hits / tot) if tot else 0.0
