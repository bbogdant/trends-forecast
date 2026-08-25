"""
Per-keyword reliability tiers for the surge early-warning signal.

Every number here was measured, not chosen. Protocol: rolling-origin backtest
on the trend-robust surge definition (scripts/surge_def.py), weekly origins,
2 years minimum training, 149 origins per keyword, everything fitted train-only.
"Coverage" is how often the ACTUAL next surge fell inside the announced window,
given the model announced one and a surge occurred.

WHY TIERS AT ALL
----------------
Skill is not uniform across the portfolio and pretending otherwise would put a
confident-looking window on apps where it is worthless. Measured coverage with a
4-week window ran from 98% (hesi_exam) down to 10% (aapc_cpc), and five keywords
never produce a forecast surge at all -- their Prophet seasonal component never
clears the surge threshold, so there is nothing to bracket:

    pmp_exam 0.0% of origins, emt_certification 0.0%,
    pharmacy_tech_certification 3.1%, comptia_certification 4.7%,
    aswb_exam 11.8%

Those are not tuning failures. Those keywords are professional certifications
taken on demand year-round; the seasonal surge marketing wants to front-run does
not exist in the series. The three strongest keywords (TEAS, HESI, NCLEX) are
nursing-school entrance and licensure exams tied to rigid academic cohorts.

WHY WINDOWS AND NOT DATES
-------------------------
Binary "surge within 2-3 weeks?" detection maxed out at recall 0.22-0.24. The
signed timing error has a tight core (p25 = 0, p75 = +2 weeks, median exactly 0)
with fat tails (p05 = -7, p95 = +9), so a bracket around the predicted week
captures most of the mass while a point estimate does not. Aggregate: a 4-week
window covers 72% with a median 3-week lead; 6 weeks covers 77% with a 4-week
lead and clears 2 weeks of lead in 97% of cases.

CAVEAT THAT MATTERS
-------------------
Each keyword's numbers rest on 3-9 distinct surge episodes across five years of
history -- origin-weeks are not independent, since one episode spans several
consecutive origins. Treat these as provisional and re-derive them as
state/peak_ledger.json accumulates:
    python3 scripts/backtest_peaks.py --from-dashboard-data docs/data
"""

# tier -> window geometry. `before`/`after` are weeks around the predicted peak.
TIER_WINDOWS = {
    "tier1": {"before": 3, "after": 1},   # 4-week bracket
    "tier2": {"before": 4, "after": 2},   # 6-week bracket
}

# Measured per keyword. `coverage` and `median_lead_weeks` are for that
# keyword's assigned tier width, `signal_rate` is the share of origins on which
# the model announced anything at all.
KEYWORD_TIERS = {
    # --- tier 1: 4-week window, coverage >= 95%, median timing error 0.0 weeks
    "teas_test": {
        "tier": "tier1", "coverage": 0.97, "median_lead_weeks": 3,
        "signal_rate": 0.583, "surge_episodes": 3,
        "note": "median timing error 0.0wk; 100% of hits gave >=2wk lead",
    },
    "hesi_exam": {
        "tier": "tier1", "coverage": 0.98, "median_lead_weeks": 3,
        "signal_rate": 0.850, "surge_episodes": 5,
        "note": "highest signal rate in the portfolio; median timing error 0.0wk",
    },

    # --- tier 2: 6-week window, coverage >= 75%
    "ancc_certification": {
        "tier": "tier2", "coverage": 0.92, "median_lead_weeks": 6,
        "signal_rate": 0.197, "surge_episodes": 6,
        "note": "collapses to 32% on a 4-week window -- needs the wider bracket; "
                "speaks rarely but is reliable when it does",
    },
    "nclex_exam": {
        "tier": "tier2", "coverage": 0.80, "median_lead_weeks": 3,
        "signal_rate": 0.630, "surge_episodes": 7,
        "note": "median timing error 2.0wk, hence tier 2 rather than tier 1",
    },
    "cna_test": {
        "tier": "tier2", "coverage": 0.78, "median_lead_weeks": 4,
        "signal_rate": 0.181, "surge_episodes": 1,
        "note": "only 1 clean surge episode in five years -- weakest evidence in "
                "tier 2, revisit as the ledger grows",
    },
}

# Explicitly not enabled, with the measured reason. Kept in code so the
# dashboard can say WHY there is no signal instead of rendering an empty box.
NO_SIGNAL_REASONS = {
    "real_estate_exam": "window coverage only 41% (median timing error 7wk)",
    "aapc_cpc": "window coverage 10-15% (median timing error 9wk)",
    "servsafe_exam": "window coverage 11% (median timing error 11wk)",
    "ccrn_exam": "no measurable early-warning skill in backtest",
    "aswb_exam": "model produces a forecast surge on only 11.8% of weeks",
    "comptia_certification": "model produces a forecast surge on only 4.7% of weeks",
    "pharmacy_tech_certification": "model produces a forecast surge on only 3.1% of weeks",
    "emt_certification": "model never produces a forecast surge (0% of weeks)",
    "pmp_exam": "model never produces a forecast surge (0% of weeks)",
    "bcen_certification": "98% of weeks are zero -- series is not forecastable",
}


def tier_for(slug):
    """Returns (tier_name, window_spec, meta) or (None, None, reason)."""
    info = KEYWORD_TIERS.get(slug)
    if info:
        return info["tier"], TIER_WINDOWS[info["tier"]], info
    return None, None, NO_SIGNAL_REASONS.get(
        slug, "not yet evaluated in the early-warning backtest")
