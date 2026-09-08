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

THE 20 KEYWORDS ADDED LATER (GED, ASVAB, LSAT, cosmetology, CCNA, firefighter,
praxis, SAT, personal trainer, ACT, ASE, CSCS, journeyman electrician, NCE,
phlebotomy, MBLEX, NREMT paramedic, SHRM, HiSET, plumber) were run through the
same trend-robust rolling-origin backtest (52 weekly origins each -- fewer than
the original 15's 149, because these had no pre-existing ledger history and the
scoring window needs 54 weeks of held-out future data past every origin). Only
cosmetology_exam cleared a tier threshold (tier2, 86% coverage on 22 fired
signals -- comparable evidence to nclex_exam's tier2 entry). The other 19 came
back below both the 95%/75% thresholds, several of them by a wide margin, and
two (mblex_exam, plumber_exam) are mostly-zero series like bcen_certification.
hiset_test came closest to tier2 (61% coverage) without clearing it.
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
    "cosmetology_exam": {
        "tier": "tier2", "coverage": 0.86, "median_lead_weeks": 8,
        "signal_rate": 0.423, "surge_episodes": 24,
        "note": "fires often (42% of origins) and has an unusually high full-"
                "history episode count (24) -- this series swings past the "
                "surge threshold much more frequently than the rest of the "
                "portfolio, so the 86% coverage rests on more repeat evidence "
                "than most tier-2 entries but the series itself is noisier",
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

    # --- the 20 keywords added later (see module docstring) ---
    "ged_test": "window coverage 14% (median timing error 13wk)",
    "asvab_test": "window coverage 0% (median timing error 44wk)",
    "lsat_test": "window coverage 17% (median timing error 9wk)",
    "ccna_certification": "no measurable early-warning skill in backtest",
    "firefighter_test": "model never produces a forecast surge (0% of weeks)",
    "praxis_test": "window coverage 48% (median timing error 7wk)",
    "sat_prep": "model produces a forecast surge on only 11.5% of weeks",
    "personal_trainer_certification": "window coverage 29% (median timing error 28wk)",
    "act_test": "window coverage 48% (median timing error 4wk); fires on every "
                "origin but the bracket is too tight for how far the peak actually lands",
    "ase_certification": "model never produces a forecast surge (0% of weeks)",
    "cscs_certification": "window coverage 0% (median timing error 22wk); fires on "
                "69% of origins but never inside the bracket -- timing, not detection, is the failure",
    "journeyman_electrician_test": "window coverage 24% (median timing error 23wk)",
    "nce_exam": "window coverage 0% (median timing error 18wk)",
    "phlebotomy_certification": "model produces a forecast surge on only 1.9% of weeks",
    "mblex_exam": "62% of weeks are zero -- series is not forecastable",
    "nremt_paramedic": "window coverage 53% (median timing error 0wk); close but "
                "below the 75% tier-2 bar",
    "shrm_certification": "window coverage 44% (median timing error 2wk); fires on "
                "every origin but under-clears the coverage bar",
    "hiset_test": "window coverage 61% (median timing error 4wk); closest of the 20 "
                "to clearing tier 2, revisit as more data accumulates",
    "plumber_exam": "72% of weeks are zero -- series is not forecastable",
}


def tier_for(slug):
    """Returns (tier_name, window_spec, meta) or (None, None, reason)."""
    info = KEYWORD_TIERS.get(slug)
    if info:
        return info["tier"], TIER_WINDOWS[info["tier"]], info
    return None, None, NO_SIGNAL_REASONS.get(
        slug, "not yet evaluated in the early-warning backtest")
