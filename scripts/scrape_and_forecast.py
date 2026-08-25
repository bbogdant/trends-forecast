"""
End-to-end: call the Apify "agenscrape/google-trends-scraper" actor for a list
of keywords, then run each result through the same Prophet forecast pipeline
as compute_forecast.py, and write one <key>_dashboard_data.json per keyword
(ready to be embedded into trend_dashboard.html's APPS object).

Requires an Apify API token in the APIFY_TOKEN environment variable — never
hardcode it in this file. Get one at https://console.apify.com/account/integrations.

Usage:
    export APIFY_TOKEN="your_token_here"
    python3 scrape_and_forecast.py "ATI TEAS" "CNA Exam" "NCLEX Exam" ...

Each keyword becomes its own output file named after a slugified key
(lowercase, spaces -> underscores), e.g. "ATI TEAS" -> ati_teas_dashboard_data.json
"""
import sys, os, json, time, re, statistics, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from datetime import datetime, timedelta
import urllib.request
import urllib.error

# Make sure compute_forecast.py (same folder) is importable regardless of the
# working directory the script is invoked from (e.g. repo root in CI).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_forecast import build_forecast_payload, DEFAULT_LEDGER_PATH
import peak_ledger

ACTOR = "agenscrape~google-trends-scraper"
API_BASE = "https://api.apify.com/v2"
POLL_INTERVAL_SECS = 10
MAX_WAIT_SECS = 900  # 15 minutes ceiling for a batch run


def slugify(keyword):
    return re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")


def _http_json(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Apify API error {e.code}: {e.read().decode()[:500]}")


def run_actor(token, keywords, geo="US", time_range="today 5-y"):
    input_payload = {
        "keywords": keywords,
        "geo": geo,
        "timeRange": time_range,
        "includeRelatedSearches": True,
        "includeRelatedTopics": False,   # not used by our pipeline, skip to save time/cost
        "includeGeoData": False,         # not used by our pipeline, skip to save time/cost
        "includeInterestOverTime": True,
    }
    start_url = f"{API_BASE}/acts/{ACTOR}/runs?token={token}"
    run = _http_json(start_url, method="POST", body=input_payload)["data"]
    run_id = run["id"]
    dataset_id = run["defaultDatasetId"]
    print(f"Started actor run {run_id} for {len(keywords)} keyword(s)...")

    waited = 0
    status = run["status"]
    while status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
        time.sleep(POLL_INTERVAL_SECS)
        waited += POLL_INTERVAL_SECS
        status = _http_json(f"{API_BASE}/actor-runs/{run_id}?token={token}")["data"]["status"]
        print(f"  [{waited}s] status: {status}")
        if waited >= MAX_WAIT_SECS:
            raise RuntimeError(f"Actor run {run_id} did not finish within {MAX_WAIT_SECS}s (last status: {status})")

    if status != "SUCCEEDED":
        raise RuntimeError(f"Actor run {run_id} ended with status {status}")

    items = _http_json(f"{API_BASE}/datasets/{dataset_id}/items?token={token}&format=json")
    return items


def process_entry(entry, app_label=None, ledger=None, ledger_path=None, slug=None):
    """Parse one Apify dataset item and run it through the shared pipeline.

    The forecasting logic itself lives in compute_forecast.build_forecast_payload
    so this entrypoint and the manual one cannot drift apart -- they did once
    before, which is how the marketing-window feature shipped without actually
    reaching the weekly cron output.
    """
    keyword = entry["keyword"]
    points = []
    for p in entry.get("interestOverTime", []):
        # accept either an epoch-seconds "time" field or an ISO "formattedTime"/"date"
        if "time" in p:
            dt = datetime.fromtimestamp(int(p["time"]), tz=None)
        else:
            dt = datetime.fromisoformat(p["date"])
        points.append({"date": dt.date().isoformat(), "value": p["value"], "partial": p.get("isPartial", False)})

    breakouts = [r["query"] for r in entry.get("relatedSearches", {}).get("rising", [])
                 if r.get("formattedValue") == "Breakout"]

    out = build_forecast_payload(keyword, points, breakouts, app_label=app_label,
                                 ledger=ledger, ledger_path=ledger_path, slug=slug)
    out.pop("_train_residuals", None)   # not used by the dashboard; keeps files smaller
    return out


# Dropdown label shown on the dashboard per app -- distinct from the actual
# Google Trends search phrase (the "winner_keyword"), which is chosen purely
# for forecast/correlation quality and can read oddly if title-cased raw
# (e.g. "Aapc Cpc"). Keyed by keyword.lower() so it survives case variations
# in how the keyword is passed on the command line. Finalized via
# lag-correlation testing against real revenue, Aug 2026 -- see
# keyword_research/top15_compare_groups.csv for the reasoning per app.
APP_LABELS = {
    "teas test": "ATI TEAS",
    "comptia certification": "CompTIA",
    "aswb exam": "ASWB",
    "emt certification": "EMT",
    "ancc certification": "ANCC",
    "cna test": "CNA",
    "hesi exam": "HESI A2",
    "pharmacy tech certification": "PTCB",
    "nclex exam": "NCLEX",
    "servsafe exam": "ServSafe",
    "aapc cpc": "AAPC CPC",
    "bcen certification": "BCEN",
    "real estate exam": "Real Estate",
    "ccrn exam": "CCRN",
    "pmp exam": "PMP",
}


def main():
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        print("Set APIFY_TOKEN as an environment variable first (never hardcode it in this file).")
        sys.exit(1)

    # Output directory: defaults to docs/data so this drops straight into the
    # GitHub Pages folder when run from the repo root (see .github/workflows/
    # weekly-trends.yml). Override with OUTPUT_DIR for local/manual runs.
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join("docs", "data"))
    os.makedirs(out_dir, exist_ok=True)

    keywords = sys.argv[1:]
    if not keywords:
        print('Usage: python3 scrape_and_forecast.py "ATI TEAS" "CNA Exam" ...')
        sys.exit(1)

    # One shared ledger across all keywords, loaded once and saved once at the
    # end, so a mid-run failure cannot leave it half-written.
    ledger_path = os.environ.get("PEAK_LEDGER", DEFAULT_LEDGER_PATH)
    ledger = peak_ledger.load_ledger(ledger_path)
    n_before = len(ledger["entries"])

    items = run_actor(token, keywords)
    print(f"\nGot {len(items)} dataset item(s) back from Apify.")

    by_keyword = {item["keyword"].lower(): item for item in items}
    for kw in keywords:
        entry = by_keyword.get(kw.lower())
        if entry is None:
            print(f"  WARNING: no result returned for '{kw}', skipping.")
            continue
        key = slugify(kw)
        output = process_entry(entry, app_label=APP_LABELS.get(kw.lower(), kw.title()),
                               ledger=ledger, ledger_path=None, slug=key)
        out_path = os.path.join(out_dir, f"{key}_dashboard_data.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        o = output.get("surge_outlook")
        sig = (f"{o['tier']}/{o['state']}" if o else f"no-signal ({output['no_signal_reason']})")
        print(f"  {key}: wrote {out_path}  (baseline={output['recent_baseline']}, "
              f"seas_ref={output['seasonal_reference']}, yoy={output['yoy_growth_pct']}%, "
              f"windows={len(output['spike_windows'])}, signal={sig})")

    peak_ledger.save_ledger(ledger, ledger_path)
    resolved = sum(1 for e in ledger["entries"] if e.get("resolved_on"))
    print(f"\nLedger {ledger_path}: {len(ledger['entries'])} entries "
          f"(+{len(ledger['entries']) - n_before} new this run), {resolved} resolved.")


if __name__ == "__main__":
    main()
