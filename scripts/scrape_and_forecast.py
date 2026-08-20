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
from compute_forecast import historical_anomalies, fit_prophet_forecast, detect_spike_windows

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


def process_entry(entry, app_label=None):
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

    complete = [p for p in points if not p["partial"]]
    last_partial = points[-1] if points and points[-1]["partial"] else None

    recent_baseline = statistics.mean(p["value"] for p in complete[-8:])
    last_date = datetime.fromisoformat(complete[-1]["date"])
    one_year_ago = last_date - timedelta(days=365)
    prior_window = [p["value"] for p in complete if abs((datetime.fromisoformat(p["date"]) - one_year_ago).days) <= 28]
    prior_mean = statistics.mean(prior_window) if prior_window else 0.0
    yoy_growth = ((recent_baseline - prior_mean) / prior_mean) if prior_mean > 0 else 0.0

    anomalies = historical_anomalies(complete)
    forecast, residuals, cap = fit_prophet_forecast(complete)
    spike_windows = detect_spike_windows(forecast, recent_baseline, last_date)

    return {
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

    items = run_actor(token, keywords)
    print(f"\nGot {len(items)} dataset item(s) back from Apify.")

    by_keyword = {item["keyword"].lower(): item for item in items}
    for kw in keywords:
        entry = by_keyword.get(kw.lower())
        if entry is None:
            print(f"  WARNING: no result returned for '{kw}', skipping.")
            continue
        output = process_entry(entry, app_label=kw.title())
        key = slugify(kw)
        out_path = os.path.join(out_dir, f"{key}_dashboard_data.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  {key}: wrote {out_path}  (baseline={output['recent_baseline']}, "
              f"yoy={output['yoy_growth_pct']}%, spike_windows={len(output['spike_windows'])})")


if __name__ == "__main__":
    main()
