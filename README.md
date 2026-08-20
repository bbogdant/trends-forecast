# Exam-Prep Trends Dashboard

Google Trends demand forecasting (Prophet, logistic growth) for 15 exam-prep
keywords, refreshed weekly via GitHub Actions and published with GitHub Pages.

## How it works

1. **Every Monday 06:00 UTC**, `.github/workflows/weekly-trends.yml` runs
   `scripts/scrape_and_forecast.py`, which:
   - calls the Apify `agenscrape/google-trends-scraper` actor for all 15
     tracked keywords (5-year weekly window, US),
   - fits a Prophet forecast (logistic growth, capped at 0-100) per keyword,
   - detects predicted spike windows and historical anomalies,
   - writes one `docs/data/<key>_dashboard_data.json` per keyword.
2. The workflow commits and pushes the updated JSON files.
3. GitHub Pages serves `docs/index.html`, which `fetch()`es the JSON files
   in `docs/data/` at load time — so the page itself never needs to change,
   only the data files do.

## One-time setup

1. **Add the Apify token as a secret**: repo Settings → Secrets and
   variables → Actions → New repository secret → name `APIFY_TOKEN`, value
   = your Apify API token. Never commit this token to the repo.
2. **Enable GitHub Pages**: repo Settings → Pages → Build and deployment →
   Source: "Deploy from a branch" → Branch: `main`, folder: `/docs`.
3. **Run once manually** to seed the data (optional — `docs/data/` already
   ships with a snapshot from 2026-08-09): Actions tab → "Weekly Trends
   Refresh" → Run workflow.

## Adding a new keyword

Add it to the keyword list in `.github/workflows/weekly-trends.yml`'s
`Run scrape + forecast` step, and to `APP_KEYS` near the top of the
`<script>` block in `docs/index.html` (slugified: lowercase, spaces → `_`).

## Local / manual run

```
export APIFY_TOKEN="your_token_here"
pip install -r requirements.txt
python scripts/scrape_and_forecast.py "ATI TEAS" "CNA Exam"
```

Writes into `docs/data/` by default (override with `OUTPUT_DIR`).

## Access control

This repo can be private, but a **GitHub Pages site published from it is
still public to anyone with the URL** — private repo != private site,
except on GitHub Enterprise Cloud. If the dashboard should only be visible
to your team, see the options discussed with Claude (Cloudflare Access in
front of the Pages URL, or Cloudflare Pages + Cloudflare Access) before
sharing the link widely.
