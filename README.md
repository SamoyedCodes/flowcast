# Challenge 5: Predictive Airport Capacity & Delay-Risk Model (ATL, DFW)

Forecasts, 3 to 12 hours ahead,
1. the hourly **Arrival Acceptance Rate (AAR)**, scored by RMSE,
2. whether a flight will suffer a **severe delay (> 45 min)**, with a calibrated probability,

and shows both to an air-traffic duty manager in a dashboard with history and plain-language recommendations.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py                 # download -> clean -> features -> train -> evaluate (cached after first run)
python run_pipeline.py --skip-download # offline replay from the verified cache
python -m pytest -q tests test_stage1_pipeline.py   # leakage / physics / data-integrity checks
streamlit run dashboard/app.py         # the ATC dashboard
python run_pipeline.py --sensitivity   # optional: re-test the AAR assumptions (about 8 min)
```

Inputs: `data/clean/flights_clean.csv` (organiser flight data, already cleaned) and public data the pipeline
downloads by itself (2023-01 to 2025-07, no manual steps):

* METAR observations from the Iowa State Mesonet (`data/weather/raw/`)
* **TAF forecasts** with their issuance/validity metadata from the Iowa State Mesonet, **BTS reporting-carrier
  on-time archives** and OpenFlights airport time zones, fetched by `stage1_pipeline.py` into `data/external/`.
  The first run downloads about 1 GB. Every file is validated and gets a `.source.json` receipt with its URL and
  SHA-256; later runs reuse the cache, and `--skip-download` refuses to run if anything is missing or changed.
  `data/external/` is not committed.

## What is in the repo

| Path | Purpose |
|---|---|
| `src/config.py` | Airports (ICAO, time zone, runway heading, FAA AAR table), time split, leads. Add an airport here and the pipeline picks it up |
| `src/download.py` | Cached METAR downloads (retry) |
| `stage1_pipeline.py` | Verified TAF / TAF-metadata / BTS / airport downloads, TAF validity normalisation, BTS airport-traffic features (tests: `test_stage1_pipeline.py`) |
| `src/weather.py` | METAR cleaning, runway head/cross/tailwind from the runway heading, hourly table, weather-adjusted AAR |
| `src/aar.py` | The AAR rule (FAA called rates by flight category + hazard reductions) |
| `src/taf.py` | Parses TAFs; for each target hour reads the newest TAF issued *before* the forecast time, if it is valid for that hour |
| `src/features.py` | AAR table and delay table (leak-free joins, recent hub performance, BTS airport traffic) |
| `src/train_aar.py` | LightGBM AAR forecaster + baselines, RMSE, event detection, unseen-airport test |
| `src/train_delay.py` | Severe-delay classifier: picks the model family out-of-fold (regularised logistic regression beat LightGBM), ablations, calibration, bootstrap, failure cases |
| `src/sensitivity.py` | Does the conclusion survive different AAR assumptions? Does the AAR rule agree with real BTS arrivals? |
| `src/advisory.py` | Rule-based recommendations for the controller |
| `dashboard/app.py` | Streamlit decision-support dashboard (replays the held-out 2025 period) |
| `tests/` | Invariant tests (no look-ahead, wind geometry, AAR rule, chronological splits, TAF validity, stale weather, purging) |
| `reports/` | All metrics as CSV |
| `presentation/` | `Challenge5_Capacity_and_Delay_Risk.pptx` (14 slides, built by `build_deck.py` from `reports/`) and `LEARNING_BRIEF.md` (what was built and why, with Q&A prep) |

`weather_download.ipynb` and `data_cleaning.ipynb` are the earlier exploratory notebooks; the scripts above supersede them.

## Design decisions in one place

* **Leakage:** every weather feature comes from the latest METAR hour already *published* at `issue time = event - lead`; TAFs must have been issued at least `lead` hours before the target hour; "recent hub performance" and BTS traffic only count flights already completed at the issue time. Enforced by assertions and `tests/`.
* **TAF validity:** the newest TAF issued before the cutoff must also be valid for the target hour, using the Mesonet issuance/validity metadata. An expired, not-yet-valid or malformed newest TAF gives missing values; there is no fallback to an older forecast.
* **Stale or missing weather stays unknown:** if the last real METAR is more than 2 h old at the issue time, the row is kept but its weather is blank (`wx_stale = 1`). "Not reported" defaults (no ceiling = 12,000 ft, no gust/precipitation = 0) apply only to hours that had a report. Local times that daylight saving makes ambiguous or nonexistent are dropped, not shifted.
* **Airport traffic:** BTS reporting-carrier archives give scheduled movements in the event hour and operations completed in the 3 h before the issue time. BTS is a demand proxy, not all traffic and not AAR; final schedules are retrospective.
* **Runway directionality:** wind is converted to head/tail/crosswind with the assigned runway headings (ATL 80, DFW 180), and the AAR rule penalises crosswind and tailwind, not raw wind speed.
* **Not tied to one airport:** the AAR target is a ratio to the airport's best-case rate, the model has no airport-ID feature, and an unseen-airport test is reported. Unknown runway heading falls back to the prevailing wind direction.
* **Normalised delays:** delay is a rate (share of flights > 45 min), never raw minutes; results are shown per airport and per direction.
* **Time-based evaluation:** train 2023-01..2024-06, validation 2024-07..12, test 2025-01..07, never shuffled. The first day of 2025 is purged, as is any flight whose departure and arrival fall in different splits. The delay model uses expanding-window cross-validation over 2024 for hyper-parameters; Platt calibration is kept only if a fit on 2024 Q1-Q3 lowers the Brier score on 2024 Q4.
* **AAR ground truth:** no public dataset provides the called hourly AAR, so the target is derived from FAA-published called rates by weather category plus documented hazard reductions. This is the biggest assumption, so it is stress-tested (`reports/aar_sensitivity.csv`) and checked against real airport-wide BTS arrivals in busy hours (`reports/aar_vs_traffic.csv`). The direction holds: in the busiest quarter of hours, the share of scheduled arrivals that actually land falls steadily as the rule's AAR falls (ATL 94% → 87%, DFW 81% → 71% from the top to the bottom AAR band). The rank correlation is weak (0.12 ATL, 0.17 DFW), and BTS demand (about 70 arrivals/h) sits well below the rule's VFR rates, so the rule's absolute level is not confirmed, only its direction.
