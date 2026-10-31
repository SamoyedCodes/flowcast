# Challenge 5: Predictive Airport Capacity & Delay-Risk Model (ATL, DFW)

Forecasts, 3 to 12 hours ahead,
1. the hourly **Arrival Acceptance Rate (AAR)**, scored by RMSE,
2. whether a flight will suffer a **severe delay (> 45 min)**, with a calibrated probability,

and shows both to an air-traffic duty manager in a dashboard with history and plain-language recommendations.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py                 # download -> clean -> features -> train -> evaluate (cached after first run)
python -m pytest -q                    # leakage / physics / data-integrity checks
streamlit run dashboard/app.py         # the ATC dashboard
python run_pipeline.py --sensitivity   # optional: re-test the AAR assumptions (about 8 min)
```

Inputs: `data/clean/flights_clean.csv` (organiser flight data, already cleaned) and public weather that the
pipeline downloads by itself from the Iowa State Mesonet archive: METAR observations and **TAF forecasts**
(2023-01 to 2025-07). No manual steps.

## What is in the repo

| Path | Purpose |
|---|---|
| `src/config.py` | Airports (ICAO, time zone, runway heading, FAA AAR table), time split, leads. Add an airport here and the pipeline picks it up |
| `src/download.py` | Cached METAR + TAF downloads (retry, splits failed requests) |
| `src/weather.py` | METAR cleaning, runway head/cross/tailwind from the runway heading, hourly table, weather-adjusted AAR |
| `src/aar.py` | The AAR rule (FAA called rates by flight category + hazard reductions) |
| `src/taf.py` | Parses TAFs; for each target hour reads the newest TAF issued *before* the forecast time |
| `src/features.py` | AAR table and delay table (leak-free joins, recent hub performance) |
| `src/train_aar.py` | LightGBM AAR forecaster + baselines, RMSE, event detection, unseen-airport test |
| `src/train_delay.py` | Severe-delay classifier: picks the model family out-of-fold (regularised logistic regression beat LightGBM), ablations, calibration, bootstrap, failure cases |
| `src/sensitivity.py` | Does the conclusion survive different AAR assumptions? |
| `src/advisory.py` | Rule-based recommendations for the controller |
| `dashboard/app.py` | Streamlit decision-support dashboard (replays the held-out 2025 period) |
| `tests/` | Invariant tests (no look-ahead, wind geometry, AAR rule, chronological splits) |
| `reports/` | All metrics as CSV |
| `presentation/` | `Challenge5_Capacity_and_Delay_Risk.pptx` (14 slides, built by `build_deck.py` from `reports/`) and `LEARNING_BRIEF.md` (what was built and why, with Q&A prep) |

`weather_download.ipynb` and `data_cleaning.ipynb` are the earlier exploratory notebooks; the scripts above supersede them.

## Design decisions in one place

* **Leakage:** every weather feature comes from the latest METAR hour already *published* at `issue time = event - lead`; TAFs must have been issued at least `lead` hours before the target hour; "recent hub performance" only counts flights already completed at the issue time. Enforced by assertions and `tests/`.
* **Runway directionality:** wind is converted to head/tail/crosswind with the assigned runway headings (ATL 80, DFW 180), and the AAR rule penalises crosswind and tailwind, not raw wind speed.
* **Not tied to one airport:** the AAR target is a ratio to the airport's best-case rate, the model has no airport-ID feature, and an unseen-airport test is reported. Unknown runway heading falls back to the prevailing wind direction.
* **Normalised delays:** delay is a rate (share of flights > 45 min), never raw minutes; results are shown per airport and per direction.
* **Time-based evaluation:** train 2023-01..2024-06, validation 2024-07..12, test 2025-01..07, never shuffled. The delay model uses expanding-window cross-validation over 2024 for hyper-parameters and calibration.
* **AAR ground truth:** no public dataset provides the called hourly AAR, so the target is derived from FAA-published called rates by weather category plus documented hazard reductions. This is the biggest assumption, so it is validated against real flights and stress-tested (`reports/aar_sensitivity.csv`).
