# AeroCast data pipeline (Airbus Challenge #5)

This pipeline pulls only the source types the challenge brief names: METAR/TAF weather logs, NOAA storm logs and historical flight performance data. Every download runs from a script, with no manual steps, and is logged for traceability.

## Sources (all public, no account needed)

| Source | What we take | Endpoint | Time basis |
|---|---|---|---|
| BTS Reporting Carrier On-Time Performance (1987-present) | every flight to/from the airports | `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip` | local clock times; converted to UTC here |
| METAR (routine + specials) via Iowa Environmental Mesonet | temp, dewpoint, **wind direction**, speed, gust, **visibility**, **cloud layers (ceiling)**, weather codes (TS, FZ, SN, FG), altimeter | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | UTC |
| TAF via IEM (decoded per forecast group) | forecast wind, gust, visibility, ceiling, TS, TEMPO/PROB groups | `https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py` | UTC; filtered by issuance time |
| NOAA NCEI Storm Events (details) | severe events within 50 km of each airport (plus zone events in the same state) | `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/` | local standard time, CZ_TIMEZONE; converted to UTC |

## Run

```bash
pip install pandas pyarrow requests airportsdata
python run_all.py --start 2023-01 --end 2025-07 --airports ATL DFW --provided Flight_Data_With_Weather_Final.csv
```

- **BTS download size:** about 25-35 MB per month, so the full 31-month window is roughly 1 GB. If the network is slow, start with `--start 2024-01`.
- **IEM rate limit:** 1 request per second, and the code already respects it.
- **Caching:** every file is cached in `data/raw/`, so re-runs are instant.
- **Provenance:** `data/manifest.jsonl` records the URL, retrieval time, byte count and SHA-256 of each file.
- **Other airports:** you can pass any airport code. To get wind components for it, add its runway heading to `airports.json`.

## Outputs (`data/processed/`)

| File | Grain | Use |
|---|---|---|
| `airport_hourly.parquet` | airport x UTC hour | scheduled vs actual arrivals and departures, `throughput_ratio`, `arrivals_norm` (divided by the airport's own 95th percentile, so ATL and DFW are comparable), severe-delay share |
| `aar_horizon.parquet` | airport x target hour x horizon (3/6/9/12 h) | **AAR model table**: features as of t0 = target − horizon, targets = actual arrivals / throughput ratio |
| `flights_horizon.parquet` | flight x horizon | **severe-delay classifier table**: `y_severe` = delay > 45 min, or cancelled, or diverted |
| `metar_hourly.parquet` | airport x UTC hour | flight category (VFR/MVFR/IFR/LIFR), crosswind/headwind vs the assumed heading, best-end headwind, flow-change flag, pressure tendency |
| `storm_events.parquet` | event | **evaluation labels only**: compiled after the event, never a model input |
| `audit_report.csv` | check | exactness audit against the organiser CSV |

## Leakage rules (enforced in code and unit-tested)

1. Weather features come only from METARs at or before t0. Hour H means the last report in (H−1h, H].
2. TAF features come only from TAFs issued at or before t0 that cover the target hour.
3. Airport-state features (arrivals, mean delay and share > 45 min over the last 2 h) count only flights that had actually landed by t0.
4. The inbound-aircraft delay (same tail, true rotation within 18 h) is used only if that leg had landed by t0.
5. Post-event fields are removed from the flight feature table. These are: actual times, delays, delay causes, taxi times, cancelled and diverted.
6. Train/validation/test splits are chronological: train before 2024-07, validation 2024-07 to 2024-12, test 2025.

## Physics

- Crosswind = |V·sin(θ−h)|.
- Headwind = V·cos(θ−h); a negative value is a tailwind.
- θ is the direction the wind blows from (METAR, degrees true) and V is the speed in knots.
- h is the challenge's assumed heading: ATL 80°, DFW 180°.
- The opposite runway end (h+180°) is also reported, as `best_end_headwind_kt`. This matters because controllers land into the wind, so a sustained tailwind on h usually means a flow change rather than tailwind landings.
- Calm wind gives zero components. Variable or missing direction gives NaN, never an invented value.

## Known limits (say these out loud in the pitch)

- **The "AAR" here is a proxy:** actual arrivals of BTS reporting carriers per hour. It is not the FAA-called Airport Arrival Rate (FAA ASPM). Cargo, general aviation and non-reporting carriers are missing.
- **Throughput is capped by demand.** In quiet hours, arrivals measure the schedule rather than capacity. Use `high_demand_hour == 1` when scoring capacity.
- **TAF archive:** IEM's TAF archive is complete for major US airports in this period. `audit_report.csv` reports issuances per day, so any gaps are visible.

## Tests

Run `python -m pytest -q tests`. The tests use synthetic inputs in each source's documented format and check:
- wind trigonometry, including wrap-around, calm and variable winds
- flight category thresholds
- Storm Events time-zone handling
- that the TAF as-of lookup never uses a TAF issued after t0
- that post-event columns are absent from the feature table
- the inbound-aircraft availability rule
