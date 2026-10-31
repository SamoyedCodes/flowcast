# AeroCast rubric audit (Challenge #5), 31 Oct 2026, 12:20 SGT

Verdict: **not yet a valid submission.** The code covers every Challenge #5 requirement. But the headline metric (AAR RMSE) has not been measured on real data, because the full BTS flight data has not been downloaded yet. Everything in this file was run and checked unless it says otherwise.

## Blockers (in priority order)

| # | Blocker | Effect | Fix | Owner |
|---|---|---|---|---|
| 1 | Full BTS on-time data not downloaded. The cloud session's network policy denies transtats.bts.gov, mesonet.agron.iastate.edu and www.ncei.noaa.gov. | No valid AAR RMSE, no TAF features, no Storm Events evaluation. This touches Honest evaluation (20%), Technical soundness (25%) and Impact (10%). | On a laptop: `cd jho/aerocast_data && pip install pandas pyarrow requests airportsdata lightgbm scikit-learn streamlit plotly && python run_all.py --start 2024-01 --end 2025-07 --airports ATL DFW`. Use `--start 2024-01` to halve the download. | Keith (his network already reached IEM) |
| 2 | Push to GitHub refused (403: the Claude GitHub App is not installed on kkeeiitthh/LARP). | Work exists only in this session's local commit. | Reconnect at https://claude.ai/connect-github and install the app on the repo. | Repo owner |

## Want-to-see checklist (Challenge #5)

| Requirement | Status | Evidence |
|---|---|---|
| Multi-source integration on shared timestamps and station IDs | Done in code; METAR part run on real data | `build_datasets.py` joins BTS (local times converted to UTC per airport) with METAR (UTC) on airport + hour, and TAF by as-of issuance time. METAR: 52,843 real reports processed. |
| AAR prediction measured by RMSE | **Code done; real metric missing** | `train_models.py` reports RMSE/MAE/nRMSE by airport x horizon, overall and on high-demand hours, against persistence, climatology and schedule baselines. Only smoke-tested on the 0.5% sample (about 0.2 arrivals/h), which cannot measure capacity. The dashboard shows a red banner whenever the data is this sparse. |
| Severe delay (>45 min) classification | Done in code | LightGBM classifier with PR-AUC, ROC-AUC, Brier, precision/recall/F1, confusion counts and a calibration table. The threshold is chosen on validation and frozen for test. Smoke test (sample data, so indicative only): PR-AUC beats climatology in all 16 airport x direction x horizon cells. |
| Interactive dashboard with risk scores, historical trend and ATC recommendations | Done, headless-tested | `streamlit run dashboard.py`. Selectors for airport, horizon and hour. Shows predicted arrivals with a 90% interval, flights above the risk threshold, weather at t0, a 48 h trend and a daily completion ratio. Recommendations are rule-based and show their evidence. No exceptions in AppTest for ATL, DFW or horizons 3 and 12. |
| Automatic download and processing | Done in code; BTS/TAF/NOAA not run live | `run_all.py` covers download, build, train and audit, with a SHA-256 provenance manifest. METAR reuses the notebook cache. |

## Want-to-avoid checklist

| Pitfall | Status | Evidence |
|---|---|---|
| Leakage across time steps | Guarded and tested | Features use only data at or before t0 = target - horizon. METAR hour H = last report in (H-1h, H]. TAF uses an as-of join on issuance time with an assert. Airport state counts only flights already landed. Inbound delay is used only if the previous leg had landed. Post-event columns are dropped. Splits are chronological. Smoke build: all t0 < target. Notebook as-of join: median weather age 29 min at t0, assert passes. |
| Ignoring runway directionality | Done | Crosswind, headwind and tailwind on the assumed headings (ATL 80, DFW 180), plus gust crosswind, best-end headwind and a flow-change flag. Spot check: ATL 2024-07-15 20:52Z wind 320/19G27 matches NOAA (50 km/h gust). |
| Over-fitting to one airport | Done | `load_airports` accepts any IATA/ICAO code (KXXX too); the runway heading comes from `airports.json`. Leave-one-airport-out test added: train on ATL, predict DFW and vice versa, on a normalised target. |
| Un-normalised cross-airport comparison | Done | `arrivals_norm` (arrivals / own p95), nRMSE, daily completion ratio, severe rates as shares. |

## Rubric self-score (honest, current state)

| Criterion (weight) | Now | After a real BTS run | What moves it |
|---|---|---|---|
| Problem realism (20%) | 3 | 4 | User: airport ops duty manager. Status quo: capacity cut reactively when weather arrives. Pitch needs one concrete hour (e.g. an ATL thunderstorm day) costed at USD 98.41/block minute (A4A 2025). |
| Technical soundness (25%) | 3 | 4 | End-to-end build exists. Statistics/ML forecast; rules recommend; Claude only rephrases on-screen drivers (tested; it repeats the numbers given). Needs the real run. |
| Honest evaluation (20%) | 2 | 4 | Baselines, intervals, a failure tab (largest over-predictions; cells where the model loses to climatology), leave-one-airport-out. Stuck at 2 until real metrics exist. |
| Trust and traceability (20%) | 4 | 4 | Every number has a t0 timestamp, observation age, TAF age, interval, model hash and manifest of source URLs and SHA-256. Rules show threshold and source. Data-validity banner refuses to present sample-based AAR as capacity. Advisory-only wording. |
| Impact (10%) | 1 | 3 | Needs measured before/after: e.g. shortfall hours flagged 6 h ahead vs schedule-only, false alarms per week. Keep measured numbers apart from cost estimates. |
| Time discipline (5%) | ? | ? | Feature freeze 15:00, valid submission 15:45. |

## Defects found and fixed during this audit

1. **Stale gusts.** Forward-fill copied the previous hour's gust into hours whose report had no gust group, inventing gusts after storms. Now only hours with no report at all are filled (max 2 h). Verified on ATL 2024-07-15 22Z.
2. **TAF timestamp parsing.** tz-aware IEM timestamps would have crashed `astype("datetime64[ns]")` on pandas 2+. All source timestamps now go through `to_utc_naive`.
3. **Wrong rule citation.** The crosswind threshold cited "Order 8400.9: 20 kt". The order says 15 kt including gusts (clear, dry runway). The 5 kt tailwind trigger comes from JO 7110.65 3-5-1. Both corrected.
4. **Claude summary silently empty.** A 300-token cap was used up by thinking, and the UI then wrongly said "key not set". Cap raised; an empty reply now shows an explicit warning. No key is stored in the repo (git grep checked).
5. **Notebook weather timing.** Section 0 claimed the weather matched the scheduled local hour. It is UTC: the IEM comparison across all 17,713 hours gives mean |temp diff| 0.03 C read as UTC, 3.48 C read as local. The lag join is now a true as-of join.

## Known limits to say out loud

- The AAR is a proxy (actual BTS-carrier arrivals per hour), not the FAA-called rate. Cargo and GA are missing.
- In quiet hours arrivals measure demand, not capacity, so capacity is scored on high-demand hours.
- One assumed heading per airport (challenge rule), not the real multi-runway configuration.
- Storm Events are compiled after the event, so they are evaluation labels only.
- Conformal intervals are fitted on all hours. On the sample, coverage on high-demand hours was below 90%. Check this on real data and fit per demand band if it persists.

## Suggestions from published work

- The FAA's current tool matches the TAF against historical analog hours and takes the most common AAR (KBR comparison paper). A "TAF-analog" baseline would be the most credible status quo to beat. Add it once TAFs are downloaded.
- Aerospace (MDPI, 2024) notes hourly throughput prediction is under-studied. Use this as a one-line novelty claim, not an accuracy claim.

## Order of work from now

1. 12:30 Keith runs `run_all.py` locally (around 30-60 min for 2024-01 to 2025-07).
2. Read `data/reports/aar_metrics.csv`: does the model beat climatology on high-demand hours at 3 h and 12 h? Pick the failure case from the dashboard's evaluation tab.
3. 14:30 Cost one real shortfall day for the impact slide.
4. 15:00 Freeze. 15:45 Submit.
