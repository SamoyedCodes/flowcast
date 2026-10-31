# Learning brief: what I built for Challenge 5, and why

Read this top to bottom once (about 25 minutes). Part 6 is the one to rehearse before presenting.

---

## 1. The whole thing in one paragraph

An airport can land only so many aircraft per hour. That number is the **Arrival Acceptance Rate (AAR)**. Bad weather lowers it, and controllers usually lower it *after* the weather arrives, so planes circle or queue. The challenge asks for a model that predicts the AAR and the risk of a long delay **3 to 12 hours before** the weather hits, for Atlanta (ATL) and Dallas-Fort Worth (DFW). I built a pipeline that downloads public weather (observations *and* forecasts), turns it into leak-free features, trains two models (a LightGBM AAR regressor and a regularised logistic-regression severe-delay classifier), scores them honestly on a year they never saw, and shows the result in a dashboard for a traffic-management duty manager.

## 2. What the task sheet asked, and where each item lives

| Requirement on the sheet | What I did | Where |
|---|---|---|
| Merge weather and flights on shared timestamps and airport ID | Join on (airport, issue time) using the *publication* time of each observation | `src/features.py` |
| Predict AAR, scored by RMSE | LightGBM per lead hour (3 to 12), RMSE in arrivals/hour | `src/train_aar.py`, `reports/aar_metrics.csv` |
| Classify flights with severe delay risk (> 45 min) | Regularised logistic regression per lead hour (chosen out-of-fold over LightGBM), calibrated probabilities | `src/train_delay.py` |
| Dashboard with risk scores, historical trends, recommendations for ATC | Streamlit app with five tabs | `dashboard/app.py`, `src/advisory.py` |
| Scripts that download and process public data automatically | One command, cached downloads | `run_pipeline.py`, `src/download.py` |
| Avoid leakage across time steps | Only information published before the forecast time; asserted and tested | `src/features.py`, `tests/` |
| Do not ignore runway directionality | Head/tail/crosswind from the assigned headings (ATL 80°, DFW 180°) | `src/weather.py` |
| Do not overfit to one airport | Airport config, ratio target, no airport-ID feature, unseen-airport test | `src/config.py`, `reports/aar_unseen_airport.csv` |
| Do not compare raw delay minutes across airports | Delays are rates (share of flights > 45 min), results shown per airport | `reports/delay_metrics.csv` |

## 3. Vocabulary you need

* **METAR**: the routine weather observation an airport publishes about hourly (wind, visibility, cloud base, rain, storms). What *was*.
* **TAF**: the airport's forecast for the next 24 to 30 hours. What forecasters *expect*. I download these too; they are the single most useful input at long lead times.
* **Flight category**: VFR (clear), MVFR (marginal), IFR (poor), LIFR (very poor), defined by cloud base and visibility. Airports have a different capacity in each.
* **AAR**: arrivals per hour the airport accepts. **Called rate** = what the facility actually announces.
* **Crosswind / tailwind**: the parts of the wind across and behind the runway. Aircraft can land in strong headwind but not in strong crosswind or tailwind, so those, not raw wind speed, cut capacity.
* **Lead time (h)**: how many hours before the event the forecast is made. A forecast at lead 6 uses only information available 6 hours before.
* **Data leakage**: accidentally using information from the future (or from the answer) when training or scoring. It makes results look great and worthless.
* **RMSE**: typical size of the error, in the unit of the thing predicted (arrivals/hour here). Lower is better.
* **PR-AUC (average precision)**: how well a score ranks rare positives. With about 13% severe delays, random scores about 0.13, perfect scores 1.0. It is the honest metric for imbalanced problems; accuracy would be misleading (always saying "no delay" is 87% accurate and useless).
* **ROC-AUC**: chance that a random delayed flight is scored higher than a random on-time one (0.5 = coin flip).
* **Calibration**: when the model says 30% risk, does it happen about 30% of the time?
* **Baselines**: *persistence* (carry the last value forward, i.e. the reactive status quo), *climatology* (the usual value for that month and hour), *TAF-implied* (apply my AAR rule to the TAF forecast).
* **Ablation**: add information step by step and see what each step buys.
* **Bootstrap**: re-score on many resamples of the test days to see how uncertain a number is.

## 4. The pipeline, step by step

1. **Download** (`src/download.py`). METAR and TAF from the Iowa State Mesonet archive, cached on disk. Learned the hard way: the server sometimes truncates a response, so the script retries and, if needed, splits the date range in half.
2. **Clean weather** (`src/weather.py`). Converts units, finds the cloud ceiling, computes head/cross/tailwind from the runway heading, aggregates to one row per airport-hour, and records `known_at_utc` (when the last observation of that hour was published).
3. **TAF features** (`src/taf.py`). For a target hour T and lead h, find the newest TAF issued at least h hours before T and read what it says about hour T: base conditions, plus worst case from TEMPO/PROB30 groups. Checked against a slow brute-force version: 0 mismatches.
4. **AAR target** (`src/aar.py`). See part 5, decision 1.
5. **Feature tables** (`src/features.py`).
   * *AAR table*: one row per (airport, target hour, lead). 452,300 rows, 76 features.
   * *Delay table*: one row per (flight event, lead). A departure from ATL/DFW *and* an arrival at ATL/DFW each count. 25,231 labelled flight events, 96 features. Cancelled and diverted flights have no delay value, so they are dropped.
   * *Recent hub performance*: mean delay and severe-delay share of flights that had **already** departed or landed at the issue time.
6. **Train and evaluate** (`src/train_aar.py`, `src/train_delay.py`). Time split: train 2023-01 to 2024-06, validation 2024-07 to 2024-12, test 2025-01 to 2025-07. Never shuffled. For the delay model, the model family and its settings are chosen by out-of-fold score on 2024, not by assumption.
7. **Stress-test** (`src/sensitivity.py`), **explain** (SHAP-style contributions in the dashboard), **advise** (`src/advisory.py`).
8. **Verify** (`tests/test_pipeline.py`, 10 tests): wind geometry, AAR rule, chronological splits, "weather published before issue time", "TAF issued at least h hours before", no leaky columns, own flight excluded from recent-performance windows.

## 5. The decisions that matter, and why

**Decision 1: how to define the AAR target.** No public dataset gives the hourly AAR a facility called. Options were (a) invent a rule, (b) use flight counts, (c) find real logs. Flight counts are useless here: the flight file holds about 17 departures a day per hub (a roughly 3% sample), so hourly counts say nothing about capacity. Real logs (FAA ASPM) are behind a login. So I derived it from published practice: the **FAA Airport Capacity Profiles** list the facility-called AAR by weather category (ATL 132/124/110/98; DFW 114/102/90, with DFW's lowest category scaled from ATL's ratio and flagged as an assumption). I then multiply by reductions for crosswind, tailwind, thunderstorms, winter weather and gusts. The *sizes* of those reductions are my assumptions. I handled that in three ways: validation against real flights (hours with a low proxy AAR had more severe departure delays: 16.1% at ≤60% of best case versus 10.0% above 95%), a sensitivity study (8 settings; the model wins in all of them), and saying so openly everywhere.

**Decision 2: use TAFs.** The task sheet mentions METARs *and* TAFs. A METAR tells you the present; at +12 hours it is stale. A TAF is a professional forecast issued before the event, so using it is legitimate and strong. Ablation: for delay risk at +12 h, PR-AUC goes from 0.236 (METAR only) to 0.269 (with TAF).

**Decision 3: leak-free by construction.** Every feature has a timestamp and the join refuses anything published after the forecast time. This is the single most important thing the judges warned about. If a team's accuracy looks too good, leakage is the usual reason.

**Decision 4: a separate model for each lead time.** Information quality differs (a 3-hour-old observation is fresher than a 12-hour-old one), so one set of weights fits poorly. It also gives a clean "accuracy versus lead time" story. Result: AAR error rises only gently from 12.3 (3 h) to 13.0 (12 h) because the TAF carries the long horizons.

**Decision 5: ratio target, no airport ID.** The AAR model predicts the *share of the airport's best-case rate*, then converts back to arrivals/hour. That lets one model serve any airport, which is what "do not overfit to one airport" asks. The honest check (train without one airport, test on it) is mixed, see part 7.

**Decision 6: up-weight capacity-cut hours.** My first model scored well overall but badly in the hours that matter: it predicted 97/hour at DFW on 30 April where the real value was 38. A least-squares model pulls rare extremes toward the average. Fix: count hours with a real cut (AAR ≥ 20% below best case) twice as much in training. I picked the weight on *validation* data with a rule written down first (smallest error on cut hours, as long as overall error rises by at most 5%): weight 2 passed, weight 4 did not. Test result: overall RMSE unchanged (12.26 → 12.28), error on cut hours 23.6 → 21.4, now better than the TAF alone (23.0).

**Decision 7: two alert settings.** Alerting when the central forecast says "cut" gives 71% of alerts real and catches 69% of real cuts (3 h ahead). Alerting on the low end of the forecast range gives 47% real and catches 90%. The right choice depends on the cost of a false alarm versus a miss, which is the controller's call, so the dashboard exposes it.

**Decision 8: calibrated probabilities.** A risk score is only useful if "30%" means 30%. I fitted a Platt (logistic) correction on out-of-fold predictions. On the test year, the highest-risk tenth of flights was predicted at about 40% risk on average and showed 37% severe delays.

**Decision 9: machine learning forecasts, rules advise, a person decides.** No language model anywhere. The recommendations in `src/advisory.py` are plain if-then rules you can read and audit. That is the "right tool for each job" the rubric rewards.

**Decision 10: for the delay model, simple beat fancy, and I let the data decide.** My first instinct (and first build) was LightGBM for both tasks. A check I ran late showed a plain L2-regularised logistic regression on the same 96 features beat it on the untouched 2025 test year at every horizon (PR-AUC 0.316 vs 0.301 at +3 h; 0.269 vs 0.248 at +12 h). I did not just swap models because of the test result: I rebuilt the selection so the pipeline compares candidates **out-of-fold on 2024** (four expanding-window folds) and picks the winner. Logistic regression won there too (0.285 vs 0.258 at +3 h), so the test result was confirming, not choosing. Why it wins: only about 1,900 severe events, a weak signal that is mostly additive, and trees overfit that. Bonus: the per-flight reasons are now exact (coefficient times how unusual the value is), not an approximation. The AAR task is different (45,000 hourly rows per lead, sharp non-linear thresholds such as thunderstorm and low cloud), and there boosting does win. The full selection table is `reports/delay_model_selection.csv`.

## 6. Results, and the honest reading of them (test year 2025-01 to 2025-07, never used for tuning)

**AAR forecast (RMSE in arrivals/hour, lower is better)**

| | +3 h | +6 h | +12 h |
|---|---|---|---|
| Our model | **12.3** | **12.7** | **13.0** |
| TAF-implied | 15.2 | 15.0 | 15.2 |
| Climatology | 18.4 | 18.4 | 18.4 |
| Persistence (today's reactive baseline) | 17.9 | 20.7 | 23.3 |

* Error is 31% lower than persistence at +3 h and 44% lower at +12 h.
* In hours with a real capacity cut, RMSE is 21.4 (+3 h) and 22.9 (+12 h), versus TAF-implied 23.0 and 25.0.
* The 80% forecast range actually contains the truth about 89% of the time (a bit too wide, which is the safe direction).

**Severe-delay classifier (PR-AUC; random = 0.13)**

| Information used | +3 h | +12 h |
|---|---|---|
| Schedule only (airline, route, time of day) | 0.216 | 0.216 |
| + recent hub delays | 0.245 | 0.224 |
| + METAR weather | 0.303 | 0.236 |
| + TAF forecast (full model) | **0.316** | **0.269** |
| *For comparison: best LightGBM, same features* | *0.301* | *0.248* |

* The full model's gain over schedule-only is +0.101 (95% interval +0.063 to +0.137) at +3 h and +0.052 (+0.023 to +0.083) at +12 h. Both exclude zero, so the weather signal is real, but modest.
* The riskiest 10% of flights are 37% severely delayed (13% overall). Departures are easier to score than arrivals (PR-AUC 0.35 vs 0.25), because an arrival's delay also contains the delay of its previous leg.
* Predicted and observed risk track each other across the ten risk groups, and the daily average predicted risk rank-correlates 0.60 with the observed daily share (419 airport-days).
* Be careful with the story: most of the delay signal is schedule-driven (airline, route, time of day). Weather helps most at short lead; beyond about +6 h the skill is roughly flat, carried by the TAF.

## 7. Where it is weak (say this before they ask)

1. **The AAR "truth" is a proxy**, not logged facility data. Mitigated by validation and sensitivity, not removed. The top next step is real ASPM rates.
2. **Severe storms are under-called.** Even after the fix, deep one-hour drops are only partly forecast (e.g. forecast 97 vs 38 on 30 April at DFW, six hours ahead). The cautious setting and the TAF line exist for this.
3. **Some severe delays have no weather cause.** The biggest miss in 2025 was a 1,419-minute departure delay at DFW (19 March, AA to ORD) scored at 6% risk. No weather model could have caught it.
4. **Partial transfer to a new airport.** Trained without DFW, the model is *worse* than the TAF alone on DFW (13.2 vs 11.7 RMSE at +3 h); trained without ATL it still wins (14.9 vs 18.1). With only two airports I can claim the pipeline works for any airport code, not that one trained model transfers.
5. **The flight file is a ~3% sample of traffic** (about 17 departures a day per hub), so I could not normalise by true flight volume and the dashboard shows risk rates, not counts.
6. **Test period is seven months (Jan to Jul 2025)**: no autumn, and the severe-delay rate is higher than in training (13.2% vs 10.1%). The calibration held up anyway, but seasonality is not fully tested.
7. **Absolute AAR error depends on my hazard assumptions** (tailwind and storm factors move RMSE a lot). I claim the ranking and the size of the gain, not the absolute number.

## 8. Questions judges may ask

* *"Is there leakage?"* Show part 4 step 8 and the timeline slide. Features are joined on publication time; a test fails if anything is published after the issue time or a TAF is issued less than h hours before the target.
* *"Where does the AAR label come from?"* FAA-published called rates by weather category plus reductions I chose. Proxy, validated against real flights, stress-tested, and flagged.
* *"Why not predict raw wind speed effects?"* Because crosswind and tailwind relative to the runway matter, not speed. That is why the headings are in the config.
* *"Why not an LLM?"* The task is numeric forecasting; an LLM would be the wrong tool. The explanations come from the model's own feature contributions and rule-based text.
* *"Why logistic regression and not boosting or a neural network for delay?"* Because I tested it: out-of-fold on 2024 and on the 2025 test year it beat LightGBM, with only about 1,900 severe events and a weak, mostly additive signal. The selection table is in the repo. It is also easier to explain: each flight's risk drivers are exact coefficient contributions.
* *"What would you do with more time?"* Real AAR logs; NOAA storm logs (not used: I did not integrate them); more airports; a live feed; a trial with traffic managers to measure delay saved.
* *"What did you not do?"* No NOAA storm-log integration; no live feed; no claim of minutes or fuel saved.

## 9. Things I changed my mind about while building (so you know the pitfalls)

* My first delay baseline "random guess" row used a tie-breaker that accidentally created a ranking and gave it a fake PR-AUC of 0.166. I caught it, replaced it with the true constant predictor (PR-AUC = prevalence = 0.132).
* My first AAR model looked good by RMSE but failed in the worst hours; checking *where* it fails (not just the average) led to the weighting fix. Average error can hide the cases that matter.
* A script edit with the wrong text encoding truncated `dashboard/app.py` to zero bytes. I rebuilt it from the earlier version and re-ran the dashboard tests. Lesson: always specify the encoding on Windows.
* I assumed gradient boosting would be best for both tasks. It was not for delay risk: a regularised logistic regression beat it. I only found out because I ran a plain baseline against my final model late on. Lesson: always put a simple model on the same features next to the fancy one, and choose between them out-of-fold, not by reputation.
* LightGBM was not installed; I installed it, plus streamlit, plotly, shap, python-pptx, pytest.
* I re-computed the headline RMSE and PR-AUC independently from the saved predictions to be sure my metric code agreed. It did.

## 10. Run it, and what I could not fully verify

```bash
pip install -r requirements.txt
python run_pipeline.py          # about 5 minutes after the first download (add --sensitivity for the stress test)
python -m pytest -q             # 10 tests
streamlit run dashboard/app.py  # deep link: ?airport=KDFW&t0=2025-02-18T17
python presentation/build_deck.py   # rebuild the slides from reports/
```

* Re-running the pipeline can move numbers in the third decimal (the logistic delay model is deterministic, but the AAR LightGBM models can move slightly; an earlier LightGBM delay model moved from 0.298 to 0.300 PR-AUC between two identical runs, which is part of why the simpler model is attractive). The slides rebuild from `reports/`, but the numbers typed in this brief and in the dashboard help text are from the final run.
* Nothing is committed to git; the working tree has new folders `src/`, `dashboard/`, `tests/`, `reports/`, `models/`, `presentation/`, `data/processed/`, `data/reference/`.
* The AAR reduction sizes, the DFW low-IFR rate and the severity thresholds are assumptions, labelled in `src/aar.py` and `src/config.py`.
* A stale file from my first prototype, `data/model/delay_table.csv`, could not be deleted because another program had it open. It is not used by anything and is safe to delete by hand.
* I checked the slides by exporting them to images and viewing them, and the dashboard by clicking through it and by automated tests. I did not run the deck on a different machine or in a different PowerPoint version.
