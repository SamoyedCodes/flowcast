# Autonomous SG 2026 — Airbus Problem Statements

**Track:** AI & Aerospace  
**Organisers:** Acacia Hackathon Society × NUS Product Club  
**Institution:** National University of Singapore (NUS)  
**Date:** 31 October 2026

**Datasets:** [Google Drive folder](https://drive.google.com/drive/u/1/folders/13h0JZnSqhYTX3rV8dGDGy5AiwaADQvJR)

---

## Challenge #1 — Satellite Collision Avoidance

> How could we leverage AI-driven technologies for satellite collision avoidance?

### Description

With low earth orbit (LEO) being crowded with satellites and space debris, satellite operators receive thousands of data messages daily warning of potential close encounters. Evaluating every encounter and determining if corrective action is required necessitates too much manual labor and is vulnerable to human error.

### Want to see

A collision risk evaluator that reads orbital conjunction telemetry and subsequently calculates the collision probability (Pc) for satellite-debris pairs, automatically flagging high-risk events that require maneuver planning.

**Risk classification:** Algorithm predicting whether an event is high risk, requires maneuvering, or safe (backed by probability).

In accordance with the *NASA Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook* (NASA/SP-20205011318) and ESA Space Debris Office Standards:

| Risk tier | Criteria |
| --- | --- |
| **Safe** | $P_c < 10^{-6}$ **and** miss distance > 5.0 km |
| **Elevated risk** | $10^{-6} \leq P_c < 10^{-4}$ **or** miss distance between 1.0 km and 5.0 km |
| **Critical (Maneuver Required)** | $P_c \geq 10^{-4}$ (i.e. `risk >= -4.0` in the dataset) **or** miss distance < 1.0 km |

**Dashboard UI:** Visual platform showing trajectories, collision risk scores and miss distances.

**Warnings:** In a 24-hour frame, show how many encounters come within the defined dangerous distance and generate insights.

### Provided data

**Collision Avoidance Challenge dataset** (file in `Challenge #1\archive(1).zip`).

- Can use only the following columns for simplicity:
  - `event_id`
  - `time_to_tca`
  - `miss_distance`
  - `relative_speed`
  - `relative_position_r`
  - `relative_position_t`
  - `relative_position_n`
  - `c_object_type`
  - `c_sigma_r`
  - `t_sigma_r`
  - `max_risk_estimate`
  - `F10`
  - `AP`
- In `archive(1).zip`:
  - Use `train_data` to train models, fit parameters and relationships between telemetry features, etc.
  - The `risk` column represents $\log_{10}(P_c)$; map this to operational risk tiers to train and validate models.
  - Use `test_data` to generate predictions on rows, for outputs to be scored to see how well the model generalises to unseen encounters.

---

## Challenge #2 — Multi-Agent Air Traffic Management

> How might AI help air traffic controllers, pilots and ground operations staff make faster and better decisions?

### Description

Today, when a runway closes, weather delays occur or ground equipment fail unexpectedly, communications are reliant on human phone calls, manual software updates etc., leading to cascading delays, excessive holding patterns, and excessive fuel consumption. The future of Air Traffic Management relies on AI assistants that can autonomously communicate and negotiate trade-offs across organisational boundaries.

### Want to see

A **Multi-Agent Communication Framework** where ATC AI agents, Pilot AI agents, and ground ops agents are able to negotiate resolutions to real-time disruptions without human intervention until a consensus is reached.

| Agent | Responsibilities |
| --- | --- |
| **ATC agent** | Take care of runway wake separation, taxiway blockage; maximise sector throughput. |
| **Pilot agent** | Preserve fuel reserves, ensure flight safety, minimise turbulence and protect passengers' connections. |
| **Ground agent** | Optimise gate utilisation, avoid ground-crew overtime, minimise turnaround delays. |

These three agents must exchange structured data, evaluate cross-domain constraints, and converge on an agreed solution. This includes:

- Structured messages
- Explicit constraint checking
- Traceable consensus log
- Deterministic non-AI fallback

### What to avoid

- Combining three roles into a single prompt call; the point is to test multi-agent coordination across independent domains.
- LLM agents talking to each other in endless, unconstrained loops with no convergence logic or turn limits.
- Disregard of safety margins; AI agents should not be allowed to accept solutions that ignore operational constraints.

### Provided data

**US Department of Transportation data for 2015 flight delays and cancellations**.

- File in `Challenge #2\DATA`.
- Download this and pick any row with `WEATHER_DELAY > 15` or `LATE_AIRCRAFT_DELAY > 20`.
- Pass the parameters in that row into the three AI agents.
- Demonstrate how agents negotiate revised arrival time, gate assignment, or flight plan adjustment in **under three turns**.

---

## Challenge #3 — Sustainable Aviation Fuel (SAF) Supply Chain in Kalimantan

> How might we optimise the SAF supply chain in Kalimantan?

### Description

There is a need to aggressively scale Sustainable Aviation Fuel (SAF) adoption to meet decarbonisation goals. Kalimantan (Indonesian Borneo) possesses abundant agricultural and forestry residues, making it a potential hub for regional SAF production. However, transporting raw biomass, establishing infrastructure, and delivering finished SAF to international airports requires strategic decision-making, and often comes with complex logistical and capital trade-offs.

### Want to see

- A SAF supply chain optimisation model across Kalimantan that minimises the **total cost (OpEx + CapEx + Logistics)**, while meeting SAF demand at Kalimantan's four international airports:
  - Supadio Intl
  - Iskandar Intl
  - Sultan Aji Muhammad Sulaiman Sepinggan Intl
  - Juwata Intl
- Optimal coordinates of depots and refineries.
- Facility sizing decisions.
- Material flow network.
- **Infrastructure placement:** Deciding spatial locations and sizes (Small vs Large) for pre-processing depots and SAF refineries.
- Optimal routing of raw feedstock to depots, depots to refineries, and finally refineries to airports.

**Feedstock quantity:** 6,800 kt.

### Facility costs and production capacity

| Facility type | Size | Cost (USD) | Production capacity |
| --- | --- | ---: | ---: |
| Depot | Small | $2.25M | 150 kt |
| Depot | Large | $4.5M | 300 kt |
| Refinery | Small | $480M | 500 kt |
| Refinery | Large | $765M | 1,087 kt |

### Transport specifications and costs

| Truck size | Farm to depot | Cost (USD/tonne/km) | Depot to refinery | Cost (USD/tonne/km) | Refinery to airport | Cost (USD/tonne/km) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 3 t | 0.10 | 18 t | 0.12 | 16 t | 0.08 |
| Large | 9 t | 0.10 | 30 t | 0.12 | 32 t | 0.08 |

### Data provided

**Airport locations**

- Download `data/airports.dat` for precise airport locations (including Name, City, Country, Latitude, Longitude).
- **Only four airports to be considered:** Supadio Intl, Iskandar Intl, Sultan Aji Muhammad Sulaiman Sepinggan Intl, Juwata Intl.
- File location: `Challenge #3 - SAF\Airport location\airport.dat`.

**Biomass locations**

- 80 locations, with a range of 20–100 kilotons of biomass per location; total biomass is 6,800 kilotons.
- File location: `Challenge #3 - SAF\Biomass locations\kalimantan_80_locations_6800kt_full_range.csv`.

---

## Challenge #4 — Trustworthy Search for Regulated Technical Information

> How might AI help people find, understand, and verify regulated technical information?

### Description

Sensitive regulatory documents include airworthiness directives, maintenance manuals and accident reports. Often, mechanics and engineers need to find these precise instructions quickly with full trust in the output. GenAI models often hallucinate or blend instructions across different types of equipment. This can lead to dangerous mishaps in operational situations.

### Want to see

A **traceable and context-aware search engine** that ingests multi-page technical documents, parses exact applicability rules, and answers natural language queries with citations. If the information is ambiguous, it should confidently refuse to answer instead of giving an unreliable answer.

- **Specific references:** Every claim or sentence must link to its exact source metadata.
- **Zero hallucination:** System must not output hallucinated responses.
- **Configuration filtering:** System must methodically evaluate conditional applicability rules (e.g. Directive A only applies to serial numbers S-1000 to S-1500).
- **Revision logic:** User must be aware if the document has multiple versions, and what the latest version is.
- **Clear communication of uncertainty:** If a prompt falls outside the scope of the data, contradicts existing constraints, or involves ambiguous terms, the system must clearly output a message expressing its inability to complete the task.
- **Sub-3-second retrieval latency:** Satisfactory search performance across 1,000+ pages.

### Want to avoid

- **LLM guessing:** LLM attempting to fill knowledge gaps using its pre-trained weight memory.
- **Unformatted data:** Returned data must follow an appropriate structure; it cannot just be randomly written text.

### Provided data

**Airworthiness Directives (ADs)**

- File is in `Challenge #4\432 entries of FAA Airworthiness Directives`.

---

## Challenge #5 — Predictive Airport Capacity and Weather Delay Risk

> How might we leverage historical weather observation logs and flight performance datasets to predict airport runway capacity reductions and flight delay risks 3 to 12 hours before severe weather events occur?

### Description

Severe weather accounts for over 70% of all air traffic delays globally. When severe weather hits major airport hubs, air traffic controllers and airport operators must dynamically reduce hourly arrival and departure rates. Because capacity reductions are conducted reactively when weather actually arrives, aircraft end up stuck in costly airborne holding patterns or stranded on taxiways with engines running, wasting fuel and causing severe downstream flight cancellations.

In this challenge, participants must build a **Predictive Airport Capacity & Risk Model** that ingests publicly available weather observations (METARs/TAFs and NOAA storm logs) and historical flight performance data to forecast hourly runway arrival acceptance rates (AAR) and departure delay risks **3 to 12 hours in advance**.

### Want to see

- **Multi-source data integration:** Merge airport weather observation logs with flight delay records using shared timestamps and airport station identifiers.
- **Predictive model performance and accuracy:** Accurate prediction of Arrival Acceptance Rates, measured by Root Mean Squared Error.
- **Delay risk classification:** Specific classification of flights that face a severe delay risk (**> 45 minutes**).
- **Interactive dashboard and decision support for ATC:** Dashboard displaying risk scores alongside historical trends and recommendations for air traffic controllers.
- **Automatic processing of public datasets:** Scripts that download and process public datasets automatically without manual human intervention.

### Want to avoid

- **Data leakage across time steps:** Using weather measurements recorded when a flight arrives to predict delays. Instead, use historical forecast/observation logs available prior to the event.
- **Ignoring runway directionality:** Evaluating raw wind speed without calculating actual crosswind and tailwind components relative to specific airport runway orientations.
- **Over-fitting to a single airport:** Building a model that only works on one specific airport layout rather than designing a generalised pipeline that accepts any airport ICAO code in the dataset.
- **Evaluating un-normalized delays:** Comparing raw delay minutes between massive hub airports (e.g. Chicago O'Hare) and smaller regional airports without normalizing by overall flight volume.

### Provided data

**Airport weather data from 2023 to 2025**

- Select only two airports: **Atlanta Hartsfield-Jackson (ATL)** and **Dallas Fort-Worth (DFW)**, for departing and arriving flights.
- Weather telemetry includes air temperature, precipitation, wind speed, wind gust peak, air pressure, snow depth, and weather condition nodes.
- Assume the following runway headings for all take-offs and landings:

  | Airport | Assumed heading |
  | --- | ---: |
  | Hartsfield-Jackson Atlanta International Airport | 80° |
  | Dallas Fort Worth International Airport | 180° |

- File in `Challenge #5\Flight_Data_With_Weather_Final.csv`.

---

# Judging Rubric — AI & Aerospace Technology Track

Six criteria. Score each from **1 to 5** and apply the weights below. **Total weight: 100%.**

| Criterion | Weight | What judges look for |
| --- | ---: | --- |
| Problem realism | **20%** | Specific user, credible pain, clear status quo |
| Technical soundness | **25%** | Right tool for each job, working end to end |
| Honest evaluation | **20%** | Tested on something, failure cases shown |
| Trust and traceability | **20%** | Evidence visible, uncertainty handled, human decides |
| Impact | **10%** | Measured or reasoned improvement over today |
| Time discipline | **5%** | Efficient submission and a focused presentation |

## 1. Problem realism — 20% (score 1–5)

**Does this solve a specific aerospace workflow problem?**

Name the user, current process and cost in time, money or risk. Explain the constraints of the chosen challenge, rather than describing aviation in general.

| Strong | Weak |
| --- | --- |
| A credible user, narrow workflow and clear status quo. | A generic aviation chatbot or a solution looking for a problem. |

## 2. Technical soundness — 25% (score 1–5)

**Does the prototype work, with the right method for each task?**

Show an end-to-end build. Use LLMs for reading and explanation; use solvers, rules or statistics for scheduling, constraints and detection. Test beyond a scripted demo.

| Strong | Weak |
| --- | --- |
| Working inputs and outputs; explicit constraints and a sound LLM / non-LLM split. | LLM-only legality checks, opaque risk scores or a hard-coded demo. |

## 3. Honest evaluation — 20% (score 1–5)

**What was tested, against what baseline, and where does it fail?**

State metrics, test data and results. Label synthetic data and its limits. Show one case where the system is wrong, uncertain or declines, and explain how the user knows.

| Strong | Weak |
| --- | --- |
| A baseline, measured results and a visible failure with a reason. | Only success cases, unsupported accuracy claims or hidden data limits. |

## 4. Trust and traceability — 20% (score 1–5)

**Can a qualified person verify the output before acting?**

Trace claims to a document, revision and section, a timestamped signal or an applied rule. Handle applicability and missing evidence. Keep the qualified person in control.

| Strong | Weak |
| --- | --- |
| Checkable sources, visible uncertainty and a clear refusal when needed. | Confident unsourced answers or a claim to replace expert judgement. |

## 5. Impact — 10% (score 1–5)

**What improves over the current process, and by how much?**

Compare against the stated baseline: delay minutes, supplier impact, retrieval quality or false alarms, as relevant. Distinguish measured improvement from estimates.

| Strong | Weak |
| --- | --- |
| A credible before/after measure and realistic adoption constraints. | Large benefit claims with no baseline or unstated assumptions. |

## 6. Time discipline — 5% (score 1–5)

**Was the project submitted early and presented efficiently?**

Earlier valid submission is better, using the recorded submission time. Also assess a clear, focused presentation completed within 3 minutes. Give one combined 1–5 score.

| Strong | Weak |
| --- | --- |
| Earlier submission and a focused, complete presentation within 3 minutes. | Later submission and a rushed, unfocused or over-time presentation. |

## Scoring and Judges' Notes

**Scale:** 1 = weak; 2 = limited; 3 = credible; 4 = strong; 5 = exceptional evidence.

**Weighted total / 100** = sum of `(criterion score / 5) × criterion weight`.

**Example:** Technical soundness scored 4/5 earns 20 of its 25 points.

- Score criteria separately. Ask: *"Where does this fail, and how would your user know?"*
- Reward sound restraint, not presentation polish.
- Each project is evaluated by **3 judges**.
- Time is only **5%**: earlier submission does not substitute for quality or evidence.
