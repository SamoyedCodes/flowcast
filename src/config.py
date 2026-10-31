"""Central configuration. Add an airport to AIRPORTS and every stage of the pipeline picks it up."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_WX = DATA / "weather" / "raw"
RAW_TAF = DATA / "weather" / "raw_taf"
PROC = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FLIGHTS_CSV = DATA / "clean" / "flights_clean.csv"

# Study window (END exclusive). Flights in the organiser file run to 2025-07-31.
START, END = "2023-01-01", "2025-08-01"

# icao -> airport metadata. runway_heading_deg: single assumed take-off/landing heading from the brief.
# For an airport not listed here the pipeline falls back to the prevailing wind direction
# (see weather.estimate_heading) and warns, so any ICAO code in the flight file can be processed.
AIRPORTS = {
    "KATL": {"iata": "ATL", "tz": "America/New_York", "runway_heading_deg": 80,
             # FAA Airport Capacity Profile (ATL, 2018): facility-called AAR by weather category
             "aar": {"VFR": 132, "MVFR": 124, "IFR": 110, "LIFR": 98}},
    "KDFW": {"iata": "DFW", "tz": "America/Chicago", "runway_heading_deg": 180,
             # FAA Airport Capacity Profile (DFW, 2018). LIFR is not published separately; scaled from
             # ATL's low-IMC/IMC ratio (98/110) and flagged as an assumption.
             "aar": {"VFR": 114, "MVFR": 102, "IFR": 90, "LIFR": 80}},
}
IATA_TO_ICAO = {v["iata"]: k for k, v in AIRPORTS.items()}

LEADS = (3, 6, 9, 12)          # forecast horizons in hours
SEVERE_DELAY_MIN = 45          # severe-delay threshold (minutes)

# time-based split (UTC): never shuffle
TRAIN_END = "2024-07-01"
VAL_END = "2025-01-01"
