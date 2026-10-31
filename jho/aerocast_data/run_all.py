"""
One command, no manual steps: download -> process -> train/evaluate -> audit. Then: streamlit run dashboard.py

  python run_all.py --start 2023-01 --end 2025-07 --airports ATL DFW --provided Flight_Data_With_Weather_Final.csv

Any airport in airports.json (or any IATA/ICAO code, without runway heading) can be passed.
Downloads are cached in data/raw and logged with URL, time and SHA-256 in data/manifest.jsonl.
"""
import argparse
import subprocess
import sys

STEPS = [
    ["pull_bts.py", "--start", "{start}", "--end", "{end}", "--airports", "{airports}"],
    ["pull_metar.py", "--start", "{start}", "--end", "{end}", "--airports", "{airports}"],
    ["pull_taf.py", "--start", "{start}", "--end", "{end}", "--airports", "{airports}"],
    ["pull_storm_events.py", "--years", "{years}", "--airports", "{airports}"],
    ["build_datasets.py", "--airports", "{airports}", "--flight-sample", "{sample}"],
    ["train_models.py"],
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2025-07")
    ap.add_argument("--airports", nargs="+", default=["ATL", "DFW"])
    ap.add_argument("--flight-sample", default="0.25")
    ap.add_argument("--provided", help="organiser CSV, enables the exactness audit")
    ap.add_argument("--skip", nargs="*", default=[], help="e.g. --skip pull_storm_events.py")
    a = ap.parse_args()
    years = [str(y) for y in range(int(a.start[:4]), int(a.end[:4]) + 1)]
    for step in STEPS:
        if step[0] in a.skip:
            continue
        cmd = [sys.executable]
        for tok in step:
            if tok == "{airports}":
                cmd += a.airports
            elif tok == "{years}":
                cmd += years
            else:
                cmd.append(tok.format(start=a.start, end=a.end, sample=a.flight_sample))
        print("\n>>>", " ".join(cmd[1:]))
        subprocess.run(cmd, check=True)
    if a.provided:
        subprocess.run([sys.executable, "audit_pulled_data.py", "--provided", a.provided], check=True)
