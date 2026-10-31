"""One command, no manual steps:  download public data -> clean -> features -> train -> evaluate.

    python run_pipeline.py                 # everything (downloads are cached; reruns skip files already on disk)
    python run_pipeline.py --sensitivity   # also re-run the AAR-assumption sensitivity study (~8 min)
    python run_pipeline.py --skip-download # use the cached raw weather only

Then:  streamlit run dashboard/app.py
"""
import argparse
import time

import pandas as pd

from src import config as C, download, features, sensitivity, taf as tafmod, train_aar, train_delay, weather


def stage(name):
    print(f"\n=== {name} " + "=" * (70 - len(name)))
    return time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()

    for d in (C.PROC, C.MODELS, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    t = stage("1/5 download METAR + TAF (Iowa State Mesonet)")
    if args.skip_download:
        print("skipped")
    else:
        download.download_asos()
        download.download_taf()
    print(f"{time.time() - t:.0f}s")

    t = stage("2/5 clean METAR -> hourly weather, runway wind components, weather-adjusted AAR")
    hourly = weather.build_hourly()
    hourly.to_csv(C.PROC / "hourly_weather.csv", index=False)
    print(hourly.groupby("airport").agg(hours=("hour_utc", "size"), with_obs=("n_obs", lambda s: (s > 0).sum())))
    print(f"{time.time() - t:.0f}s")

    t = stage("3/5 TAF forecasts + feature tables (AAR and delay)")
    taf_all = tafmod.build_taf_table(hourly, features.AAR_LEADS)
    taf_all.to_parquet(C.PROC / "taf_features.parquet", index=False)
    aar_t = features.build_aar_table(hourly, taf_all)
    aar_t.to_parquet(C.PROC / "aar_table.parquet", index=False)
    delay_t = features.build_delay_table(hourly, taf_all, features.load_flights())
    delay_t.to_parquet(C.PROC / "delay_table.parquet", index=False)
    print("AAR table", aar_t.shape, "| delay table", delay_t.shape, f"| {time.time() - t:.0f}s")

    t = stage("4/5 train + evaluate AAR forecaster")
    train_aar.main()
    print(f"{time.time() - t:.0f}s")

    t = stage("5/5 train + evaluate severe-delay classifier")
    train_delay.main()
    print(f"{time.time() - t:.0f}s")

    if args.sensitivity:
        t = stage("extra: AAR assumption sensitivity")
        sensitivity.main()
        print(f"{time.time() - t:.0f}s")
    print("\nDone. Launch the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
