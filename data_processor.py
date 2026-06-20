"""
ClearPath :: data_processor.py
================================
Loads the raw Astram event feed and turns it into model-ready features.

WHY this module exists:
The raw Astram CSV is an operational log (one row per reported road event). Machine-
learning models cannot consume timestamps, free-text causes, or corridor names directly.
This module is the single, deterministic place where raw rows become numeric features so
that the XGBoost classifier and the LSTM forecaster both train on *identical* assumptions.
Keeping all feature logic here (and re-using it at inference time) avoids train/serve skew.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

SEED = 42
np.random.seed(SEED)

# Resolve paths relative to this file so the script runs from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, "data", "Astram_event_data_anonymized.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "processed_events.csv")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
HOURLY_CSV = os.path.join(BASE_DIR, "hourly_corridor_counts.csv")
RESOLUTION_STATS_PATH = os.path.join(BASE_DIR, "resolution_stats.json")


def load_data(path=RAW_CSV):
    """
    WHAT: Read the raw Astram CSV into a DataFrame and drop rows with no usable target.
    WHY: `low_memory=False` prevents pandas from guessing mixed dtypes column-by-column
    (the file has many NULL-heavy columns). Rows missing `priority` cannot be used for
    supervised training, so we drop those 2 rows rather than guess a label.
    """
    print(f"[load] reading raw events from {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[load] {len(df):,} raw rows, {df.shape[1]} columns")
    before = len(df)
    df = df.dropna(subset=["priority"]).reset_index(drop=True)
    print(f"[load] dropped {before - len(df)} rows with missing priority -> {len(df):,} rows")
    return df


def clean_missing(df):
    """
    WHAT: Fill the known null gaps (corridor ~20 nulls, zone ~4729 nulls) with 'Unknown'.
    WHY: Tree models and encoders need a concrete category. 'Unknown' keeps these rows in
    the training set (dropping 4729 zone-null rows would throw away >half the data) while
    letting the model learn that 'Unknown' is itself a weak signal.
    """
    fill_cols = ["corridor", "zone", "junction", "police_station", "event_cause"]
    for col in fill_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna("Unknown")
            if n:
                print(f"[clean] filled {n} nulls in '{col}' with 'Unknown'")
    # requires_road_closure arrives as bool/str -> coerce to clean 0/1 int.
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    )
    return df


def add_time_features(df):
    """
    WHAT: Parse `start_datetime` into hour, day_of_week, month, is_weekend.
    WHY: Congestion is fundamentally cyclical (rush hours, weekday commutes). Exposing these
    cycles as explicit integer features lets the classifier learn 'morning peak on a weekday'
    patterns that a raw timestamp would hide. `utc=True` normalises the mixed-offset stamps.
    """
    print("[time] parsing start_datetime into cyclical features")
    dt = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"] = dt.dt.hour.fillna(0).astype(int)
    df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)  # Mon=0 .. Sun=6
    df["month"] = dt.dt.month.fillna(1).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["_start_dt"] = dt  # kept internally for rolling/timeseries steps, dropped before save
    return df


def encode_categoricals(df):
    """
    WHAT: Map `event_cause` and `corridor` to dense integer codes; persist the encoders.
    WHY: XGBoost needs numeric inputs. LabelEncoder gives a compact, reproducible mapping.
    We SAVE the fitted encoders so the API/recommender can transform a live request the same
    way the model was trained — this is the contract that prevents train/serve skew.
    """
    print("[encode] label-encoding event_cause and corridor")
    cause_enc, corr_enc = LabelEncoder(), LabelEncoder()
    df["event_cause_encoded"] = cause_enc.fit_transform(df["event_cause"].astype(str))
    df["corridor_encoded"] = corr_enc.fit_transform(df["corridor"].astype(str))
    joblib.dump({"event_cause": cause_enc, "corridor": corr_enc}, ENCODERS_PATH)
    print(f"[encode] {len(corr_enc.classes_)} corridors, "
          f"{len(cause_enc.classes_)} causes -> saved encoders to {ENCODERS_PATH}")
    return df


def add_rolling_count(df):
    """
    WHAT: For each row, count how many events hit the SAME corridor in the prior 2 hours.
    WHY: This is the strongest "is something brewing here right now" signal. A corridor that
    already logged several events in the last 2h is far more likely to escalate. We compute
    it per-corridor on a time-indexed rolling window so the feature is leak-free (only past
    events count). It feeds the classifier as `events_in_corridor_last_2hrs`.
    """
    print("[rolling] computing events_in_corridor_last_2hrs")
    df = df.sort_values("_start_dt").reset_index(drop=True)
    parts = []
    for corridor, g in df.groupby("corridor", sort=False):
        g = g.dropna(subset=["_start_dt"]).set_index("_start_dt").sort_index()
        # rolling('2h') counts rows in the trailing 2h window (inclusive of current row),
        # subtract 1 so the count reflects PRIOR events only.
        g["events_in_corridor_last_2hrs"] = (
            g["id"].rolling("2h").count().astype(int) - 1
        ).clip(lower=0)
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    print(f"[rolling] max prior-2h count = {out['events_in_corridor_last_2hrs'].max()}")
    return out


def add_target(df):
    """
    WHAT: Map priority -> binary target (High=1, Low=0).
    WHY: The classifier's job is to flag high-priority events. A clean 1/0 target makes the
    business meaning ('does this need urgent response?') explicit and lets us read precision/
    recall directly in operational terms.
    """
    df["target"] = (df["priority"].astype(str).str.strip().str.lower() == "high").astype(int)
    print(f"[target] High={int(df['target'].sum())}  Low={int((df['target'] == 0).sum())}")
    return df


def build_hourly_timeseries(df):
    """
    WHAT: Build a per-corridor hourly event-count timeseries and save it.
    WHY: The LSTM forecaster predicts HOW BUSY a corridor will be next hour. It needs a
    regular, gap-free hourly grid per corridor (reindexed so missing hours = 0 events).
    Resampling here guarantees the LSTM trains on evenly-spaced sequences.
    """
    print("[timeseries] building hourly event counts per corridor")
    ts = df.dropna(subset=["_start_dt"]).copy()
    ts["hour_bucket"] = ts["_start_dt"].dt.floor("h")
    grid = (
        ts.groupby(["corridor", "hour_bucket"]).size().rename("event_count").reset_index()
    )
    # Reindex each corridor onto a continuous hourly range so gaps become explicit zeros.
    filled = []
    for corridor, g in grid.groupby("corridor", sort=False):
        full = pd.date_range(g["hour_bucket"].min(), g["hour_bucket"].max(), freq="h")
        g = g.set_index("hour_bucket").reindex(full, fill_value=0)
        g["corridor"] = corridor
        g.index.name = "hour_bucket"
        filled.append(g.reset_index())
    hourly = pd.concat(filled, ignore_index=True)
    hourly = hourly[["corridor", "hour_bucket", "event_count"]]
    hourly.to_csv(HOURLY_CSV, index=False)
    print(f"[timeseries] {len(hourly):,} corridor-hours -> saved to {HOURLY_CSV}")
    return hourly


def print_eda(df):
    """
    WHAT: Print top-5 corridors by event count, peak hours, and the most common cause per corridor.
    WHY: Quick sanity-check / demo value — confirms the pipeline parsed reality correctly and
    surfaces the hotspots that the rest of ClearPath is built to manage.
    """
    print("\n================ EDA SUMMARY ================")
    print("\nTop 5 corridors by event count:")
    print(df["corridor"].value_counts().head(5).to_string())

    print("\nPeak hours (top 5 by event count):")
    print(df["hour"].value_counts().head(5).sort_index(ascending=False).to_string())

    print("\nMost common event cause per (top 5) corridor:")
    top5 = df["corridor"].value_counts().head(5).index
    for c in top5:
        cause = df.loc[df["corridor"] == c, "event_cause"].mode()
        print(f"  {c:<20} -> {cause.iloc[0] if len(cause) else 'n/a'}")
    print("=============================================\n")


def compute_resolution_stats(path=PROCESSED_CSV, save=True):
    """
    WHAT: Compute how long events actually took to resolve (created_date -> closed_datetime),
    overall and per corridor, and save the baseline to resolution_stats.json.
    WHY: ClearPath's officer/barricade/diversion recommendations are only "impressive" if we
    can show a real before/after number. This baseline (the city's ACTUAL historical response
    time) is what the recommender later compares its estimated response time against.
    """
    print("[resolution] computing historical resolution times")
    df = pd.read_csv(path)
    created = pd.to_datetime(df["created_date"], utc=True, errors="coerce")
    closed = pd.to_datetime(df["closed_datetime"], utc=True, errors="coerce")
    df["resolution_minutes"] = (closed - created).dt.total_seconds() / 60.0

    valid = df[(df["resolution_minutes"] > 0) & (df["resolution_minutes"] < 1440)]
    print(f"[resolution] {len(valid):,}/{len(df):,} events have a usable resolution time "
          f"(closed, 0 < duration < 24h)")

    avg_baseline = float(valid["resolution_minutes"].mean())
    per_corridor = (
        valid.groupby("corridor")["resolution_minutes"].mean().round(2).to_dict()
    )

    stats = {
        "avg_baseline_minutes": round(avg_baseline, 2),
        "per_corridor": {k: float(v) for k, v in per_corridor.items()},
        "total_events_with_resolution": int(len(valid)),
    }
    if save:
        with open(RESOLUTION_STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[resolution] saved -> {RESOLUTION_STATS_PATH}")
    print(f"[resolution] Avg event resolution time: {avg_baseline:.1f} minutes")
    return stats


def process(save=True):
    """
    WHAT: Run the full pipeline end-to-end and (optionally) persist processed_events.csv.
    WHY: One callable entrypoint keeps the ordering correct and is reusable from models.py.
    """
    df = load_data()
    df = clean_missing(df)
    df = add_time_features(df)
    df = encode_categoricals(df)
    df = add_rolling_count(df)
    df = add_target(df)
    build_hourly_timeseries(df)
    print_eda(df)

    feature_cols = [
        "id", "corridor", "event_cause", "priority", "target",
        "hour", "day_of_week", "month", "is_weekend",
        "corridor_encoded", "event_cause_encoded",
        "requires_road_closure", "events_in_corridor_last_2hrs",
        "latitude", "longitude", "created_date", "closed_datetime",
    ]
    processed = df[feature_cols].copy()
    if save:
        processed.to_csv(PROCESSED_CSV, index=False)
        print(f"[save] processed dataset ({processed.shape}) -> {PROCESSED_CSV}")
        compute_resolution_stats(PROCESSED_CSV, save=True)
    return processed


if __name__ == "__main__":
    print("=" * 60)
    print("ClearPath :: Data Processor")
    print("=" * 60)
    process()
    print("[done] data_processor complete.")
