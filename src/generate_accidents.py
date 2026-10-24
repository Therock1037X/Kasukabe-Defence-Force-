
import numpy as np
import pandas as pd
import geopandas as gpd

from config import DATA_DIR, RANDOM_SEED, DATA_IS_SYNTHETIC


def main():
    input_path = DATA_DIR / "road_segments_with_pois.geoparquet"
    output_path = DATA_DIR / "synthetic_accident_events.parquet"

    if not DATA_IS_SYNTHETIC:
        raise ValueError(
            "This script is intended only for synthetic-data generation."
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Road dataset not found: {input_path}"
        )

    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading road segments and POI features...")
    roads = gpd.read_parquet(input_path)
    print(f"Road segments loaded: {len(roads):,}")

    # Normalize road classifications.
    highway = roads["highway"].fillna("unknown").astype(str).str.lower()

    # Approximate road-class factors for simulation only.
    road_factor = np.select(
        [
            highway.str.contains("motorway|trunk", regex=True).to_numpy(),
            highway.str.contains("primary", regex=False).to_numpy(),
            highway.str.contains("secondary", regex=False).to_numpy(),
            highway.str.contains("tertiary", regex=False).to_numpy(),
        ],
        [2.0, 1.7, 1.4, 1.15],
        default=1.0,
    )

    # POI counts are used as synthetic simulation signals,
    # not as proof of actual accident risk.
    def poi_count(column):
        if column not in roads.columns:
            return np.zeros(len(roads), dtype=float)
        return (
            pd.to_numeric(roads[column], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

    schools = poi_count("school")
    markets = poi_count("marketplace")
    signals = poi_count("traffic_signals")
    bus_stops = poi_count("bus_stop")

    # Bounded multipliers prevent unusually high POI counts
    # from dominating the simulation.
    poi_factor = (
        1.0
        + 0.08 * np.minimum(schools, 3)
        + 0.06 * np.minimum(markets, 3)
        + 0.03 * np.minimum(bus_stops, 5)
    )

    # Simulate three calendar years of events.
    years = 3
    length_factor = np.clip(
        roads["length_m"].fillna(150).to_numpy() / 150.0,
        0.05,
        1.0,
    )

    # Fictional baseline rate, chosen for a manageable demo dataset.
    annual_rate = 0.015
    expected_events = (
        annual_rate * years * road_factor * poi_factor * length_factor
    )

    event_counts = rng.poisson(expected_events)
    road_indices = np.repeat(
        np.arange(len(roads)),
        event_counts,
    )

    if len(road_indices) == 0:
        raise RuntimeError(
            "No events were generated. Check the simulation parameters."
        )

    print(f"Synthetic events generated: {len(road_indices):,}")

    # Select road features for each generated event.
    events = roads.iloc[road_indices].copy().reset_index(drop=True)

    events["event_id"] = [
        f"SYN-{i:07d}" for i in range(1, len(events) + 1)
    ]

    # Generate timestamps across 2023–2025.
    start_date = pd.Timestamp("2023-01-01")
    random_days = rng.integers(0, 365 * years, size=len(events))
    random_hours = rng.choice(
        np.arange(24),
        size=len(events),
        p=np.array([
            0.015, 0.010, 0.008, 0.008, 0.010, 0.020,
            0.040, 0.065, 0.070, 0.050, 0.040, 0.040,
            0.045, 0.040, 0.040, 0.045, 0.060, 0.085,
            0.095, 0.080, 0.055, 0.040, 0.025, 0.014,
        ]),
    )

    events["event_datetime"] = (
        start_date
        + pd.to_timedelta(random_days, unit="D")
        + pd.to_timedelta(random_hours, unit="h")
    )

    events["year"] = events["event_datetime"].dt.year
    events["month"] = events["event_datetime"].dt.month
    events["hour"] = events["event_datetime"].dt.hour
    events["day_of_week"] = events["event_datetime"].dt.dayofweek

    # Simulate weather with a simplified monsoon pattern.
    monsoon = events["month"].isin([6, 7, 8, 9]).to_numpy()
    rain_probability = np.where(monsoon, 0.35, 0.08)
    cloudy_probability = np.where(monsoon, 0.35, 0.22)

    random_weather = rng.random(len(events))
    events["weather"] = np.select(
        [
            random_weather < rain_probability,
            random_weather < rain_probability + cloudy_probability,
        ],
        ["Rain", "Cloudy"],
        default="Clear",
    )

    events["visibility_km"] = np.select(
        [
            events["weather"].eq("Rain").to_numpy(),
            events["weather"].eq("Cloudy").to_numpy(),
        ],
        [3.0, 7.0],
        default=10.0,
    )

    # Generate fictional severity labels using simulation rules.
    # These labels are NOT learned from real accident records.
    high_speed_road = np.isin(
        road_indices,
        np.array([], dtype=int),
    )
    event_highway = events["highway"].fillna("unknown").astype(str).str.lower()

    high_speed_road = event_highway.str.contains(
        "motorway|trunk|primary", regex=True
    ).to_numpy()
    night = (
        (events["hour"].to_numpy() >= 22)
        | (events["hour"].to_numpy() <= 5)
    )
    rain = events["weather"].eq("Rain").to_numpy()

    p_fatal = np.clip(
        0.06
        + 0.035 * high_speed_road
        + 0.025 * rain
        + 0.02 * night,
        0.04,
        0.18,
    )

    p_serious = np.clip(
        0.28
        + 0.04 * high_speed_road
        + 0.025 * rain
        + 0.02 * night,
        0.20,
        0.45,
    )

    draw = rng.random(len(events))
    events["severity"] = np.select(
        [
            draw < p_fatal,
            draw < p_fatal + p_serious,
        ],
        ["Fatal", "Serious"],
        default="Minor",
    )

    # Keep only useful event attributes and mapping coordinates.
    keep_columns = [
        "event_id",
        "segment_id",
        "event_datetime",
        "year",
        "month",
        "hour",
        "day_of_week",
        "weather",
        "visibility_km",
        "severity",
        "highway",
        "length_m",
        "lanes",
        "maxspeed",
        "oneway",
        "surface",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "centroid_lon",
        "centroid_lat",
    ]

    for column in [
        "school",
        "hospital",
        "marketplace",
        "bus_station",
        "bus_stop",
        "fuel",
        "police",
        "traffic_signals",
        "near_school",
        "near_market",
    ]:
        if column in events.columns:
            keep_columns.append(column)

    events = events[keep_columns].copy()
    events["data_source"] = "SYNTHETIC"
    events["is_synthetic"] = True

    events.to_parquet(output_path, index=False)

    print("\n===== SYNTHETIC EVENT SUMMARY =====")
    print(f"Total events: {len(events):,}")
    print("\nSeverity distribution:")
    print(events["severity"].value_counts())
    print("\nWeather distribution:")
    print(events["weather"].value_counts())
    print("\nEvents by year:")
    print(events["year"].value_counts().sort_index())
    print(f"\nSaved to: {output_path}")
    print("\nWARNING: These are simulated events, not real accident records.")


if __name__ == "__main__":
    main()
