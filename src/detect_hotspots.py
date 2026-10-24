
import numpy as np
import pandas as pd
import geopandas as gpd

from config import DATA_DIR, OUTPUT_DIR


def main():
    # Input and output paths
    roads_path = DATA_DIR / "road_segments_with_pois.geoparquet"
    events_path = DATA_DIR / "synthetic_accident_events.parquet"

    risk_output_path = OUTPUT_DIR / "road_segments_with_risk.geoparquet"
    hotspots_output_path = OUTPUT_DIR / "hotspots.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load road segments and synthetic accident events
    # ---------------------------------------------------------
    print("Loading road segments and synthetic events...")

    if not roads_path.exists():
        raise FileNotFoundError(
            f"Road dataset not found: {roads_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Event dataset not found: {events_path}"
        )

    roads = gpd.read_parquet(roads_path)
    events = pd.read_parquet(events_path)

    required_road_columns = {"segment_id", "length_m", "geometry"}
    required_event_columns = {"segment_id", "severity"}

    missing_roads = required_road_columns - set(roads.columns)
    missing_events = required_event_columns - set(events.columns)

    if missing_roads:
        raise ValueError(
            f"Road dataset is missing columns: {sorted(missing_roads)}"
        )

    if missing_events:
        raise ValueError(
            f"Event dataset is missing columns: {sorted(missing_events)}"
        )

    # Ensure the event source is clearly identified.
    if "is_synthetic" not in events.columns:
        raise ValueError(
            "Missing is_synthetic column. "
            "Cannot verify that events are simulated."
        )

    if not events["is_synthetic"].fillna(False).all():
        raise ValueError(
            "Expected synthetic events only. "
            "Check the input dataset before continuing."
        )

    print(f"Road segments loaded: {len(roads):,}")
    print(f"Synthetic events loaded: {len(events):,}")

    # ---------------------------------------------------------
    # 2. Clean identifiers and severity labels
    # ---------------------------------------------------------
    roads = roads.copy()
    events = events.copy()

    roads["segment_id"] = roads["segment_id"].astype(str)
    events["segment_id"] = events["segment_id"].astype(str)

    roads["length_m"] = pd.to_numeric(
        roads["length_m"], errors="coerce"
    )

    if roads["segment_id"].duplicated().any():
        raise ValueError(
            "Road segment IDs are not unique. "
            "Resolve duplicate segments before scoring."
        )

    valid_severities = {"Minor", "Serious", "Fatal"}

    events = events.loc[
        events["severity"].isin(valid_severities)
    ].copy()

    if events.empty:
        raise ValueError(
            "No events with valid Minor, Serious or Fatal labels."
        )

    # ---------------------------------------------------------
    # 3. Aggregate events for each road segment
    # ---------------------------------------------------------
    print("Aggregating events by road segment...")

    event_counts = (
        events.groupby("segment_id")
        .size()
        .rename("simulated_event_count")
        .reset_index()
    )

    severity_counts = (
        pd.crosstab(events["segment_id"], events["severity"])
        .reindex(columns=["Minor", "Serious", "Fatal"], fill_value=0)
        .rename(columns={
            "Minor": "minor_count",
            "Serious": "serious_count",
            "Fatal": "fatal_count",
        })
        .reset_index()
    )

    event_summary = event_counts.merge(
        severity_counts,
        on="segment_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 4. Join event statistics to every road segment
    # ---------------------------------------------------------
    print("Joining event statistics to road segments...")

    result = roads.merge(
        event_summary,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "simulated_event_count",
        "minor_count",
        "serious_count",
        "fatal_count",
    ]

    result[count_columns] = (
        result[count_columns].fillna(0).astype(int)
    )

    # ---------------------------------------------------------
    # 5. Calculate a transparent severity-weighted index
    # ---------------------------------------------------------
    # These are illustrative weights, not medically or
    # statistically validated severity costs.
    #
    # Minor = 1 point
    # Serious = 2 points
    # Fatal = 4 points

    result["severity_weighted_events"] = (
        result["minor_count"]
        + 2 * result["serious_count"]
        + 4 * result["fatal_count"]
    )

    # Adjust for road-segment length to reduce the bias
    # that longer segments have more opportunities for events.
    result["length_km"] = result["length_m"] / 1000.0

    result["weighted_events_per_km"] = np.where(
        result["length_km"] > 0,
        result["severity_weighted_events"] / result["length_km"],
        np.nan,
    )

    # ---------------------------------------------------------
    # 6. Convert the index into a relative score from 0 to 100
    # ---------------------------------------------------------
    # Percentile ranking is calculated only among segments
    # with at least one simulated event.
    # It is NOT a probability of an accident.

    result["risk_score"] = 0.0

    has_events = (
        (result["simulated_event_count"] > 0)
        & result["weighted_events_per_km"].notna()
    )

    if has_events.any():
        result.loc[has_events, "risk_score"] = (
            result.loc[has_events, "weighted_events_per_km"]
            .rank(method="average", pct=True)
            .mul(100)
            .round(2)
        )

    # ---------------------------------------------------------
    # 7. Assign relative hotspot categories
    # ---------------------------------------------------------
    result["risk_category"] = "No simulated events"

    result.loc[
        has_events & (result["risk_score"] < 50),
        "risk_category",
    ] = "Low"

    result.loc[
        has_events
        & (result["risk_score"] >= 50)
        & (result["risk_score"] < 80),
        "risk_category",
    ] = "Moderate"

    result.loc[
        has_events
        & (result["risk_score"] >= 80)
        & (result["risk_score"] < 95),
        "risk_category",
    ] = "High"

    result.loc[
        has_events & (result["risk_score"] >= 95),
        "risk_category",
    ] = "Very High"

    # ---------------------------------------------------------
    # 8. Save the complete geospatial dataset
    # ---------------------------------------------------------
    result.to_parquet(
        risk_output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 9. Save a shortlist of the top 100 hotspots
    # ---------------------------------------------------------
    hotspot_columns = [
        column
        for column in [
            "segment_id",
            "risk_score",
            "risk_category",
            "simulated_event_count",
            "minor_count",
            "serious_count",
            "fatal_count",
            "severity_weighted_events",
            "weighted_events_per_km",
            "length_m",
            "highway",
            "centroid_lon",
            "centroid_lat",
        ]
        if column in result.columns
    ]

    hotspots = (
        result.loc[has_events, hotspot_columns]
        .sort_values(
            ["risk_score", "weighted_events_per_km"],
            ascending=False,
        )
        .head(100)
    )

    hotspots.to_csv(
        hotspots_output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 10. Print the summary
    # ---------------------------------------------------------
    print("\n===== ROADWATCH NAGPUR: HOTSPOT SUMMARY =====")
    print(f"Road segments analysed: {len(result):,}")
    print(f"Segments with simulated events: {int(has_events.sum()):,}")
    print(
        "Segments without simulated events: "
        f"{int((~has_events).sum()):,}"
    )

    print("\nRisk category distribution:")
    print(
        result["risk_category"]
        .value_counts()
        .reindex(
            [
                "Very High",
                "High",
                "Moderate",
                "Low",
                "No simulated events",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\nTop 10 simulated hotspots:")
    display_columns = [
        column
        for column in [
            "segment_id",
            "risk_score",
            "risk_category",
            "simulated_event_count",
            "serious_count",
            "fatal_count",
            "highway",
        ]
        if column in hotspots.columns
    ]

    print(hotspots[display_columns].head(10).to_string(index=False))

    print(f"\nGeospatial output saved to: {risk_output_path}")
    print(f"Hotspot CSV saved to: {hotspots_output_path}")

    print(
        "\nIMPORTANT: Scores are relative rankings derived from "
        "synthetic events. They are not validated real-world "
        "accident probabilities or confirmed dangerous locations."
    )


if __name__ == "__main__":
    main()
    