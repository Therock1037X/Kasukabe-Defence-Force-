
import osmnx as ox
import geopandas as gpd
import pandas as pd

from config import DATA_DIR, CITY_NAME, PROJECTED_CRS, POI_BUFFER_M

ox.settings.use_cache = True
ox.settings.log_console = True

POI_TAGS = {
    "amenity": [
        "school",
        "hospital",
        "marketplace",
        "bus_station",
        "fuel",
        "police",
    ],
    "highway": ["bus_stop", "traffic_signals"],
}


def main():
    roads_path = DATA_DIR / "road_segments.geoparquet"
    output_path = DATA_DIR / "road_segments_with_pois.geoparquet"

    print("Loading road segments...")
    roads = gpd.read_parquet(roads_path).to_crs(PROJECTED_CRS)
    print(f"Loaded {len(roads):,} road segments.")

    print(f"Downloading POIs for {CITY_NAME}...")
    pois = ox.features_from_place(CITY_NAME, tags=POI_TAGS)

    if pois.empty:
        raise RuntimeError("No POIs were returned by OpenStreetMap.")

    # Convert POI features into representative points.
    pois = pois.reset_index()
    pois = pois[pois.geometry.notna()].copy()
    pois = gpd.GeoDataFrame(
        pois,
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    pois["poi_type"] = None

    for tag, values in POI_TAGS.items():
        if tag in pois.columns:
            mask = pois[tag].isin(values)
            pois.loc[mask, "poi_type"] = pois.loc[mask, tag]

    pois = pois[pois["poi_type"].notna()].copy()
    pois["geometry"] = pois.geometry.representative_point()

    print(f"POI features found: {len(pois):,}")
    print("Counting POIs within 100 metres of each road segment...")

    # Buffer each road segment by the configured distance.
    buffers = roads[["segment_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(POI_BUFFER_M)

    # Count nearby POIs by type.
    joined = gpd.sjoin(
        pois[["poi_type", "geometry"]],
        buffers,
        how="inner",
        predicate="within",
    )

    if joined.empty:
        counts = pd.DataFrame(index=pd.Index([], name="segment_id"))
    else:
        counts = (
            joined.groupby(["segment_id", "poi_type"])
            .size()
            .unstack(fill_value=0)
        )

    # Add the counts to every road segment.
    result = roads.merge(
        counts,
        left_on="segment_id",
        right_index=True,
        how="left",
    )

    poi_types = sorted(pois["poi_type"].unique())

    for poi_type in poi_types:
        if poi_type not in result.columns:
            result[poi_type] = 0
        result[poi_type] = result[poi_type].fillna(0).astype(int)

    result["near_school"] = (
        result["school"].gt(0)
        if "school" in result.columns
        else False
    )
    result["near_market"] = (
        result["marketplace"].gt(0)
        if "marketplace" in result.columns
        else False
    )

    result = result.to_crs("EPSG:4326")
    result.to_parquet(output_path, index=False)

    print("\n===== POI SUMMARY =====")
    print(f"Road segments enriched: {len(result):,}")
    print(f"POI types: {poi_types}")
    print(f"Segments near schools: {result['near_school'].sum():,}")
    print(f"Segments near markets: {result['near_market'].sum():,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
