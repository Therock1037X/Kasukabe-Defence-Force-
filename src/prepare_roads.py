
import osmnx as ox
import geopandas as gpd
import pandas as pd

from shapely.ops import substring

from config import (
    DATA_DIR,
    CITY_NAME,
    DISPLAY_CRS,
    PROJECTED_CRS,
    TARGET_SEGMENT_LENGTH_M,
)

# Reproducibility and OSM settings
ox.settings.use_cache = True
ox.settings.log_console = True


def split_line(line, segment_length):
    """Split a road LineString into smaller segments."""
    if line is None or line.is_empty or line.length == 0:
        return []

    segments = []
    start = 0.0

    while start < line.length:
        end = min(start + segment_length, line.length)
        part = substring(line, start, end)

        if part.geom_type == "LineString" and part.length > 0:
            segments.append(part)

        start = end

    return segments


def normalize_osm_value(value):
    """Convert OSM tags into consistent, GeoParquet-friendly values."""
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)

    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None

    return str(value)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading drivable roads for {CITY_NAME}...")

    graph = ox.graph_from_place(
        CITY_NAME,
        network_type="drive",
        simplify=True,
    )

    nodes, edges = ox.graph_to_gdfs(graph)

    print(f"Downloaded road edges: {len(edges):,}")
    print(f"Downloaded road nodes: {len(nodes):,}")

    # Use projected coordinates for measurements in metres.
    edges = edges.to_crs(PROJECTED_CRS)

    records = []
    segment_id = 1

    for _, edge in edges.iterrows():
        geometry = edge.geometry

        if geometry is None or geometry.is_empty:
            continue

        if geometry.geom_type != "LineString":
            continue

        pieces = split_line(
            geometry,
            TARGET_SEGMENT_LENGTH_M,
        )

        for piece in pieces:
            record = {
                "segment_id": segment_id,
                "geometry": piece,
                "length_m": piece.length,
                "highway": normalize_osm_value(edge.get("highway")),
                "lanes": normalize_osm_value(edge.get("lanes")),
                "maxspeed": normalize_osm_value(edge.get("maxspeed")),
                "oneway": normalize_osm_value(edge.get("oneway")),
                "surface": normalize_osm_value(edge.get("surface")),
                "source_u": int(edge.name[0]),
                "source_v": int(edge.name[1]),
            }

            records.append(record)
            segment_id += 1

    segments = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=PROJECTED_CRS,
    )

    if segments.empty:
        raise RuntimeError("No road segments were created.")

    # Convert road geometry to geographic coordinates for display.
    geographic = segments.to_crs(DISPLAY_CRS)

    # Calculate centroids in the projected CRS to avoid geographic warnings.
    centroids = segments.geometry.centroid.to_crs(DISPLAY_CRS)

    geographic["start_lon"] = geographic.geometry.apply(
        lambda geom: geom.coords[0][0]
    )
    geographic["start_lat"] = geographic.geometry.apply(
        lambda geom: geom.coords[0][1]
    )
    geographic["end_lon"] = geographic.geometry.apply(
        lambda geom: geom.coords[-1][0]
    )
    geographic["end_lat"] = geographic.geometry.apply(
        lambda geom: geom.coords[-1][1]
    )
    geographic["centroid_lon"] = centroids.x
    geographic["centroid_lat"] = centroids.y

    output_path = DATA_DIR / "road_segments.geoparquet"
    geographic.to_parquet(output_path, index=False)

    print("\n===== ROAD NETWORK SUMMARY =====")
    print(f"Road segments created: {len(geographic):,}")
    print(
        f"Total road length: "
        f"{geographic['length_m'].sum() / 1000:.2f} km"
    )

    print("\nSegments by road class:")
    print(
        geographic["highway"]
        .astype(str)
        .value_counts()
        .head(15)
    )

    print("\nMissing-value percentages:")
    for column in ["lanes", "maxspeed", "surface"]:
        missing_pct = geographic[column].isna().mean() * 100
        print(f"{column}: {missing_pct:.2f}%")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
