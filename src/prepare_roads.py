"""Stage 1: Road Network Extraction and Segmentation.

Downloads or loads the Nagpur drivable road network, partitions roads into
~150m segments, computes curvature (0-1) and junction count, and imputes
missing lanes and speeds based on road hierarchy.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring

from config import (
    CITY_NAME,
    DATA_DIR,
    DISPLAY_CRS,
    FULL_CITY,
    PROJECTED_CRS,
    TARGET_SEGMENT_LENGTH_M,
    TEST_AREA_BBOX,
)

# Imputation dictionaries based on standard urban road hierarchy
DEFAULT_LANES_BY_HIGHWAY: Dict[str, int] = {
    "motorway": 6,
    "trunk": 4,
    "primary": 4,
    "secondary": 2,
    "tertiary": 2,
    "residential": 2,
    "living_street": 1,
    "service": 1,
    "unclassified": 2,
}

DEFAULT_SPEED_BY_HIGHWAY: Dict[str, float] = {
    "motorway": 80.0,
    "trunk": 65.0,
    "primary": 50.0,
    "secondary": 40.0,
    "tertiary": 35.0,
    "residential": 25.0,
    "living_street": 20.0,
    "service": 15.0,
    "unclassified": 30.0,
}


def split_line(line: LineString, segment_length: float) -> List[LineString]:
    """Split a road LineString into smaller segments of target length."""
    if line is None or line.is_empty or line.length == 0:
        return []

    segments: List[LineString] = []
    start = 0.0

    while start < line.length:
        end = min(start + segment_length, line.length)
        part = substring(line, start, end)

        if part.geom_type == "LineString" and part.length > 0:
            segments.append(part)

        start = end

    return segments


def compute_curvature(geometries: gpd.GeoSeries, lengths_m: np.ndarray) -> np.ndarray:
    """Calculate normalized curvature (0.0 straight to 1.0 sharply curved).
    
    Curvature = 1.0 - (Euclidean straight-line distance / road length).
    """
    coords = [np.array(geom.coords) for geom in geometries]
    euclidean = np.array([np.linalg.norm(c[-1] - c[0]) for c in coords])
    lengths = np.maximum(lengths_m, 1.0)
    sinuosity = euclidean / lengths
    curvature = np.clip(1.0 - sinuosity, 0.0, 1.0)
    return curvature


def compute_junction_counts(
    start_lats: np.ndarray,
    start_lons: np.ndarray,
    end_lats: np.ndarray,
    end_lons: np.ndarray,
) -> np.ndarray:
    """Compute junction counts (intersections of >= 3 road segments)."""
    # Round coordinates to ~1.1 meters (5 decimal places) to identify shared nodes
    start_keys = [f"{lat:.5f}_{lon:.5f}" for lat, lon in zip(start_lats, start_lons)]
    end_keys = [f"{lat:.5f}_{lon:.5f}" for lat, lon in zip(end_lats, end_lons)]

    all_keys = pd.Series(start_keys + end_keys)
    degree_counts = all_keys.value_counts().to_dict()

    start_deg = np.array([degree_counts.get(k, 1) for k in start_keys])
    end_deg = np.array([degree_counts.get(k, 1) for k in end_keys])

    # A junction is a node connecting 3 or more segments
    junction_count = (start_deg >= 3).astype(int) + (end_deg >= 3).astype(int)
    return junction_count


def parse_numeric_lanes(raw_val: Optional[str], highway_class: str) -> Tuple[int, bool]:
    """Parse lanes from OSM string or impute based on road class."""
    if pd.notna(raw_val) and str(raw_val).strip() not in ("", "None", "nan"):
        first_token = str(raw_val).split(";")[0].split(",")[0].strip()
        try:
            val = int(float(first_token))
            if 1 <= val <= 10:
                return val, False
        except ValueError:
            pass

    highway_clean = str(highway_class).lower().split(";")[0].strip()
    return DEFAULT_LANES_BY_HIGHWAY.get(highway_clean, 2), True


def parse_numeric_speed(raw_val: Optional[str], highway_class: str) -> Tuple[float, bool]:
    """Parse maxspeed from OSM string or impute based on road class."""
    if pd.notna(raw_val) and str(raw_val).strip() not in ("", "None", "nan"):
        first_token = str(raw_val).split(";")[0].split()[0].strip()
        try:
            val = float(first_token)
            if 10.0 <= val <= 150.0:
                return val, False
        except ValueError:
            pass

    highway_clean = str(highway_class).lower().split(";")[0].strip()
    return DEFAULT_SPEED_BY_HIGHWAY.get(highway_clean, 30.0), True


def normalize_highway_class(raw_val: Optional[str]) -> str:
    """Normalize multi-value or messy highway tags to primary class."""
    if not raw_val or pd.isna(raw_val):
        return "unclassified"
    val = str(raw_val).lower().split(";")[0].strip()
    return val if val else "unclassified"


def prepare_road_segments(
    full_city: bool = FULL_CITY,
    input_existing_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Prepare and save normalized road segments."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    in_path = input_existing_path or (DATA_DIR / "road_segments.geoparquet")
    out_path = output_path or (DATA_DIR / "segments.parquet")

    if in_path.exists():
        print(f"Loading cached road network from: {in_path}")
        roads = gpd.read_parquet(in_path)
    else:
        print(f"Downloading drivable road network for {CITY_NAME}...")
        import osmnx as ox

        ox.settings.use_cache = True
        graph = ox.graph_from_place(CITY_NAME, network_type="drive", simplify=True)
        nodes, edges = ox.graph_to_gdfs(graph)
        edges = edges.to_crs(PROJECTED_CRS)

        records = []
        seg_id = 1
        for _, edge in edges.iterrows():
            geom = edge.geometry
            if geom is None or geom.is_empty or geom.geom_type != "LineString":
                continue
            pieces = split_line(geom, TARGET_SEGMENT_LENGTH_M)
            for piece in pieces:
                records.append({
                    "segment_id": seg_id,
                    "geometry": piece,
                    "length_m": piece.length,
                    "highway": str(edge.get("highway")),
                    "lanes": str(edge.get("lanes")),
                    "maxspeed": str(edge.get("maxspeed")),
                    "oneway": str(edge.get("oneway")),
                    "surface": str(edge.get("surface")),
                    "lit": str(edge.get("lit")),
                })
                seg_id += 1
        roads = gpd.GeoDataFrame(records, geometry="geometry", crs=PROJECTED_CRS)
        roads = roads.to_crs(DISPLAY_CRS)
        centroids = roads.geometry.to_crs(PROJECTED_CRS).centroid.to_crs(DISPLAY_CRS)
        roads["centroid_lon"] = centroids.x
        roads["centroid_lat"] = centroids.y
        roads["start_lon"] = roads.geometry.apply(lambda g: g.coords[0][0])
        roads["start_lat"] = roads.geometry.apply(lambda g: g.coords[0][1])
        roads["end_lon"] = roads.geometry.apply(lambda g: g.coords[-1][0])
        roads["end_lat"] = roads.geometry.apply(lambda g: g.coords[-1][1])

    # Filter to test area if full_city is False
    if not full_city:
        min_lon, min_lat, max_lon, max_lat = TEST_AREA_BBOX
        print(f"Filtering to test area: Dharampeth + Sitabuldi + Wardha Road BBox {TEST_AREA_BBOX}...")
        mask = (
            (roads["centroid_lon"] >= min_lon)
            & (roads["centroid_lon"] <= max_lon)
            & (roads["centroid_lat"] >= min_lat)
            & (roads["centroid_lat"] <= max_lat)
        )
        roads = roads[mask].copy().reset_index(drop=True)
        print(f"Selected {len(roads):,} segments in test area.")
    else:
        print(f"Processing FULL CITY dataset: {len(roads):,} segments.")

    # Re-index segment IDs
    roads["segment_id"] = np.arange(1, len(roads) + 1)

    # Normalize highway
    roads["highway"] = roads["highway"].apply(normalize_highway_class)

    # Compute curvature using projected metric geometries
    metric_geoms = roads.to_crs(PROJECTED_CRS).geometry
    roads["curvature"] = compute_curvature(metric_geoms, roads["length_m"].values)

    # Compute junction count
    roads["junction_count"] = compute_junction_counts(
        roads["start_lat"].values,
        roads["start_lon"].values,
        roads["end_lat"].values,
        roads["end_lon"].values,
    )

    # Normalize lit tag
    if "lit" not in roads.columns:
        roads["lit"] = "no"
    else:
        roads["lit"] = roads["lit"].fillna("no").astype(str).str.lower()
        roads["lit"] = roads["lit"].apply(lambda v: "yes" if v in ("yes", "true", "1") else "no")

    # Impute missing lanes and speed
    parsed_lanes = [
        parse_numeric_lanes(raw, hwy)
        for raw, hwy in zip(roads["lanes"], roads["highway"])
    ]
    parsed_speeds = [
        parse_numeric_speed(raw, hwy)
        for raw, hwy in zip(roads["maxspeed"], roads["highway"])
    ]

    roads["lanes"] = [val for val, _ in parsed_lanes]
    roads["lanes_imputed"] = [imp for _, imp in parsed_lanes]

    roads["maxspeed"] = [val for val, _ in parsed_speeds]
    roads["maxspeed_imputed"] = [imp for _, imp in parsed_speeds]

    # Surface & oneway normalization
    roads["oneway"] = roads["oneway"].fillna("no").astype(str).str.lower().apply(
        lambda v: True if v in ("yes", "true", "1") else False
    )
    roads["surface"] = roads["surface"].fillna("paved").astype(str)

    # Save to data/segments.parquet
    final_columns = [
        "segment_id",
        "geometry",
        "length_m",
        "highway",
        "lanes",
        "maxspeed",
        "oneway",
        "surface",
        "lit",
        "curvature",
        "junction_count",
        "centroid_lon",
        "centroid_lat",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "lanes_imputed",
        "maxspeed_imputed",
    ]
    result = roads[final_columns].copy()
    result = result.to_crs(DISPLAY_CRS)
    result.to_parquet(out_path, index=False)

    # Print summary statistics
    total_segments = len(result)
    total_km = result["length_m"].sum() / 1000.0
    pct_lanes_imputed = result["lanes_imputed"].mean() * 100.0
    pct_speed_imputed = result["maxspeed_imputed"].mean() * 100.0

    print("\n================ STAGE 1: ROAD NETWORK SUMMARY ================")
    print(f"Total Road Segments: {total_segments:,}")
    print(f"Total Road Length:   {total_km:,.2f} km")
    print(f"Lanes Imputed:       {pct_lanes_imputed:.1f}%")
    print(f"Speed Imputed:       {pct_speed_imputed:.1f}%")
    print(f"Mean Curvature:      {result['curvature'].mean():.4f} (range: {result['curvature'].min():.3f} - {result['curvature'].max():.3f})")
    print(f"Junctions / Seg:     {result['junction_count'].value_counts().to_dict()}")
    print("\nRoad Classes (Segment count and total km):")
    class_stats = result.groupby("highway")["length_m"].agg(
        Segments="count",
        Total_km=lambda x: round(x.sum() / 1000.0, 2),
    ).sort_values(by="Segments", ascending=False)
    print(class_stats)
    print(f"\nSaved successfully to: {out_path}")
    print("===============================================================\n")

    return result


def main():
    prepare_road_segments()


if __name__ == "__main__":
    main()
