"""Stage 2: Feature Engineering & Environmental Layers.

Enriches road segments with:
1. OpenStreetMap POI counts within 100m (schools, hospitals, markets, bus stops, signals),
   is_market_zone, and school_zone.
2. Google Earth Engine environmental rasters (sampled in batches of 5,000 with retries):
   - SRTM Elevation and Slope (~30m resolution)
   - Dynamic World built-up fraction (~10m resolution)
   - WorldPop population density (~100m resolution)
   - VIIRS Nighttime Lights (~500m native resolution, so neighboring segments share values)
3. Derived lighting features:
   - ntl_mean, ntl_log, ntl_percentile, ntl_gradient (vs segments within 500m)
   - lit_tag, dark_exposure, unlit_night

Saves results to data/segments_features.parquet so the runtime backend never
depends on Earth Engine.
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import warnings
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from config import DATA_DIR, DISPLAY_CRS, PROJECTED_CRS
from src.gee_init import init_ee


def load_pois_features(segments_df: gpd.GeoDataFrame) -> pd.DataFrame:
    """Join POI counts within 100m from precomputed POI road dataset."""
    pois_path = DATA_DIR / "road_segments_with_pois.geoparquet"

    if not pois_path.exists():
        warnings.warn(f"POI dataset not found at {pois_path}. Using zero POIs.")
        zeros = np.zeros(len(segments_df), dtype=int)
        return pd.DataFrame({
            "schools": zeros,
            "hospitals": zeros,
            "markets": zeros,
            "bus_stops": zeros,
            "signals": zeros,
            "school_zone": False,
            "is_market_zone": False,
        }, index=segments_df.index)

    pois_df = gpd.read_parquet(pois_path)

    # Align by coordinate key for 100% precision (deduplicating parallel lines)
    pois_df["coord_key"] = (
        pois_df["start_lat"].round(5).astype(str)
        + "_"
        + pois_df["start_lon"].round(5).astype(str)
        + "_"
        + pois_df["end_lat"].round(5).astype(str)
        + "_"
        + pois_df["end_lon"].round(5).astype(str)
    )
    segments_df["coord_key"] = (
        segments_df["start_lat"].round(5).astype(str)
        + "_"
        + segments_df["start_lon"].round(5).astype(str)
        + "_"
        + segments_df["end_lat"].round(5).astype(str)
        + "_"
        + segments_df["end_lon"].round(5).astype(str)
    )

    pois_dedup = pois_df.drop_duplicates(subset=["coord_key"])
    merged = segments_df[["coord_key"]].merge(
        pois_dedup,
        on="coord_key",
        how="left",
    )

    schools = merged["school"].fillna(0).astype(int).values if "school" in merged else np.zeros(len(segments_df), dtype=int)
    hospitals = merged["hospital"].fillna(0).astype(int).values if "hospital" in merged else np.zeros(len(segments_df), dtype=int)
    markets = merged["marketplace"].fillna(0).astype(int).values if "marketplace" in merged else np.zeros(len(segments_df), dtype=int)
    bus_stops = (
        merged["bus_stop"].fillna(0).astype(int)
        + (merged["bus_station"].fillna(0).astype(int) if "bus_station" in merged else 0)
    ).values
    signals = merged["traffic_signals"].fillna(0).astype(int).values if "traffic_signals" in merged else np.zeros(len(segments_df), dtype=int)

    near_school = merged["near_school"].fillna(False).astype(bool).values if "near_school" in merged else (schools > 0)
    near_market = merged["near_market"].fillna(False).astype(bool).values if "near_market" in merged else (markets > 0)

    school_zone = (schools > 0) | near_school
    is_market_zone = (markets > 0) | near_market

    poi_features = pd.DataFrame({
        "schools": schools,
        "hospitals": hospitals,
        "markets": markets,
        "bus_stops": bus_stops,
        "signals": signals,
        "school_zone": school_zone,
        "is_market_zone": is_market_zone,
    }, index=segments_df.index)

    return poi_features


def safe_float(val: Any, default: float) -> float:
    """Safely convert value to float or return default."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if not np.isnan(f) else default
    except (ValueError, TypeError):
        return default


def sample_earth_engine_batch(
    composite_image: Any,
    batch_coords: List[Tuple[int, float, float]],
    max_retries: int = 3,
) -> Dict[int, Dict[str, float]]:
    """Sample environmental rasters for a batch of points with exponential retries."""
    import ee

    features = [
        ee.Feature(ee.Geometry.Point([lon, lat]), {"segment_id": int(sid)})
        for sid, lon, lat in batch_coords
    ]
    fc = ee.FeatureCollection(features)

    # Sample all bands at 30m scale
    sampled_fc = composite_image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=30,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            res = sampled_fc.getInfo()
            batch_results = {}
            for f in res.get("features", []):
                props = f.get("properties", {})
                sid = props.get("segment_id")
                if sid is not None:
                    batch_results[int(sid)] = {
                        "elevation": safe_float(props.get("elevation"), 310.0),
                        "slope": safe_float(props.get("slope"), 1.2),
                        "built_up": safe_float(props.get("built_up"), 0.65),
                        "population_density": safe_float(props.get("population_density"), 100.0),
                        "night_lights": safe_float(props.get("night_lights"), 30.0),
                    }
            return batch_results
        except Exception as err:
            last_error = err
            sleep_time = 2 ** (attempt + 1)
            print(f"  [GEE Retry {attempt + 1}/{max_retries}] Error: {err}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    raise RuntimeError(f"Earth Engine sampling failed after {max_retries} attempts: {last_error}")


def extract_earth_engine_features(
    segments_df: gpd.GeoDataFrame,
    batch_size: int = 5000,
) -> pd.DataFrame:
    """Extract environmental raster features from Earth Engine in batches with fallback."""
    print("\nConnecting to Earth Engine for environmental features...")
    try:
        import ee
        init_ee()

        # 1. SRTM Elevation & Slope (~30m resolution)
        srtm = ee.Image("USGS/SRTMGL1_003")
        elevation = srtm.select("elevation").rename("elevation").unmask(310.0)
        slope = ee.Terrain.slope(elevation).rename("slope").unmask(1.2)

        # 2. Dynamic World Built-up Fraction (~10m resolution, 2023 annual mean)
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate("2023-01-01", "2023-12-31")
            .select("built")
            .mean()
            .rename("built_up")
            .unmask(0.65)
        )

        # 3. WorldPop Population Density (~100m resolution)
        wp = (
            ee.ImageCollection("WorldPop/GP/100m/pop")
            .filter(ee.Filter.eq("country", "IND"))
            .first()
            .rename("population_density")
            .unmask(100.0)
        )

        # 4. VIIRS Nighttime Lights (monthly composite, 12-month median)
        # Note: VIIRS native resolution is ~500m, so multiple adjacent ~150m road segments share values.
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate("2023-01-01", "2024-01-01")
            .select("avg_rad")
            .median()
            .rename("night_lights")
            .unmask(30.0)
        )

        composite = ee.Image.cat([elevation, slope, dw, wp, viirs])

        # Prepare coordinate tuples: (segment_id, centroid_lon, centroid_lat)
        points_data = [
            (int(row.segment_id), float(row.centroid_lon), float(row.centroid_lat))
            for row in segments_df.itertuples()
        ]

        total_pts = len(points_data)
        print(f"Sampling {total_pts:,} points in batches of up to {batch_size:,}...")

        all_results: Dict[int, Dict[str, float]] = {}
        for start_idx in range(0, total_pts, batch_size):
            end_idx = min(start_idx + batch_size, total_pts)
            batch = points_data[start_idx:end_idx]
            print(f"  Sampling points {start_idx + 1:,} to {end_idx:,} of {total_pts:,}...")
            t0 = time.time()
            batch_res = sample_earth_engine_batch(composite, batch)
            all_results.update(batch_res)
            print(f"  Batch completed in {time.time() - t0:.2f}s ({len(batch_res):,} results).")

        # Build DataFrame
        records = []
        for sid in segments_df["segment_id"].values:
            props = all_results.get(int(sid), {
                "elevation": 310.0,
                "slope": 1.2,
                "built_up": 0.65,
                "population_density": 100.0,
                "night_lights": 30.0,
            })
            records.append(props)

        ee_df = pd.DataFrame(records, index=segments_df.index)
        print("Earth Engine sampling completed successfully.")
        return ee_df

    except Exception as exc:
        warnings.warn(
            f"Earth Engine extraction encountered an error: {exc}.\n"
            "Using calibrated environmental placeholders per fallback specification."
        )
        # Synthetic fallback values calibrated for Nagpur
        rng = np.random.default_rng(42)
        n = len(segments_df)
        return pd.DataFrame({
            "elevation": rng.normal(310.0, 15.0, size=n).clip(280.0, 360.0),
            "slope": rng.exponential(1.2, size=n).clip(0.1, 12.0),
            "built_up": rng.beta(5, 2, size=n).clip(0.1, 1.0),
            "population_density": rng.lognormal(4.6, 0.5, size=n).clip(10.0, 500.0),
            "night_lights": rng.lognormal(3.2, 0.6, size=n).clip(2.0, 90.0),
        }, index=segments_df.index)


def derive_night_light_features(
    segments_df: gpd.GeoDataFrame,
    night_lights: np.ndarray,
    lit_series: pd.Series,
) -> pd.DataFrame:
    """Derive lighting and visibility features: ntl_mean, ntl_log, ntl_percentile,

    ntl_gradient (vs segments within 500m), lit_tag, dark_exposure, unlit_night.
    """
    ntl = np.maximum(night_lights, 0.0)
    ntl_mean = float(np.mean(ntl))
    ntl_log = np.log1p(ntl)

    # Percentile rank (0.0 to 1.0)
    sorted_indices = np.argsort(ntl)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(ntl))
    ntl_percentile = ranks / max(len(ntl) - 1, 1)

    # Spatial neighborhood gradient within 500m using KDTree
    projected = segments_df.to_crs(PROJECTED_CRS)
    coords = np.column_stack([projected.geometry.centroid.x, projected.geometry.centroid.y])
    tree = KDTree(coords)
    neighbors_list = tree.query_ball_point(coords, r=500.0)

    local_means = np.array([np.mean(ntl[indices]) if len(indices) > 0 else ntl[i] for i, indices in enumerate(neighbors_list)])
    ntl_gradient = ntl - local_means

    # lit_tag normalization
    lit_tag = lit_series.fillna("no").astype(str).str.lower().apply(
        lambda x: "yes" if x in ("yes", "true", "1") else "no"
    )

    # Default dark exposure & unlit night (at night: is_night=1)
    dark_exposure_night = 1.0 * (1.0 - ntl_percentile)
    unlit_night = (lit_tag == "no").astype(int)

    return pd.DataFrame({
        "night_lights": ntl,
        "ntl_mean": ntl_mean,
        "ntl_log": ntl_log,
        "ntl_percentile": ntl_percentile,
        "ntl_gradient": ntl_gradient,
        "lit_tag": lit_tag,
        "dark_exposure_night": dark_exposure_night,
        "unlit_night": unlit_night,
    }, index=segments_df.index)


def prepare_features(
    input_segments_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Run Stage 2: enrich road segments with POIs and environmental layers."""
    in_path = input_segments_path or (DATA_DIR / "segments.parquet")
    out_path = output_path or (DATA_DIR / "segments_features.parquet")

    if not in_path.exists():
        raise FileNotFoundError(f"Input road segments not found at {in_path}. Run Stage 1 first.")

    print(f"Loading Stage 1 road segments from: {in_path}")
    segments = gpd.read_parquet(in_path)
    print(f"Loaded {len(segments):,} road segments.")

    # 1. POI enrichment
    print("\nEnriching segments with OpenStreetMap POI buffers (100m)...")
    poi_df = load_pois_features(segments)
    print(f"  Segments in school zones: {poi_df['school_zone'].sum():,}")
    print(f"  Segments in market zones: {poi_df['is_market_zone'].sum():,}")

    # 2. Earth Engine environmental layers
    ee_df = extract_earth_engine_features(segments, batch_size=5000)

    # 3. Derived lighting features
    print("\nDeriving lighting & spatial gradient features...")
    lighting_df = derive_night_light_features(
        segments,
        ee_df["night_lights"].values,
        segments["lit"] if "lit" in segments else pd.Series("no", index=segments.index),
    )

    # Combine everything
    result = segments.copy()
    for col in poi_df.columns:
        result[col] = poi_df[col]

    result["elevation"] = ee_df["elevation"].values
    result["slope"] = ee_df["slope"].values
    result["built_up"] = ee_df["built_up"].values
    result["population_density"] = ee_df["population_density"].values

    for col in lighting_df.columns:
        result[col] = lighting_df[col].values

    # Clean up temporary columns if any
    if "coord_key" in result.columns:
        result = result.drop(columns=["coord_key"])

    result = result.to_crs(DISPLAY_CRS)
    result.to_parquet(out_path, index=False)

    print("\n================ STAGE 2: FEATURES SUMMARY ================")
    print(f"Enriched Segments:   {len(result):,}")
    print(f"Mean Elevation:      {result['elevation'].mean():.1f} m (range: {result['elevation'].min():.1f} - {result['elevation'].max():.1f} m)")
    print(f"Mean Slope:          {result['slope'].mean():.2f}° (max: {result['slope'].max():.2f}°)")
    print(f"Mean Built-up Ratio: {result['built_up'].mean():.3f}")
    print(f"Mean Population Den: {result['population_density'].mean():.1f} people/100m")
    print(f"Mean Night Lights:   {result['night_lights'].mean():.2f} nW/cm²/sr")
    print(f"Night Light Gradient (500m): mean={result['ntl_gradient'].mean():.3f}, std={result['ntl_gradient'].std():.3f}")
    print(f"School Zones:        {result['school_zone'].sum():,} ({result['school_zone'].mean() * 100:.1f}%)")
    print(f"Market Zones:        {result['is_market_zone'].sum():,} ({result['is_market_zone'].mean() * 100:.1f}%)")
    print(f"Lit Tag:             {result['lit_tag'].value_counts().to_dict()}")
    print(f"Saved successfully to: {out_path}")
    print("===========================================================\n")

    return result


def main():
    prepare_features()


if __name__ == "__main__":
    main()
