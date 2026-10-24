"""Stage 5: High-Performance Vectorized Scoring Engine.

Loads the calibrated severity model and precomputed segment features once,
and rapidly evaluates risk under given environmental conditions:
  score(conditions) -> p_minor, p_serious, p_fatal, risk_index, band, color, top_drivers.

- Vectorized execution across all network segments in < 3 seconds.
- LRU caching by (hour, weather, month, day_type).
- Quantile risk bands:
    Green  (0-50%):   #2ecc71
    Yellow (50-80%):  #f1c40f
    Orange (80-95%):  #e67e22
    Red    (95-100%): #e74c3c
- GeoJSON output format for direct consumption by MapLibre GL frontend.
"""

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, DISPLAY_CRS, MODEL_DIR, OUTPUT_DIR
from src.train import explain_top_drivers

BAND_COLORS = {
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "orange": "#e67e22",
    "red": "#e74c3c",
}


class ScoringEngine:
    """In-memory scoring engine for fast condition-based risk inference."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_meta_path: Optional[Path] = None,
        segments_path: Optional[Path] = None,
    ):
        self.model_path = model_path or (MODEL_DIR / "model.joblib")
        self.meta_path = feature_meta_path or (MODEL_DIR / "feature_list.json")
        self.segments_path = segments_path or (DATA_DIR / "segments_features.parquet")

        print(f"Initializing ScoringEngine...")
        t0 = time.time()
        self.model = joblib.load(self.model_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.segments_gdf = gpd.read_parquet(self.segments_path)
        self.n_segments = len(self.segments_gdf)

        # Pre-extract static numeric feature matrix
        self.num_cols = self.meta["numeric_features"]
        self.cat_cols = self.meta["categorical_features"]
        self.feat_names = self.meta["feature_names"]
        self.categories = self.meta["categories"]

        # Cache category to code mappings
        self.cat_maps = {
            col: {val: i for i, val in enumerate(cats)}
            for col, cats in self.categories.items()
        }

        # Build base numerical array for static segment features
        if "historical_crash_rate" not in self.segments_gdf.columns:
            acc_path = ROOT_DIR / "data" / "synthetic_accidents.parquet"
            if acc_path.exists():
                try:
                    acc_df = pd.read_parquet(acc_path, columns=["segment_id", "month"])
                    train_acc = acc_df[acc_df["month"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
                    rate_map = (train_acc["segment_id"].value_counts() / 8.0).to_dict()
                    self.segments_gdf["historical_crash_rate"] = (
                        self.segments_gdf["segment_id"].map(rate_map).fillna(0.0).astype(float)
                    )
                except Exception:
                    self.segments_gdf["historical_crash_rate"] = 0.1
            else:
                self.segments_gdf["historical_crash_rate"] = 0.1

        # Separate dynamic runtime numerical features from static segment attributes
        self.dynamic_num_names = {"visibility_km", "vehicle_speed", "dark_exposure"}
        self.static_num_cols = [c for c in self.num_cols if c not in self.dynamic_num_names]
        for col in self.static_num_cols:
            if col not in self.segments_gdf.columns:
                self.segments_gdf[col] = 0.0

        self.static_num_matrix = self.segments_gdf[self.static_num_cols].fillna(0.0).values.astype(float)
        self.static_num_indices = np.array([self.num_cols.index(c) for c in self.static_num_cols], dtype=int)
        self.vis_idx = self.num_cols.index("visibility_km")
        self.spd_idx = self.num_cols.index("vehicle_speed")
        self.dark_idx = self.num_cols.index("dark_exposure")

        # Precompute static categorical column arrays
        self.static_cat_arrays = {}
        for col in self.cat_cols:
            if col in ("highway", "lit_tag", "school_zone", "is_market_zone"):
                mapping = self.cat_maps.get(col, {})
                if col in self.segments_gdf.columns:
                    series = self.segments_gdf[col].astype(str)
                    self.static_cat_arrays[col] = np.array([mapping.get(val, 0) for val in series], dtype=float)
                else:
                    self.static_cat_arrays[col] = np.zeros(self.n_segments, dtype=float)

        self.cache: Dict[Tuple[int, str, int, str], Dict[str, Any]] = {}
        print(f"ScoringEngine initialized with {self.n_segments:,} segments in {time.time() - t0:.2f}s.")

    def score(
        self,
        hour: int = 18,
        weather: str = "clear",
        month: int = 10,
        day_type: str = "weekday",
        as_geojson: bool = False,
    ) -> Dict[str, Any]:
        """Score all segments under current conditions."""
        weather_clean = weather.strip().lower()
        if weather_clean not in ("clear", "rain", "fog", "storm"):
            weather_clean = "clear"

        day_of_week = 5 if day_type == "weekend" else 2
        cache_key = (hour, weather_clean, month, day_type)

        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return cached_result if not as_geojson else cached_result["geojson"]

        t_start = time.time()

        # Dynamic condition variables
        is_night = 1 if (hour < 6 or hour >= 19) else 0
        surface_condition = "wet" if weather_clean in ("rain", "storm") else ("wet" if weather_clean == "fog" else "dry")
        light_condition = "dark" if is_night else "daylight"
        visibility_km = (
            10.0 if weather_clean == "clear" else (
                3.5 if weather_clean == "rain" else (
                    1.0 if weather_clean == "storm" else 0.4
                )
            )
        )

        # Vectorized feature matrix construction
        X = np.empty((self.n_segments, len(self.feat_names)), dtype=float)

        # 1. Populate static numerical columns in one vectorized slice
        X[:, self.static_num_indices] = self.static_num_matrix

        # 2. Populate dynamic numerical columns
        X[:, self.vis_idx] = visibility_km
        ntl_pct = self.segments_gdf["ntl_percentile"].values
        X[:, self.dark_idx] = is_night * (1.0 - ntl_pct)
        base_spd = self.segments_gdf["maxspeed"].values.astype(float)
        delta = 6.0 * is_night - (8.0 if weather_clean in ("rain", "storm") else (12.0 if weather_clean == "fog" else 0.0))
        X[:, self.spd_idx] = np.clip(base_spd + delta, 15.0, 120.0)

        # 3. Populate categorical columns
        cat_offset = len(self.num_cols)
        for c_idx, col in enumerate(self.cat_cols):
            out_col_idx = cat_offset + c_idx
            mapping = self.cat_maps.get(col, {})

            if col in self.static_cat_arrays:
                X[:, out_col_idx] = self.static_cat_arrays[col]
            elif col == "weather":
                code = mapping.get(weather_clean, 0)
                X[:, out_col_idx] = code
            elif col == "light":
                code = mapping.get(light_condition, 0)
                X[:, out_col_idx] = code
            elif col == "surface_condition":
                code = mapping.get(surface_condition, 0)
                X[:, out_col_idx] = code
            elif col == "is_night":
                code = mapping.get(str(is_night), mapping.get(is_night, is_night))
                X[:, out_col_idx] = code
            elif col == "hour":
                code = mapping.get(hour, mapping.get(str(hour), hour))
                X[:, out_col_idx] = code
            elif col == "day_of_week":
                code = mapping.get(day_of_week, mapping.get(str(day_of_week), day_of_week))
                X[:, out_col_idx] = code
            elif col == "month":
                code = mapping.get(month, mapping.get(str(month), month))
                X[:, out_col_idx] = code
            else:
                X[:, out_col_idx] = 0.0

        # Model Inference
        probs = self.model.predict_proba(X)
        p_minor = probs[:, 0]
        p_serious = probs[:, 1]
        p_fatal = probs[:, 2]

        # Unified Risk Index formula: P(serious)*1.0 + P(fatal)*2.0
        risk_index = p_serious * 1.0 + p_fatal * 2.0

        # Quantile risk banding: Green 0-50%, Yellow 50-80%, Orange 80-95%, Red 95-100%
        q50 = float(np.quantile(risk_index, 0.50))
        q80 = float(np.quantile(risk_index, 0.80))
        q95 = float(np.quantile(risk_index, 0.95))

        bands = np.where(
            risk_index < q50, "green",
            np.where(risk_index < q80, "yellow",
            np.where(risk_index < q95, "orange", "red"))
        )

        colors = np.array([BAND_COLORS[b] for b in bands])

        # Precompute top drivers for orange and red segments
        top_drivers_list = []
        for i in range(self.n_segments):
            if bands[i] in ("orange", "red"):
                feat_dict = {
                    "is_night": is_night,
                    "night_lights": X[i, self.num_cols.index("night_lights")],
                    "lit_tag": self.segments_gdf["lit_tag"].iloc[i],
                    "vehicle_speed": X[i, self.num_cols.index("vehicle_speed")],
                    "curvature": X[i, self.num_cols.index("curvature")],
                    "junction_count": X[i, self.num_cols.index("junction_count")],
                    "weather": weather_clean,
                    "school_zone": self.segments_gdf["school_zone"].iloc[i],
                    "is_market_zone": self.segments_gdf["is_market_zone"].iloc[i],
                    "slope": X[i, self.num_cols.index("slope")],
                }
                drivers = explain_top_drivers(feat_dict)
            else:
                drivers = ["normal traffic conditions", "standard geometry", "adequate sight distance"]
            top_drivers_list.append(drivers)

        inference_time = time.time() - t_start

        # Build GeoJSON
        features = []
        seg_ids = self.segments_gdf["segment_id"].values
        hwys = self.segments_gdf["highway"].values
        geoms = self.segments_gdf.geometry.values

        for i in range(self.n_segments):
            geom = geoms[i]
            if geom is None or geom.is_empty:
                continue

            coords = list(geom.coords)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(pt[0], 6), round(pt[1], 6)] for pt in coords],
                },
                "properties": {
                    "segment_id": int(seg_ids[i]),
                    "name": f"{str(hwys[i]).capitalize()} Road #{seg_ids[i]}",
                    "risk_index": round(float(risk_index[i]), 4),
                    "p_minor": round(float(p_minor[i]), 4),
                    "p_serious": round(float(p_serious[i]), 4),
                    "p_fatal": round(float(p_fatal[i]), 4),
                    "band": str(bands[i]),
                    "color": str(colors[i]),
                    "top_drivers": top_drivers_list[i],
                },
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_segments": self.n_segments,
                "hour": hour,
                "weather": weather_clean,
                "month": month,
                "day_type": day_type,
                "inference_time_seconds": round(inference_time, 3),
                "thresholds": {"q50": round(q50, 4), "q80": round(q80, 4), "q95": round(q95, 4)},
                "band_counts": pd.Series(bands).value_counts().to_dict(),
            },
        }

        result = {
            "p_minor": p_minor,
            "p_serious": p_serious,
            "p_fatal": p_fatal,
            "risk_index": risk_index,
            "band": bands,
            "color": colors,
            "top_drivers": top_drivers_list,
            "geojson": geojson,
            "metadata": geojson["metadata"],
        }

        self.cache[cache_key] = result
        return geojson if as_geojson else result


_ENGINE_INSTANCE: Optional[ScoringEngine] = None


def get_scoring_engine() -> ScoringEngine:
    """Singleton accessor for ScoringEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ScoringEngine()
    return _ENGINE_INSTANCE


def test_scorer():
    """Test scoring engine performance and latencies."""
    engine = get_scoring_engine()

    print("\nBenchmarking Scoring Engine...")
    t0 = time.time()
    res1 = engine.score(hour=18, weather="rain", month=7, day_type="weekday", as_geojson=True)
    t1 = time.time() - t0
    print(f"Cold evaluation (20k segments + GeoJSON): {t1:.3f}s (Target: < 3.0s)")

    t0 = time.time()
    res2 = engine.score(hour=18, weather="rain", month=7, day_type="weekday", as_geojson=True)
    t2 = time.time() - t0
    print(f"Cached evaluation: {t2 * 1000:.2f}ms")

    meta = res1["metadata"]
    print("\nBand Counts under Rainy Evening Peak:")
    print(meta["band_counts"])
    print("Risk Thresholds:", meta["thresholds"])

    assert t1 < 3.0, f"Latency violation: {t1:.2f}s > 3.0s"
    print("\n[PASS] Scoring Engine passed sub-3 second latency validation!\n")


if __name__ == "__main__":
    test_scorer()
