"""Stage 9: Automated Verification Test Suite for KDF Navigation (RoadWatch Nagpur).

Validates:
1. Road segmentation schema integrity and deduplication.
2. Synthetic accident meteorological constraints and class distribution (~60/30/10).
3. ScoringEngine sub-3s latency, probability normalization, and quantile banding.
4. FastAPI endpoints (/health, /risk, /segment/{id}, /route, /ingest, /live, /alert).
"""

import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app, startup_event, state
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from src.score import get_scoring_engine


@pytest.fixture(scope="session", autouse=True)
def init_application():
    """Ensure startup event has run and application state is populated."""
    startup_event()
    return state


# ---------------------------------------------------------------------------
# 1. Road Segmentation & Feature Data Tests
# ---------------------------------------------------------------------------

def test_segments_data_integrity():
    """Verify segment parquet files exist and have no duplicate segment_ids."""
    seg_file = DATA_DIR / "segments.parquet"
    feat_file = DATA_DIR / "segments_features.parquet"

    assert seg_file.exists(), "segments.parquet must exist"
    assert feat_file.exists(), "segments_features.parquet must exist"

    df_seg = pd.read_parquet(seg_file)
    df_feat = pd.read_parquet(feat_file)

    assert len(df_seg) == len(df_feat), "Segment counts between stages must match"
    assert df_seg["segment_id"].is_unique, "Duplicate segment_ids found in segments.parquet"
    assert df_feat["segment_id"].is_unique, "Duplicate segment_ids found in segments_features.parquet"

    # Physical attribute bounds
    assert (df_seg["lanes"] >= 1).all(), "Lanes must be >= 1"
    assert (df_seg["maxspeed"] >= 10.0).all(), "Maxspeed must be >= 10 km/h"
    assert (df_seg["curvature"] >= 0.0).all() and (df_seg["curvature"] <= 1.0).all(), "Curvature must be between 0 and 1"
    assert (df_seg["length_m"] > 0).all(), "Segment length must be positive"


# ---------------------------------------------------------------------------
# 2. Synthetic Accidents Data Tests
# ---------------------------------------------------------------------------

def test_synthetic_accident_distribution():
    """Verify accident distribution matches ~60% minor, ~30% serious, ~10% fatal."""
    acc_file = DATA_DIR / "synthetic_accidents.parquet"
    assert acc_file.exists(), "synthetic_accidents.parquet must exist"

    df = pd.read_parquet(acc_file)
    assert len(df) >= 10_000, "Should have sufficient accident records"

    dist = df["severity"].value_counts(normalize=True).to_dict()
    minor_pct = dist.get("minor", 0.0)
    serious_pct = dist.get("serious", 0.0)
    fatal_pct = dist.get("fatal", 0.0)

    # Allow 5% tolerance around 60/30/10 target
    assert 0.55 <= minor_pct <= 0.65, f"Minor class ratio {minor_pct:.3f} outside expected 0.55-0.65"
    assert 0.25 <= serious_pct <= 0.35, f"Serious class ratio {serious_pct:.3f} outside expected 0.25-0.35"
    assert 0.06 <= fatal_pct <= 0.14, f"Fatal class ratio {fatal_pct:.3f} outside expected 0.06-0.14"

    # Assert zero impossible meteorological states
    summer_noon_fog = df[(df["month"].isin([4, 5])) & (df["hour"] == 12) & (df["weather"] == "fog")]
    assert len(summer_noon_fog) == 0, "Impossible state: fog at noon in summer"

    rain_dry_surface = df[(df["weather"].isin(["rain", "storm"])) & (df["surface_condition"] == "dry")]
    assert len(rain_dry_surface) == 0, "Impossible state: rain on dry road surface"


# ---------------------------------------------------------------------------
# 3. Scoring Engine Tests
# ---------------------------------------------------------------------------

def test_scoring_engine_performance_and_validity():
    """Verify scoring engine latency < 3.0s and output contract."""
    engine = get_scoring_engine()

    t0 = time.time()
    result = engine.score(hour=19, weather="rain", month=8, day_type="weekday", as_geojson=False)
    cold_latency = time.time() - t0

    assert cold_latency < 3.0, f"Cold evaluation latency {cold_latency:.2f}s exceeded 3.0s target"

    # Test cache performance
    t0 = time.time()
    cached_result = engine.score(hour=19, weather="rain", month=8, day_type="weekday", as_geojson=False)
    cached_latency = time.time() - t0
    assert cached_latency < 0.02, f"Cached evaluation {cached_latency * 1000:.1f}ms exceeded 20ms limit"

    # Probability contract
    p_minor = result["p_minor"]
    p_serious = result["p_serious"]
    p_fatal = result["p_fatal"]
    p_sum = p_minor + p_serious + p_fatal

    np.testing.assert_allclose(p_sum, 1.0, rtol=1e-4, err_msg="Probabilities must sum to 1.0")
    assert np.all(result["risk_index"] >= 0.0), "Risk index cannot be negative"
    assert np.all(result["risk_index"] <= 2.0), "Risk index cannot exceed 2.0 (p_fatal * 2)"

    # Banding distribution
    bands = pd.Series(result["band"]).value_counts()
    assert "green" in bands and "yellow" in bands and "orange" in bands and "red" in bands


# ---------------------------------------------------------------------------
# 4. FastAPI Endpoints Integration Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    """Test /health endpoint returns healthy status and loaded segment counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["segments_loaded"] > 0
    assert data["graph_nodes"] > 0
    assert data["model_loaded"] is True


def test_api_risk_geojson(client):
    """Test /risk endpoint produces valid GeoJSON FeatureCollection."""
    response = client.get("/risk?hour=18&weather=rain")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0
    first_feat = data["features"][0]
    assert "geometry" in first_feat
    assert "properties" in first_feat
    assert "risk_index" in first_feat["properties"]
    assert "band" in first_feat["properties"]
    assert "color" in first_feat["properties"]


def test_api_segment_details(client):
    """Test /segment/{id} endpoint returns detailed attributes and top drivers."""
    sid = int(state.segments_df["segment_id"].iloc[0])
    response = client.get(f"/segment/{sid}?hour=20&weather=rain")
    assert response.status_code == 200
    data = response.json()
    assert data["segment_id"] == sid
    assert "risk_index" in data
    assert "probabilities" in data
    assert "environmental_factors" in data
    assert isinstance(data["top_drivers"], list)


def test_api_route_optimization(client):
    """Test dual-route endpoint returns both fastest and safest metrics with tradeoff summary."""
    # Dharampeth to Sitabuldi
    response = client.get("/route?from=21.144,79.066&to=21.105,79.075&lambda=2.0&weather=rain&hour=19")
    assert response.status_code == 200
    data = response.json()

    assert "routes" in data
    assert "fastest" in data["routes"]
    assert "safest" in data["routes"]
    assert "summary" in data

    fastest = data["routes"]["fastest"]
    safest = data["routes"]["safest"]

    assert fastest["distance_km"] > 0.0
    assert safest["distance_km"] > 0.0
    assert "worst_segment" in fastest
    assert "worst_segment" in safest
    assert "share_red_orange_pct" in fastest
    assert "share_red_orange_pct" in safest


def test_api_telemetry_ingest_and_live(client):
    """Test POST /ingest updates vehicle state and logs harsh braking."""
    payload = {
        "lat": 21.144,
        "lon": 79.066,
        "speed": 42.0,
        "rain_sensor": 750.0,
        "ldr": 200.0,
        "harsh_brake": 1,
    }
    post_resp = client.post("/ingest", json=payload)
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["status"] == "ok"
    assert post_data["inferred_weather"] == "rain"
    assert post_data["inferred_light"] == "dark"

    # Query GET /live
    live_resp = client.get("/live")
    assert live_resp.status_code == 200
    live_data = live_resp.json()
    assert live_data["harsh_brake"] == 1
    assert live_data["speed"] == 42.0
    assert live_data["inferred_weather"] == "rain"


def test_api_proximity_alert(client):
    """Test GET /alert queries proximity to danger zones."""
    # Test alert near known central coordinate
    response = client.get("/alert?lat=21.144&lon=79.066&hour=22&weather=storm")
    assert response.status_code == 200
    data = response.json()
    assert "alert" in data
    assert isinstance(data["alert"], bool)
    assert "message" in data
