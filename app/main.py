
from functools import lru_cache
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

RISK_FILE = OUTPUT_DIR / "road_segments_with_risk.geoparquet"
HOTSPOTS_FILE = OUTPUT_DIR / "hotspots.csv"
METRICS_FILE = OUTPUT_DIR / "model_metrics.json"

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Roadwatch Nagpur API",
    description=(
        "Backend API for road-segment risk analysis "
        "and simulated accident hotspot data in Nagpur."
    ),
    version="1.0.0",
)

# Development configuration. Restrict origins before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_roads():
    """Load and cache the road risk dataset."""
    if not RISK_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Risk dataset not found: {RISK_FILE.name}",
        )

    try:
        roads = gpd.read_parquet(RISK_FILE)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read road risk dataset: {exc}",
        )

    required = {"segment_id", "risk_score", "risk_category", "geometry"}
    missing = required - set(roads.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Risk dataset is missing columns: {sorted(missing)}",
        )

    return roads


@lru_cache(maxsize=1)
def load_model_metrics():
    """Read saved model metrics if available."""
    import json

    if not METRICS_FILE.exists():
        return None

    try:
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def json_safe(value):
    """Convert pandas/numpy values into JSON-compatible values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    return value


def record_to_dict(record):
    """Convert a pandas record to a JSON-safe dictionary."""
    return {key: json_safe(value) for key, value in record.items()}


def get_event_count_column(roads):
    """Find the saved event-count column, if present."""
    for name in ("simulated_event_count", "event_count", "accident_count"):
        if name in roads.columns:
            return name
    return None


def get_count(roads, column):
    """Safely sum an existing numeric count column."""
    if column not in roads.columns:
        return 0
    return int(pd.to_numeric(roads[column], errors="coerce").fillna(0).sum())


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Roadwatch Nagpur API",
        "version": "1.0.0",
    }


# --------------------------------------------------
# DASHBOARD SUMMARY
# --------------------------------------------------

@app.get("/api/v1/summary", tags=["Dashboard"])
def get_summary():
    roads = load_roads()

    category_counts = (
        roads["risk_category"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    event_column = get_event_count_column(roads)

    if event_column:
        event_counts = pd.to_numeric(
            roads[event_column], errors="coerce"
        ).fillna(0)
        segments_with_events = int((event_counts > 0).sum())
        total_simulated_events = int(event_counts.sum())
    else:
        segments_with_events = None
        total_simulated_events = None

    summary = {
        "city": "Nagpur, Maharashtra, India",
        "total_road_segments": int(len(roads)),
        "segments_with_simulated_events": segments_with_events,
        "segments_without_simulated_events": (
            int(len(roads) - segments_with_events)
            if segments_with_events is not None
            else None
        ),
        "total_simulated_events": total_simulated_events,
        "risk_category_counts": category_counts,
        "data_source": "SYNTHETIC",
        "is_synthetic": True,
        "risk_score_note": (
            "Risk scores are relative rankings derived from simulated "
            "events; they are not probabilities of an accident."
        ),
    }

    return summary


# --------------------------------------------------
# HOTSPOT LIST
# --------------------------------------------------

@app.get("/api/v1/hotspots", tags=["Hotspots"])
def get_hotspots(
    risk_category: Optional[str] = Query(
        default=None,
        description="Filter by risk category, e.g. Very High or High",
    ),
    min_score: float = Query(default=0, ge=0, le=100),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    roads = load_roads()
    filtered = roads.copy()

    # By default, only return segments with simulated events.
    event_column = get_event_count_column(filtered)

    if event_column:
        counts = pd.to_numeric(
            filtered[event_column], errors="coerce"
        ).fillna(0)
        filtered = filtered.loc[counts > 0].copy()

    filtered = filtered[
        pd.to_numeric(filtered["risk_score"], errors="coerce")
        .fillna(0) >= min_score
    ]

    if risk_category:
        filtered = filtered[
            filtered["risk_category"].astype(str).str.casefold()
            == risk_category.casefold()
        ]

    filtered = filtered.sort_values(
        "risk_score", ascending=False
    )

    total = int(len(filtered))
    page = filtered.iloc[offset:offset + limit]

    fields = [
        "segment_id",
        "risk_score",
        "risk_category",
        "simulated_event_count",
        "minor_count",
        "serious_count",
        "fatal_count",
        "weighted_events_per_km",
        "length_m",
        "highway",
        "centroid_lon",
        "centroid_lat",
    ]

    available_fields = [c for c in fields if c in page.columns]
    results = [
        record_to_dict(row)
        for _, row in page[available_fields].iterrows()
    ]

    return {
        "count": len(results),
        "total": total,
        "limit": limit,
        "offset": offset,
        "data_source": "SYNTHETIC",
        "is_synthetic": True,
        "results": results,
    }


# --------------------------------------------------
# ROAD GEOMETRY FOR THE FRONTEND MAP
# --------------------------------------------------

@app.get("/api/v1/roads", tags=["Map"])
def get_roads(
    risk_category: Optional[str] = Query(default=None),
    min_score: float = Query(default=0, ge=0, le=100),
    limit: int = Query(default=2000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
):
    roads = load_roads()
    filtered = roads.copy()

    filtered = filtered[
        pd.to_numeric(filtered["risk_score"], errors="coerce")
        .fillna(0) >= min_score
    ]

    if risk_category:
        filtered = filtered[
            filtered["risk_category"].astype(str).str.casefold()
            == risk_category.casefold()
        ]

    total = int(len(filtered))
    page = filtered.iloc[offset:offset + limit].copy()

    # GeoJSON uses longitude/latitude coordinates.
    if page.crs is not None and page.crs.to_epsg() != 4326:
        page = page.to_crs(epsg=4326)

    properties = [
        "segment_id",
        "risk_score",
        "risk_category",
        "simulated_event_count",
        "minor_count",
        "serious_count",
        "fatal_count",
        "length_m",
        "highway",
    ]

    available = [c for c in properties if c in page.columns]
    page = page[available + ["geometry"]]

    try:
        geojson = page.to_json(drop_id=True)
        import json
        geojson_object = json.loads(geojson)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate GeoJSON: {exc}",
        )

    return {
        "type": "FeatureCollection",
        "metadata": {
            "count": len(page),
            "total": total,
            "limit": limit,
            "offset": offset,
            "data_source": "SYNTHETIC",
            "is_synthetic": True,
        },
        "features": geojson_object.get("features", []),
    }


# --------------------------------------------------
# INDIVIDUAL ROAD SEGMENT DETAILS
# --------------------------------------------------

@app.get("/api/v1/hotspots/{segment_id}", tags=["Hotspots"])
def get_hotspot_details(segment_id: int):
    roads = load_roads()

    matches = roads[roads["segment_id"].astype(str) == str(segment_id)]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Road segment {segment_id} was not found.",
        )

    row = matches.iloc[0]
    details = record_to_dict(
        row.drop(labels=["geometry"], errors="ignore")
    )

    details["data_source"] = "SYNTHETIC"
    details["is_synthetic"] = True

    return details


# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------

@app.get("/api/v1/model-metrics", tags=["Machine Learning"])
def get_model_metrics():
    metrics = load_model_metrics()

    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="Saved model metrics are not available.",
        )

    return {
        "data_source": "SYNTHETIC",
        "is_synthetic": True,
        "metrics": metrics,
        "warning": (
            "These metrics evaluate training on synthetic data. "
            "They do not establish real-world accident prediction accuracy."
        ),
    }
