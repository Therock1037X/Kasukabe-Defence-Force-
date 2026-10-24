"""Stage 6: FastAPI Backend for KDF Navigation (RoadWatch Nagpur).

Provides real-time road risk analysis, Fastest vs. Safest routing,
live ESP32 sensor telemetry ingestion, and forward collision/risk alerts.
"""

import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.spatial import KDTree

from config import DATA_DIR, DISPLAY_CRS, MODEL_DIR, OUTPUT_DIR, PROJECTED_CRS
from src.score import ScoringEngine, get_scoring_engine
from src.train import explain_top_drivers

# Initialize FastAPI application
app = FastAPI(
    title="KDF Navigation API",
    description="Safety-first vehicle navigation backend with ML risk prediction and dual-route optimization.",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# SQLite Database for live sensor ingestion
DB_PATH = OUTPUT_DIR / "sensor_logs.db"


def init_db():
    """Initialize SQLite database for sensor telemetry logging."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            lat REAL,
            lon REAL,
            rain_sensor REAL,
            ldr REAL,
            harsh_brake INTEGER,
            speed REAL,
            nearest_segment_id INTEGER,
            inferred_weather TEXT,
            inferred_light TEXT
        );
    """)
    conn.commit()
    conn.close()


init_db()

# Pydantic Schemas
class SensorTelemetryInput(BaseModel):
    lat: float = Field(..., description="Vehicle latitude", ge=-90.0, le=90.0)
    lon: float = Field(..., description="Vehicle longitude", ge=-180.0, le=180.0)
    rain_sensor: float = Field(..., description="Analog rain sensor reading (0-1023)")
    ldr: float = Field(..., description="Light dependent resistor reading (0-1023)")
    harsh_brake: int = Field(0, description="1 if deceleration > 0.4g, else 0")
    speed: float = Field(..., description="Vehicle speed in km/h", ge=0.0)


# Global Application State (Loaded once on startup)
class AppState:
    scoring_engine: Optional[ScoringEngine] = None
    graph: Optional[nx.DiGraph] = None
    node_list: List[str] = []
    node_coords: np.ndarray = np.empty((0, 2))
    node_kdtree: Optional[KDTree] = None
    segments_kdtree: Optional[KDTree] = None
    segments_coords: np.ndarray = np.empty((0, 2))
    segments_df: Optional[gpd.GeoDataFrame] = None
    live_state: Dict[str, Any] = {
        "timestamp": None,
        "lat": 21.1458,
        "lon": 79.0882,
        "speed": 0.0,
        "rain_sensor": 0.0,
        "ldr": 800.0,
        "harsh_brake": 0,
        "inferred_weather": "clear",
        "inferred_light": "daylight",
        "nearest_segment_id": None,
    }


state = AppState()


@app.on_event("startup")
def startup_event():
    """Load model, segment database, and routing graph once at startup."""
    print("FastAPI Startup: Loading ML ScoringEngine and Road Graph...")
    t0 = time.time()
    state.scoring_engine = get_scoring_engine()
    state.segments_df = state.scoring_engine.segments_gdf

    # Build KDTree for segment centroids (spatial snapping)
    state.segments_coords = np.column_stack([
        state.segments_df["centroid_lat"].values,
        state.segments_df["centroid_lon"].values,
    ])
    state.segments_kdtree = KDTree(state.segments_coords)

    # Build networkx DiGraph for routing
    G = nx.DiGraph()
    for row in state.segments_df.itertuples():
        u = f"{row.start_lat:.5f}_{row.start_lon:.5f}"
        v = f"{row.end_lat:.5f}_{row.end_lon:.5f}"
        spd = max(float(row.maxspeed), 10.0)
        length_m = float(row.length_m)
        travel_time_sec = length_m / (spd * 1000.0 / 3600.0)

        # Coordinate list for GeoJSON geometry reconstruction
        coords = list(row.geometry.coords) if hasattr(row.geometry, "coords") else []

        edge_attrs = {
            "segment_id": int(row.segment_id),
            "length_m": length_m,
            "maxspeed": spd,
            "travel_time": travel_time_sec,
            "coords": coords,
            "highway": str(row.highway),
        }

        G.add_edge(u, v, **edge_attrs)
        if not bool(row.oneway):
            rev_attrs = dict(edge_attrs)
            rev_attrs["coords"] = coords[::-1]
            G.add_edge(v, u, **rev_attrs)

    state.graph = G
    state.node_list = list(G.nodes())
    node_lat_lons = np.array([
        [float(k.split("_")[0]), float(k.split("_")[1])]
        for k in state.node_list
    ])
    state.node_coords = node_lat_lons
    state.node_kdtree = KDTree(node_lat_lons)

    print(f"Startup complete: Graph has {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges (loaded in {time.time() - t0:.2f}s).")


def snap_to_nearest_node(lat: float, lon: float) -> str:
    """Snap geographic coordinate to nearest road graph node."""
    if state.node_kdtree is None or len(state.node_list) == 0:
        raise HTTPException(status_code=503, detail="Routing graph not loaded")
    _, idx = state.node_kdtree.query([lat, lon])
    return state.node_list[idx]


# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    """Serve car-dashboard frontend."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>KDF Navigation API is Running. Access <a href='/docs'>/docs</a> or <a href='/static/index.html'>/static/index.html</a></h2>")


@app.get("/health")
def health_check():
    """Health and system readiness endpoint."""
    return {
        "status": "healthy",
        "city": "Nagpur, Maharashtra, India",
        "test_area_active": True,
        "segments_loaded": state.scoring_engine.n_segments if state.scoring_engine else 0,
        "graph_nodes": state.graph.number_of_nodes() if state.graph else 0,
        "graph_edges": state.graph.number_of_edges() if state.graph else 0,
        "model_loaded": state.scoring_engine is not None,
    }


@app.get("/risk")
def get_risk_layers(
    hour: int = Query(18, ge=0, le=23, description="Hour of day (0-23)"),
    weather: str = Query("clear", description="clear | rain | fog | storm"),
    month: int = Query(10, ge=1, le=12, description="Month of year (1-12)"),
    day_type: str = Query("weekday", description="weekday | weekend"),
    bbox: Optional[str] = Query(None, description="Optional min_lon,min_lat,max_lon,max_lat"),
):
    """Retrieve color-coded road segment risk strips under specified conditions."""
    if not state.scoring_engine:
        raise HTTPException(status_code=503, detail="Scoring engine not ready")

    scored = state.scoring_engine.score(
        hour=hour,
        weather=weather,
        month=month,
        day_type=day_type,
        as_geojson=False,
    )

    geojson = scored["geojson"]

    # Spatial BBox filtering if supplied
    if bbox:
        try:
            min_x, min_y, max_x, max_y = [float(x.strip()) for x in bbox.split(",")]
            filtered_features = []
            for f in geojson["features"]:
                coords = f["geometry"]["coordinates"]
                # Check if any coordinate is within bbox
                if any(min_x <= pt[0] <= max_x and min_y <= pt[1] <= max_y for pt in coords):
                    filtered_features.append(f)
            return {
                "type": "FeatureCollection",
                "features": filtered_features,
                "metadata": {**geojson["metadata"], "filtered_count": len(filtered_features)},
            }
        except Exception:
            pass

    return geojson


@app.get("/segment/{segment_id}")
def get_segment_details(
    segment_id: int,
    hour: int = Query(18, ge=0, le=23),
    weather: str = Query("clear"),
    month: int = Query(10, ge=1, le=12),
):
    """Get detailed risk profile, probabilities, and top 3 drivers for a road segment."""
    if not state.scoring_engine or state.segments_df is None:
        raise HTTPException(status_code=503, detail="Scoring engine not ready")

    match = state.segments_df[state.segments_df["segment_id"] == segment_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Segment #{segment_id} not found")

    row = match.iloc[0]
    scored = state.scoring_engine.score(hour=hour, weather=weather, month=month)
    idx = int(row.name) if row.name in range(state.scoring_engine.n_segments) else 0

    drivers = scored["top_drivers"][idx]

    return {
        "segment_id": segment_id,
        "road_name": f"{str(row.highway).capitalize()} Road #{segment_id}",
        "highway": row.highway,
        "length_m": round(float(row.length_m), 1),
        "lanes": int(row.lanes),
        "maxspeed_kmh": float(row.maxspeed),
        "curvature": round(float(row.curvature), 4),
        "junction_count": int(row.junction_count),
        "risk_index": round(float(scored["risk_index"][idx]), 4),
        "band": str(scored["band"][idx]),
        "color": str(scored["color"][idx]),
        "probabilities": {
            "p_minor": round(float(scored["p_minor"][idx]), 4),
            "p_serious": round(float(scored["p_serious"][idx]), 4),
            "p_fatal": round(float(scored["p_fatal"][idx]), 4),
        },
        "environmental_factors": {
            "night_lights_radiance": round(float(row.night_lights), 2),
            "ntl_percentile": round(float(row.ntl_percentile), 2),
            "elevation_m": round(float(row.elevation), 1),
            "slope_deg": round(float(row.slope), 2),
            "built_up": round(float(row.built_up), 2),
            "population_density": round(float(row.population_density), 1),
            "lit_tag": row.lit_tag,
            "school_zone": bool(row.school_zone),
            "market_zone": bool(row.is_market_zone),
        },
        "top_drivers": drivers,
    }


def compute_route_metrics(
    G: nx.DiGraph,
    path_nodes: List[str],
    segment_risk_map: Dict[int, Tuple[float, str]],
) -> Dict[str, Any]:
    """Calculate distance, travel time, route risk, red/orange percentage, and worst segment."""
    total_dist_m = 0.0
    total_time_sec = 0.0
    weighted_risk_sum = 0.0
    red_orange_dist_m = 0.0

    worst_segment_id = None
    worst_risk = -1.0
    worst_highway = "unknown"

    all_coords = []

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge = G[u][v]

        sid = edge["segment_id"]
        length = edge["length_m"]
        t_sec = edge["travel_time"]
        coords = edge.get("coords", [])

        if coords:
            if not all_coords:
                all_coords.extend(coords)
            else:
                all_coords.extend(coords[1:])

        total_dist_m += length
        total_time_sec += t_sec

        risk_val, band = segment_risk_map.get(sid, (0.5, "yellow"))
        weighted_risk_sum += risk_val * length

        if band in ("orange", "red"):
            red_orange_dist_m += length

        if risk_val > worst_risk:
            worst_risk = risk_val
            worst_segment_id = sid
            worst_highway = edge.get("highway", "road")

    dist_km = total_dist_m / 1000.0
    eta_min = total_time_sec / 60.0
    length_weighted_mean_risk = weighted_risk_sum / max(total_dist_m, 1.0)
    share_red_orange = red_orange_dist_m / max(total_dist_m, 1.0)
    route_risk = length_weighted_mean_risk + 0.5 * share_red_orange

    return {
        "geometry": {
            "type": "LineString",
            "coordinates": all_coords,
        },
        "distance_km": round(dist_km, 2),
        "eta_minutes": round(eta_min, 1),
        "length_weighted_mean_risk": round(length_weighted_mean_risk, 4),
        "share_red_orange_pct": round(share_red_orange * 100.0, 1),
        "red_orange_km": round(red_orange_dist_m / 1000.0, 2),
        "route_risk": round(route_risk, 4),
        "worst_segment": {
            "segment_id": worst_segment_id,
            "road_name": f"{str(worst_highway).capitalize()} Road #{worst_segment_id}",
            "risk_index": round(worst_risk, 4),
        },
    }


@app.get("/route")
def get_dual_route(
    from_coord: str = Query(..., alias="from", description="Start 'lat,lon' (e.g. 21.144,79.066)"),
    to_coord: str = Query(..., alias="to", description="Destination 'lat,lon' (e.g. 21.105,79.075)"),
    lambda_param: float = Query(1.5, alias="lambda", ge=0.0, le=10.0, description="Safety vs speed tradeoff (0=fastest only)"),
    hour: int = Query(18, ge=0, le=23),
    weather: str = Query("clear"),
    month: int = Query(10, ge=1, le=12),
):
    """Compute Fastest vs. Safest routes side-by-side with risk comparison metrics."""
    if not state.graph or not state.scoring_engine:
        raise HTTPException(status_code=503, detail="Routing service not initialized")

    try:
        from_lat, from_lon = [float(x.strip()) for x in from_coord.split(",")]
        to_lat, to_lon = [float(x.strip()) for x in to_coord.split(",")]
    except Exception:
        raise HTTPException(status_code=400, detail="Coordinates must be formatted as 'lat,lon'")

    start_node = snap_to_nearest_node(from_lat, from_lon)
    end_node = snap_to_nearest_node(to_lat, to_lon)

    if start_node == end_node:
        raise HTTPException(status_code=400, detail="Start and destination snap to the same road node")

    # Evaluate current risk for all segments
    scored = state.scoring_engine.score(hour=hour, weather=weather, month=month)
    risk_indices = scored["risk_index"]
    bands = scored["band"]
    seg_ids = state.scoring_engine.segments_gdf["segment_id"].values

    segment_risk_map = {
        int(seg_ids[i]): (float(risk_indices[i]), str(bands[i]))
        for i in range(len(seg_ids))
    }

    # 1. Fastest Route (cost = travel_time)
    try:
        fastest_nodes = nx.shortest_path(
            state.graph,
            source=start_node,
            target=end_node,
            weight="travel_time",
        )
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No route found between start and destination")

    fastest_metrics = compute_route_metrics(state.graph, fastest_nodes, segment_risk_map)

    # 2. Safest Route (cost = travel_time + lambda * (risk_index * 100) * (length_m / 100))
    def safety_weight(u, v, d):
        sid = d["segment_id"]
        risk_val, _ = segment_risk_map.get(sid, (0.5, "yellow"))
        time_cost = d["travel_time"]
        safety_penalty = lambda_param * (risk_val * 100.0) * (d["length_m"] / 100.0)
        return time_cost + safety_penalty

    try:
        safest_nodes = nx.shortest_path(
            state.graph,
            source=start_node,
            target=end_node,
            weight=safety_weight,
        )
    except nx.NetworkXNoPath:
        safest_nodes = fastest_nodes

    safest_metrics = compute_route_metrics(state.graph, safest_nodes, segment_risk_map)

    # One-line trade-off summary
    avoided_risk_km = max(0.0, fastest_metrics["red_orange_km"] - safest_metrics["red_orange_km"])
    added_time_min = max(0.0, safest_metrics["eta_minutes"] - fastest_metrics["eta_minutes"])

    if avoided_risk_km > 0.05:
        if added_time_min < 0.5:
            summary = f"Safest route avoids {avoided_risk_km:.1f} km of high-risk road with virtually no time penalty."
        else:
            summary = f"Safest route avoids {avoided_risk_km:.1f} km of high-risk road, adds {added_time_min:.0f} min."
    else:
        summary = "Fastest route is already optimal for safety under these conditions."

    return {
        "summary": summary,
        "lambda": lambda_param,
        "conditions": {"hour": hour, "weather": weather, "month": month},
        "routes": {
            "fastest": fastest_metrics,
            "safest": safest_metrics,
        },
    }


@app.post("/ingest")
def ingest_telemetry(reading: SensorTelemetryInput):
    """Receive live telemetry from ESP32, update live conditions, and log harsh braking."""
    # Infer environmental conditions from physical sensors
    inferred_weather = "rain" if reading.rain_sensor > 500 else "clear"
    inferred_light = "dark" if reading.ldr < 350 else "daylight"

    # Snap to nearest road segment using centroid KDTree
    nearest_sid = None
    if state.segments_kdtree is not None and state.segments_df is not None:
        _, idx = state.segments_kdtree.query([reading.lat, reading.lon])
        nearest_sid = int(state.segments_df["segment_id"].iloc[idx])

    # Update live state
    state.live_state.update({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lat": reading.lat,
        "lon": reading.lon,
        "speed": reading.speed,
        "rain_sensor": reading.rain_sensor,
        "ldr": reading.ldr,
        "harsh_brake": reading.harsh_brake,
        "inferred_weather": inferred_weather,
        "inferred_light": inferred_light,
        "nearest_segment_id": nearest_sid,
    })

    # Log into SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sensor_telemetry 
            (lat, lon, rain_sensor, ldr, harsh_brake, speed, nearest_segment_id, inferred_weather, inferred_light)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reading.lat,
            reading.lon,
            reading.rain_sensor,
            reading.ldr,
            reading.harsh_brake,
            reading.speed,
            nearest_sid,
            inferred_weather,
            inferred_light,
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"Telemetry DB log error: {exc}")

    return {
        "status": "ok",
        "nearest_segment_id": nearest_sid,
        "inferred_weather": inferred_weather,
        "inferred_light": inferred_light,
    }


@app.get("/live")
def get_live_telemetry():
    """Retrieve current live environmental conditions and vehicle position."""
    return state.live_state


@app.get("/alert")
def get_collision_risk_alert(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    heading: Optional[float] = Query(None, ge=0.0, le=360.0),
    hour: int = Query(18, ge=0, le=23),
    weather: str = Query("clear"),
    month: int = Query(10, ge=1, le=12),
):
    """Check if vehicle is within 300m of a Red or Orange accident-prone road segment."""
    if state.segments_kdtree is None or state.segments_df is None or state.scoring_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    scored = state.scoring_engine.score(hour=hour, weather=weather, month=month)
    bands = scored["band"]

    # 300 meters is approx 0.0027 degrees in lat/lon
    # Query spatial tree within 300m
    indices = state.segments_kdtree.query_ball_point([lat, lon], r=0.0027)

    danger_segments = []
    for idx in indices:
        band = bands[idx]
        if band in ("orange", "red"):
            seg_lat, seg_lon = state.segments_coords[idx]
            # Approximate distance in meters
            d_meters = math.hypot((seg_lat - lat) * 111_000, (seg_lon - lon) * 103_000)
            if d_meters <= 300.0:
                danger_segments.append((d_meters, band, state.segments_df["segment_id"].iloc[idx]))

    if danger_segments:
        danger_segments.sort(key=lambda x: x[0])
        nearest_d, nearest_band, sid = danger_segments[0]
        return {
            "alert": True,
            "distance_m": round(nearest_d, 1),
            "band": nearest_band,
            "segment_id": int(sid),
            "message": "Accident-prone zone ahead, reduce speed",
        }

    return {
        "alert": False,
        "distance_m": None,
        "band": None,
        "segment_id": None,
        "message": "Route clear",
    }
