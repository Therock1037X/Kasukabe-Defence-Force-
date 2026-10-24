#!/usr/bin/env python3
"""Stage 8: ESP32 Telemetry Simulator for KDF Navigation.

Streams simulated vehicle telemetry (GPS, speed, rain, light, harsh braking)
along a driving route in Nagpur to the FastAPI backend at POST /ingest.
Allows full end-to-end testing of live vehicle tracking without physical hardware.
"""

import argparse
import math
import random
import sys
import time
from typing import List, Tuple
import urllib.request
import json


def fetch_route_coordinates(
    host: str,
    port: int,
    from_coord: str = "21.144,79.066",
    to_coord: str = "21.105,79.075",
) -> List[Tuple[float, float]]:
    """Fetch real road coordinates from /route to drive along."""
    url = f"http://{host}:{port}/route?from={from_coord}&to={to_coord}&lambda=1.5"
    print(f"[SIMULATOR] Fetching route from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ESP32-Simulator/1.0"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Get geometry coordinates from safest route
            coords = data["routes"]["safest"]["geometry"]["coordinates"]
            # coords is [[lon, lat], ...]
            lat_lons = [(pt[1], pt[0]) for pt in coords]
            print(f"[SIMULATOR] Successfully retrieved {len(lat_lons)} route waypoints.")
            return lat_lons
    except Exception as exc:
        print(f"[SIMULATOR WARNING] Could not fetch route from API ({exc}). Using fallback corridor waypoints.")
        # Fallback waypoints from Dharampeth to Sitabuldi
        return [
            (21.1442, 79.0664),
            (21.1430, 79.0690),
            (21.1415, 79.0725),
            (21.1395, 79.0760),
            (21.1370, 79.0785),
            (21.1330, 79.0805),
            (21.1280, 79.0815),
            (21.1220, 79.0810),
            (21.1150, 79.0790),
            (21.1080, 79.0770),
            (21.1050, 79.0750),
        ]


def interpolate_route(waypoints: List[Tuple[float, float]], step_meters: float = 25.0) -> List[Tuple[float, float]]:
    """Interpolate waypoints so steps correspond to ~2 seconds of driving at ~45 km/h."""
    dense_path = []
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]

        # Approximate distance
        d_lat_m = (lat2 - lat1) * 111_000.0
        d_lon_m = (lon2 - lon1) * 103_000.0
        dist_m = math.hypot(d_lat_m, d_lon_m)

        steps = max(int(dist_m / step_meters), 1)
        for s in range(steps):
            frac = s / steps
            interp_lat = lat1 + (lat2 - lat1) * frac
            interp_lon = lon1 + (lon2 - lon1) * frac
            dense_path.append((interp_lat, interp_lon))

    dense_path.append(waypoints[-1])
    return dense_path


def main():
    parser = argparse.ArgumentParser(description="KDF Navigation ESP32 Telemetry Simulator")
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI server host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--interval", type=float, default=2.0, help="Dispatch interval in seconds")
    parser.add_argument(
        "--scenario",
        default="rainy_commute",
        choices=["clear_day", "rainy_commute", "night_fog", "harsh_brake_spike"],
        help="Simulated environmental scenario",
    )
    parser.add_argument("--loop", action="store_true", default=True, help="Continuously loop the route")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  KDF Navigation — ESP32 Telemetry Simulator")
    print(f"  Target: http://{args.host}:{args.port}/ingest")
    print(f"  Scenario: {args.scenario} | Interval: {args.interval}s")
    print("=" * 60 + "\n")

    waypoints = fetch_route_coordinates(args.host, args.port)
    driving_path = interpolate_route(waypoints, step_meters=25.0)
    print(f"[SIMULATOR] Prepared smooth vehicle path with {len(driving_path)} steps.\n")

    step_idx = 0
    total_steps = len(driving_path)

    while True:
        lat, lon = driving_path[step_idx]

        # Base speed ~ 35 - 55 km/h with subtle noise
        speed = round(random.uniform(36.0, 52.0), 1)

        # Environmental sensors by scenario
        harsh_brake = 0
        if args.scenario == "clear_day":
            rain_sensor = round(random.uniform(50.0, 150.0), 1)  # Dry
            ldr = round(random.uniform(750.0, 950.0), 1)         # Bright
        elif args.scenario == "rainy_commute":
            rain_sensor = round(random.uniform(650.0, 880.0), 1) # Wet / Rain
            ldr = round(random.uniform(320.0, 480.0), 1)         # Overcast / Dusk
            speed = round(speed * 0.82, 1)                       # Slower in rain
        elif args.scenario == "night_fog":
            rain_sensor = round(random.uniform(350.0, 520.0), 1) # Damp surface
            ldr = round(random.uniform(80.0, 220.0), 1)          # Dark night
            speed = round(speed * 0.75, 1)
        elif args.scenario == "harsh_brake_spike":
            rain_sensor = 120.0
            ldr = 850.0
            # Trigger harsh brake every 8 steps
            if step_idx % 8 == 0:
                harsh_brake = 1
                speed = 18.0

        payload = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed": speed,
            "rain_sensor": rain_sensor,
            "ldr": ldr,
            "harsh_brake": harsh_brake,
        }

        # Send POST request to /ingest
        url = f"http://{args.host}:{args.port}/ingest"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ESP32-Simulator/1.0"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                nearest_sid = resp_data.get("nearest_segment_id")
                inf_w = resp_data.get("inferred_weather")
                inf_l = resp_data.get("inferred_light")

                brake_indicator = " ⚠️ HARSH BRAKE!" if harsh_brake else ""
                print(
                    f"[{time.strftime('%H:%M:%S')}] Step {step_idx + 1:03d}/{total_steps:03d} | "
                    f"Pos: ({lat:.5f}, {lon:.5f}) | Spd: {speed:4.1f} km/h | "
                    f"Rain: {rain_sensor:4.0f} ({inf_w}) | LDR: {ldr:4.0f} ({inf_l}) | "
                    f"Segment: #{nearest_sid}{brake_indicator}"
                )
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Ingest failed: {exc}")

        step_idx += 1
        if step_idx >= total_steps:
            if args.loop:
                print("\n[SIMULATOR] Route completed. Looping back to start...\n")
                step_idx = 0
            else:
                print("\n[SIMULATOR] Route simulation completed.")
                break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
