
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Reproducibility
RANDOM_SEED = 42

# Development area
CITY_NAME = "Nagpur, Maharashtra, India"
FULL_CITY = False

# Initial test-area focus (Dharampeth + Sitabuldi + part of Wardha Road)
TEST_AREAS = [
    "Dharampeth",
    "Sitabuldi",
    "Wardha Road",
]
TEST_AREA_BBOX = (79.050, 21.085, 79.095, 21.160)  # min_lon, min_lat, max_lon, max_lat

# Geospatial configuration
DISPLAY_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32644"
TARGET_SEGMENT_LENGTH_M = 150

# POI search
POI_BUFFER_M = 100

# Synthetic-data disclosure
DATA_IS_SYNTHETIC = True
