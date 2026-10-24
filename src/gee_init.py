import os
import ee

DEFAULT_PROJECT = "kdf-navigation"


def init_ee(project: str | None = None) -> None:
    """Initialize Earth Engine with project ID from environment variable or default."""
    project_id = project or os.getenv("EE_PROJECT", DEFAULT_PROJECT)
    ee.Initialize(project=project_id)


if __name__ == "__main__":
    init_ee()
    print("Earth Engine initialized successfully with project:", os.getenv("EE_PROJECT", DEFAULT_PROJECT))
