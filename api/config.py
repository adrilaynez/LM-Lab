"""
API Configuration
Environment-based settings for the FastAPI service.
"""

import os
from pathlib import Path


# ============ Paths ============
# Resolve project root relative to this file (api/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

# ============ Server ============
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# ============ CORS ============
# Comma-separated list of allowed origins
_cors_raw = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
)
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# ============ Model Inference ============
DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_GENERATE_LENGTH = int(os.getenv("MAX_GENERATE_LENGTH", "200"))

# ============ API Metadata ============
API_TITLE = "LM-Lab API"
API_DESCRIPTION = "Interactive ML model inference & interpretability API"
API_VERSION = "1.0.0"
