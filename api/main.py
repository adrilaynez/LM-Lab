"""
LM-Lab API — FastAPI Application Factory

Entry point: uvicorn api.main:app --reload
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,

)
from api.routers import health, meta, mlp_grid, models


# --------------------------------------------------------------------------- #
#  Lifespan — pre-load models on startup
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm model cache on startup."""
    from api.services.inference import get_available_model_ids, _load_model

    available = get_available_model_ids()
    for model_id in available:
        try:
            _load_model(model_id)
            print(f"  [OK] Pre-loaded model: {model_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to pre-load {model_id}: {e}")

    # Pre-load N-Gram models (N=1..5)
    from api.services.inference import _load_ngram_model
    print("  Pre-loading N-Gram models (N=1..5)...")
    for n in range(1, 6):
        try:
            _load_ngram_model(n)
            print(f"  [OK] Pre-loaded NGram-N{n}")
        except Exception as e:
            print(f"  [ERROR] Failed to pre-load NGram-N{n}: {e}")

    print(f"\n[STARTUP] LM-Lab API ready - {len(available)} model(s) loaded\n")
    yield  # App runs
    print("\n[SHUTDOWN] LM-Lab API shutting down\n")


# --------------------------------------------------------------------------- #
#  App Factory
# --------------------------------------------------------------------------- #

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
# CORS Configuration
# Replace with the ACTUAL URL of your frontend (e.g.: https://your-app.vercel.app)
# Or use ["*"] to allow everything (testing only, not recommended for production)
origins = [
    "https://your-deployed-frontend.com",
    "http://localhost:3000", # To keep working locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers under /api/v1
app.include_router(health.router,    prefix="/api/v1")
app.include_router(meta.router,      prefix="/api/v1")
app.include_router(mlp_grid.router,  prefix="/api/v1")
app.include_router(models.router,    prefix="/api/v1")


# Debug: print all registered routes at startup
for route in app.routes:
    print(route.path, getattr(route, "methods", None))


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "LM-Lab API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
