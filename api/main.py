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
    CORS_ORIGINS,
)
from api.routers import health, meta, models


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
            print(f"  ✅ Pre-loaded model: {model_id}")
        except Exception as e:
            print(f"  ⚠️  Failed to pre-load {model_id}: {e}")

    print(f"\n🚀 LM-Lab API ready — {len(available)} model(s) loaded\n")
    yield  # App runs
    print("\n🛑 LM-Lab API shutting down\n")


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers under /api/v1
app.include_router(health.router, prefix="/api/v1")
app.include_router(meta.router,   prefix="/api/v1")
app.include_router(models.router,  prefix="/api/v1")


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "LM-Lab API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
