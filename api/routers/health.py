"""
Health check router.
"""

from fastapi import APIRouter

from api.config import API_VERSION, DEVICE
from api.schemas.responses import HealthResponse
from api.services.inference import get_available_model_ids

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """System health check — returns API version, device, and loaded model count."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        device=DEVICE,
        models_loaded=len(get_available_model_ids()),
    )
