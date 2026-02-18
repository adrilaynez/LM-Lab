"""
FastAPI Dependency Injection
Shared dependencies injected into route handlers.
"""

from fastapi import Depends, HTTPException

from api.services.inference import model_exists, checkpoint_exists, get_all_model_ids


async def require_model(model_id: str) -> str:
    """Validate that model_id exists in the registry."""
    if not model_exists(model_id):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_FOUND",
                "message": f"Model '{model_id}' is not registered",
                "available_models": get_all_model_ids(),
            },
        )
    return model_id


async def require_checkpoint(model_id: str = Depends(require_model)) -> str:
    """Validate that model_id has a trained checkpoint."""
    if not checkpoint_exists(model_id):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CHECKPOINT_NOT_AVAILABLE",
                "message": f"Model '{model_id}' exists but no checkpoint is available",
            },
        )
    return model_id
