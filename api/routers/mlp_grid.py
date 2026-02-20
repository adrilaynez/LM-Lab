"""
MLP Grid router — endpoints for the MLP Model Zoo (108 precomputed configurations).
"""

from fastapi import APIRouter, HTTPException

from api.schemas.requests import (
    MLPGridPredictRequest,
    MLPGridGenerateRequest,
    MLPGridInternalsRequest,
)
from api.schemas.responses import (
    MLPGridListResponse,
    MLPGridConfigSummary,
    MLPGridPredictResponse,
    MLPGridGenerateResponse,
    MLPGridTimelineResponse,
    MLPGridEmbeddingResponse,
    MLPGridInternalsResponse,
    MLPGridEmbeddingQualityResponse,
)
from api.services import inference

router = APIRouter(prefix="/models/mlp-grid", tags=["MLP Grid"])


# --------------------------------------------------------------------------- #
#  1. Configuration Listing (Model Zoo)
# --------------------------------------------------------------------------- #

@router.get("", response_model=MLPGridListResponse)
async def list_mlp_grid():
    """
    List all available trained MLP configurations from checkpoints/mlp_grid.
    Returns embedding_dim, hidden_size, learning_rate, final_loss, perplexity, etc.
    """
    try:
        configs = inference.list_mlp_grid_configurations()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_LIST_ERROR", "message": str(e)},
        )

    return MLPGridListResponse(
        configurations=[MLPGridConfigSummary(**c) for c in configs],
        total=len(configs),
    )


# --------------------------------------------------------------------------- #
#  2. Inference (Next-Token Prediction)
# --------------------------------------------------------------------------- #

@router.post("/predict", response_model=MLPGridPredictResponse)
async def mlp_grid_predict(body: MLPGridPredictRequest):
    """
    Next-token prediction for a selected MLP grid configuration.
    Returns top-k predictions and full probability distribution.
    """
    try:
        result = inference.mlp_grid_predict(
            text=body.text,
            emb_dim=body.embedding_dim,
            hidden_size=body.hidden_size,
            lr=body.learning_rate,
            top_k=body.top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "CONFIG_NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_PREDICT_ERROR", "message": str(e)},
        )

    return result


# --------------------------------------------------------------------------- #
#  3. Text Generation
# --------------------------------------------------------------------------- #

@router.post("/generate", response_model=MLPGridGenerateResponse)
async def mlp_grid_generate(body: MLPGridGenerateRequest):
    """
    Text generation using a selected MLP grid configuration.
    Accepts seed text, max_tokens, and temperature.
    """
    try:
        result = inference.mlp_grid_generate(
            emb_dim=body.embedding_dim,
            hidden_size=body.hidden_size,
            lr=body.learning_rate,
            seed_text=body.seed_text,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "CONFIG_NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_GENERATE_ERROR", "message": str(e)},
        )

    return result


# --------------------------------------------------------------------------- #
#  4. Training Snapshot Timeline
# --------------------------------------------------------------------------- #

@router.get("/timeline", response_model=MLPGridTimelineResponse)
async def mlp_grid_timeline(
    embedding_dim: int,
    hidden_size: int,
    learning_rate: float,
):
    """
    Return the full training timeline for a selected configuration.
    Includes loss history, grad norms, dead neurons, activation stats,
    and per-snapshot summaries — all from stored data.
    """
    try:
        result = inference.mlp_grid_training_timeline(
            emb_dim=embedding_dim,
            hidden_size=hidden_size,
            lr=learning_rate,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "CONFIG_NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_TIMELINE_ERROR", "message": str(e)},
        )

    return result


# --------------------------------------------------------------------------- #
#  5. Embedding Matrix
# --------------------------------------------------------------------------- #

@router.get("/embedding", response_model=MLPGridEmbeddingResponse)
async def mlp_grid_embedding(
    embedding_dim: int,
    hidden_size: int,
    learning_rate: float,
    snapshot_step: str | None = None,
):
    """
    Return the embedding matrix for a selected configuration and optional snapshot.
    Defaults to the final training snapshot. Frontend handles visualization (PCA/t-SNE).
    """
    try:
        result = inference.mlp_grid_embedding(
            emb_dim=embedding_dim,
            hidden_size=hidden_size,
            lr=learning_rate,
            snapshot_step=snapshot_step,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "CONFIG_NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_EMBEDDING_ERROR", "message": str(e)},
        )

    return result


# --------------------------------------------------------------------------- #
#  6. Internals Inspection
# --------------------------------------------------------------------------- #

@router.post("/internals", response_model=MLPGridInternalsResponse)
async def mlp_grid_internals(body: MLPGridInternalsRequest):
    """
    For a given input text and configuration, return hidden activations,
    pre-activations, and weight statistics via MLPModel.get_internals().
    """
    try:
        result = inference.mlp_grid_internals(
            text=body.text,
            emb_dim=body.embedding_dim,
            hidden_size=body.hidden_size,
            lr=body.learning_rate,
            top_k=body.top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "CONFIG_NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_INTERNALS_ERROR", "message": str(e)},
        )

    return result


# --------------------------------------------------------------------------- #
#  7. Embedding Quality Metrics
# --------------------------------------------------------------------------- #

@router.get("/embedding-quality", response_model=MLPGridEmbeddingQualityResponse)
async def mlp_grid_embedding_quality(
    embedding_dim: int,
    hidden_size: int,
    learning_rate: float,
    snapshot_step: str | None = None,
):
    """
    Return embedding quality metrics (pairwise distance, norms,
    vowel/consonant separation) using MLPModel.get_embedding_quality_metrics().
    Uses pre-computed metrics from snapshot when available.
    """
    try:
        result = inference.mlp_grid_embedding_quality(
            emb_dim=embedding_dim,
            hidden_size=hidden_size,
            lr=learning_rate,
            snapshot_step=snapshot_step,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "CONFIG_NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "MLP_GRID_QUALITY_ERROR", "message": str(e)},
        )

    return result
