"""
Transformer router — endpoints for GPT grid models (§08–§09 visualizers).
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from api.services import transformer_inference as tfi

router = APIRouter(prefix="/transformer", tags=["Transformer"])


# --------------------------------------------------------------------------- #
#  Request / Response schemas
# --------------------------------------------------------------------------- #

class GenerateRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.8, ge=0.01, le=5.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=200)
    return_probs: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    generated: str
    probabilities: Optional[list] = None


class ConfigSummary(BaseModel):
    config_id: str
    n_blocks: int
    d_model: int
    n_heads: int
    d_ff: int
    block_size: int
    dropout: float
    total_params: int
    best_val_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_train_loss: Optional[float] = None
    training_time_sec: Optional[float] = None
    max_steps: int
    purpose: str


class ConfigListResponse(BaseModel):
    configurations: list[ConfigSummary]
    total: int


# --------------------------------------------------------------------------- #
#  1. Configuration Listing
# --------------------------------------------------------------------------- #

@router.get("/configs", response_model=ConfigListResponse)
async def list_configs():
    """List all trained GPT grid configurations with summary stats."""
    try:
        configs = tfi.list_gpt_configs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    summaries = []
    for c in configs:
        try:
            summaries.append(ConfigSummary(**c))
        except Exception:
            continue

    return ConfigListResponse(configurations=summaries, total=len(summaries))


# --------------------------------------------------------------------------- #
#  2. Text Generation
# --------------------------------------------------------------------------- #

@router.post("/{config_id}/generate", response_model=GenerateResponse)
async def generate(config_id: str, body: GenerateRequest):
    """Generate text from a trained GPT model."""
    try:
        result = tfi.generate_text(
            config_id=config_id,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            return_probs=body.return_probs,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(**result)


# --------------------------------------------------------------------------- #
#  3. Training Timeline
# --------------------------------------------------------------------------- #

@router.get("/{config_id}/timeline")
async def get_timeline(config_id: str):
    """
    Get training timeline: loss curves, checkpoints, generated samples.
    Used by TrainingTimelapseViz.
    """
    try:
        return tfi.get_gpt_timeline(config_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
#  4. Attention Maps
# --------------------------------------------------------------------------- #

@router.get("/{config_id}/attention")
async def get_attention(
    config_id: str,
    text: Optional[str] = Query(default=None, max_length=1024),
    step: Optional[int] = Query(default=None, ge=0),
):
    """
    Get attention maps for a given text.
    If step is provided, returns pre-computed maps from that checkpoint.
    If text is provided, computes fresh attention maps from the final model.
    """
    try:
        return tfi.get_attention_maps(config_id, text=text, step=step)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
#  5. Checkpoint Data
# --------------------------------------------------------------------------- #

@router.get("/{config_id}/checkpoint/{step}")
async def get_checkpoint(config_id: str, step: int):
    """Get full checkpoint data (samples, stats) for a specific training step."""
    try:
        return tfi.get_checkpoint_data(config_id, step)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
#  6. Available Steps
# --------------------------------------------------------------------------- #

@router.get("/{config_id}/steps")
async def get_steps(config_id: str):
    """List all available checkpoint steps for a config."""
    steps = tfi.get_available_steps(config_id)
    if not steps:
        raise HTTPException(status_code=404, detail=f"No checkpoints found for {config_id}")
    return {"config_id": config_id, "steps": steps}
