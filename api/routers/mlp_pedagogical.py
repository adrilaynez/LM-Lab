"""
Pedagogical MLP endpoints — serves pre-trained data for narrative visualizers.

Routes:
  GET /api/v1/mlp/depth-comparison  — Group 1: varying depth, no stability
  GET /api/v1/mlp/stability-grid    — Group 2: depth × technique grid
  GET /api/v1/mlp/big-models        — Group 3: large configs / MLP limits
  GET /api/v1/mlp/lr-sweep          — Group 4: learning rate sweep
  GET /api/v1/mlp/dropout-experiment — Group 5: dropout regularization
  GET /api/v1/mlp/overtraining-timeline — Group 6: quality evolution
"""

from fastapi import APIRouter, HTTPException

from api.services import inference

router = APIRouter(prefix="/mlp", tags=["MLP Pedagogical"])


@router.get("/depth-comparison")
async def depth_comparison():
    """
    Returns all depth-comparison models (n_layers 1-6, no stability techniques).
    Each entry includes: config, loss_curve, final_loss, params, generated_text.
    """
    try:
        return inference.get_depth_comparison()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DEPTH_COMPARISON_ERROR", "message": str(e)},
        )


@router.get("/stability-grid")
async def stability_grid():
    """
    Returns the full stability technique grid:
    depths [1,2,3,4,6] × techniques [none, kaiming, kaiming+BN, kaiming+BN+residual].
    Includes grid_axes metadata for 2-D visualization.
    """
    try:
        return inference.get_stability_grid()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "STABILITY_GRID_ERROR", "message": str(e)},
        )


@router.get("/big-models")
async def big_models():
    """
    Returns large MLP configs (high hidden_size / long context_size).
    Shows param explosion and quality plateau even with all stability techniques.
    """
    try:
        return inference.get_big_models()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "BIG_MODELS_ERROR", "message": str(e)},
        )


@router.get("/lr-sweep")
async def lr_sweep():
    """
    Returns learning rate sweep models (same arch, 5 LRs).
    Shows optimal LR selection: too low → slow, too high → diverges.
    """
    try:
        return inference.get_lr_sweep()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "LR_SWEEP_ERROR", "message": str(e)},
        )


@router.get("/dropout-experiment")
async def dropout_experiment():
    """
    Returns dropout experiment models (dropout=0/0.2/0.5).
    Shows regularization effect: overfitting without, optimal with moderate.
    """
    try:
        return inference.get_dropout_experiment()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DROPOUT_EXPERIMENT_ERROR", "message": str(e)},
        )


@router.get("/overtraining-timeline")
async def overtraining_timeline():
    """
    Returns a single model trained for 200K steps with text snapshots
    at milestones showing quality evolution from gibberish to coherent text.
    """
    try:
        return inference.get_overtraining_timeline()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "OVERTRAINING_TIMELINE_ERROR", "message": str(e)},
        )
