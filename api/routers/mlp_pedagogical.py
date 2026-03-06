"""
Pedagogical MLP endpoints — serves pre-trained data for narrative visualizers.

Routes:
  GET /api/v1/mlp/depth-comparison       — Group 1: varying depth, no stability
  GET /api/v1/mlp/stability-grid         — Group 2: depth × technique grid
  GET /api/v1/mlp/big-models             — Group 3: large configs / MLP limits
  GET /api/v1/mlp/lr-sweep               — Group 4: learning rate sweep
  GET /api/v1/mlp/dropout-experiment     — Group 5: dropout regularization
  GET /api/v1/mlp/overtraining-timeline  — Group 6: quality evolution
  GET /api/v1/mlp/scale-stability        — Group 8: H=256/512, depths 4-20, SGD
  GET /api/v1/mlp/data-size              — Group 9: same model, different data amounts
  GET /api/v1/mlp/activation-battle      — 5 activation functions compared
  GET /api/v1/mlp/embedding-bottleneck   — E=2,4,8,32,128 dimension sweep
  GET /api/v1/mlp/network-shape          — pyramid/funnel/cylinder shape comparison
  GET /api/v1/mlp/advanced-embeddings    — mlp_advanced 2D–128D embedding matrices
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


@router.get("/weight-tying")
async def weight_tying():
    """
    Returns weight tying experiment on base corpus (vocab=28).
    Tied vs untied output weights — shows parameter efficiency trade-off.
    """
    try:
        return inference.get_weight_tying()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "WEIGHT_TYING_ERROR", "message": str(e)},
        )


@router.get("/weight-tying-graham")
async def weight_tying_graham():
    """
    Returns weight tying experiment on Paul Graham corpus (vocab=96).
    Shows how weight tying becomes beneficial with larger vocabularies.
    """
    try:
        return inference.get_weight_tying_graham()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "WEIGHT_TYING_GRAHAM_ERROR", "message": str(e)},
        )


@router.get("/scale-stability")
async def scale_stability():
    """
    Returns scale stability experiment: H=256/512 × depths 4-20 ×
    {kaiming, kaiming+BN+residual}, all SGD lr=0.001, no clipping.
    Shows how techniques interact with optimizer choice at scale.
    """
    try:
        return inference.get_scale_stability()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SCALE_STABILITY_ERROR", "message": str(e)},
        )


@router.get("/data-size")
async def data_size():
    """
    Returns data size experiment: same model (H=256, L=4, ctx=8)
    trained on 100K to 1.7M chars. Shows dataset size impact.
    """
    try:
        return inference.get_data_size()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DATA_SIZE_ERROR", "message": str(e)},
        )


@router.get("/activation-battle")
async def activation_battle():
    """
    Returns 5-way activation function comparison (tanh, relu, gelu, sigmoid, linear).
    Same architecture trained with each activation. Sorted best → worst val_loss.
    """
    try:
        return inference.get_activation_battle()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "ACTIVATION_BATTLE_ERROR", "message": str(e)},
        )


@router.get("/embedding-bottleneck")
async def embedding_bottleneck():
    """
    Returns embedding dimension bottleneck experiment (E=2,4,8,32,128).
    Shows how embedding dimension affects quality and cluster structure.
    Sorted by emb_dim ascending.
    """
    try:
        return inference.get_embedding_bottleneck()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "EMBEDDING_BOTTLENECK_ERROR", "message": str(e)},
        )


@router.get("/network-shape")
async def network_shape():
    """
    Returns three network shape comparison models: pyramid, funnel, cylinder.
    Same parameter budget, different layer size distributions.
    Shows how architecture shape affects learning and information flow.
    """
    try:
        return inference.get_network_shape()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "NETWORK_SHAPE_ERROR", "message": str(e)},
        )


@router.get("/advanced-embeddings")
async def advanced_embeddings(dims: str | None = None):
    """
    Returns embedding matrices from mlp_advanced checkpoints (2D–128D).
    Each model: 3-layer H=256, orthogonal init, context_size=8, 50K steps.
    Use to compare how embedding dimension affects structure formation.

    Query params:
      dims: comma-separated list of dimensions (e.g. "2,10,32,128").
            Defaults to all available: 2,4,6,10,16,24,32,50,128.
    """
    parsed_dims: list[int] | None = None
    if dims:
        try:
            parsed_dims = [int(d.strip()) for d in dims.split(",") if d.strip()]
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_DIMS", "message": f"dims must be comma-separated integers, got: {dims}"},
            )
    try:
        return inference.get_advanced_embeddings(dims=parsed_dims)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_TRAINED_YET", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "ADVANCED_EMBEDDINGS_ERROR", "message": str(e)},
        )
