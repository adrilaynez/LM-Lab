"""
Models router — CRUD-style endpoints for model catalog, inference, and generation.
"""

from fastapi import APIRouter, HTTPException

from api.config import DEVICE
from api.schemas.requests import (
    PredictRequest,
    GenerateRequest,
    BigramGenerateRequest,
    BigramStepwiseRequest,
    NGramVisualizeRequest,
    DatasetLookupRequest,
)
from api.schemas.responses import (
    ModelSummary,
    ModelListResponse,
    ModelDetailResponse,
    PredictResponse,
    InternalsResponse,
    GenerateResponse,
    BigramInferenceResponse,
    BigramGenerationResponse,
    BigramStepwisePredictionResponse,
    PredictionResult,
    TokenInfo,
    InferenceMetadata,
    NGramInferenceResponse,
    DatasetLookupResponse
)
from api.services import inference
from api.services.serializer import serialize_internals

router = APIRouter(prefix="/models", tags=["Models"])


# --------------------------------------------------------------------------- #
#  Catalog
# --------------------------------------------------------------------------- #

@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all registered models with availability status."""
    all_ids = inference.get_all_model_ids()
    available_ids = set(inference.get_available_model_ids())

    summaries = []
    for mid in all_ids:
        detail = inference.get_model_detail(mid)
        summaries.append(
            ModelSummary(
                id=mid,
                name=detail.get("name", mid),
                description=detail.get("description", ""),
                type=detail.get("type", ""),
                complexity=detail.get("complexity", ""),
                available=mid in available_ids,
            )
        )

    return ModelListResponse(models=summaries, total=len(summaries))


@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(model_id: str):
    """Get detailed information about a specific model."""
    if not inference.model_exists(model_id):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_FOUND",
                "message": f"Model '{model_id}' is not registered",
                "available_models": inference.get_all_model_ids(),
            },
        )

    detail = inference.get_model_detail(model_id)
    return ModelDetailResponse(
        id=model_id,
        name=detail.get("name", model_id),
        description=detail.get("description", ""),
        type=detail.get("type", ""),
        complexity=detail.get("complexity", ""),
        parameters=detail.get("parameters"),
        training_time=detail.get("training_time"),
        how_it_works=detail.get("how_it_works", []),
        strengths=detail.get("strengths", []),
        limitations=detail.get("limitations", []),
        use_cases=detail.get("use_cases", []),
        visualization=detail.get("visualization"),
        available=inference.checkpoint_exists(model_id),
    )


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _require_checkpoint(model_id: str):
    """Raise 404 if model has no checkpoint."""
    if not inference.model_exists(model_id):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_FOUND",
                "message": f"Model '{model_id}' is not registered",
                "available_models": inference.get_all_model_ids(),
            },
        )
    if not inference.checkpoint_exists(model_id):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CHECKPOINT_NOT_AVAILABLE",
                "message": f"Model '{model_id}' exists but has no trained checkpoint yet",
            },
        )


# --------------------------------------------------------------------------- #
#  Bigram-Specific (visualization-ready)
#  ⚠️  These MUST come before /{model_id} routes so FastAPI doesn't
#  match "bigram" as a path parameter.
# --------------------------------------------------------------------------- #

@router.post("/bigram/visualize", response_model=BigramInferenceResponse)
async def bigram_visualize(body: PredictRequest):
    """
    Bigram-specific endpoint.
    Returns predictions + full visualization data (transition matrix,
    training loss curve, architecture info).
    """
    _require_checkpoint("bigram")

    try:
        result = inference.run_bigram_inference(body.text, body.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "BIGRAM_INFERENCE_ERROR", "message": str(e)},
        )

    return BigramInferenceResponse(**result)


@router.post("/bigram/generate", response_model=BigramGenerationResponse)
async def bigram_generate(body: BigramGenerateRequest):
    """
    Generate text from a single start character using the Bigram model
    with temperature-controlled sampling.
    """
    _require_checkpoint("bigram")

    try:
        result = inference.bigram_generate(
            body.start_char, body.num_tokens, body.temperature
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "BIGRAM_GENERATION_ERROR", "message": str(e)},
        )

    return BigramGenerationResponse(**result)


@router.post("/bigram/predict_stepwise", response_model=BigramStepwisePredictionResponse)
async def bigram_predict_stepwise(body: BigramStepwiseRequest):
    """
    Step-by-step character prediction from the last character of input text.
    Returns each predicted character with its sampling probability.
    """
    _require_checkpoint("bigram")

    try:
        result = inference.bigram_predict_stepwise(body.text, body.steps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "BIGRAM_STEPWISE_ERROR", "message": str(e)},
        )

    return BigramStepwisePredictionResponse(**result)


@router.post("/bigram/dataset_lookup", response_model=DatasetLookupResponse)
async def bigram_dataset_lookup(body: DatasetLookupRequest):
    """
    Find examples of specific Bigram sequence in the dataset.
    """
    try:
        # Bigram context is just 1 character
        # Validate context length? run_dataset_lookup doesn't strictly enforce it, 
        # but for bigram it makes sense to be 1 char context + 1 char next_token.
        
        result = inference.run_dataset_lookup(
            context_tokens=body.context,
            next_token=body.next_token
        )
        return result
    except Exception as e:
         raise HTTPException(
             status_code=500,
             detail={"code": "DATASET_LOOKUP_ERROR", "message": str(e)}
         )


# --------------------------------------------------------------------------- #
#  N-Gram & Interpretability
# --------------------------------------------------------------------------- #

@router.post("/ngram/visualize", response_model=NGramInferenceResponse)
async def ngram_visualize(body: NGramVisualizeRequest):
    """
    Visualize N-Gram model inference (N=1..5).
    Returns active slice of transition tensor.
    """
    try:
        result = inference.run_ngram_inference(
            text=body.text,
            context_size=body.context_size,
            top_k=body.top_k
        )
        return result
    except ValueError as e:
        msg = str(e)
        if msg.startswith("CONTEXT_TOO_LARGE"):
            parts = msg.split(":")
            req_n = int(parts[1])
            max_n = int(parts[2])
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "context_too_large",
                    "message": "N-gram context size exceeds practical limits due to combinatorial explosion.",
                    "details": {
                        "requested_n": req_n,
                        "max_supported_n": max_n,
                        "explanation": f"The number of possible contexts grows exponentially with N (VocabularySize^{req_n}), making higher-order N-grams impractical without massive data and storage."
                    }
                }
            )
        raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": str(e)})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "NGRAM_INFERENCE_ERROR", "message": str(e)}
        )


@router.post("/ngram/dataset_lookup", response_model=DatasetLookupResponse)
async def dataset_lookup(body: DatasetLookupRequest):
    """
    Find examples of specific N-gram sequence in the dataset.
    """
    try:
        result = inference.run_dataset_lookup(
            context_tokens=body.context,
            next_token=body.next_token
        )
        return result
    except Exception as e:
         raise HTTPException(
             status_code=500,
             detail={"code": "DATASET_LOOKUP_ERROR", "message": str(e)}
         )


# --------------------------------------------------------------------------- #
#  Inference (generic)
# --------------------------------------------------------------------------- #

@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(model_id: str, body: PredictRequest):
    """Run forward pass and return top-k next-token predictions."""
    _require_checkpoint(model_id)

    try:
        predictions, full_dist, tokenizer, elapsed_ms = inference.predict(
            model_id, body.text, body.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INFERENCE_ERROR", "message": str(e)})

    return PredictResponse(
        model_id=model_id,
        model_name=inference.get_model_detail(model_id).get("name", model_id),
        input=TokenInfo(
            text=body.text,
            token_ids=tokenizer.encode(body.text),
        ),
        predictions=[PredictionResult(**p) for p in predictions],
        full_distribution=full_dist,
        metadata=InferenceMetadata(
            inference_time_ms=round(elapsed_ms, 2),
            device=DEVICE,
            vocab_size=tokenizer.vocab_size,
        ),
    )


@router.post("/{model_id}/internals", response_model=InternalsResponse)
async def internals(model_id: str, body: PredictRequest):
    """Run forward pass + return full internal state (weights, activations, etc.)."""
    _require_checkpoint(model_id)

    try:
        predictions, raw_internals, tokenizer, elapsed_ms = inference.get_internals(
            model_id, body.text, body.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INFERENCE_ERROR", "message": str(e)})

    serialized = serialize_internals(raw_internals)

    return InternalsResponse(
        model_id=model_id,
        model_name=inference.get_model_detail(model_id).get("name", model_id),
        input=TokenInfo(
            text=body.text,
            token_ids=tokenizer.encode(body.text),
        ),
        predictions=[PredictionResult(**p) for p in predictions],
        internals=serialized,
        metadata=InferenceMetadata(
            inference_time_ms=round(elapsed_ms, 2),
            device=DEVICE,
            vocab_size=tokenizer.vocab_size,
        ),
    )


# --------------------------------------------------------------------------- #
#  Generation (generic)
# --------------------------------------------------------------------------- #

@router.post("/{model_id}/generate", response_model=GenerateResponse)
async def generate(model_id: str, body: GenerateRequest):
    """Auto-regressive text generation."""
    _require_checkpoint(model_id)

    try:
        generated_text, elapsed_ms = inference.generate(
            model_id, body.seed_text, body.max_length, body.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "GENERATION_ERROR", "message": str(e)})

    _, tokenizer, _, _ = inference._load_model(model_id)

    return GenerateResponse(
        model_id=model_id,
        generated_text=generated_text,
        seed_text=body.seed_text,
        temperature=body.temperature,
        length=len(generated_text),
        metadata=InferenceMetadata(
            inference_time_ms=round(elapsed_ms, 2),
            device=DEVICE,
            vocab_size=tokenizer.vocab_size,
        ),
    )

