"""
Response schemas — Pydantic models defining the API JSON contract.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ============ Shared ============

class TensorData(BaseModel):
    """JSON-safe representation of a PyTorch tensor."""
    shape: list[int]
    data: list  # Nested list of floats
    dtype: str = "float32"


class ErrorResponse(BaseModel):
    """Standardized error envelope."""
    error: ErrorDetail


class ErrorDetail(BaseModel):
    code: str
    message: str
    available_models: list[str] | None = None


# ============ Health ============

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    device: str
    models_loaded: int


# ============ Model Catalog ============

class ModelSummary(BaseModel):
    """Compact model info for catalog listing."""
    id: str
    name: str
    description: str
    type: str
    complexity: str
    available: bool = Field(
        description="True if a trained checkpoint exists for this model"
    )


class ModelListResponse(BaseModel):
    models: list[ModelSummary]
    total: int


class ModelDetailResponse(BaseModel):
    """Full model info including how-it-works, strengths, etc."""
    id: str
    name: str
    description: str
    type: str
    complexity: str
    parameters: str | None = None
    training_time: str | None = None
    how_it_works: list[str] = []
    strengths: list[str] = []
    limitations: list[str] = []
    use_cases: list[str] = []
    visualization: str | None = None
    available: bool = True


# ============ Inference ============

class PredictionResult(BaseModel):
    token: str
    probability: float


class TokenInfo(BaseModel):
    text: str
    token_ids: list[int]


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    input: TokenInfo
    predictions: list[PredictionResult]
    full_distribution: list[float] | None = None
    metadata: InferenceMetadata


class InferenceMetadata(BaseModel):
    inference_time_ms: float
    device: str
    vocab_size: int


# ============ Internals ============

class InternalsResponse(BaseModel):
    """Full internal state returned by get_internals()."""
    model_id: str
    model_name: str
    input: TokenInfo
    predictions: list[PredictionResult]
    internals: dict[str, TensorData]
    metadata: InferenceMetadata


# ============ Generation ============

class GenerateResponse(BaseModel):
    model_id: str
    generated_text: str
    seed_text: str
    temperature: float
    length: int
    metadata: InferenceMetadata


# ============ Bigram-Specific Visualization ============

class TransitionMatrix(BaseModel):
    """Softmaxed character transition probability matrix with labels."""
    shape: list[int]
    data: list[list[float]]
    row_labels: list[str] = Field(description="Current character labels (rows)")
    col_labels: list[str] = Field(description="Next character labels (columns)")


class TrainingHistory(BaseModel):
    """Training curve and configuration metadata."""
    loss_history: list[float] = []
    final_loss: float | None = None
    training_steps: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    raw_text_size: int | None = None
    train_data_size: int | None = None
    val_data_size: int | None = None
    unique_characters: int | None = None


class ModelArchitectureInfo(BaseModel):
    """Model registry metadata (how it works, strengths, etc.)."""
    name: str
    description: str
    type: str
    complexity: str
    how_it_works: list[str] = []
    strengths: list[str] = []
    limitations: list[str] = []
    use_cases: list[str] = []


class BigramVisualization(BaseModel):
    """All visualization-ready data for the Bigram model."""
    transition_matrix: TransitionMatrix
    training: TrainingHistory
    architecture: ModelArchitectureInfo


class BigramInferenceResponse(BaseModel):
    """Full Bigram inference result: predictions + visualization data."""
    model_id: str
    model_name: str
    input: TokenInfo
    predictions: list[PredictionResult]
    full_distribution: list[float] | None = None
    visualization: BigramVisualization
    metadata: InferenceMetadata


# ============ Bigram Generation ============

class BigramGenerationResponse(BaseModel):
    """Bigram text generation result with temperature sampling."""
    model_id: str
    generated_text: str
    length: int
    temperature: float
    start_char: str
    metadata: InferenceMetadata


# ============ Bigram Stepwise Prediction ============

class StepwisePredictionStep(BaseModel):
    """Single step in a stepwise character prediction."""
    step: int
    char: str
    probability: float


class BigramStepwisePredictionResponse(BaseModel):
    """Multi-step character-by-character prediction result."""
    model_id: str
    input_text: str
    steps: list[StepwisePredictionStep]
    final_prediction: str
    metadata: InferenceMetadata


# ============ N-Gram Visualization ============

class HistoricalContext(BaseModel):
    description: str
    limitations: list[str]
    modern_evolution: str


class NGramDiagnostics(BaseModel):
    vocab_size: int
    context_size: int
    estimated_context_space: int  # V^(context_size)
    sparsity: float | None = None  # Optional


class ActiveSlice(BaseModel):
    """Visualization of the active slice of the N-gram tensor."""
    context_tokens: list[str]
    matrix: TransitionMatrix
    next_token_probs: dict[str, float]


class NGramInferenceResponse(BaseModel):
    """Response for N-Gram visualization endpoint."""
    model_id: str
    context_size: int
    context: list[str]
    active_slice: ActiveSlice
    diagnostics: NGramDiagnostics
    historical_context: HistoricalContext
    metadata: InferenceMetadata


# ============ Dataset Explorer ============

class DatasetLookupResponse(BaseModel):
    """Response for dataset example lookup."""
    query: str
    count: int
    examples: list[str]
    source: str
