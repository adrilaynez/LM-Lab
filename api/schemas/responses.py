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


class HistoricalContext(BaseModel):
    description: str
    limitations: list[str]
    modern_evolution: str


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
    historical_context: HistoricalContext


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




class NGramDiagnostics(BaseModel):
    vocab_size: int
    context_size: int
    estimated_context_space: int  # V^(context_size)
    sparsity: float | None = None
    perplexity: float | None = None
    context_utilization: float | None = None  # unique_contexts / context_space
    corpus_name: str = "Paul Graham Essays"
    smoothing_alpha: float = 1.0  # Laplace smoothing alpha used during training


class ContextDistribution(BaseModel):
    """Probability distribution for a specific context."""
    context: str
    probabilities: list[float]  # Full vocab distribution
    row_labels: list[str] | None = None  # Optional labels if not global


class NGramTrainingInfo(BaseModel):
    """Training statistics for N-Gram models."""
    total_tokens: int | None = None
    unique_chars: int | None = None
    unique_contexts: int | None = None
    context_space_size: int | None = None
    context_utilization: float | None = None
    sparsity: float | None = None
    transition_density: float | None = None
    loss_history: list[float] = []
    final_loss: float | None = None
    perplexity: float | None = None
    smoothing_alpha: float | None = None  # Laplace alpha (forwarded from diagnostics)
    corpus_name: str | None = None        # Forwarded for convenience


class ActiveSlice(BaseModel):
    """The active context slice shown in the transition matrix panel."""
    context_tokens: list[str] | None = None
    matrix: TransitionMatrix | None = None


class NGramVisualization(BaseModel):
    """Visualization-ready data for N-Gram models."""
    transition_matrix: TransitionMatrix | None = None  # For N=1
    active_slice: ActiveSlice | None = None            # For N>1: current context slice
    context_distributions: dict[str, ContextDistribution] | None = None
    training: NGramTrainingInfo
    diagnostics: NGramDiagnostics
    architecture: ModelArchitectureInfo
    historical_context: HistoricalContext | None = None


class NGramInferenceResponse(BaseModel):
    """Response for N-Gram visualization endpoint."""
    model_id: str
    model_name: str
    context_size: int
    input: TokenInfo
    predictions: list[PredictionResult]
    full_distribution: list[float] | None = None
    visualization: NGramVisualization
    metadata: InferenceMetadata


# ============ Dataset Explorer ============

class DatasetLookupResponse(BaseModel):
    """Response for dataset example lookup."""
    query: str
    count: int
    examples: list[str]
    source: str


# ============ N-Gram Stepwise Prediction ============

class NGramStepwiseStep(BaseModel):
    """Single step in N-Gram stepwise prediction."""
    step: int
    char: str
    probability: float
    context_window: list[str] = Field(description="The N characters used as context for this step")
    top_k: list[PredictionResult] = Field(default=[], description="Top-k predictions at this step")


class NGramStepwisePredictionResponse(BaseModel):
    """N-Gram step-by-step character prediction result."""
    model_id: str
    context_size: int
    input_text: str
    steps: list[NGramStepwiseStep]
    final_prediction: str
    metadata: InferenceMetadata


# ============ N-Gram Generation ============

class NGramGenerationResponse(BaseModel):
    """N-Gram text generation result with temperature sampling."""
    model_id: str
    context_size: int
    generated_text: str
    length: int
    temperature: float
    start_text: str
    metadata: InferenceMetadata
# ============ MLP-Specific Visualization ============

class MLPActivationStats(BaseModel):
    mean: float
    std: float

class MLPVisualization(BaseModel):
    """All visualization-ready data for the MLP model."""
    embedding_matrix: list[list[float]] # [vocab_size, emb_dim]
    loss_history: list[float]
    dead_neurons_history: list[float]
    activation_stats_history: list[MLPActivationStats]
    grad_norm_history: list[float]
    training_metadata: dict
    expected_uniform_loss: float
    token_frequency_distribution: list[float]
    architecture: ModelArchitectureInfo
    historical_context: HistoricalContext

class MLPInferenceResponse(BaseModel):
    """Full MLP inference result: predictions + visualization data."""
    model_id: str
    model_name: str
    config: dict
    input: TokenInfo
    predictions: list[PredictionResult]
    visualization: MLPVisualization
    metadata: InferenceMetadata


# ============ MLP Grid (Model Zoo) ============

class MLPGridConfigSummary(BaseModel):
    """Summary of a single trained MLP configuration from the grid.
    
    final_loss = val_loss when available, else train_loss (clearly labeled in frontend).
    perplexity is computed from final_loss for consistency.
    """
    embedding_dim: int
    hidden_size: int
    learning_rate: float
    context_size: int
    batch_size: int | None = None
    final_loss: float
    final_val_loss: float | None = None
    final_train_loss: float | None = None
    perplexity: float
    initial_loss: float | None = None
    initial_val_loss: float | None = None
    expected_uniform_loss: float | None = None
    generalization_gap: float | None = None
    train_time_sec: float | None = None
    total_parameters: int | None = None
    score: float | None = None
    snapshot_steps: list[str] = []
    filename: str


class MLPGridListResponse(BaseModel):
    """Response listing all available MLP grid configurations."""
    configurations: list[MLPGridConfigSummary]
    total: int


class MLPGridPredictResponse(BaseModel):
    """Next-token prediction for a specific MLP grid configuration."""
    model_id: str
    config: dict
    input: TokenInfo
    predictions: list[PredictionResult]
    full_distribution: list[float] | None = None
    metadata: InferenceMetadata


class MLPGridGenerateResponse(BaseModel):
    """Text generation result for a specific MLP grid configuration."""
    model_id: str
    config: dict
    generated_text: str
    seed_text: str
    temperature: float
    length: int
    metadata: InferenceMetadata


class MLPGridTimelineResponse(BaseModel):
    """Training snapshot timeline for a specific MLP grid configuration."""
    model_id: str
    config: dict
    metrics_log: dict
    snapshots: dict
    metadata: dict


class MLPGridEmbeddingResponse(BaseModel):
    """Embedding matrix for a specific MLP grid configuration."""
    model_id: str
    config: dict
    embedding_matrix: list[list[float]]
    vocab: list[str]
    shape: list[int]
    snapshot_step: str


class MLPGridInternalsResponse(BaseModel):
    """Internal activations for a specific MLP grid configuration."""
    model_id: str
    config: dict
    input: TokenInfo
    predictions: list[PredictionResult]
    internals: dict
    metadata: InferenceMetadata


class MLPGridEmbeddingQualityResponse(BaseModel):
    """Embedding quality metrics for a specific MLP grid configuration."""
    model_id: str
    config: dict
    metrics: dict
    snapshot_step: str
