"""
Request schemas — Pydantic models for API input validation.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input for prediction and internals endpoints."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input text to feed into the model",
        examples=["hello"]
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=65,
        description="Number of top predictions to return"
    )


class GenerateRequest(BaseModel):
    """Input for text generation endpoint."""
    seed_text: str = Field(
        default="",
        max_length=200,
        description="Optional seed text to start generation from"
    )
    max_length: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of characters to generate"
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)"
    )


class BigramGenerateRequest(BaseModel):
    """Input for bigram-specific text generation with temperature."""
    start_char: str = Field(
        default="A",
        min_length=1,
        max_length=1,
        description="Single character to start generation from",
        examples=["A"],
    )
    num_tokens: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of characters to generate",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )


class BigramStepwiseRequest(BaseModel):
    """Input for step-by-step character prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input text; prediction starts from last character",
        examples=["The quick"],
    )
    steps: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of characters to predict step-by-step",
    )


class NGramVisualizeRequest(BaseModel):
    """Input for N-Gram visualization."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Input text (context)",
        examples=["hello"],
    )
    context_size: int = Field(
        ...,
        ge=1,
        le=20,
        description="N-gram context size (N-1). 1=Bigram, 2=Trigram.",
        examples=[2]
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Top predictions to return"
    )


class DatasetLookupRequest(BaseModel):
    """Input for dataset explorer."""
    context: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of context tokens",
        examples=[["Q"]]
    )
    next_token: str = Field(
        ...,
        min_length=1,
        max_length=1,
        description="Next token to look for",
        examples=["u"]
    )


class NGramStepwiseRequest(BaseModel):
    """Input for N-Gram step-by-step character prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input text; prediction uses last N characters as context",
        examples=["The quick"],
    )
    context_size: int = Field(
        ...,
        ge=1,
        le=5,
        description="N-gram context size (1=Bigram, 2=Trigram, etc.)",
        examples=[2],
    )
    steps: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of characters to predict step-by-step",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Top predictions per step",
    )


class NGramGenerateRequest(BaseModel):
    """Input for N-Gram text generation."""
    start_text: str = Field(
        default="The ",
        min_length=1,
        max_length=100,
        description="Seed text to start generation from (at least N characters recommended)",
        examples=["The "],
    )
    context_size: int = Field(
        ...,
        ge=1,
        le=5,
        description="N-gram context size (1=Bigram, 2=Trigram, etc.)",
        examples=[2],
    )
    num_tokens: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of characters to generate",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )


# ============ MLP Grid Requests ============

class MLPGridPredictRequest(BaseModel):
    """Input for MLP grid next-token prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input text to feed into the model",
        examples=["hello"],
    )
    embedding_dim: int = Field(..., ge=1, description="Embedding dimension", examples=[10])
    hidden_size: int = Field(..., ge=1, description="Hidden layer size", examples=[128])
    learning_rate: float = Field(..., gt=0.0, description="Learning rate used during training", examples=[0.1])
    top_k: int = Field(default=10, ge=1, le=65, description="Number of top predictions to return")


class MLPGridGenerateRequest(BaseModel):
    """Input for MLP grid text generation."""
    embedding_dim: int = Field(..., ge=1, description="Embedding dimension", examples=[10])
    hidden_size: int = Field(..., ge=1, description="Hidden layer size", examples=[128])
    learning_rate: float = Field(..., gt=0.0, description="Learning rate used during training", examples=[0.1])
    seed_text: str = Field(
        default="The ",
        max_length=200,
        description="Seed text to start generation from",
    )
    max_tokens: int = Field(default=100, ge=1, le=500, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description="Sampling temperature")


class MLPGridInternalsRequest(BaseModel):
    """Input for MLP grid internals inspection."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input text to feed into the model",
        examples=["hello"],
    )
    embedding_dim: int = Field(..., ge=1, description="Embedding dimension", examples=[10])
    hidden_size: int = Field(..., ge=1, description="Hidden layer size", examples=[128])
    learning_rate: float = Field(..., gt=0.0, description="Learning rate used during training", examples=[0.1])
    top_k: int = Field(default=10, ge=1, le=65, description="Number of top predictions to return")
