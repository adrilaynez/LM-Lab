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
