"""Public response schemas for the Speech Emotion Recognition API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmotionProbabilities(BaseModel):
    """Probability for every fixed emotion class in model-label order."""

    model_config = ConfigDict(extra="forbid")

    neutral: float = Field(ge=0.0, le=1.0)
    calm: float = Field(ge=0.0, le=1.0)
    happy: float = Field(ge=0.0, le=1.0)
    sad: float = Field(ge=0.0, le=1.0)
    angry: float = Field(ge=0.0, le=1.0)
    fearful: float = Field(ge=0.0, le=1.0)
    disgust: float = Field(ge=0.0, le=1.0)
    surprised: float = Field(ge=0.0, le=1.0)


class ModelIdentity(BaseModel):
    """Stable identifier for the frozen local inference model."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(min_length=1)
    version: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Validated prediction returned for one uploaded recording."""

    model_config = ConfigDict(extra="forbid")

    emotion: str = Field(min_length=1)
    class_index: int = Field(ge=0, le=7)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: EmotionProbabilities
    model: ModelIdentity
