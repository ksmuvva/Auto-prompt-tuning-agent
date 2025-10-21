"""Configuration management for the prompt tuning agent."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMConfig(BaseModel):
    """Configuration for LLM interface."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = "default-test-key"  # Default for testing
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=100000)
    timeout: int = Field(default=30, ge=1, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within bounds."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AgentConfig(BaseModel):
    """Configuration for the prompt tuning agent."""
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    max_iterations: int = Field(default=10, ge=1, le=100)
    convergence_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    initial_prompts: List[str] = Field(default_factory=list)
    optimization_metric: str = Field(default="score")
    batch_size: int = Field(default=5, ge=1, le=50)
    enable_logging: bool = True
    log_level: str = "INFO"

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max_iterations is positive."""
        if v <= 0:
            raise ValueError("max_iterations must be positive")
        return v

    @field_validator("convergence_threshold")
    @classmethod
    def validate_convergence_threshold(cls, v: float) -> float:
        """Validate convergence_threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("convergence_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary."""
        return cls(**data)
