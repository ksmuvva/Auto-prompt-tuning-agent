"""Pytest configuration and fixtures."""

import pytest
from src.agent.config import AgentConfig, LLMConfig, LLMProvider
from src.agent.llm_interface import LLMInterface
from src.agent.metrics import MetricsCalculator, PromptMetrics
from src.agent.prompt_tuner import PromptTuner
from src.agent.core import PromptTuningAgent


@pytest.fixture
def default_llm_config():
    """Provide default LLM configuration."""
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        api_key="test-key",
        temperature=0.7,
        max_tokens=1000,
    )


@pytest.fixture
def default_agent_config(default_llm_config):
    """Provide default agent configuration."""
    return AgentConfig(
        llm_config=default_llm_config,
        max_iterations=10,
        convergence_threshold=0.01,
    )


@pytest.fixture
def llm_interface(default_llm_config):
    """Provide LLM interface instance."""
    return LLMInterface(default_llm_config)


@pytest.fixture
def metrics_calculator():
    """Provide metrics calculator instance."""
    return MetricsCalculator()


@pytest.fixture
def prompt_tuner(llm_interface, metrics_calculator, default_agent_config):
    """Provide prompt tuner instance."""
    return PromptTuner(llm_interface, metrics_calculator, default_agent_config)


@pytest.fixture
def agent(default_agent_config):
    """Provide agent instance."""
    return PromptTuningAgent(default_agent_config)


@pytest.fixture
def sample_prompt():
    """Provide sample prompt for testing."""
    return "Write a function to calculate fibonacci numbers"


@pytest.fixture
def sample_prompts():
    """Provide list of sample prompts."""
    return [
        "Explain quantum computing",
        "Write a sorting algorithm",
        "Describe machine learning",
    ]


@pytest.fixture
def sample_metrics():
    """Provide sample PromptMetrics."""
    return PromptMetrics(
        score=0.75,
        length=100,
        clarity=0.8,
        specificity=0.7,
        token_efficiency=0.75,
        metadata={"test": True}
    )
