"""Tests for configuration module."""

import pytest
from src.agent.config import LLMConfig, AgentConfig, LLMProvider


class TestLLMConfig:
    """Tests for LLMConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default LLM config initialization."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test custom LLM config initialization."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-2",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2000,
        )
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-2"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000

    @pytest.mark.edge_case
    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    @pytest.mark.edge_case
    def test_temperature_extreme_values(self):
        """Test temperature at extreme valid values."""
        config_min = LLMConfig(temperature=0.0)
        assert config_min.temperature == 0.0

        config_max = LLMConfig(temperature=2.0)
        assert config_max.temperature == 2.0

    @pytest.mark.edge_case
    def test_max_tokens_bounds(self):
        """Test max_tokens validation."""
        # Valid
        LLMConfig(max_tokens=1)
        LLMConfig(max_tokens=100000)

        # Invalid
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=-1)

    @pytest.mark.micro
    def test_timeout_validation(self):
        """Test timeout parameter validation."""
        config = LLMConfig(timeout=60)
        assert config.timeout == 60

        with pytest.raises(ValueError):
            LLMConfig(timeout=0)

    @pytest.mark.micro
    def test_max_retries_validation(self):
        """Test max_retries validation."""
        config = LLMConfig(max_retries=5)
        assert config.max_retries == 5

        # Should accept 0 retries
        config_zero = LLMConfig(max_retries=0)
        assert config_zero.max_retries == 0

    @pytest.mark.edge_case
    def test_provider_enum_values(self):
        """Test all provider enum values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.LOCAL == "local"


class TestAgentConfig:
    """Tests for AgentConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default agent config initialization."""
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.convergence_threshold == 0.01
        assert config.batch_size == 5
        assert config.enable_logging is True
        assert config.log_level == "INFO"

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test custom agent config initialization."""
        llm_config = LLMConfig(api_key="test")
        config = AgentConfig(
            llm_config=llm_config,
            max_iterations=20,
            convergence_threshold=0.05,
            batch_size=10,
        )
        assert config.max_iterations == 20
        assert config.convergence_threshold == 0.05
        assert config.batch_size == 10

    @pytest.mark.edge_case
    def test_max_iterations_bounds(self):
        """Test max_iterations validation."""
        # Valid
        AgentConfig(max_iterations=1)
        AgentConfig(max_iterations=100)

        # Invalid
        with pytest.raises(ValueError):
            AgentConfig(max_iterations=0)
        with pytest.raises(ValueError):
            AgentConfig(max_iterations=-1)

    @pytest.mark.edge_case
    def test_convergence_threshold_bounds(self):
        """Test convergence_threshold validation."""
        # Valid
        AgentConfig(convergence_threshold=0.0)
        AgentConfig(convergence_threshold=0.5)
        AgentConfig(convergence_threshold=1.0)

        # Invalid
        with pytest.raises(ValueError):
            AgentConfig(convergence_threshold=-0.1)
        with pytest.raises(ValueError):
            AgentConfig(convergence_threshold=1.1)

    @pytest.mark.edge_case
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = AgentConfig(log_level=level)
            assert config.log_level == level

        # Test lowercase (should be converted to uppercase)
        config = AgentConfig(log_level="debug")
        assert config.log_level == "DEBUG"

        # Invalid level
        with pytest.raises(ValueError):
            AgentConfig(log_level="INVALID")

    @pytest.mark.micro
    def test_to_dict(self):
        """Test config serialization to dict."""
        config = AgentConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "max_iterations" in config_dict
        assert "convergence_threshold" in config_dict

    @pytest.mark.micro
    def test_from_dict(self):
        """Test config deserialization from dict."""
        config_dict = {
            "max_iterations": 15,
            "convergence_threshold": 0.02,
        }
        config = AgentConfig.from_dict(config_dict)
        assert config.max_iterations == 15
        assert config.convergence_threshold == 0.02

    @pytest.mark.micro
    def test_round_trip_serialization(self):
        """Test config serialization round trip."""
        original = AgentConfig(max_iterations=20)
        config_dict = original.to_dict()
        restored = AgentConfig.from_dict(config_dict)
        assert original.max_iterations == restored.max_iterations

    @pytest.mark.edge_case
    def test_empty_initial_prompts(self):
        """Test with empty initial prompts list."""
        config = AgentConfig(initial_prompts=[])
        assert config.initial_prompts == []

    @pytest.mark.edge_case
    def test_large_batch_size(self):
        """Test batch size at boundary."""
        config = AgentConfig(batch_size=50)
        assert config.batch_size == 50

        with pytest.raises(ValueError):
            AgentConfig(batch_size=51)

    @pytest.mark.edge_case
    def test_small_batch_size(self):
        """Test batch size at lower boundary."""
        config = AgentConfig(batch_size=1)
        assert config.batch_size == 1

        with pytest.raises(ValueError):
            AgentConfig(batch_size=0)
