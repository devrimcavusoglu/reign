"""
Test module for REIGN model configurations.

This module tests the different REIGN configuration classes to ensure they
instantiate correctly with expected parameters.
"""
from typing import Any, Dict

import pytest
import torch

from reign.configuration import ReignBaseConfig, ReignConfig, ReignLargeConfig, ReignSmallConfig


class TestReignConfigurations:
    """Test cases for REIGN model configurations."""

    def test_reign_small_config(self) -> None:
        """Test ReignSmallConfig initialization and parameters."""
        config = ReignSmallConfig()

        # Test expected parameter values
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 6
        assert config.intermediate_size == 1536
        assert config.gn_projection_dim == 384

        # Test head dimension calculation
        head_dim = config.hidden_size // config.num_attention_heads
        assert head_dim == 64  # Standard head dimension

    def test_reign_base_config(self) -> None:
        """Test ReignBaseConfig initialization and parameters."""
        config = ReignBaseConfig()

        # Test expected parameter values
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072
        assert config.gn_projection_dim == 768

        # Test head dimension calculation
        head_dim = config.hidden_size // config.num_attention_heads
        assert head_dim == 64  # Standard head dimension

        # Test intermediate size ratio
        assert config.intermediate_size == config.hidden_size * 4

    def test_reign_large_config(self) -> None:
        """Test ReignLargeConfig initialization and parameters."""
        config = ReignLargeConfig()

        # Test expected parameter values
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 4096
        assert config.gn_projection_dim == 1024

        # Test head dimension calculation
        head_dim = config.hidden_size // config.num_attention_heads
        assert head_dim == 64  # Standard head dimension

        # Test intermediate size ratio
        assert config.intermediate_size == config.hidden_size * 4

    def test_original_reign_config(self) -> None:
        """Test original ReignConfig still works with default values."""
        config = ReignConfig()

        # Test default parameter values
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 4096
        assert config.gn_projection_dim == 1024

    def test_config_inheritance(self) -> None:
        """Test that all configs properly inherit from ReignConfig."""
        configs = [
            ReignSmallConfig(),
            ReignBaseConfig(),
            ReignLargeConfig(),
        ]

        for config in configs:
            assert isinstance(config, ReignConfig)
            # Test common inherited attributes
            assert hasattr(config, "model_type")
            assert config.model_type == "reign"
            assert hasattr(config, "torch_dtype")

    @pytest.mark.parametrize(
        "config_class,expected_params",
        [
            (
                ReignSmallConfig,
                {
                    "hidden_size": 384,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 6,
                    "intermediate_size": 1536,
                },
            ),
            (
                ReignBaseConfig,
                {
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                },
            ),
            (
                ReignLargeConfig,
                {
                    "hidden_size": 1024,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "intermediate_size": 4096,
                },
            ),
        ],
    )
    def test_config_parameters(self, config_class: type, expected_params: Dict[str, Any]) -> None:
        """Test configuration parameters with parameterized inputs."""
        config = config_class()

        for param_name, expected_value in expected_params.items():
            actual_value = getattr(config, param_name)
            assert actual_value == expected_value, (
                f"{config_class.__name__}.{param_name} = {actual_value}, "
                f"expected = {expected_value}"
            )

    def test_config_with_custom_parameters(self) -> None:
        """Test that custom parameters can be passed to configs."""
        custom_max_pos_embeddings = 256

        configs = [
            ReignSmallConfig(max_position_embeddings=custom_max_pos_embeddings),
            ReignBaseConfig(max_position_embeddings=custom_max_pos_embeddings),
            ReignLargeConfig(max_position_embeddings=custom_max_pos_embeddings),
        ]

        for config in configs:
            assert config.max_position_embeddings == custom_max_pos_embeddings

    def test_parameter_count_relationships(self) -> None:
        """Test that parameter counts follow expected relationships."""
        small_config = ReignSmallConfig()
        base_config = ReignBaseConfig()
        large_config = ReignLargeConfig()

        # Helper function to estimate parameter count
        def estimate_params(config):
            # Simplified parameter count estimation
            # (not exact, but good for relative comparison)
            embed_params = config.hidden_size * config.vocab_size
            transformer_params = config.num_hidden_layers * (
                4 * config.hidden_size * config.hidden_size
                + 2 * config.hidden_size * config.intermediate_size
            )
            return embed_params + transformer_params

        small_params = estimate_params(small_config)
        base_params = estimate_params(base_config)
        large_params = estimate_params(large_config)

        # Test parameter count relationships
        assert small_params < base_params < large_params

        # Small should be significantly smaller than base
        assert small_params < base_params * 0.5

        # Large should be significantly larger than base
        assert large_params > base_params * 1.5
