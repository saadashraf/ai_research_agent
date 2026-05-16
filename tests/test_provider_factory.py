"""Tests for the get_provider() factory in src/providers/__init__.py."""

from unittest.mock import patch

import pytest

from src.providers import get_provider
from src.providers.anthropic_provider import AnthropicProvider


class TestGetProvider:
    def test_returns_anthropic_provider_when_configured(self):
        with patch("src.providers.config") as mock_config, \
             patch("src.providers.anthropic_provider.anthropic.Anthropic"):
            mock_config.PROVIDER = "anthropic"
            mock_config.MODEL = "claude-test"
            result = get_provider()
        assert isinstance(result, AnthropicProvider)

    def test_passes_model_from_config(self):
        with patch("src.providers.config") as mock_config, \
             patch("src.providers.anthropic_provider.anthropic.Anthropic"):
            mock_config.PROVIDER = "anthropic"
            mock_config.MODEL = "claude-haiku-4-5"
            result = get_provider()
        assert result.model_name == "claude-haiku-4-5"

    def test_raises_for_unknown_provider(self):
        with patch("src.providers.config") as mock_config:
            mock_config.PROVIDER = "openai"
            with pytest.raises(ValueError, match="Unknown provider"):
                get_provider()

    def test_error_message_mentions_bad_provider_name(self):
        with patch("src.providers.config") as mock_config:
            mock_config.PROVIDER = "nonsense"
            with pytest.raises(ValueError, match="nonsense"):
                get_provider()
