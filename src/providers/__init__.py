import config
from src.providers.base import ModelProvider


def get_provider() -> ModelProvider:
    """
    Factory function — reads config and returns the right provider.
    Your agent calls this once at startup. That's the only
    place the provider name string ever appears.
    """
    if config.PROVIDER == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=config.MODEL)

    # Add more providers here in later phases:
    # elif config.PROVIDER == "openai":
    #     from src.providers.openai_provider import OpenAIProvider
    #     return OpenAIProvider(model=config.MODEL)

    raise ValueError(f"Unknown provider: '{config.PROVIDER}'. "
                     f"Check config.py — valid options: anthropic")