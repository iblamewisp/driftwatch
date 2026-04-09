import fnmatch

from app.providers.base import AbstractProvider
from app.providers.openai import OpenAIProvider
from app.providers.anthropic import AnthropicProvider

# Patterns are matched in order — first match wins.
_ROUTING_TABLE: list[tuple[str, AbstractProvider]] = [
    ("claude-*", AnthropicProvider()),
    ("gpt-*",    OpenAIProvider()),
    ("o1*",      OpenAIProvider()),
    ("o3*",      OpenAIProvider()),
    ("o4*",      OpenAIProvider()),
]

_openai_default = OpenAIProvider()


def get_provider(model: str) -> AbstractProvider:
    for pattern, provider in _ROUTING_TABLE:
        if fnmatch.fnmatch(model, pattern):
            return provider
    return _openai_default
