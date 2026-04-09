from typing import Any
from pydantic import BaseModel


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 1024
    stream: bool = False
    system: str | None = None

    model_config = {"extra": "allow"}


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicContentBlock(BaseModel):
    type: str
    text: str = ""


class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[AnthropicContentBlock]
    stop_reason: str | None = None
    usage: AnthropicUsage

    model_config = {"extra": "allow"}
