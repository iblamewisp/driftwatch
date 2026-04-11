import hashlib

from fastapi import APIRouter, Request

from app.forwarding import forward_streaming, forward_unary
from app.providers.registry import get_provider
from app.schemas.anthropic import AnthropicRequest
from app.schemas.openai import ChatCompletionRequest
from app.utils import extract_request_text, forwarded_headers
from monitoring.metrics import REQUEST_COUNTER
from services.circuit_breaker import get_provider_breaker

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    parsed = ChatCompletionRequest.model_validate(body)
    provider = get_provider(parsed.model)
    REQUEST_COUNTER.labels(model=parsed.model, stream=str(parsed.stream)).inc()
    prompt_hash = hashlib.sha256(parsed.model_dump_json(include={"messages"}).encode()).hexdigest()

    ctx = dict(
        provider=provider,
        breaker=get_provider_breaker(provider.__class__.__name__.replace("Provider", "").lower()),
        body=body,
        headers=provider.prepare_headers(forwarded_headers(request)),
        http_client=request.app.state.http_client,
        redis=request.app.state.redis,
        request_id=request.state.request_id,
        request_text=extract_request_text(parsed.messages),
        prompt_hash=prompt_hash,
        model=parsed.model,
    )
    if parsed.stream:
        return await forward_streaming(**ctx)
    return await forward_unary(**ctx)


@router.post("/v1/messages")
async def messages(request: Request):
    from app.providers.anthropic import AnthropicProvider
    body = await request.json()
    parsed = AnthropicRequest.model_validate(body)
    provider = AnthropicProvider()
    REQUEST_COUNTER.labels(model=parsed.model, stream=str(parsed.stream)).inc()
    prompt_hash = hashlib.sha256(parsed.model_dump_json(include={"messages"}).encode()).hexdigest()

    ctx = dict(
        provider=provider,
        breaker=get_provider_breaker("anthropic"),
        body=body,
        headers=provider.prepare_headers(forwarded_headers(request)),
        http_client=request.app.state.http_client,
        redis=request.app.state.redis,
        request_id=request.state.request_id,
        request_text=extract_request_text(parsed.messages),
        prompt_hash=prompt_hash,
        model=parsed.model,
    )
    if parsed.stream:
        return await forward_streaming(**ctx)
    return await forward_unary(**ctx)
