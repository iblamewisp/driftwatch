"""Pure helper functions — no I/O, no dependencies on app state."""

from fastapi import Request


def extract_request_text(messages: list) -> str:
    """Extract the last user message as plain text for embedding."""
    for msg in reversed(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "user":
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


def forwarded_headers(request: Request) -> dict[str, str]:
    """Strip internal headers before forwarding to the upstream provider."""
    skip = {"x-driftwatch-key", "host", "content-length"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}
