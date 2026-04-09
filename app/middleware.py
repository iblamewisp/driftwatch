import hmac
import hashlib
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from monitoring.logging import request_id_var


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Only guard proxied routes
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        key = request.headers.get("X-DriftWatch-Key", "")
        if not key:
            return JSONResponse(status_code=401, content={"detail": "Missing X-DriftWatch-Key header"})

        key_hash = hashlib.sha256(key.encode()).hexdigest()
        if not hmac.compare_digest(key_hash, settings.DRIFTWATCH_KEY_HASH):
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request_id_var.set(request_id)

        return await call_next(request)
