from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import settings
from app.middleware import AuthMiddleware
from app.proxy import router as proxy_router
from monitoring.logging import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    app.state.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    yield
    await app.state.http_client.aclose()
    await app.state.redis.aclose()


app = FastAPI(title="Driftwatch", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(AuthMiddleware)
app.include_router(proxy_router)


if settings.ENABLE_METRICS:
    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
