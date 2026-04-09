from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class GoldenSetEntry(BaseModel):
    id: UUID
    description: str
    created_at: datetime


class KeygenResult(BaseModel):
    api_key: str   # dwk_{hex}
    key_hash: str  # sha256, goes into .env
