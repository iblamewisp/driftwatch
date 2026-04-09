from app.config import settings
from db.repositories.base import AbstractVectorRepository
from db.repositories.pgvector_repo import PgvectorRepository
from db.repositories.qdrant_repo import QdrantRepository


def get_vector_store() -> AbstractVectorRepository:
    if settings.VECTOR_BACKEND == "pgvector":
        return PgvectorRepository()
    if settings.VECTOR_BACKEND == "qdrant":
        return QdrantRepository()
    raise ValueError(f"Unknown VECTOR_BACKEND: {settings.VECTOR_BACKEND}")
