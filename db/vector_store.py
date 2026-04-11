from app.config import settings
from db.repositories.base import AbstractVectorRepository
from db.repositories.cluster_base import AbstractClusterRepository
from db.repositories.cluster_postgres import PostgresClusterRepository
from db.repositories.pgvector_repo import PgvectorRepository
from db.repositories.qdrant_repo import QdrantRepository


def get_vector_store() -> AbstractVectorRepository:
    if settings.VECTOR_BACKEND == "pgvector":
        return PgvectorRepository()
    if settings.VECTOR_BACKEND == "qdrant":
        return QdrantRepository()
    raise ValueError(f"Unknown VECTOR_BACKEND: {settings.VECTOR_BACKEND}")


def get_cluster_repository() -> AbstractClusterRepository:
    if settings.VECTOR_BACKEND == "pgvector":
        return PostgresClusterRepository()
    raise ValueError(f"No cluster repository for VECTOR_BACKEND: {settings.VECTOR_BACKEND}")
