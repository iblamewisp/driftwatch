from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

async_engine = create_async_engine(
    settings.DATABASE_URL,
    # ── Connection health ────────────────────────────────────────────────────
    # Sends "SELECT 1" before giving out a connection — catches dead TCP
    # connections (pg restart, network drop) before the query even starts.
    pool_pre_ping=True,
    # Recycle connections older than 30 min. Without this, a long-lived worker
    # can hold a TCP socket that the OS or PgBouncer silently killed.
    pool_recycle=1800,
    # ── Pool sizing ──────────────────────────────────────────────────────────
    pool_size=10,
    max_overflow=20,
    # How long to wait for a connection from the pool before raising.
    # Prevents unbounded queue pile-up when the DB is slow.
    pool_timeout=30,
    # ── Per-statement timeout (asyncpg) ──────────────────────────────────────
    # Kills any query that hasn't returned in 20s at the driver level.
    # This is the main guard against hung in-flight queries —
    # pool_pre_ping only runs *before* a connection is handed out.
    connect_args={"command_timeout": settings.DB_COMMAND_TIMEOUT},
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
