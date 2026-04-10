"""
Postgres transaction-scoped advisory locks.

pg_advisory_xact_lock acquires an exclusive session-level lock that is
automatically released when the transaction commits or rolls back — no
manual cleanup, no risk of forgetting to release.

Usage:
    async with session.begin():
        await acquire_xact_lock(session, AdvisoryLock.CLUSTER_CREATION)
        # lock held until transaction ends

All lock keys are centralised here so collisions are visible at a glance.
Keys are arbitrary stable bigints — do not reuse or renumber them.
"""

from enum import IntEnum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


class AdvisoryLock(IntEnum):
    CLUSTER_CREATION = 7_301_948_201_738_492


async def acquire_xact_lock(session: AsyncSession, lock: AdvisoryLock) -> None:
    """Block until the advisory lock is acquired for the current transaction."""
    await session.execute(select(func.pg_advisory_xact_lock(lock.value)))
