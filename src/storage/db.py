"""Connection pool and tenant-scoped transactions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

Conn = AsyncConnection[DictRow]
Pool = AsyncConnectionPool[Conn]


def create_pool(dsn: str, *, min_size: int = 2, max_size: int = 20) -> Pool:
    # Autocommit: single-statement function calls cost one round trip; multi-statement work uses
    # an explicit transaction (tenant_tx).
    kwargs: dict[str, Any] = {"row_factory": dict_row, "autocommit": True}
    pool = AsyncConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        open=False,
        kwargs=kwargs,
        timeout=2.0,
    )
    return cast(Pool, pool)


@asynccontextmanager
async def tenant_tx(pool: Pool, tenant_id: str) -> AsyncIterator[Conn]:
    """A transaction in which RLS restricts every statement to one tenant.

    Pipeline mode is deliberately not used: a pipelined query cancelled by the request deadline
    can leave the pooled connection unusable.
    """
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
        yield conn
