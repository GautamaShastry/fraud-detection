"""Time limits that cannot be outlasted by cleanup.

asyncio.timeout and asyncio.wait_for return only once the cancelled work has finished. Database
cleanup (rolling back, returning a connection) can stall for as long as the database does, so a
stalled dependency would hold the response far past its deadline. `within` answers on time and
lets the cancelled work finish in the background.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

_background: set[asyncio.Future[Any]] = set()


def in_background(work: Awaitable[Any]) -> asyncio.Future[Any]:
    """Keep work running without a waiter (the event loop holds only weak task references)."""
    task = asyncio.ensure_future(work)
    _background.add(task)
    task.add_done_callback(_finished)
    return task


def _finished(task: asyncio.Future[Any]) -> None:
    _background.discard(task)
    if not task.cancelled():
        task.exception()  # Nobody is waiting; mark it retrieved so asyncio doesn't log it.


async def within[T](work: Awaitable[T], timeout_s: float) -> T:
    """Return work's result, or raise TimeoutError after `timeout_s` even if it won't stop."""
    task = asyncio.ensure_future(work)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
    except asyncio.CancelledError:
        task.cancel()
        in_background(task)
        raise
    if done:
        return task.result()
    task.cancel()
    in_background(task)
    raise TimeoutError
