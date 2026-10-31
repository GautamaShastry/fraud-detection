"""Limits must hold even when cancelled work refuses to finish promptly."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.domain.bounded import in_background, within


async def _stalls_on_cancel(cancelled: list[str]) -> None:
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        cancelled.append("cancelled")
        await asyncio.sleep(0.5)  # A rollback against a stalled database
        cancelled.append("cleaned up")
        raise


async def test_returns_the_result():
    async def work() -> int:
        return 7

    assert await within(work(), 1) == 7


async def test_propagates_failures():
    async def work() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await within(work(), 1)


async def test_propagates_a_timeout_raised_by_the_work():
    async def work() -> None:
        raise TimeoutError

    with pytest.raises(TimeoutError):
        await within(work(), 1)


async def test_answers_on_time_while_cleanup_finishes_in_the_background():
    log: list[str] = []
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        await within(_stalls_on_cancel(log), 0.01)
    assert time.perf_counter() - started < 0.3  # asyncio.timeout would take over 0.5 s here
    await asyncio.sleep(0.7)
    assert log == ["cancelled", "cleaned up"]  # still cancelled, still cleaned up


async def test_cancelling_the_caller_cancels_the_work():
    log: list[str] = []
    caller = asyncio.create_task(within(_stalls_on_cancel(log), 10))
    await asyncio.sleep(0.01)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await asyncio.sleep(0.01)
    assert log == ["cancelled"]


async def test_failed_background_work_is_not_reported_as_unretrieved(caplog):
    async def work() -> None:
        raise OSError("database down")

    task = in_background(work())
    await asyncio.wait({task})
    await asyncio.sleep(0)
    assert task.done()
    del task
    assert "never retrieved" not in caplog.text
