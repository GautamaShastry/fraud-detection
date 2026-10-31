"""Review finding #7: a failing worker step must not stop the others."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from src.events import worker
from src.events.archive import ExportResult


class Steps:
    """Replaces the worker's jobs; stops the loop after `iterations` passes."""

    def __init__(self, monkeypatch, stop: asyncio.Event, iterations: int = 1, **failing) -> None:
        self.calls: list[str] = []
        self.stop, self.iterations, self.failing = stop, iterations, failing

        def job(name, result):
            async def run(*_args, **_kwargs):
                self.calls.append(name)
                if self.failing.get(name):
                    raise OSError(f"{name} down")
                return result

            return run

        monkeypatch.setattr(worker, "process_inbox_once", job("inbox", 0))
        monkeypatch.setattr(worker, "export_batch", job("export", ExportResult([], 0)))
        monkeypatch.setattr(worker, "purge_expired", job("purge", {"attempts": 0}))
        backlog = job("backlog", None)

        async def backlog_then_stop(*args, **kwargs):
            await backlog(*args, **kwargs)
            if self.calls.count("backlog") >= self.iterations:
                self.stop.set()

        monkeypatch.setattr(worker, "update_backlog", backlog_then_stop)


def _config(now: list[datetime] | None = None) -> worker.WorkerConfig:
    clock = (lambda: now[0]) if now else (lambda: datetime.now(UTC))
    return worker.WorkerConfig(idle_sleep_s=0.001, clock=clock)


@pytest.mark.parametrize("failing", ["inbox", "export", "purge"])
async def test_one_failing_step_does_not_stop_the_others(monkeypatch, failing):
    stop = asyncio.Event()
    steps = Steps(monkeypatch, stop, **{failing: True})
    await asyncio.wait_for(worker.run(None, None, _config(), stop), 5)
    assert steps.calls == ["inbox", "export", "purge", "backlog"]


async def test_failing_steps_keep_the_loop_alive(monkeypatch):
    stop = asyncio.Event()
    steps = Steps(monkeypatch, stop, iterations=2)
    calls = []

    async def inbox(*_args):
        calls.append("inbox")
        if len(calls) == 2:
            stop.set()
        raise OSError("database down")

    monkeypatch.setattr(worker, "process_inbox_once", inbox)
    await asyncio.wait_for(worker.run(None, None, _config(), stop), 5)
    assert len(calls) == 2  # survived the failures and looped
    assert "export" in steps.calls


async def test_failure_is_counted_by_step(monkeypatch):
    stop = asyncio.Event()
    Steps(monkeypatch, stop, export=True)
    counter = worker.metrics.WORKER_STEP_FAILURES.labels(step="export")
    before = counter._value.get()
    await asyncio.wait_for(worker.run(None, None, _config(), stop), 5)
    assert counter._value.get() == before + 1


async def test_failed_purge_retries_and_successful_purge_waits(monkeypatch):
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    stop = asyncio.Event()
    steps = Steps(monkeypatch, stop, iterations=3, purge=True)
    await asyncio.wait_for(worker.run(None, None, _config(now), stop), 5)
    assert steps.calls.count("purge") == 3  # not marked done, so it tries again next pass

    stop = asyncio.Event()
    steps = Steps(monkeypatch, stop, iterations=3)
    await asyncio.wait_for(worker.run(None, None, _config(now), stop), 5)
    assert steps.calls.count("purge") == 1  # inside purge_every
