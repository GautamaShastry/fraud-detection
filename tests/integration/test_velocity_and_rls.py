"""T-103 F01 exactness with an injected clock, and T-101 database-level tenant isolation."""

from __future__ import annotations

import psycopg
import pytest
import redis.asyncio as aioredis

from src.features.velocity import WINDOW_MS, VelocityStore
from tests.integration.conftest import REDIS_URL

pytestmark = pytest.mark.integration

T0 = 1_800_000_000_000  # arbitrary epoch ms


@pytest.fixture
async def store(redis_client):
    client = aioredis.from_url(REDIS_URL)
    yield VelocityStore(client)
    await client.aclose()


async def test_distinct_tokens_counted_including_current(store):
    assert (await store.f01("t1", "a1", "dev", "tok1", T0)).value == 1
    assert (await store.f01("t1", "a2", "dev", "tok2", T0 + 1000)).value == 2
    assert (await store.f01("t1", "a3", "dev", "tok3", T0 + 2000)).value == 3


async def test_same_token_does_not_inflate(store):
    for i in range(5):
        result = await store.f01("t1", f"a{i}", "dev", "tok1", T0 + i * 1000)
        assert result.value == 1


async def test_retry_of_same_attempt_is_unchanged(store):
    first = await store.f01("t1", "a1", "dev", "tok1", T0)
    await store.f01("t1", "a2", "dev", "tok2", T0 + 10)
    replay = await store.f01("t1", "a1", "dev", "tok1", T0 + 20)
    assert replay.value == first.value == 1 and replay.replayed
    # The replay did not add anything either.
    assert (await store.f01("t1", "a3", "dev", "tok2", T0 + 30)).value == 2


async def test_exact_window_boundary(store):
    await store.f01("t1", "a1", "dev", "tok1", T0)
    # One millisecond before the token ages out it still counts...
    assert (await store.f01("t1", "a2", "dev", "tok2", T0 + WINDOW_MS - 1)).value == 2
    # ...exactly five minutes later it no longer does.
    assert (await store.f01("t1", "a3", "dev", "tok3", T0 + WINDOW_MS)).value == 2  # tok2 + tok3


async def test_reuse_refreshes_last_seen(store):
    await store.f01("t1", "a1", "dev", "tok1", T0)
    await store.f01("t1", "a2", "dev", "tok1", T0 + WINDOW_MS - 10)
    assert (await store.f01("t1", "a3", "dev", "tok2", T0 + WINDOW_MS + 10)).value == 2


async def test_missing_ids_remain_missing(store):
    assert (await store.f01("t1", "a1", None, "tok1", T0)).value is None
    assert (await store.f01("t1", "a2", "dev", None, T0)).value is None


async def test_tenants_and_devices_are_isolated(store):
    await store.f01("t1", "a1", "dev", "tok1", T0)
    await store.f01("t1", "a2", "dev", "tok2", T0)
    assert (await store.f01("t2", "a1", "dev", "tok3", T0)).value == 1  # other tenant
    assert (await store.f01("t1", "a9", "dev-2", "tok3", T0)).value == 1  # other device


# ------------------------------------------------------------------ RLS (T-101)


@pytest.fixture
def seeded(owner_conn):
    owner_conn.execute("DELETE FROM assessments WHERE payment_attempt_id LIKE 'rls-%'")
    owner_conn.execute("DELETE FROM attempts WHERE payment_attempt_id LIKE 'rls-%'")
    for tenant in ("tenant-a", "tenant-b"):
        owner_conn.execute(
            "INSERT INTO attempts (tenant_id, payment_attempt_id, idempotency_key, transaction_id,"
            " request_fingerprint, status, claim_token, lease_expires_at, amount_minor, currency,"
            " occurred_at, received_at, updated_at) VALUES (%s, %s, %s, 'tx', '\\x00',"
            " 'COMPLETED', gen_random_uuid(), now(), 1, 'USD', now(), now(), now())",
            (tenant, f"rls-{tenant}", f"rls-key-{tenant}"),
        )


def _app_conn(database):
    return psycopg.connect(database["app"])


def test_app_role_sees_nothing_without_tenant(database, seeded):
    with _app_conn(database) as conn:
        assert conn.execute("SELECT count(*) FROM attempts").fetchone() == (0,)


def test_app_role_sees_only_its_tenant(database, seeded):
    with _app_conn(database) as conn:
        conn.execute("SELECT set_config('app.tenant_id', 'tenant-b', true)")
        rows = conn.execute("SELECT tenant_id FROM attempts WHERE payment_attempt_id LIKE 'rls-%'")
        assert {r[0] for r in rows} == {"tenant-b"}


def test_app_role_cannot_write_other_tenant(database, seeded):
    with _app_conn(database) as conn:
        conn.execute("SELECT set_config('app.tenant_id', 'tenant-b', true)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "INSERT INTO attempts (tenant_id, payment_attempt_id, idempotency_key, "
                "transaction_id, request_fingerprint, status, claim_token, lease_expires_at, "
                "amount_minor, currency, occurred_at, received_at, updated_at) VALUES "
                "('tenant-a', 'rls-x', 'k', 'tx', '\\x00', 'COMPLETED', gen_random_uuid(), now(), "
                "1, 'USD', now(), now(), now())"
            )


def test_app_role_update_cannot_reach_other_tenant(database, seeded):
    with _app_conn(database) as conn:
        conn.execute("SELECT set_config('app.tenant_id', 'tenant-b', true)")
        cur = conn.execute(
            "UPDATE attempts SET status = 'UNAVAILABLE' WHERE tenant_id = 'tenant-a'"
        )
        assert cur.rowcount == 0


def test_app_role_cannot_bypass_or_escalate(database):
    with _app_conn(database) as conn:
        row = conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        assert row == (False, False)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("ALTER TABLE attempts DISABLE ROW LEVEL SECURITY")


def test_assessments_are_immutable(database, live_client, auth_header, owner_conn):
    from tests.integration.conftest import now_request

    body = now_request()
    response = live_client.post("/api/v1/fraud/score", json=body, headers=auth_header)
    assert response.status_code == 200
    # The runtime role has no UPDATE or DELETE privilege on assessments at all...
    with _app_conn(database) as conn:
        conn.execute("SELECT set_config('app.tenant_id', 'tenant-a', false)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("UPDATE assessments SET risk_score = 0")
        conn.rollback()
        conn.execute("SELECT set_config('app.tenant_id', 'tenant-a', false)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM assessments")
    # ...and even the owner cannot rewrite one: the trigger rejects every UPDATE.
    with pytest.raises(psycopg.errors.RaiseException):
        owner_conn.execute(
            "UPDATE assessments SET risk_score = 0 WHERE payment_attempt_id = %s",
            (body["payment_attempt_id"],),
        )
