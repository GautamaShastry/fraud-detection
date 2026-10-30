"""T-303 offline/online parity on synthetic replay, and T-306 real model inference in the API."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient

from ml.features_offline import OfflineF01, featurize
from ml.promote import approve
from ml.simulator import SimConfig, simulate
from ml.train import train
from src.api.app import create_app
from src.features.velocity import VelocityStore
from src.scoring.features import FeatureInputs, build_features
from tests.integration.conftest import REDIS_URL, now_request
from tests.support.contract import assert_response_matches

pytestmark = pytest.mark.integration

REPLAY = SimConfig(
    seed=21, days=2, customers=300, benign_per_day=200, card_testing_bursts_per_day=6
)


def _ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


async def test_offline_f01_equals_redis_lua_on_replay(redis_client):
    sim = simulate(REPLAY)
    client = aioredis.from_url(REDIS_URL)
    online, offline = VelocityStore(client), OfflineF01()
    mismatches = []
    try:
        for req in sim.requests:
            args = (
                "tenant-demo",
                req["payment_attempt_id"],
                req.get("device_id"),
                req.get("payment_method_token"),
                _ms(req["occurred_at"]),
            )
            a, b = (await online.f01(*args)).value, offline.update(*args)
            if a != b:
                mismatches.append((req["payment_attempt_id"], a, b))
        # Replaying a slice again (client retries) must return recorded values on both sides.
        for req in sim.requests[:50]:
            args = (
                "tenant-demo",
                req["payment_attempt_id"],
                req.get("device_id"),
                req.get("payment_method_token"),
                _ms(req["occurred_at"]) + 1000,
            )
            assert (await online.f01(*args)).value == offline.update(*args)
    finally:
        await client.aclose()
    assert not mismatches, mismatches[:5]
    assert max(offline._recorded.values()) >= 4  # the replay exercised card-testing bursts


def test_feature_vectors_match_between_training_and_serving_paths():
    sim = simulate(REPLAY)
    offline = featurize(sim.requests)
    f01 = OfflineF01()
    for req, expected in zip(sim.requests[:500], offline[:500], strict=True):
        occurred = datetime.fromisoformat(req["occurred_at"].replace("Z", "+00:00"))
        account = req.get("account_created_at")
        online = build_features(
            FeatureInputs(
                amount_minor=req["amount_minor"],
                occurred_at=occurred,
                account_created_at=(
                    datetime.fromisoformat(account.replace("Z", "+00:00")) if account else None
                ),
                f01=f01.update(
                    "tenant-demo",
                    req["payment_attempt_id"],
                    req.get("device_id"),
                    req.get("payment_method_token"),
                    int(occurred.timestamp() * 1000),
                ),
                has_customer=req.get("customer_id") is not None,
                has_device=req.get("device_id") is not None,
                has_payment_method=req.get("payment_method_token") is not None,
                has_ip=req.get("ip_address") is not None,
                has_shipping=req.get("shipping_destination_ref") is not None,
            )
        )
        for name, value in expected.items():
            other = online[name]
            assert (value is None and other is None) or abs(value - other) <= 1e-9, name


@pytest.fixture(scope="module")
def promoted_model(tmp_path_factory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("models")
    directory = train(seed=9, out=root, days=6)
    digest = approve(
        directory, "integration-test", synthetic_demo_only=True, models_root=root, env_file=None
    )
    return root / "current", digest


def test_api_serves_real_model_inference(
    integration_settings, promoted_model, auth_header, redis_client
):
    model_dir, digest = promoted_model
    settings = integration_settings.model_copy(
        update={"scorer_mode": "model", "model_dir": model_dir, "model_manifest_sha256": digest}
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post("/api/v1/fraud/score", json=now_request(), headers=auth_header)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scoring_mode"] == "ML"
    assert body["model_version"].startswith("synthetic-hgb-s9-")
    assert_response_matches("POST", "/api/v1/fraud/score", response)


def test_api_refuses_to_start_with_unverified_model(integration_settings, promoted_model, tmp_path):
    model_dir, digest = promoted_model
    tampered = tmp_path / "tampered"
    shutil.copytree(model_dir, tampered)
    with (tampered / "fallback.joblib").open("ab") as fh:
        fh.write(b"x")
    settings = integration_settings.model_copy(
        update={"scorer_mode": "model", "model_dir": tampered, "model_manifest_sha256": digest}
    )
    from src.scoring.artifact import ArtifactError

    with pytest.raises(ArtifactError), TestClient(create_app(settings)):
        pass  # startup must fail: the container exits non-zero and never becomes ready
