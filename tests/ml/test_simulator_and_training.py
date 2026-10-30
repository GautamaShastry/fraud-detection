"""T-302..T-307 offline: simulator, offline F01, labels, reproducible training, artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ml.evaluate import score_cutpoints, threshold_for_fpr
from ml.features_offline import OfflineF01
from ml.promote import approve, deny
from ml.sagemaker_job import (
    CostNotApprovedError,
    build_spec,
    check_approval,
    submit,
    training_job_request,
)
from ml.simulator import SimConfig, simulate
from ml.train import observed_labels, train
from src.api.schemas import ScoreRequest
from src.features.velocity import WINDOW_MS
from src.scoring.artifact import ArtifactError, load_artifact, sha256_file

SMALL = SimConfig(seed=11, days=3, customers=400, benign_per_day=150)


# ---------------------------------------------------------------- simulator


def test_simulator_is_deterministic():
    a, b = simulate(SMALL), simulate(SMALL)
    assert a.requests == b.requests and a.outcomes == b.outcomes and a.truth == b.truth
    assert simulate(SimConfig(**{**SMALL.__dict__, "seed": 12})).requests != a.requests


def test_requests_are_valid_api_payloads_without_ground_truth():
    sim = simulate(SMALL)
    for req in sim.requests:
        ScoreRequest.model_validate(req)  # rejects unknown fields such as a leaked label
        assert not {"scenario", "is_fraud", "label"} & set(req)


def test_referential_integrity_and_chronology():
    sim = simulate(SMALL)
    attempts = {r["payment_attempt_id"]: r["occurred_at"] for r in sim.requests}
    assert len(attempts) == len(sim.requests)
    assert [r["occurred_at"] for r in sim.requests] == sorted(
        r["occurred_at"] for r in sim.requests
    )
    for e in sim.outcomes:
        assert e["payment_attempt_id"] in attempts
        assert e["occurred_at"] > attempts[e["payment_attempt_id"]]


def test_scenarios_present():
    scenarios = {t["scenario"] for t in simulate(SimConfig(seed=3, days=10)).truth}
    assert {"benign", "benign_edge", "card_testing", "stolen_card", "account_takeover"} <= scenarios


# ---------------------------------------------------------------- offline F01 semantics


def test_offline_f01_matches_documented_semantics():
    f = OfflineF01()
    t0 = 1_800_000_000_000
    assert f.update("t", "a1", "d", "k1", t0) == 1
    assert f.update("t", "a2", "d", "k2", t0 + 10) == 2
    assert f.update("t", "a1", "d", "k1", t0 + 20) == 1  # retry returns recorded value
    assert f.update("t", "a3", "d", "k2", t0 + 30) == 2  # same token does not inflate
    assert f.update("t", "a4", "d", "k3", t0 + WINDOW_MS) == 2  # k1 aged out at exactly 5 min
    assert f.update("t", "a5", None, "k3", t0) is None


# ---------------------------------------------------------------- labels and evaluation


def _evt(attempt: str, kind: str, at: datetime) -> dict:
    return {
        "event_id": f"e-{attempt}-{kind}",
        "payment_attempt_id": attempt,
        "event_type": kind,
        "source": "ISSUER",
        "occurred_at": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def test_observed_labels_respect_maturity_and_never_treat_approval_as_legit():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    reqs = [
        {"payment_attempt_id": p, "occurred_at": t0.strftime("%Y-%m-%dT%H:%M:%SZ")}
        for p in ("old-legit", "fraud", "fresh", "late-fraud")
    ]
    reqs[2]["occurred_at"] = (t0 + timedelta(days=50)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [
        _evt("old-legit", "AUTHORIZATION_APPROVED", t0 + timedelta(seconds=1)),
        _evt("fraud", "CONFIRMED_FRAUD", t0 + timedelta(days=5)),
        _evt("fresh", "AUTHORIZATION_APPROVED", t0 + timedelta(days=50, seconds=1)),
        _evt("late-fraud", "CONFIRMED_FRAUD", t0 + timedelta(days=90)),
    ]
    labels = observed_labels(reqs, events, as_of=t0 + timedelta(days=70))
    # old-legit matured (0); fraud confirmed (1); fresh immature (-1); late fraud not yet known
    # at training time, and matured, so observed as 0 (label noise, as in production).
    assert labels.tolist() == [0, 1, -1, 0]


def test_threshold_for_fpr_never_exceeds_target_on_validation():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0, 1)
    thr, fpr = threshold_for_fpr(y, p, 0.01)
    assert fpr <= 0.01
    assert ((p >= thr) & (y == 0)).sum() / (y == 0).sum() <= 0.01


def test_score_cutpoints_monotonic():
    cut = score_cutpoints(np.random.default_rng(1).random(1000))
    assert len(cut) == 101 and cut == sorted(cut) and cut[0] == 0.0


# ---------------------------------------------------------------- reproducible training


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    return train(seed=5, out=tmp_path_factory.mktemp("models"), days=6)


def test_same_seed_reproduces_identical_model_files(trained, tmp_path):
    again = train(seed=5, out=tmp_path, days=6)
    for name in ("primary.joblib", "fallback.joblib"):
        if (trained / name).exists():
            assert sha256_file(trained / name) == sha256_file(again / name)
    m1 = json.loads((trained / "manifest.json").read_text())
    m2 = json.loads((again / "manifest.json").read_text())
    assert m1["policy"]["cutpoints"] == m2["policy"]["cutpoints"]
    assert m1["training"]["seeds"] == {"master": 5, "simulator": 5, "model": 5}


def test_manifest_contents(trained):
    m = json.loads((trained / "manifest.json").read_text())
    assert m["data_source"] == "SYNTHETIC" and m["approval"] == {"status": "PENDING"}
    assert m["library_versions"]["scikit-learn"]
    assert m["policy"]["decline_threshold"] is not None
    if m["fallback_policy"]:
        assert m["fallback_policy"]["decline_threshold"] is None  # I-08
    assert "Synthetic" in (trained / "report.md").read_text()


# ---------------------------------------------------------------- promotion and loading


@pytest.fixture
def workspace(trained, tmp_path) -> Path:
    root = tmp_path / "models"
    shutil.copytree(trained, root / trained.name)
    return root / trained.name


def test_unapproved_artifact_is_refused(workspace):
    with pytest.raises(ArtifactError, match="approved"):
        load_artifact(workspace, sha256_file(workspace / "manifest.json"))


def test_synthetic_requires_demo_only_flag(workspace):
    with pytest.raises(ArtifactError, match="synthetic-demo-only"):
        approve(
            workspace,
            "tester",
            synthetic_demo_only=False,
            models_root=workspace.parent,
            env_file=None,
        )


def test_approved_artifact_loads_and_scores(workspace, tmp_path):
    env = tmp_path / "model.env"
    digest = approve(
        workspace, "tester", synthetic_demo_only=True, models_root=workspace.parent, env_file=env
    )
    assert f"FRAUD_MODEL_MANIFEST_SHA256={digest}" in env.read_text()
    loaded = load_artifact(workspace.parent / "current", digest)
    assert loaded.manifest["approval"]["approver"] == "tester"
    log = (workspace.parent / "promotion_log.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["decision"] == "APPROVED"


def test_tampered_model_file_is_refused(workspace):
    digest = approve(
        workspace, "tester", synthetic_demo_only=True, models_root=workspace.parent, env_file=None
    )
    current = workspace.parent / "current"
    with (current / "primary.joblib").open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ArtifactError, match="Digest mismatch"):
        load_artifact(current, digest)


def test_wrong_pinned_manifest_digest_is_refused(workspace):
    approve(
        workspace, "tester", synthetic_demo_only=True, models_root=workspace.parent, env_file=None
    )
    with pytest.raises(ArtifactError, match="pinned"):
        load_artifact(workspace.parent / "current", "0" * 64)


def test_feature_schema_mismatch_is_refused(workspace):
    manifest = json.loads((workspace / "manifest.json").read_text())
    manifest["feature_schema"] = "online-features-2.0.0"
    (workspace / "manifest.json").write_text(json.dumps(manifest))
    approve_path = workspace / "manifest.json"
    manifest["approval"] = {"status": "APPROVED", "synthetic_demo_only": True}
    approve_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="feature schema"):
        load_artifact(workspace, sha256_file(approve_path))


def test_denied_artifact_moves_to_staging(workspace):
    staged = deny(workspace, "reviewer", "metrics below target", models_root=workspace.parent)
    assert staged.parent.name == "staging" and not workspace.exists()
    assert json.loads((staged / "manifest.json").read_text())["approval"]["status"] == "DENIED"


# ---------------------------------------------------------------- SageMaker cost gate (no AWS)


def _spec(**kw):
    return build_spec(
        image_uri="123.dkr.ecr.us-east-1.amazonaws.com/fraud:abc",
        role_arn="arn:x",
        output_s3="s3://b/models",
        git_commit="abcdef1234",
        hours=0.5,
        now=datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
        **kw,
    )


def test_job_spec_name_cost_and_runtime_cap():
    spec = _spec()
    assert spec.job_name == "fraud-train-20261029-120000-abcdef12"
    assert spec.max_runtime_s == 2700
    assert spec.estimated_cost_usd == pytest.approx(0.23 * 0.75)
    req = training_job_request(spec, "owner")
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 2700
    assert req["EnableManagedSpotTraining"] is False
    assert {"Key": "ApprovedBy", "Value": "owner"} in req["Tags"]


def test_unknown_instance_type_rejected():
    with pytest.raises(ValueError):
        _spec(instance_type="ml.p3.2xlarge")


@pytest.mark.parametrize("budget,approver", [(None, None), (0.01, "owner"), (1.0, None)])
def test_cost_gate_blocks_without_sufficient_named_approval(budget, approver):
    with pytest.raises(CostNotApprovedError):
        check_approval(_spec(), budget, approver)


def test_submit_uses_client_only_after_approval():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def create_training_job(self, **kw):
            self.calls.append(kw)
            return {"TrainingJobArn": "arn:aws:sagemaker:job/x"}

    spec = _spec()
    check_approval(spec, 1.0, "owner")
    client = FakeClient()
    assert submit(spec, "owner", client)["TrainingJobArn"].endswith("job/x")
    assert client.calls[0]["TrainingJobName"] == spec.job_name
