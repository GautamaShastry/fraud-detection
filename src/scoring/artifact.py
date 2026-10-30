"""Load a promoted model artifact with integrity and compatibility checks (FR-06, T-306).

Artifact directory layout (written by ml/train.py):

    manifest.json        model/policy versions, feature schema, file digests, provenance
    primary.joblib       primary classifier
    fallback.joblib      fallback classifier (no velocity features)

Trust chain: the deployment pins the SHA-256 of manifest.json; the manifest pins the SHA-256
of each model file. Model files are verified *before* deserialization, because joblib files
are pickles and must never be loaded from an unverified source.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from src.scoring.compiled import CompiledHGB
from src.scoring.features import FALLBACK_FEATURE_NAMES, FEATURE_NAMES, FEATURE_SCHEMA
from src.scoring.model import EstimatorScorer, ScoringBundle, ScoringPolicy, SignalRule

logger = logging.getLogger(__name__)


class ArtifactError(Exception):
    """The artifact is untrusted, corrupt or incompatible; the service must not become ready."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedArtifact:
    bundle: ScoringBundle
    manifest: dict[str, Any]


def _verified_load(directory: Path, entry: dict[str, Any]) -> Any:
    path = directory / entry["file"]
    if path.parent.resolve() != directory.resolve():
        raise ArtifactError("Artifact file must live in the artifact directory")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, entry["sha256"]):
        logger.critical(
            "SECURITY: model file digest mismatch for %s; refusing to load", entry["file"]
        )
        raise ArtifactError(f"Digest mismatch for {entry['file']}")
    return joblib.load(path)


def load_artifact(directory: Path, expected_manifest_sha256: str) -> LoadedArtifact:
    manifest_path = directory / "manifest.json"
    actual = sha256_file(manifest_path)
    if not hmac.compare_digest(actual, expected_manifest_sha256.lower()):
        logger.critical("SECURITY: manifest digest mismatch; refusing to load model")
        raise ArtifactError("Manifest digest does not match the pinned value")
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise ArtifactError(
            f"Artifact feature schema {manifest.get('feature_schema')} != service {FEATURE_SCHEMA}"
        )
    built_with = manifest.get("library_versions", {}).get("scikit-learn")
    if built_with != sklearn.__version__:
        raise ArtifactError(
            f"Artifact built with scikit-learn {built_with}, runtime {sklearn.__version__}"
        )
    if manifest.get("approval", {}).get("status") != "APPROVED":
        raise ArtifactError("Artifact has not been approved for promotion")
    if manifest.get("data_source") == "SYNTHETIC" and not manifest.get("approval", {}).get(
        "synthetic_demo_only"
    ):
        raise ArtifactError("Synthetic-only artifact lacks an explicit demo-only approval")

    primary_entry, fallback_entry = manifest["primary"], manifest.get("fallback")
    if tuple(primary_entry["feature_names"]) != FEATURE_NAMES:
        raise ArtifactError("Primary model feature order does not match the service schema")
    primary = _scorer(
        manifest["model_version"], FEATURE_NAMES, _verified_load(directory, primary_entry)
    )
    fallback = None
    if fallback_entry is not None:
        if tuple(fallback_entry["feature_names"]) != FALLBACK_FEATURE_NAMES:
            raise ArtifactError("Fallback model feature order does not match the service schema")
        fallback = _scorer(
            fallback_entry["model_version"],
            FALLBACK_FEATURE_NAMES,
            _verified_load(directory, fallback_entry),
        )

    bundle = ScoringBundle(
        primary=primary,
        primary_policy=ScoringPolicy.from_json(manifest["policy"]),
        fallback=fallback,
        fallback_policy=(
            ScoringPolicy.from_json(manifest["fallback_policy"]) if fallback is not None else None
        ),
        signal_rules=tuple(SignalRule(**r) for r in manifest.get("signal_rules", [])),
    )
    if bundle.fallback_policy is not None and bundle.fallback_policy.decline_threshold is not None:
        raise ArtifactError("Fallback policy may not recommend decline (I-08)")
    warmup(bundle)
    return LoadedArtifact(bundle, manifest)


def _scorer(version: str, names: tuple[str, ...], estimator: Any) -> EstimatorScorer:
    compiled = None
    if type(estimator).__name__ == "HistGradientBoostingClassifier":
        compiled = CompiledHGB.from_estimator(estimator)
        # Parity gate: random rows with injected missing values must match predict_proba.
        rng = np.random.default_rng(0)
        rows = rng.normal(0, 3, size=(256, len(names)))
        rows[rng.random(rows.shape) < 0.2] = np.nan
        try:
            compiled.verify(estimator, rows)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
    return EstimatorScorer(version, names, estimator, compiled)


def warmup(bundle: ScoringBundle) -> None:
    """Score a synthetic feature vector through each model; fail loudly on bad output."""
    sample: dict[str, float | None] = {name: 0.0 for name in FEATURE_NAMES}
    sample.update({"amount_log": math.log1p(5000), "hour_cos": 1.0, "account_age_days": None})
    for scorer, policy in (
        (bundle.primary, bundle.primary_policy),
        (bundle.fallback, bundle.fallback_policy),
    ):
        if scorer is None or policy is None:
            continue
        output = scorer.predict(sample)
        if not 0.0 <= output <= 1.0:
            raise ArtifactError(f"Warmup produced invalid output {output}")
        policy.risk_score(output)
