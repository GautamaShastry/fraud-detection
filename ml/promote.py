"""Manual model promotion gate (FR-13, T-306).

    python -m ml.promote models/<version> --approver "Name" [--synthetic-demo-only] [--deny REASON]

Approval verifies every file digest, shows metrics and provenance, requires a named approver
and an interactive "yes" (or --yes), then records the decision in the manifest and the
append-only log models/promotion_log.jsonl. Approved artifacts are copied to models/current
and the pinned manifest digest is written to infra/local/.env for the local stack.
Denied artifacts are moved to models/staging/.

Synthetic-only artifacts can only be approved with --synthetic-demo-only, which marks them as
demo models that make no real-world performance claim.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.scoring.artifact import ArtifactError, load_artifact, sha256_file

LOG = "promotion_log.jsonl"


def summarize(manifest: dict[str, Any]) -> str:
    m = manifest["metrics"]
    lines = [
        f"Model version:   {manifest['model_version']}",
        f"Feature schema:  {manifest['feature_schema']}",
        f"Data source:     {manifest['data_source']}",
        f"Trained:         {manifest['training']['timestamp']} "
        f"@ {manifest['training']['git_commit'][:12]}",
        f"Fallback:        {'yes' if manifest.get('fallback') else 'no'}",
    ]
    prefix = "Synthetic: " if manifest["data_source"] == "SYNTHETIC" else ""
    for name in ("primary", "fallback"):
        if name in m:
            rep = m[name]["synthetic_ground_truth"]
            op = next(p for p in rep["operating_points"] if p["target_fpr"] == 0.005)
            lines.append(
                f"{prefix}{name}: test ROC-AUC {rep['test']['roc_auc']:.4f}, recall "
                f"{op['test']['recall']:.1%} at val-selected 0.5% FPR "
                f"(test FPR {op['test']['fpr']:.3%})"
            )
    return "\n".join(lines)


def _verify_digests(directory: Path, manifest: dict[str, Any]) -> None:
    for key in ("primary", "fallback"):
        entry = manifest.get(key)
        if entry and sha256_file(directory / entry["file"]) != entry["sha256"]:
            raise ArtifactError(f"{entry['file']} does not match its recorded digest")


def _update_env(path: Path, values: dict[str, str]) -> None:
    """Set keys in a dotenv file, preserving every other line."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in lines if ln.split("=", 1)[0].strip() not in values]
    kept += [f"{k}={v}" for k, v in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _log(models_root: Path, record: dict[str, Any]) -> None:
    with (models_root / LOG).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def approve(
    directory: Path,
    approver: str,
    *,
    synthetic_demo_only: bool,
    models_root: Path,
    env_file: Path | None,
) -> str:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_digests(directory, manifest)
    if manifest["data_source"] == "SYNTHETIC" and not synthetic_demo_only:
        raise ArtifactError("Synthetic-only model: approval requires --synthetic-demo-only")
    now = datetime.now(UTC).isoformat()
    manifest["approval"] = {
        "status": "APPROVED",
        "approver": approver,
        "approved_at": now,
        "synthetic_demo_only": synthetic_demo_only,
    }
    approved = json.dumps(manifest, indent=2).encode("utf-8")
    digest = hashlib.sha256(approved).hexdigest()

    # Validate a private copy: the artifact says APPROVED only after every check has passed.
    staging = Path(tempfile.mkdtemp(prefix=".promote-", dir=models_root))
    try:
        candidate = staging / "artifact"
        shutil.copytree(directory, candidate)
        (candidate / "manifest.json").write_bytes(approved)
        load_artifact(candidate, digest)  # the exact checks the API performs at startup

        # The log goes first: if it can't be written, nothing has been approved.
        _log(
            models_root,
            {
                "decision": "APPROVED",
                "model_version": manifest["model_version"],
                "approver": approver,
                "at": now,
                "manifest_sha256": digest,
                "synthetic_demo_only": synthetic_demo_only,
            },
        )
        temp_manifest = staging / "manifest.json"
        temp_manifest.write_bytes(approved)
        os.replace(temp_manifest, manifest_path)
        current = models_root / "current"
        if current.exists():
            current.rename(staging / "previous")
        candidate.rename(current)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if env_file is not None:
        _update_env(env_file, {"FRAUD_SCORER_MODE": "model", "FRAUD_MODEL_MANIFEST_SHA256": digest})
    return digest


def deny(directory: Path, reviewer: str, reason: str, *, models_root: Path) -> Path:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    now = datetime.now(UTC).isoformat()
    manifest["approval"] = {
        "status": "DENIED",
        "reviewer": reviewer,
        "denied_at": now,
        "reason": reason,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging = models_root / "staging" / directory.name
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), staging)
    _log(
        models_root,
        {
            "decision": "DENIED",
            "model_version": manifest["model_version"],
            "reviewer": reviewer,
            "at": now,
            "reason": reason,
        },
    )
    return staging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--approver", default=None)
    parser.add_argument("--synthetic-demo-only", action="store_true")
    parser.add_argument("--deny", metavar="REASON")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    parser.add_argument("--env-file", type=Path, default=Path("infra/local/.env"))
    args = parser.parse_args()

    models_root = args.artifact.parent
    manifest = json.loads((args.artifact / "manifest.json").read_text(encoding="utf-8"))
    print(summarize(manifest))
    print(f"Report: {args.artifact / 'report.md'}\n")
    person = args.approver or getpass.getuser()
    if args.deny:
        print(f"Denied -> {deny(args.artifact, person, args.deny, models_root=models_root)}")
        return
    if not args.approver:
        sys.exit("--approver is required to approve a promotion")
    if not args.yes and input(f"Approve promotion as {person}? Type 'yes': ").strip() != "yes":
        sys.exit("Promotion not approved; nothing changed.")
    digest = approve(
        args.artifact,
        person,
        synthetic_demo_only=args.synthetic_demo_only,
        models_root=models_root,
        env_file=args.env_file,
    )
    print(f"Approved. models/current pinned manifest sha256={digest}")


if __name__ == "__main__":
    main()
