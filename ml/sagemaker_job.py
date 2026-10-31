"""On-demand SageMaker training jobs with a cost approval gate (FR-13, T-305).

    # Dry run (default): print the job spec and cost estimate; no AWS call is made.
    python -m ml.sagemaker_job --image <ecr-uri> --role <arn> --output s3://bucket/models \
        --git-commit $(git rev-parse HEAD) --hours 0.5

    # Submit: only with an explicit approved budget that covers the estimate.
    python -m ml.sagemaker_job ... --approve-cost 0.50 --approver "Name"

The training image is this repository's container (same pinned scikit-learn as the API, so the
artifact loads in the service). Every approval or refusal is appended to
evidence/cost_approvals.jsonl. Paid training must never run without an owner-approved budget.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# On-demand USD/hour, us-east-1, at authoring time. Verify current pricing before approving.
INSTANCE_PRICES = {"ml.m5.xlarge": 0.23, "ml.m5.2xlarge": 0.461}
PRICING_NOTE = "Static us-east-1 on-demand list price at authoring time; verify before approval."
APPROVAL_LOG = Path("evidence/cost_approvals.jsonl")


@dataclass(frozen=True)
class JobSpec:
    job_name: str
    image_uri: str
    role_arn: str
    instance_type: str
    output_s3: str
    input_s3: str | None
    hyperparameters: dict[str, str]
    git_commit: str
    estimated_hours: float
    max_runtime_s: int
    estimated_cost_usd: float


def job_name(git_commit: str, now: datetime) -> str:
    short = re.sub(r"[^0-9a-f]", "", git_commit.lower())[:8] or "nogit"
    return f"fraud-train-{now:%Y%m%d-%H%M%S}-{short}"


def build_spec(
    *,
    image_uri: str,
    role_arn: str,
    output_s3: str,
    git_commit: str,
    hours: float,
    instance_type: str = "ml.m5.xlarge",
    input_s3: str | None = None,
    seed: int = 7,
    now: datetime | None = None,
) -> JobSpec:
    if instance_type not in INSTANCE_PRICES:
        raise ValueError(f"Instance type must be one of {sorted(INSTANCE_PRICES)}")
    if hours <= 0:
        raise ValueError("Estimated duration must be positive")
    now = now or datetime.now(UTC)
    return JobSpec(
        job_name=job_name(git_commit, now),
        image_uri=image_uri,
        role_arn=role_arn,
        instance_type=instance_type,
        output_s3=output_s3,
        input_s3=input_s3,
        hyperparameters={"seed": str(seed)},
        git_commit=git_commit,
        estimated_hours=hours,
        max_runtime_s=int(math.ceil(hours * 1.5 * 3600)),  # 1.5x buffer; also the cost cap
        # Billing is per second of runtime; the max-runtime cap bounds the worst case.
        estimated_cost_usd=round(INSTANCE_PRICES[instance_type] * hours * 1.5, 4),
    )


def training_job_request(spec: JobSpec, approver: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "TrainingJobName": spec.job_name,
        "AlgorithmSpecification": {
            "TrainingImage": spec.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "ml.sagemaker_entry"],
        },
        "RoleArn": spec.role_arn,
        "OutputDataConfig": {"S3OutputPath": spec.output_s3},
        "ResourceConfig": {
            "InstanceType": spec.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": spec.max_runtime_s},
        "HyperParameters": spec.hyperparameters,
        "EnableManagedSpotTraining": False,
        "Tags": [
            {"Key": "Project", "Value": "FraudDetection"},
            {"Key": "GitCommit", "Value": spec.git_commit[:40]},
            {"Key": "ApprovedBy", "Value": approver[:128]},
        ],
    }
    if spec.input_s3:
        request["InputDataConfig"] = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": spec.input_s3,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ]
    return request


class CostNotApprovedError(Exception):
    pass


def check_approval(spec: JobSpec, approved_usd: float | None, approver: str | None) -> None:
    if approved_usd is None or not approver:
        raise CostNotApprovedError("No cost approval given; dry run only")
    if not math.isfinite(approved_usd):  # NaN compares false to everything, so it would pass
        raise CostNotApprovedError("Approved budget must be a finite amount")
    if approved_usd < spec.estimated_cost_usd:
        raise CostNotApprovedError(
            f"Approved budget ${approved_usd:.2f} is below the estimate "
            f"${spec.estimated_cost_usd:.2f}"
        )


def log_decision(
    spec: JobSpec, decision: str, approver: str | None, detail: str, path: Path = APPROVAL_LOG
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "decision": decision,
                    "approver": approver,
                    "detail": detail,
                    **asdict(spec),
                }
            )
            + "\n"
        )


def submit(spec: JobSpec, approver: str, client: Any = None) -> dict[str, Any]:
    if client is None:
        import boto3

        client = boto3.client("sagemaker")
    response: dict[str, Any] = client.create_training_job(**training_job_request(spec, approver))
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--approve-cost", type=float)
    parser.add_argument("--approver")
    args = parser.parse_args()

    spec = build_spec(
        image_uri=args.image,
        role_arn=args.role,
        output_s3=args.output,
        git_commit=args.git_commit,
        hours=args.hours,
        instance_type=args.instance_type,
        input_s3=args.input,
    )
    print(json.dumps({**asdict(spec), "pricing_note": PRICING_NOTE}, indent=2))
    try:
        check_approval(spec, args.approve_cost, args.approver)
    except CostNotApprovedError as exc:
        log_decision(spec, "NOT_SUBMITTED", args.approver, str(exc))
        print(f"Not submitted: {exc}")
        sys.exit(0 if args.approve_cost is None else 2)
    response = submit(spec, args.approver)
    log_decision(spec, "SUBMITTED", args.approver, response.get("TrainingJobArn", ""))
    print(f"Submitted {response.get('TrainingJobArn')}")


if __name__ == "__main__":
    main()
