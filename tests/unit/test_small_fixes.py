"""Regressions from the CodeRabbit review: small behaviors with no other natural home."""

from __future__ import annotations

import importlib.util
import os
import re
import stat
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.demo.app import FAIL_OPEN
from src.demo.app import create_app as create_demo

ROOT = Path(__file__).resolve().parents[2]


class _Tokens:
    def get(self) -> str:
        return "token"


@pytest.mark.parametrize("status", [200, 502])
def test_demo_fails_open_on_a_non_json_response(status):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, text="<html>bad gateway"))
    with TestClient(create_demo("http://api", _Tokens(), transport, 1.0)) as demo:
        r = demo.post(
            "/checkout",
            json={"amount_minor": 1, "device_id": "d", "payment_method_token": "t"},
        )
    assert r.status_code == 200
    assert r.json()["payment_decision"] == FAIL_OPEN
    assert r.json()["error"] == "JSONDecodeError"


def test_dashboard_cdn_scripts_are_pinned_by_hash():
    html = (ROOT / "src/dashboard/static/index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script[^>]*https://[^>]*>", html)
    assert len(scripts) == 4
    for tag in scripts:
        assert 'integrity="sha384-' in tag and 'crossorigin="anonymous"' in tag


def _dev_env():
    spec = importlib.util.spec_from_file_location("dev_env", ROOT / "scripts/dev_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_generated_credentials_are_owner_only_whatever_the_umask(tmp_path):
    dev_env = _dev_env()
    old = os.umask(0o022)
    try:
        env = tmp_path / ".env"
        env.write_text("KEEP=1\n")
        env.chmod(0o644)  # An existing, too-open file gets fixed too
        dev_env.ensure_env(env)
        dev_env.ensure_dev_users(tmp_path / "secrets")
    finally:
        os.umask(old)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "secrets/dev-users.txt").stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_containers_can_read_dev_users_under_a_strict_umask(tmp_path):
    dev_env = _dev_env()
    old = os.umask(0o077)
    try:
        dev_env.ensure_dev_users(tmp_path / "secrets")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "secrets").stat().st_mode) == 0o755
    assert stat.S_IMODE((tmp_path / "secrets/dev-users.json").stat().st_mode) == 0o644


def test_dev_env_keeps_existing_values(tmp_path):
    dev_env = _dev_env()
    env = tmp_path / ".env"
    env.write_text("POSTGRES_PASSWORD=existing\n", encoding="utf-8")
    dev_env.ensure_env(env)
    assert "POSTGRES_PASSWORD=existing\n" in env.read_text(encoding="utf-8")
