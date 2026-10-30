"""Minimal OIDC provider for LOCAL DEVELOPMENT AND TESTS ONLY (I-04 local half).

Implements discovery, JWKS, an HTML login form, the authorization-code flow with mandatory PKCE
(S256) and RS256 tokens. Users come from a JSON file; each has a tenant and dashboard roles that
appear as `tenant_id` and `roles` claims. It must never be exposed beyond localhost: replace it
with a managed OIDC issuer (Keycloak, Cognito, ...) by configuration for any shared deployment.

    uvicorn src.devauth.oidc:create_app --factory --port 8081
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

CODE_TTL_S = 60
TOKEN_TTL_S = 900


@dataclass(frozen=True)
class DevUser:
    username: str
    password_sha256: str
    tenant_id: str
    roles: tuple[str, ...]
    sub: str


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def load_users(path: Path) -> dict[str, DevUser]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        u["username"]: DevUser(
            u["username"],
            u["password_sha256"],
            u["tenant_id"],
            tuple(u["roles"]),
            u.get("sub", u["username"]),
        )
        for u in raw["users"]
    }


def create_app(
    issuer: str | None = None,
    users_path: Path | None = None,
    clients: dict[str, list[str]] | None = None,
) -> FastAPI:
    issuer = str(issuer or os.environ.get("DEV_OIDC_ISSUER") or "http://localhost:8081").rstrip("/")
    users = load_users(
        users_path or Path(os.environ.get("DEV_OIDC_USERS", "infra/local/dev-users.json"))
    )
    # client_id -> allowed redirect URIs (exact match)
    clients = clients or json.loads(
        os.environ.get(
            "DEV_OIDC_CLIENTS",
            '{"fraud-dashboard": ["http://localhost:8000/dashboard/", '
            '"http://127.0.0.1:8000/dashboard/"]}',
        )
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": "dev-oidc-1", "alg": "RS256", "use": "sig"})
    codes: dict[str, dict[str, Any]] = {}

    app = FastAPI(title="Local dev OIDC (test only)", openapi_url=None, docs_url=None)

    @app.get("/.well-known/openid-configuration")
    async def discovery() -> dict[str, Any]:
        return {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "jwks_uri": f"{issuer}/jwks",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "subject_types_supported": ["public"],
            "scopes_supported": ["openid", "profile"],
            "token_endpoint_auth_methods_supported": ["none"],
        }

    @app.get("/jwks")
    async def jwks() -> dict[str, Any]:
        return {"keys": [jwk]}

    def _check_client(client_id: str, redirect_uri: str) -> None:
        if redirect_uri not in clients.get(client_id, []):
            raise HTTPException(400, "unknown client or redirect_uri")

    @app.get("/authorize", response_class=HTMLResponse)
    async def authorize(request: Request) -> HTMLResponse:
        q = request.query_params
        _check_client(q.get("client_id", ""), q.get("redirect_uri", ""))
        if (
            q.get("response_type") != "code"
            or q.get("code_challenge_method") != "S256"
            or not q.get("code_challenge")
        ):
            raise HTTPException(400, "authorization code with PKCE S256 is required")
        hidden = "".join(
            f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">'
            for k, v in q.items()
        )
        return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><title>Dev sign-in</title>
<body style="font-family:system-ui;max-width:360px;margin:64px auto">
<h2>Local dev sign-in</h2><p style="color:#a00">Test-only identity provider.</p>
<form method="post" action="/authorize">{hidden}
<label>Username <input name="username" autofocus></label><br><br>
<label>Password <input name="password" type="password"></label><br><br>
<button>Sign in</button></form></body>""")

    @app.post("/authorize")
    async def authorize_submit(
        request: Request, username: str = Form(...), password: str = Form(...)
    ) -> RedirectResponse:
        form = dict((await request.form()).items())
        client_id, redirect_uri = str(form.get("client_id", "")), str(form.get("redirect_uri", ""))
        _check_client(client_id, redirect_uri)
        user = users.get(username)
        digest = hashlib.sha256(password.encode()).hexdigest()
        if user is None or not hmac.compare_digest(digest, user.password_sha256):
            raise HTTPException(401, "invalid credentials")
        code = secrets.token_urlsafe(24)
        codes[code] = {
            "user": user,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "challenge": form.get("code_challenge"),
            "nonce": form.get("nonce"),
            "exp": time.time() + CODE_TTL_S,
        }
        params = {"code": code, "state": form.get("state", "")}
        return RedirectResponse(f"{redirect_uri}?{urllib.parse.urlencode(params)}", 302)

    @app.post("/token")
    async def token(
        grant_type: str = Form(...),
        code: str = Form(...),
        redirect_uri: str = Form(...),
        client_id: str = Form(...),
        code_verifier: str = Form(...),
    ) -> JSONResponse:
        entry = codes.pop(code, None)  # single use
        if grant_type != "authorization_code" or entry is None or entry["exp"] < time.time():
            raise HTTPException(400, "invalid_grant")
        if entry["client_id"] != client_id or entry["redirect_uri"] != redirect_uri:
            raise HTTPException(400, "invalid_grant")
        challenge = _b64(hashlib.sha256(code_verifier.encode()).digest())
        if not hmac.compare_digest(challenge, str(entry["challenge"])):
            raise HTTPException(400, "invalid_grant (PKCE)")
        user: DevUser = entry["user"]
        now = int(time.time())
        common = {
            "iss": issuer,
            "sub": user.sub,
            "iat": now,
            "exp": now + TOKEN_TTL_S,
            "preferred_username": user.username,
        }
        access = jwt.encode(
            {
                **common,
                "aud": client_id,
                "tenant_id": user.tenant_id,
                "roles": list(user.roles),
                "azp": client_id,
                "jti": secrets.token_hex(8),
            },
            key,
            algorithm="RS256",
            headers={"kid": "dev-oidc-1"},
        )
        id_token = jwt.encode(
            {**common, "aud": client_id, "nonce": entry["nonce"]},
            key,
            algorithm="RS256",
            headers={"kid": "dev-oidc-1"},
        )
        return JSONResponse(
            {
                "access_token": access,
                "id_token": id_token,
                "token_type": "Bearer",
                "expires_in": TOKEN_TTL_S,
            },
            headers={"Cache-Control": "no-store"},
        )

    return app
