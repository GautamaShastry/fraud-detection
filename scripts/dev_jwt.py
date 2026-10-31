"""Local test-only OAuth token issuer (I-04 local half). Never use these keys outside dev/CI.

python scripts/dev_jwt.py init --out infra/local/secrets
python scripts/dev_jwt.py mint --keys infra/local/secrets --client-id demo-merchant
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

DEFAULT_ISSUER = "https://local-dev-issuer.invalid"
DEFAULT_AUDIENCE = "fraud-api"
KID = "local-dev-1"


def init(out: Path, force: bool) -> None:
    private_path = out / "private.pem"
    if private_path.exists() and not force:
        print(f"Keys already exist in {out}; use --force to rotate")
        return
    out.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public_jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    (out / "jwks.json").write_text(json.dumps({"keys": [public_jwk]}, indent=2))
    print(f"Wrote {private_path} and {out / 'jwks.json'}")


def mint(keys: Path, client_id: str, scope: str, ttl_s: int, issuer: str, audience: str) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": client_id,
        "client_id": client_id,
        "scope": scope,
        "iat": now,
        "exp": now + ttl_s,
        "jti": str(uuid.uuid4()),
    }
    private_key = (keys / "private.pem").read_bytes()
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": KID})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--out", type=Path, required=True)
    p_init.add_argument("--force", action="store_true")
    p_mint = sub.add_parser("mint")
    p_mint.add_argument("--keys", type=Path, required=True)
    p_mint.add_argument("--client-id", required=True)
    p_mint.add_argument("--scope", default="fraud:score fraud:outcomes")
    p_mint.add_argument("--ttl", type=int, default=600)
    p_mint.add_argument("--issuer", default=DEFAULT_ISSUER)
    p_mint.add_argument("--audience", default=DEFAULT_AUDIENCE)
    args = parser.parse_args()

    if args.command == "init":
        init(args.out, args.force)
    else:
        print(mint(args.keys, args.client_id, args.scope, args.ttl, args.issuer, args.audience))


if __name__ == "__main__":
    main()
