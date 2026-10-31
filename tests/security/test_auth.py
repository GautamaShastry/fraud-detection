"""FR-01 negative tests: every invalid credential is rejected and auth never fails open."""

import base64
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.conftest import valid_score_request
from tests.support.contract import assert_response_matches

SCORE = "/api/v1/fraud/score"


def _post(client, token=None, body=None, raw_header=None):
    headers = {}
    if raw_header is not None:
        headers["Authorization"] = raw_header
    elif token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(SCORE, json=body or valid_score_request(), headers=headers)


def _assert_401(response):
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHENTICATED"
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert_response_matches("POST", SCORE, response)


def test_valid_token_passes_authentication(client, make_token):
    # Authenticated and valid: reaches the scorer, which is explicitly unavailable in Phase 0.
    response = _post(client, make_token())
    assert response.status_code == 503


def test_es256_token_accepted(client, make_token, ec_key):
    response = _post(client, make_token(key=ec_key, kid="ec-1", algorithm="ES256"))
    assert response.status_code == 503


def test_missing_token(client):
    _assert_401(_post(client))


@pytest.mark.parametrize("header", ["Basic abc", "Bearer", "Bearer   ", "token abc"])
def test_malformed_authorization_header(client, header):
    _assert_401(_post(client, raw_header=header))


def test_garbage_token(client):
    _assert_401(_post(client, "not-a-jwt"))


def test_expired_token(client, make_token):
    past = int(time.time()) - 3600
    _assert_401(_post(client, make_token(iat=past - 300, exp=past)))


def test_not_yet_valid_token(client, make_token):
    _assert_401(_post(client, make_token(nbf=int(time.time()) + 3600)))


def test_wrong_issuer(client, make_token):
    _assert_401(_post(client, make_token(iss="https://evil.invalid")))


def test_wrong_audience(client, make_token):
    _assert_401(_post(client, make_token(aud="some-other-api")))


@pytest.mark.parametrize("claim", ["exp", "iat", "iss", "aud", "sub"])
def test_required_claims(client, make_token, claim):
    _assert_401(_post(client, make_token(**{claim: None})))


def test_signature_from_untrusted_key(client, make_token):
    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _assert_401(_post(client, make_token(key=attacker)))


def test_unknown_kid(client, make_token):
    _assert_401(_post(client, make_token(kid="nope")))


def test_alg_none_rejected(client):
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    now = int(time.time())
    claims = {
        "iss": "https://test-issuer.invalid",
        "aud": "fraud-api",
        "sub": "merchant-a",
        "scope": "fraud:score",
        "iat": now,
        "exp": now + 300,
    }
    token = f"{b64({'alg': 'none', 'kid': 'rsa-1'})}.{b64(claims)}."
    _assert_401(_post(client, token))


def test_hs256_with_public_key_as_secret_rejected(client, rsa_key, make_token):
    # Classic algorithm-confusion attack: sign HS256 using the published RSA public key.
    public_pem = rsa_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    claims = jwt.decode(make_token(), options={"verify_signature": False})
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","kid":"rsa-1"}').rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    import hashlib
    import hmac

    signing_input = header + b"." + payload
    sig = base64.urlsafe_b64encode(
        hmac.new(public_pem, signing_input, hashlib.sha256).digest()
    ).rstrip(b"=")
    _assert_401(_post(client, (signing_input + b"." + sig).decode()))


def test_algorithm_must_match_key_type(client, make_token, ec_key):
    # Valid ES256 signature but presented under the RSA key's kid.
    _assert_401(_post(client, make_token(key=ec_key, kid="rsa-1", algorithm="ES256")))


def test_missing_scope_forbidden(client, make_token):
    response = _post(client, make_token(scope="fraud:outcomes"))
    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert_response_matches("POST", SCORE, response)


def test_scp_list_claim_supported(client, make_token):
    response = _post(client, make_token(scope=None, scp=["fraud:score"]))
    assert response.status_code == 503


def test_unmapped_client_forbidden(client, make_token):
    response = _post(client, make_token(client_id="unknown-merchant"))
    assert response.status_code == 403


def test_payload_cannot_assert_tenant(client, make_token):
    response = _post(client, make_token(), valid_score_request(tenant_id="tenant-b"))
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "tenant_id"


def test_authentication_precedes_body_validation(client):
    # An unauthenticated caller learns nothing about payload validation rules.
    _assert_401(_post(client, body={"junk": True}))


def test_auth_errors_do_not_echo_token(client, make_token):
    token = make_token(aud="other")
    response = _post(client, token)
    assert token not in response.text
