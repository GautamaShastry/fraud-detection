"""T-004: the contract is valid, its examples conform, and the implementation agrees with it."""

import pytest
from openapi_spec_validator import validate
from pydantic import ValidationError

from src.api.schemas import ScoreRequest
from tests.conftest import valid_score_request
from tests.support.contract import _resolve, schema_validator, spec


def test_contract_is_valid_openapi_31():
    validate(spec())


def _example_pairs():
    """Yield (id, schema pointer target, example value) for every example in the contract."""
    for path, ops in spec()["paths"].items():
        for method, op in ops.items():
            containers = []
            if "requestBody" in op:
                containers.append(op["requestBody"])
            containers += [_resolve(r, "")[0] for r in op["responses"].values()]
            for container in containers:
                for media in container.get("content", {}).values():
                    for name, ex in media.get("examples", {}).items():
                        value = _resolve(ex, "")[0]["value"]
                        yield f"{method} {path} {name}", media["schema"], value


EXAMPLES = list(_example_pairs())


def test_contract_has_examples():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize(
    "schema_ref,value", [e[1:] for e in EXAMPLES], ids=[e[0] for e in EXAMPLES]
)
def test_examples_conform_to_schema(schema_ref, value):
    name = schema_ref["$ref"].rsplit("/", 1)[-1]
    errors = list(schema_validator(name).iter_errors(value))
    assert not errors, [e.message for e in errors]


VALID_REQUESTS = [
    valid_score_request(),
    valid_score_request(amount_minor=0),
    valid_score_request(occurred_at="2026-10-28T12:00:00.123456+00:00"),
    valid_score_request(ip_address="2001:db8::1", device_id="dev-1", customer_id="c:1"),
    valid_score_request(account_created_at="2025-01-01T00:00:00Z"),
]

INVALID_REQUESTS = {
    "tenant_id asserted": valid_score_request(tenant_id="tenant-b"),
    "unknown field": valid_score_request(card_number="4111111111111111"),
    "missing transaction_id": valid_score_request(transaction_id=None),
    "missing occurred_at": valid_score_request(occurred_at=None),
    "amount as string": valid_score_request(amount_minor="4599"),
    "amount as float": valid_score_request(amount_minor=45.99),
    "amount as bool": valid_score_request(amount_minor=True),
    "negative amount": valid_score_request(amount_minor=-1),
    "amount too large": valid_score_request(amount_minor=10**13),
    "lowercase currency": valid_score_request(currency="usd"),
    "non-UTC offset": valid_score_request(occurred_at="2026-10-28T12:00:00+02:00"),
    "naive timestamp": valid_score_request(occurred_at="2026-10-28T12:00:00"),
    "date only": valid_score_request(occurred_at="2026-10-28"),
    "bad id chars": valid_score_request(payment_attempt_id="pa 1/../"),
    "empty id": valid_score_request(transaction_id=""),
    "id too long": valid_score_request(transaction_id="x" * 129),
    "short idempotency key": valid_score_request(idempotency_key="short"),
    "bad ip": valid_score_request(ip_address="999.1.1.1"),
}


@pytest.mark.parametrize("body", VALID_REQUESTS)
def test_valid_requests_accepted_by_contract_and_model(body):
    assert not list(schema_validator("ScoreRequest").iter_errors(body))
    ScoreRequest.model_validate(body)


@pytest.mark.parametrize("body", INVALID_REQUESTS.values(), ids=INVALID_REQUESTS.keys())
def test_invalid_requests_rejected_by_model(body):
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(body)


# The bad IP is rejected by the model; plain JSON Schema treats `format` as an annotation.
@pytest.mark.parametrize(
    "body",
    [v for k, v in INVALID_REQUESTS.items() if k != "bad ip"],
    ids=[k for k in INVALID_REQUESTS if k != "bad ip"],
)
def test_invalid_requests_rejected_by_contract(body):
    assert list(schema_validator("ScoreRequest").iter_errors(body))


def test_implemented_routes_are_in_contract(client):
    contract_paths = spec()["paths"]
    for route in client.app.routes:
        for method in getattr(route, "methods", set()) - {"HEAD"}:
            assert route.path in contract_paths, route.path
            assert method.lower() in contract_paths[route.path], f"{method} {route.path}"
