"""Validate HTTP responses and payloads against contracts/openapi.yaml."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import httpx
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "openapi.yaml"
_URI = "urn:fraud-api-contract"


@cache
def spec() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT_PATH.read_text())


@cache
def _registry() -> Registry:
    return Registry().with_resource(_URI, Resource(contents=spec(), specification=DRAFT202012))


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def validator_for(pointer: str) -> Draft202012Validator:
    """Validator for the schema at a JSON pointer inside the contract."""
    return Draft202012Validator({"$ref": f"{_URI}#{pointer}"}, registry=_registry())


def schema_validator(name: str) -> Draft202012Validator:
    return validator_for(f"/components/schemas/{name}")


def _resolve(obj: dict[str, Any], pointer: str) -> tuple[dict[str, Any], str]:
    while "$ref" in obj:
        pointer = obj["$ref"].removeprefix("#")
        obj = spec()
        for part in pointer.strip("/").split("/"):
            obj = obj[part.replace("~1", "/").replace("~0", "~")]
    return obj, pointer


def assert_response_matches(method: str, path_template: str, response: httpx.Response) -> None:
    """Assert status, media type and body match the contract for this operation."""
    op_pointer = f"/paths/{_escape(path_template)}/{method.lower()}"
    operation = spec()["paths"][path_template][method.lower()]
    status = str(response.status_code)
    assert (
        status in operation["responses"]
    ), f"{method} {path_template} returned undocumented status {status}"
    resp_obj, resp_pointer = _resolve(
        operation["responses"][status], f"{op_pointer}/responses/{status}"
    )
    media_type = response.headers["content-type"].split(";")[0].strip()
    content = resp_obj.get("content", {})
    assert media_type in content, f"Undocumented media type {media_type} for {status}"
    schema_pointer = f"{resp_pointer}/content/{_escape(media_type)}/schema"
    errors = sorted(validator_for(schema_pointer).iter_errors(response.json()), key=str)
    assert not errors, f"Response body violates contract: {[e.message for e in errors]}"
