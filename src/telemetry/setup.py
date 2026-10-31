"""Structured logging and OpenTelemetry tracing setup.

Logs are JSON with a fixed set of fields. Callers must not log raw identifiers; the formatter
additionally drops any record attribute not on the allow-list, so stray `extra=` values cannot
leak payload data.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

ALLOWED_EXTRA = frozenset({"request_id", "outcome", "tenant_id", "status", "duration_ms"})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ALLOWED_EXTRA:
            if key in record.__dict__:
                out[key] = record.__dict__[key]
        span = trace.get_current_span().get_span_context()
        if span.is_valid:
            out["trace_id"] = format(span.trace_id, "032x")
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            out["traceback"] = self.formatException(record.exc_info)
        return json.dumps(out)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def configure_tracing(service_name: str) -> None:
    """Export spans over OTLP/HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise no-op."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


tracer = trace.get_tracer("fraud-platform")
