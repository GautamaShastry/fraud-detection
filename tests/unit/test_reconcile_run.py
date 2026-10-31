"""CodeRabbit: totals can cancel out, so reconciliation must match each attempt."""

from perf.reconcile_run import read_client_200s, unmatched_200s


def test_parses_k6_console_lines():
    log = (
        'time="2026-10-31T04:21:32Z" level=info msg="ASSESSED ML pa-r-1-0" source=console\n'
        'time="2026-10-31T04:21:33Z" level=info msg="ASSESSED FALLBACK pa-r-2-1" source=console\n'
        'time="2026-10-31T04:21:33Z" level=error msg="unrelated" source=console\n'
    )
    assert read_client_200s(log) == {"pa-r-1-0": "ML", "pa-r-2-1": "FALLBACK"}


def test_all_matched():
    assert (
        unmatched_200s({"a": "ML", "b": "FALLBACK"}, {"a": "ML", "b": "FALLBACK", "c": "ML"}) == []
    )


def test_one_missing_and_one_extra_cannot_cancel_out():
    # Totals agree (one ML seen, one ML stored), but the success the client saw was never recorded.
    assert unmatched_200s({"a": "ML"}, {"z": "ML"}) == ["a"]


def test_a_different_mode_is_a_mismatch():
    assert unmatched_200s({"a": "ML"}, {"a": "FALLBACK"}) == ["a"]
