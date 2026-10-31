"""Label projection fold (I-09 provisional): order-independent and idempotent."""

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from src.storage.outcomes import fold

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TYPES = [
    "AUTHORIZATION_APPROVED",
    "AUTHORIZATION_DECLINED",
    "CHARGEBACK_REPORTED",
    "CONFIRMED_FRAUD",
    "CONFIRMED_LEGITIMATE",
    "LABEL_RETRACTED",
]
SOURCES = ["MARKETPLACE", "PROCESSOR", "ISSUER", "MANUAL_REVIEW"]

events = st.lists(
    st.builds(
        lambda i, t, s, m: {
            "event_id": f"e{i}",
            "event_type": t,
            "source": s,
            "occurred_at": T0 + timedelta(minutes=m),
            "received_at": T0,
        },
        st.integers(0, 10_000),
        st.sampled_from(TYPES),
        st.sampled_from(SOURCES),
        st.integers(0, 1000),
    ),
    max_size=12,
    unique_by=lambda e: e["event_id"],
)


@given(events, st.randoms())
def test_fold_is_order_independent(evs, rnd):
    shuffled = list(evs)
    rnd.shuffle(shuffled)
    assert fold(evs) == fold(shuffled)


def _e(i, kind, source, minute):
    return {
        "event_id": f"e{i}",
        "event_type": kind,
        "source": source,
        "occurred_at": T0 + timedelta(minutes=minute),
        "received_at": T0,
    }


def test_approval_is_not_a_legitimacy_label():
    p = fold([_e(1, "AUTHORIZATION_APPROVED", "PROCESSOR", 0)])
    assert p.authorization_status == "APPROVED" and p.label is None


def test_more_authoritative_source_wins_even_if_earlier():
    p = fold(
        [_e(1, "CONFIRMED_FRAUD", "ISSUER", 0), _e(2, "CONFIRMED_LEGITIMATE", "MARKETPLACE", 5)]
    )
    assert (p.label, p.label_source) == ("CONFIRMED_FRAUD", "ISSUER")


def test_same_source_latest_wins():
    p = fold([_e(1, "CONFIRMED_FRAUD", "ISSUER", 0), _e(2, "CONFIRMED_LEGITIMATE", "ISSUER", 5)])
    assert p.label == "CONFIRMED_LEGITIMATE"


def test_retraction_requires_equal_or_higher_authority():
    weak = fold([_e(1, "CONFIRMED_FRAUD", "ISSUER", 0), _e(2, "LABEL_RETRACTED", "MARKETPLACE", 5)])
    assert weak.label == "CONFIRMED_FRAUD"
    strong = fold([_e(1, "CONFIRMED_FRAUD", "ISSUER", 0), _e(2, "LABEL_RETRACTED", "ISSUER", 5)])
    assert strong.label is None


def test_chargeback_alone_is_not_a_fraud_label():
    p = fold([_e(1, "CHARGEBACK_REPORTED", "PROCESSOR", 0)])
    assert p.chargeback_reported_at == T0 and p.label is None
