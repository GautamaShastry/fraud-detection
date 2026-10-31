-- Durable outcome inbox, transactional outbox, outcome projection, archive ledger
-- (T-201..T-204; I-02 provisional: PostgreSQL inbox/outbox with SKIP LOCKED workers).

CREATE TABLE outcome_events (
    tenant_id          text NOT NULL REFERENCES tenants,
    event_id           text NOT NULL,
    payload_hash       bytea NOT NULL,
    payment_attempt_id text NOT NULL,
    transaction_id     text,
    event_type         text NOT NULL CHECK (event_type IN (
                           'AUTHORIZATION_APPROVED', 'AUTHORIZATION_DECLINED', 'CHARGEBACK_REPORTED',
                           'CONFIRMED_FRAUD', 'CONFIRMED_LEGITIMATE', 'LABEL_RETRACTED')),
    source             text NOT NULL CHECK (source IN ('MARKETPLACE', 'PROCESSOR', 'ISSUER', 'MANUAL_REVIEW')),
    occurred_at        timestamptz NOT NULL,
    received_at        timestamptz NOT NULL,
    reason_code        text,
    status             text NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'PROCESSED', 'DEAD')),
    delivery_attempts  integer NOT NULL DEFAULT 0,
    lease_token        uuid,
    lease_until        timestamptz,
    last_error         text,
    processed_at       timestamptz,
    PRIMARY KEY (tenant_id, event_id)
);
CREATE INDEX outcome_events_pending_idx ON outcome_events (received_at) WHERE status = 'PENDING';
CREATE INDEX outcome_events_attempt_idx ON outcome_events (tenant_id, payment_attempt_id);

-- Projection recomputed from all processed events of an attempt, so it is idempotent and
-- independent of delivery order.
CREATE TABLE attempt_outcomes (
    tenant_id                 text NOT NULL,
    payment_attempt_id        text NOT NULL,
    authorization_status      text CHECK (authorization_status IN ('APPROVED', 'DECLINED')),
    authorization_occurred_at timestamptz,
    chargeback_reported_at    timestamptz,
    label                     text CHECK (label IN ('CONFIRMED_FRAUD', 'CONFIRMED_LEGITIMATE')),
    label_source              text,
    label_occurred_at         timestamptz,
    label_event_id            text,
    event_count               integer NOT NULL,
    updated_at                timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, payment_attempt_id)
);

CREATE TABLE outbox (
    id           bigserial PRIMARY KEY,
    tenant_id    text NOT NULL,
    topic        text NOT NULL,
    payload      jsonb NOT NULL,
    created_at   timestamptz NOT NULL,
    published_at timestamptz,
    batch_id     text
);
CREATE INDEX outbox_pending_idx ON outbox (id) WHERE published_at IS NULL;

-- Which attempts have been exported to durable history; retention may purge only these.
CREATE TABLE archive_ledger (
    tenant_id          text NOT NULL,
    payment_attempt_id text NOT NULL,
    batch_id           text NOT NULL,
    archived_at        timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, payment_attempt_id)
);

CREATE TABLE archive_batches (
    batch_id     text PRIMARY KEY,
    object_key   text NOT NULL,
    manifest_key text NOT NULL,
    row_count    integer NOT NULL,
    sha256       text NOT NULL,
    created_at   timestamptz NOT NULL
);

-- Tenant isolation for the API role; workers see all tenants via role-specific policies.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['outcome_events', 'attempt_outcomes', 'outbox', 'archive_ledger'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I '
            'USING (tenant_id = current_setting(''app.tenant_id'', true)) '
            'WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))', t);
    END LOOP;
    FOREACH t IN ARRAY ARRAY['outcome_events', 'attempt_outcomes', 'outbox', 'archive_ledger',
                             'attempts', 'assessments'] LOOP
        EXECUTE format('CREATE POLICY worker_all ON %I TO fraud_worker USING (true) WITH CHECK (true)', t);
    END LOOP;
END
$$;

GRANT SELECT, INSERT ON outcome_events TO fraud_app;
GRANT SELECT ON attempt_outcomes TO fraud_app;
GRANT INSERT ON outbox TO fraud_app;
GRANT USAGE ON SEQUENCE outbox_id_seq TO fraud_app;

GRANT SELECT, UPDATE ON outcome_events TO fraud_worker;
GRANT SELECT, INSERT, UPDATE ON attempt_outcomes TO fraud_worker;
GRANT SELECT, UPDATE ON outbox TO fraud_worker;
GRANT SELECT, INSERT ON archive_ledger, archive_batches TO fraud_worker;
GRANT SELECT, DELETE ON attempts, assessments TO fraud_worker;
GRANT DELETE ON outcome_events, attempt_outcomes, archive_ledger TO fraud_worker;
GRANT DELETE ON outbox TO fraud_worker;
GRANT INSERT ON outbox TO fraud_worker;
GRANT USAGE ON SEQUENCE outbox_id_seq TO fraud_worker;
GRANT UPDATE ON archive_batches TO fraud_worker;
