-- Tenants, client identity mapping, attempts and immutable assessments (T-101, T-102, T-104).

CREATE TABLE tenants (
    tenant_id    text PRIMARY KEY CHECK (tenant_id ~ '^[a-z0-9-]{1,64}$'),
    display_name text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Trusted OAuth client -> tenant mapping. Looked up before the tenant is known, so no RLS.
CREATE TABLE api_clients (
    client_id  text PRIMARY KEY,
    tenant_id  text NOT NULL REFERENCES tenants,
    active     boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE attempts (
    tenant_id           text NOT NULL REFERENCES tenants,
    payment_attempt_id  text NOT NULL,
    idempotency_key     text NOT NULL,
    transaction_id      text NOT NULL,
    request_fingerprint bytea NOT NULL,
    status              text NOT NULL CHECK (status IN ('IN_PROGRESS', 'COMPLETED', 'UNAVAILABLE')),
    claim_token         uuid NOT NULL,
    lease_expires_at    timestamptz NOT NULL,
    amount_minor        bigint NOT NULL CHECK (amount_minor >= 0),
    currency            char(3) NOT NULL,
    occurred_at         timestamptz NOT NULL,
    received_at         timestamptz NOT NULL,
    -- Pseudonymous references (per-tenant HMAC); raw identifiers are never stored.
    customer_ref        text,
    device_ref          text,
    payment_method_ref  text,
    ip_ref              text,
    shipping_ref        text,
    account_created_at  timestamptz,
    updated_at          timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, payment_attempt_id),
    UNIQUE (tenant_id, idempotency_key)
);
CREATE INDEX attempts_received_idx ON attempts (tenant_id, received_at DESC);
CREATE INDEX attempts_transaction_idx ON attempts (tenant_id, transaction_id);

CREATE TABLE assessments (
    assessment_id      uuid PRIMARY KEY,
    tenant_id          text NOT NULL,
    payment_attempt_id text NOT NULL,
    transaction_id     text NOT NULL,
    risk_score         smallint NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    recommendation     text NOT NULL CHECK (recommendation IN ('PROCEED', 'VERIFY', 'RECOMMEND_DECLINE')),
    scoring_mode       text NOT NULL CHECK (scoring_mode IN ('ML', 'FALLBACK')),
    risk_signals       jsonb NOT NULL,
    features           jsonb NOT NULL,
    model_version      text NOT NULL,
    policy_version     text NOT NULL,
    evaluated_at       timestamptz NOT NULL,
    scoring_latency_ms real NOT NULL,
    UNIQUE (tenant_id, payment_attempt_id),
    FOREIGN KEY (tenant_id, payment_attempt_id) REFERENCES attempts (tenant_id, payment_attempt_id),
    -- I-08 provisional: fallback never recommends decline.
    CHECK (NOT (scoring_mode = 'FALLBACK' AND recommendation = 'RECOMMEND_DECLINE'))
);
CREATE INDEX assessments_evaluated_idx ON assessments (tenant_id, evaluated_at DESC);

-- Assessments are immutable once written.
CREATE FUNCTION reject_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% rows are immutable', TG_TABLE_NAME;
END
$$;
CREATE TRIGGER assessments_immutable BEFORE UPDATE ON assessments
    FOR EACH ROW EXECUTE FUNCTION reject_update();

-- Row-level security: rows are visible only when app.tenant_id matches. FORCE applies it to
-- the table owner too; an unset tenant matches nothing.
ALTER TABLE attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE attempts FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON attempts
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessments FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON assessments
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

GRANT SELECT ON tenants, api_clients TO fraud_app, fraud_worker;
GRANT SELECT, INSERT, UPDATE ON attempts TO fraud_app;
GRANT SELECT, INSERT ON assessments TO fraud_app;
