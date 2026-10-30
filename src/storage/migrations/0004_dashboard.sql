-- Dashboard access audit trail (FR-10, T-401). Tenant-isolated like every other tenant table.
CREATE TABLE dashboard_audit (
    id         bigserial PRIMARY KEY,
    tenant_id  text NOT NULL,
    subject    text NOT NULL,
    role       text NOT NULL,
    action     text NOT NULL,
    target     text,
    at         timestamptz NOT NULL
);
CREATE INDEX dashboard_audit_tenant_idx ON dashboard_audit (tenant_id, at DESC);

ALTER TABLE dashboard_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE dashboard_audit FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON dashboard_audit
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

GRANT SELECT, INSERT ON dashboard_audit TO fraud_app;
GRANT USAGE ON SEQUENCE dashboard_audit_id_seq TO fraud_app;
