-- Runtime roles. Neither role owns tables or bypasses RLS.
-- fraud_app:    API process; every tenant-scoped query runs with app.tenant_id set.
-- fraud_worker: outcome/outbox workers; cross-tenant only through role-specific policies.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fraud_app') THEN
        CREATE ROLE fraud_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fraud_worker') THEN
        CREATE ROLE fraud_worker LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO fraud_app, fraud_worker;
