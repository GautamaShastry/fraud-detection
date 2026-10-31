-- Single-round-trip attempt claim and completion (performance; semantics unchanged from
-- src/storage/attempts.py docs). SECURITY INVOKER: the caller's role and RLS apply to every
-- statement; the tenant is set transaction-locally from the authenticated principal.

CREATE FUNCTION fraud_claim_attempt(
    p_tenant text, p_attempt text, p_key text, p_transaction text, p_fingerprint bytea,
    p_token uuid, p_now timestamptz, p_lease timestamptz, p_amount bigint, p_currency text,
    p_occurred timestamptz, p_received timestamptz, p_customer text, p_device text,
    p_payment_method text, p_ip text, p_shipping text, p_account_created timestamptz
) RETURNS TABLE (outcome text, lease_expires_at timestamptz, assessment jsonb)
LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    inserted boolean;
    mismatched boolean;
    found_rows integer;
    cur record;
BEGIN
    PERFORM set_config('app.tenant_id', p_tenant, true);

    INSERT INTO attempts (tenant_id, payment_attempt_id, idempotency_key, transaction_id,
        request_fingerprint, status, claim_token, lease_expires_at, amount_minor, currency,
        occurred_at, received_at, customer_ref, device_ref, payment_method_ref, ip_ref,
        shipping_ref, account_created_at, updated_at)
    VALUES (p_tenant, p_attempt, p_key, p_transaction, p_fingerprint, 'IN_PROGRESS', p_token,
        p_lease, p_amount, p_currency, p_occurred, p_received, p_customer, p_device,
        p_payment_method, p_ip, p_shipping, p_account_created, p_received)
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS found_rows = ROW_COUNT;
    IF found_rows = 1 THEN
        RETURN QUERY SELECT 'CLAIMED'::text, p_lease, NULL::jsonb;
        RETURN;
    END IF;

    -- Existing row(s) for this attempt ID or idempotency key: lock them for the decision.
    PERFORM 1 FROM attempts a WHERE a.tenant_id = p_tenant
        AND (a.payment_attempt_id = p_attempt OR a.idempotency_key = p_key) FOR UPDATE;
    SELECT bool_or(a.request_fingerprint <> p_fingerprint), count(*)
        INTO mismatched, found_rows
        FROM attempts a WHERE a.tenant_id = p_tenant
        AND (a.payment_attempt_id = p_attempt OR a.idempotency_key = p_key);
    IF found_rows = 0 THEN
        RETURN QUERY SELECT 'RETRY'::text, NULL::timestamptz, NULL::jsonb;
        RETURN;
    END IF;
    IF mismatched THEN
        RETURN QUERY SELECT 'CONFLICT'::text, NULL::timestamptz, NULL::jsonb;
        RETURN;
    END IF;

    SELECT a.status, a.lease_expires_at INTO cur FROM attempts a
        WHERE a.tenant_id = p_tenant AND a.payment_attempt_id = p_attempt;
    IF cur.status = 'COMPLETED' THEN
        RETURN QUERY SELECT 'COMPLETED'::text, NULL::timestamptz, to_jsonb(s)
            FROM assessments s WHERE s.tenant_id = p_tenant AND s.payment_attempt_id = p_attempt;
        RETURN;
    END IF;
    IF cur.status = 'IN_PROGRESS' AND cur.lease_expires_at > p_now THEN
        RETURN QUERY SELECT 'IN_PROGRESS'::text, cur.lease_expires_at, NULL::jsonb;
        RETURN;
    END IF;

    -- Lease expired, or the previous evaluation was unavailable: take over.
    UPDATE attempts SET status = 'IN_PROGRESS', claim_token = p_token,
        lease_expires_at = p_lease, updated_at = p_now
        WHERE tenant_id = p_tenant AND payment_attempt_id = p_attempt;
    RETURN QUERY SELECT 'CLAIMED'::text, p_lease, NULL::jsonb;
END
$$;

CREATE FUNCTION fraud_complete_attempt(
    p_tenant text, p_attempt text, p_token uuid, p_now timestamptz, p_assessment_id uuid,
    p_transaction text, p_risk_score smallint, p_recommendation text, p_scoring_mode text,
    p_risk_signals jsonb, p_features jsonb, p_model_version text, p_policy_version text,
    p_evaluated_at timestamptz, p_latency_ms real, p_outbox jsonb
) RETURNS TABLE (outcome text, assessment jsonb)
LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    found_rows integer;
BEGIN
    PERFORM set_config('app.tenant_id', p_tenant, true);

    UPDATE attempts SET status = 'COMPLETED', updated_at = p_now
        WHERE tenant_id = p_tenant AND payment_attempt_id = p_attempt
          AND claim_token = p_token AND status = 'IN_PROGRESS';
    GET DIAGNOSTICS found_rows = ROW_COUNT;
    IF found_rows = 1 THEN
        INSERT INTO assessments (assessment_id, tenant_id, payment_attempt_id, transaction_id,
            risk_score, recommendation, scoring_mode, risk_signals, features, model_version,
            policy_version, evaluated_at, scoring_latency_ms)
        VALUES (p_assessment_id, p_tenant, p_attempt, p_transaction, p_risk_score,
            p_recommendation, p_scoring_mode, p_risk_signals, p_features, p_model_version,
            p_policy_version, p_evaluated_at, p_latency_ms);
        INSERT INTO outbox (tenant_id, topic, payload, created_at)
            VALUES (p_tenant, 'assessment.recorded', p_outbox, p_now);
        RETURN QUERY SELECT 'RECORDED'::text, NULL::jsonb;
        RETURN;
    END IF;

    RETURN QUERY SELECT 'EXISTING'::text, to_jsonb(s)
        FROM assessments s WHERE s.tenant_id = p_tenant AND s.payment_attempt_id = p_attempt;
    IF NOT FOUND THEN
        RETURN QUERY SELECT 'LOST'::text, NULL::jsonb;
    END IF;
END
$$;

REVOKE ALL ON FUNCTION fraud_claim_attempt FROM PUBLIC;
REVOKE ALL ON FUNCTION fraud_complete_attempt FROM PUBLIC;
GRANT EXECUTE ON FUNCTION fraud_claim_attempt TO fraud_app;
GRANT EXECUTE ON FUNCTION fraud_complete_attempt TO fraud_app;
