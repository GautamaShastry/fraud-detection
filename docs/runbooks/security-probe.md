# Runbook: authentication failure spike

**Alert:** `AuthFailureSpike` (`fraud_auth_failures_total` by reason).

- `unauthenticated` dominant from one integration: expired/rotated keys or wrong audience.
- `missing_scope` / `unmapped_client`: client registered without the tenant mapping or scope.
- Broad forged-token traffic: probing. Requests are rejected before any data access; no tenant data
  is exposed. Consider rate limits at the edge. Never relax token validation to "unblock" a merchant.
