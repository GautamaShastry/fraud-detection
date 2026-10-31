# IEEE-CIS dataset card (train files)

- Transactions: 590,540; identity rows: 144,233 (identity coverage 24.4%)
- Join key: `TransactionID`; identity IDs are a subset of transaction IDs (verified)
- Fraud rate (`isFraud`): 3.499%
- Temporal column: `TransactionDT` range 86,400–15,811,131 s (182 days). Seconds elapsed from an undisclosed reference datetime (not a real timestamp). Monotonic in event time; used for chronological ordering and splitting.
- Features: 432 (49 categorical, 383 numeric)
- Columns >90% missing: 11; >50% missing: 186
- Leakage audit exclusions: none

## Availability of online API features in IEEE-CIS

| Online feature | API source | IEEE-CIS source | Status | Note |
| --- | --- | --- | --- | --- |
| amount_log | amount_minor | TransactionAmt | AVAILABLE | IEEE amounts are in USD major units; currency is not recorded. |
| hour_sin | occurred_at | TransactionDT | PARTIAL | TransactionDT is seconds from an undisclosed reference; hour-of-day is only recoverable up to an unknown offset. |
| hour_cos | occurred_at | TransactionDT | PARTIAL | As hour_sin. |
| account_age_days | account_created_at | D1 (days since card first use?) | UNAVAILABLE | D-columns are undocumented timedeltas; none is defined as account age, so using one would invent semantics. |
| device_distinct_payment_methods_5m | device_id + payment_method_token | none | UNAVAILABLE | IEEE has no device identifier (DeviceInfo is a model string), so F01 cannot be computed from IEEE-CIS. |
| has_customer | customer_id | none | UNAVAILABLE | No customer identifier in IEEE-CIS. |
| has_device | device_id | DeviceType/DeviceInfo presence | PARTIAL | Presence of device metadata, not of a device identifier. |
| has_payment_method | payment_method_token | card1 presence | PARTIAL | card1 is an anonymized card attribute, not a token. |
| has_ip | ip_address | none | UNAVAILABLE | No IP column. |
| has_shipping | shipping_destination_ref | addr1/addr2 presence | PARTIAL | addr1/addr2 are anonymized billing-region codes, not shipping references. |

**Consequence:** the deployed scoring model cannot be trained on IEEE-CIS, because most online features (including F01) do not exist in it. IEEE-CIS is used for an offline benchmark of the modeling approach only (see `ieee_benchmark.md`).
