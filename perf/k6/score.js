// Constant-arrival-rate load test for POST /api/v1/fraud/score (T-503).
//
//   k6 run -e BASE_URL=http://api:8000 -e TOKEN=... -e RATE=100 -e DURATION=60s \
//          -e RUN_ID=run1 --summary-export=/evidence/summary.json \
//          --console-output=/evidence/attempts.log perf/k6/score.js
//
// Every iteration is a unique attempt (RUN_ID + VU + iteration), so results can be reconciled
// against PostgreSQL with perf/reconcile_run.py. Offered vs achieved rate is reported through
// dropped_iterations: a run where k6 could not offer the target rate has not tested that rate.
import http from "k6/http";
import { check } from "k6";
import { Counter, Trend } from "k6/metrics";
import exec from "k6/execution";

const RATE = parseInt(__ENV.RATE || "50");
const DURATION = __ENV.DURATION || "60s";
const RUN_ID = __ENV.RUN_ID || `run${Date.now()}`;
const DEVICES = parseInt(__ENV.DEVICES || "5000");

export const options = {
  discardResponseBodies: false,
  scenarios: {
    score: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: Math.max(10, Math.ceil(RATE * 0.2)),
      maxVUs: Math.max(50, RATE * 2),
    },
  },
  thresholds: {
    // PRD 4.1 targets for successful valid ML assessments (client-observed).
    latency_ml_ms: ["p(95)<100", "p(99)<200"],
    checks: ["rate>0.99"],
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

const ml = new Counter("assessments_ml");
const fallback = new Counter("assessments_fallback");
const unavailable = new Counter("assessments_unavailable");
const other = new Counter("responses_other");
const mlLatency = new Trend("latency_ml_ms", true);

export default function () {
  const vu = exec.vu.idInTest;
  const it = exec.scenario.iterationInTest;
  const id = `${RUN_ID}-${vu}-${it}`;
  const device = `dev-${RUN_ID}-${it % DEVICES}`;
  const body = JSON.stringify({
    transaction_id: `order-${id}`,
    payment_attempt_id: `pa-${id}`,
    idempotency_key: `idem-${id}`,
    amount_minor: 500 + (it % 50000),
    currency: "USD",
    occurred_at: new Date().toISOString().replace(/\.\d{3}Z$/, ".000Z"),
    device_id: device,
    payment_method_token: `tok-${RUN_ID}-${it % (DEVICES * 2)}`,
  });
  const res = http.post(`${__ENV.BASE_URL}/api/v1/fraud/score`, body, {
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${__ENV.TOKEN}` },
    timeout: "2s",
  });
  let outcome = "other";
  if (res.status === 200) {
    const mode = res.json("scoring_mode");
    outcome = mode === "ML" ? "ml" : "fallback";
    console.log(`ASSESSED ${mode} pa-${id}`); // Read by perf/reconcile_run.py
  } else if (res.status === 503 || res.status === 504) {
    outcome = "unavailable";
  }
  if (outcome === "ml") { ml.add(1); mlLatency.add(res.timings.duration); }
  else if (outcome === "fallback") fallback.add(1);
  else if (outcome === "unavailable") unavailable.add(1);
  else other.add(1);
  check(res, { "valid assessment": (r) => r.status === 200 });
}
