/* Minimal tenant fraud-monitoring UI. Authorization is enforced by the API (RBAC + RLS);
   the UI only hides controls a role cannot use. */
"use strict";
const html = htm.bind(React.createElement);
const { useEffect, useState, useCallback } = React;

let manager;
async function auth() {
  const cfg = await (await fetch("config.json")).json();
  manager = new oidc.UserManager({
    authority: cfg.issuer,
    client_id: cfg.client_id,
    redirect_uri: location.origin + "/dashboard/",
    response_type: "code",
    scope: "openid profile",
  });
  if (new URLSearchParams(location.search).has("code")) {
    await manager.signinRedirectCallback();
    history.replaceState({}, "", "/dashboard/");
  }
  return manager.getUser();
}

async function api(user, path) {
  const res = await fetch("/api/v1/dashboard" + path,
                          { headers: { Authorization: "Bearer " + user.access_token } });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

const pct = (v) => (v === null || v === undefined ? "n/a" : (100 * v).toFixed(1) + "%");
const ms = (v) => (v === null || v === undefined ? "n/a" : v.toFixed(1) + " ms");

function Metrics({ user }) {
  const [m, setM] = useState(null);
  const [hours, setHours] = useState(24);
  const [err, setErr] = useState(null);
  useEffect(() => { api(user, `/metrics?hours=${hours}`).then(setM, (e) => setErr(e.message)); },
            [user, hours]);
  if (err) return html`<p class="bad">${err}</p>`;
  if (!m) return html`<p class="muted">Loading…</p>`;
  const sum = (f) => m.assessments.filter(f).reduce((a, r) => a + r.count, 0);
  const unavailable = m.throughput.filter((r) => r.status === "UNAVAILABLE")
    .reduce((a, r) => a + r.count, 0);
  return html`<div>
    <p>Window: <select value=${hours} onChange=${(e) => setHours(+e.target.value)}>
      ${[1, 24, 168, 720].map((h) => html`<option value=${h}>${h} h</option>`)}</select></p>
    <div class="grid">
      <div class="card"><div class="muted">ML assessments</div>
        <div class="kpi">${sum((r) => r.scoring_mode === "ML")}</div></div>
      <div class="card"><div class="muted">Fallback assessments</div>
        <div class="kpi">${sum((r) => r.scoring_mode === "FALLBACK")}</div></div>
      <div class="card"><div class="muted">Unavailable (no score)</div>
        <div class="kpi">${unavailable}</div></div>
      <div class="card"><div class="muted">Scoring latency p95</div>
        <div class="kpi">${ms(m.server_scoring_latency_ms.p95)}</div>
        <div class="muted">${m.server_scoring_latency_ms.note}</div></div>
    </div>
    <h3>Labeled outcomes only</h3>
    <div class="banner muted">${m.labels.note}</div>
    <div class="grid">
      <div class="card"><div class="muted">Confirmed fraud</div><div class="kpi">${m.labels.confirmed_fraud}</div></div>
      <div class="card"><div class="muted">Confirmed legitimate</div><div class="kpi">${m.labels.confirmed_legitimate}</div></div>
      <div class="card"><div class="muted">Unlabeled</div><div class="kpi">${m.labels.unlabeled}</div></div>
      <div class="card"><div class="muted">Confirmed fraud flagged (verify/decline)</div>
        <div class="kpi">${pct(m.labels.fraud_flagged_rate)}</div></div>
      <div class="card"><div class="muted">Confirmed legitimate recommended decline</div>
        <div class="kpi">${pct(m.labels.legit_decline_recommendation_rate)}</div></div>
    </div></div>`;
}

function Attempts({ user, onOpen }) {
  const [rows, setRows] = useState([]);
  const [cursor, setCursor] = useState(null);
  const [filter, setFilter] = useState("");
  const load = useCallback(async (reset) => {
    const q = new URLSearchParams({ limit: "50" });
    if (filter) q.set("recommendation", filter);
    if (!reset && cursor) q.set("cursor", cursor);
    const page = await api(user, "/attempts?" + q);
    setRows((r) => (reset ? page.items : r.concat(page.items)));
    setCursor(page.next_cursor);
  }, [user, filter, cursor]);
  useEffect(() => { load(true); }, [user, filter]);
  return html`<div>
    <p>Recommendation: <select value=${filter} onChange=${(e) => setFilter(e.target.value)}>
      <option value="">all</option><option>PROCEED</option><option>VERIFY</option>
      <option>RECOMMEND_DECLINE</option></select></p>
    <table><thead><tr><th>Received</th><th>Attempt</th><th>Amount</th><th>Score</th>
      <th>Recommendation</th><th>Mode</th><th>Label</th></tr></thead>
    <tbody>${rows.map((r) => html`<tr class="click" key=${r.payment_attempt_id}
        onClick=${() => onOpen(r.payment_attempt_id)}>
      <td>${r.received_at}</td><td>${r.payment_attempt_id}</td>
      <td>${(r.amount_minor / 100).toFixed(2)} ${r.currency}</td><td>${r.risk_score ?? "—"}</td>
      <td class=${r.recommendation}>${r.recommendation ?? r.status}</td>
      <td>${r.scoring_mode ?? "—"}</td><td>${r.label}</td></tr>`)}</tbody></table>
    ${cursor && html`<p><button onClick=${() => load(false)}>Load more</button></p>`}</div>`;
}

function Detail({ user, id, onBack }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api(user, "/attempts/" + encodeURIComponent(id)).then(setD, (e) => setErr(e.message)); },
            [user, id]);
  return html`<div><button onClick=${onBack}>← Back</button>
    ${err && html`<p>${err}</p>`}
    ${d && html`<div class="card" style="margin-top:12px">
      <h3>${d.attempt.payment_attempt_id}</h3>
      <p>Status ${d.attempt.status} · label <b>${d.label}</b>${d.label_source ? " (" + d.label_source + ")" : ""}</p>
      ${d.assessment ? html`<p>Score <b>${d.assessment.risk_score}</b> ·
        <span class=${d.assessment.recommendation}>${d.assessment.recommendation}</span> ·
        ${d.assessment.scoring_mode} · model ${d.assessment.model_version} · policy ${d.assessment.policy_version}</p>
        <p class="muted">Risk signals are evidence the model considered, not causal explanations.
          The score is a rank, not a fraud probability.</p>
        <p>Signals: ${d.assessment.risk_signals.map((s) => s.code).join(", ") || "none"}</p>
        <pre>${JSON.stringify(d.assessment.features, null, 2)}</pre>`
        : html`<p>No assessment was established for this attempt.</p>`}
      <h4>Outcome events</h4>
      <pre>${JSON.stringify(d.outcomes, null, 2)}</pre></div>`}</div>`;
}

function Admin({ user }) {
  const [cfg, setCfg] = useState(null);
  const [audit, setAudit] = useState(null);
  useEffect(() => { api(user, "/admin/config").then(setCfg); api(user, "/admin/audit").then(setAudit); }, [user]);
  return html`<div><h3>Configuration</h3><pre>${JSON.stringify(cfg, null, 2)}</pre>
    <h3>Audit trail</h3><table><thead><tr><th>When</th><th>Who</th><th>Role</th><th>Action</th>
    <th>Target</th></tr></thead><tbody>${(audit?.items || []).map((a, i) => html`<tr key=${i}>
    <td>${a.at}</td><td>${a.subject}</td><td>${a.role}</td><td>${a.action}</td><td>${a.target || ""}</td></tr>`)}
    </tbody></table></div>`;
}

function App() {
  const [user, setUser] = useState(undefined);
  const [me, setMe] = useState(null);
  const [tab, setTab] = useState("metrics");
  const [open, setOpen] = useState(null);
  useEffect(() => { auth().then(setUser, () => setUser(null)); }, []);
  useEffect(() => { if (user) api(user, "/me").then(setMe, () => setMe(null)); }, [user]);
  if (user === undefined) return html`<main class="muted">Loading…</main>`;
  if (!user) return html`<main><h2>Fraud monitoring</h2>
    <button onClick=${() => manager.signinRedirect()}>Sign in</button></main>`;
  const tabs = [["metrics", "Metrics"], ["attempts", "Attempts"]];
  if (me?.role === "admin") tabs.push(["admin", "Admin"]);
  return html`<div>
    <header><b>Fraud monitoring · ${me?.tenant_id || ""}</b>
      <span>${me?.name} (${me?.role}) <button onClick=${() => manager.removeUser().then(() => location.reload())}>Sign out</button></span></header>
    <main><nav>${tabs.map(([k, label]) => html`<button class=${tab === k ? "active" : ""}
      onClick=${() => { setTab(k); setOpen(null); }}>${label}</button> `)}</nav>
      ${tab === "metrics" && html`<${Metrics} user=${user} />`}
      ${tab === "attempts" && !open && html`<${Attempts} user=${user}
          onOpen=${(id) => (me?.role === "viewer" ? alert("Attempt detail requires the analyst role") : setOpen(id))} />`}
      ${tab === "attempts" && open && html`<${Detail} user=${user} id=${open} onBack=${() => setOpen(null)} />`}
      ${tab === "admin" && html`<${Admin} user=${user} />`}
    </main></div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
