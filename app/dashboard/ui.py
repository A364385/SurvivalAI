"""SurvivalAI dashboard UI (self-contained, no external CDNs — offline-safe)."""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SurvivalAI — Paper Trading Only</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d; --text: #e6edf3;
    --muted: #8b949e; --green: #3fb950; --red: #f85149; --amber: #d29922;
    --blue: #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 14px/1.45 'Segoe UI', system-ui, sans-serif; }
  header { display: flex; align-items: center; gap: 16px; padding: 10px 20px; background: var(--panel); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10; }
  header h1 { font-size: 18px; }
  .badge { background: var(--green); color: #04140a; font-weight: 700; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
  .conn { font-size: 12px; color: var(--muted); }
  .conn.live::before { content: '● '; color: var(--green); }
  .conn.dead::before { content: '● '; color: var(--red); }
  nav { display: flex; gap: 4px; margin-left: auto; flex-wrap: wrap; }
  nav button { background: transparent; color: var(--muted); border: 1px solid transparent; padding: 6px 12px; border-radius: 6px; cursor: pointer; }
  nav button.active, nav button:hover { color: var(--text); border-color: var(--border); background: var(--bg); }
  main { padding: 16px 20px 60px; }
  .page { display: none; } .page.active { display: block; }
  .grid { display: grid; gap: 12px; }
  .grid.cols-4 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  .grid.cols-2 { grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .panel h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }
  .kpi { font-size: 24px; font-weight: 700; }
  .kpi.pos { color: var(--green); } .kpi.neg { color: var(--red); }
  .sub { color: var(--muted); font-size: 12px; }
  canvas { width: 100%; height: 220px; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .pill { padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .pill.ok { background: rgba(63,185,80,.15); color: var(--green); }
  .pill.warn { background: rgba(210,153,34,.15); color: var(--amber); }
  .pill.bad { background: rgba(248,81,73,.15); color: var(--red); }
  .pill.off { background: rgba(139,148,158,.15); color: var(--muted); }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--border); }
  .switch { position: relative; width: 40px; height: 22px; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; inset: 0; background: var(--border); border-radius: 22px; cursor: pointer; transition: .15s; }
  .slider::before { content: ''; position: absolute; height: 16px; width: 16px; left: 3px; top: 3px; background: var(--muted); border-radius: 50%; transition: .15s; }
  input:checked + .slider { background: var(--green); }
  input:checked + .slider::before { transform: translateX(18px); background: #04140a; }
  input:disabled + .slider { opacity: .4; cursor: not-allowed; }
  .btn { background: var(--blue); border: none; color: #04140a; font-weight: 700; padding: 8px 16px; border-radius: 6px; cursor: pointer; }
  .btn.danger { background: var(--red); color: #fff; }
  .btn.secondary { background: var(--border); color: var(--text); }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  input[type=text], input[type=password], select, input[type=number] {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 10px; width: 100%;
  }
  label { font-size: 12px; color: var(--muted); display: block; margin: 8px 0 3px; }
  .mono { font-family: Consolas, monospace; font-size: 12px; }
  .toolbar { display: flex; gap: 8px; margin: 10px 0; flex-wrap: wrap; align-items: center; }
  .survival-banner { padding: 12px 16px; border-radius: 8px; font-weight: 700; margin-bottom: 12px; border: 1px solid var(--border); }
  .survival-banner.HEALTHY { background: rgba(63,185,80,.1); color: var(--green); }
  .survival-banner.UNDER_PRESSURE { background: rgba(210,153,34,.1); color: var(--amber); }
  .survival-banner.CRITICAL { background: rgba(248,81,73,.15); color: var(--red); }
  .survival-banner.DYING, .survival-banner.DEAD { background: rgba(248,81,73,.25); color: var(--red); }
  #toast { position: fixed; bottom: 16px; right: 16px; background: var(--panel); border: 1px solid var(--border); padding: 10px 16px; border-radius: 8px; display: none; max-width: 380px; }
  .progress { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 4px; }
  .progress > div { height: 100%; background: var(--blue); width: 0; transition: width .3s; }
</style>
</head>
<body>
<header>
  <h1>SurvivalAI</h1>
  <span class="badge">PAPER ONLY</span>
  <span id="conn" class="conn dead">connecting…</span>
  <nav>
    <button data-page="main" class="active">Main</button>
    <button data-page="survival">Survival</button>
    <button data-page="health">Health</button>
    <button data-page="controls">Controls</button>
    <button data-page="models">Models</button>
    <button data-page="training">Training</button>
    <button data-page="settings">API &amp; Providers</button>
    <button data-page="audit">Audit</button>
  </nav>
</header>
<main>

<section id="page-main" class="page active">
  <div class="grid cols-4" id="kpis"></div>
  <div class="grid cols-2" style="margin-top:12px">
    <div class="panel"><h3>Portfolio Equity</h3><canvas id="chart-equity"></canvas></div>
    <div class="panel"><h3>Total P/L</h3><canvas id="chart-pl"></canvas></div>
    <div class="panel"><h3>Daily P/L</h3><canvas id="chart-daily"></canvas></div>
    <div class="panel"><h3>Drawdown</h3><canvas id="chart-dd"></canvas></div>
  </div>
  <div class="grid cols-2" style="margin-top:12px">
    <div class="panel"><h3>Indicators (latest)</h3><div id="indicator-table">no data yet</div></div>
    <div class="panel"><h3>System</h3><div id="system-summary">—</div></div>
  </div>
</section>

<section id="page-survival" class="page">
  <div id="survival-banner" class="survival-banner HEALTHY">—</div>
  <div class="grid cols-4" id="survival-kpis"></div>
  <div class="panel" style="margin-top:12px"><h3>Drawdown History</h3><canvas id="chart-survival-dd"></canvas></div>
</section>

<section id="page-health" class="page">
  <div class="panel"><h3>System Health</h3><table id="health-table"></table>
  <div class="toolbar"><button class="btn secondary" onclick="loadHealth()">Re-check</button></div></div>
</section>

<section id="page-controls" class="page">
  <div class="grid cols-2">
    <div class="panel"><h3>Master Control</h3>
      <div class="toolbar">
        <button class="btn" onclick="runtimeControl('resume')">RUN</button>
        <button class="btn secondary" onclick="runtimeControl('pause')">PAUSE ALL</button>
        <button class="btn secondary" onclick="runtimeControl('stop')">STOP SAFELY</button>
        <button class="btn danger" onclick="emergencyStop(true)">EMERGENCY PAPER STOP</button>
        <button class="btn secondary" onclick="emergencyStop(false)">LIFT E-STOP</button>
      </div>
      <div id="flags"></div>
    </div>
    <div class="panel"><h3>Agent On/Off</h3>
      <p class="sub" style="margin-bottom:8px">Safety-critical systems (Risk Manager, Investment Safety, Capital Protection) cannot be disabled.</p>
      <div id="agent-toggles"></div>
    </div>
  </div>
</section>

<section id="page-models" class="page">
  <div class="panel"><h3>Model Registry</h3>
    <table id="models-table"><tr><th>Model</th><th>Role</th><th>Version</th><th>Status</th><th>Eval</th><th></th></tr></table>
  </div>
  <div class="panel" style="margin-top:12px"><h3>Roles</h3><table id="roles-table"><tr><th>Role</th><th>Active Model</th></tr></table></div>
</section>

<section id="page-training" class="page">
  <div class="grid cols-2">
    <div class="panel"><h3>Start Training</h3>
      <label>Role</label><select id="train-role"></select>
      <label>Dataset version</label><input type="text" id="train-dataset" value="v1">
      <label>Model version</label><input type="text" id="train-model-version" value="v0.1">
      <label>Base model</label><select id="train-base"></select>
      <div class="toolbar" style="margin-top:12px"><button class="btn" id="btn-train" onclick="startTraining()">START TRAINING</button></div>
      <div id="train-estimate" class="sub"></div>
    </div>
    <div class="panel"><h3>Training Queue</h3><div id="train-queue">empty</div>
      <div class="panel" style="margin-top:10px;background:var(--bg)"><h3>Progress</h3><div id="train-progress" class="mono sub">idle</div><div class="progress"><div id="train-bar"></div></div></div>
    </div>
  </div>
  <div class="panel" style="margin-top:12px"><h3>Hardware</h3><div id="hw-info" class="mono sub">—</div></div>
  <div class="panel" style="margin-top:12px"><h3>Datasets</h3><div id="datasets-list" class="sub">none built yet</div></div>
</section>

<section id="page-settings" class="page">
  <div class="grid cols-2">
    <div class="panel"><h3>LLM Provider</h3>
      <label>Provider</label>
      <select id="llm-type">
        <option value="lm_studio">LM Studio (local)</option>
        <option value="ollama">Ollama (local)</option>
        <option value="gemini">Google Gemini (cloud API)</option>
        <option value="claude">Anthropic Claude (cloud API)</option>
      </select>
      <label>Endpoint (local providers only)</label><input type="text" id="llm-endpoint" value="http://127.0.0.1:1234">
      <label>Model</label><input type="text" id="llm-model" placeholder="server default">
      <p class="sub" id="remote-note">Remote providers need GEMINI_API_KEY or ANTHROPIC_API_KEY as environment variables — keys are never entered in the UI.</p>
      <div class="toolbar">
        <button class="btn secondary" onclick="testProvider('llm')">Test Connection</button>
        <button class="btn" onclick="applyLLM()">Apply Provider</button>
      </div>
    </div>
    <div class="panel"><h3>LLM Usage &amp; Cost</h3>
      <div id="llm-usage" class="mono sub">no calls yet</div>
    </div>
    <div class="panel"><h3>Alpaca PAPER</h3>
      <label>Paper API Key</label><input type="password" id="alpaca-key" placeholder="••••">
      <label>Paper API Secret</label><input type="password" id="alpaca-secret" placeholder="••••">
      <label>Paper Endpoint</label><input type="text" id="alpaca-endpoint" value="https://paper-api.alpaca.markets/v2">
      <div class="toolbar"><button class="btn secondary" onclick="testProvider('alpaca')">Test Paper Connection</button></div>
      <p class="sub">Keys are stored masked, never logged, never shown back.</p>
    </div>
  </div>
  <div class="toolbar" style="margin-top:12px"><button class="btn" onclick="saveSettings()">Save Settings</button></div>
  <div class="panel" style="margin-top:12px"><h3>Result</h3><div id="settings-result" class="mono sub">—</div></div>
</section>

<section id="page-audit" class="page">
  <div class="panel"><h3>Audit Trail (recent 100)</h3>
    <table id="audit-table"><tr><th>Time</th><th>Component</th><th>Task</th><th>Decision</th><th>Risk</th><th>Protection</th><th>Execution</th></tr></table>
  </div>
</section>

</main>
<div id="toast"></div>
<script>
'use strict';
const $ = (id) => document.getElementById(id);
const fmt = (v) => (v === null || v === undefined) ? '—' :
  (typeof v === 'number' ? v.toLocaleString(undefined, {maximumFractionDigits: 2}) : String(v));

function toast(msg, isError) {
  const t = $('toast');
  t.textContent = msg;
  t.style.borderColor = isError ? 'var(--red)' : 'var(--border)';
  t.style.display = 'block';
  clearTimeout(t._h);
  t._h = setTimeout(() => t.style.display = 'none', 4000);
}

async function api(path, opts) {
  const resp = await fetch(path, opts);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

// ---------- navigation ----------
document.querySelectorAll('nav button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('page-' + btn.dataset.page).classList.add('active');
    if (btn.dataset.page === 'health') loadHealth();
    if (btn.dataset.page === 'models') loadModels();
    if (btn.dataset.page === 'training') loadTraining();
    if (btn.dataset.page === 'audit') loadAudit();
  };
});

// ---------- charts (pure canvas, no libs) ----------
function drawLine(canvas, points, colorPositive, colorNegative, baseline) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!points || points.length < 2) {
    ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif';
    ctx.fillText('collecting data…', 12, h / 2);
    return;
  }
  const values = points.map(p => p.value);
  let min = Math.min(...values), max = Math.max(...values);
  if (baseline !== undefined) { min = Math.min(min, baseline); max = Math.max(max, baseline); }
  if (max === min) { max += 1; min -= 1; }
  const pad = 8;
  const x = (i) => pad + i * (w - 2 * pad) / (points.length - 1);
  const y = (v) => h - pad - (v - min) * (h - 2 * pad) / (max - min);
  // zero/baseline line
  if (baseline !== undefined && baseline >= min && baseline <= max) {
    ctx.strokeStyle = '#30363d'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad, y(baseline)); ctx.lineTo(w - pad, y(baseline)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // color segments above/below baseline
  const base = baseline === undefined ? min : baseline;
  for (let i = 1; i < points.length; i++) {
    const above = (values[i] >= base && values[i-1] >= base);
    const below = (values[i] <= base && values[i-1] <= base);
    ctx.strokeStyle = below ? (colorNegative || '#f85149') : (colorPositive || '#3fb950');
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x(i - 1), y(values[i - 1]));
    ctx.lineTo(x(i), y(values[i]));
    ctx.stroke();
  }
  // last value label
  ctx.fillStyle = '#e6edf3'; ctx.font = '11px sans-serif';
  const last = values[values.length - 1];
  ctx.fillText(fmt(last), Math.min(w - 60, x(points.length - 1) + 4), y(last) - 6);
}

// ---------- SSE live updates ----------
let evtSource = null;
function connectStream() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/v2/stream');
  evtSource.onopen = () => { $('conn').className = 'conn live'; $('conn').textContent = 'live'; };
  evtSource.onerror = () => { $('conn').className = 'conn dead'; $('conn').textContent = 'reconnecting…'; };
  evtSource.onmessage = (e) => {
    try { const msg = JSON.parse(e.data); renderState(msg); } catch (err) {}
  };
}

// ---------- state rendering ----------
let lastState = null;
function renderState(msg) {
  const data = msg.data || msg;
  if (data.charts) {
    drawLine($('chart-equity'), data.charts.equity, '#3fb950', '#f85149');
    drawLine($('chart-pl'), data.charts.daily_pl, '#3fb950', '#f85149', 0);
    drawLine($('chart-daily'), data.charts.daily_pl, '#3fb950', '#f85149', 0);
    drawLine($('chart-dd'), data.charts.drawdown, '#f85149', '#f85149', 0);
    drawLine($('chart-survival-dd'), data.charts.drawdown, '#f85149', '#f85149', 0);
  }
  if (data.generation) renderSurvival(data.generation);
  renderKpis(data);
}

function renderKpis(data) {
  const gen = data.generation || {};
  const equity = gen.current_capital ?? 0;
  const starting = gen.starting_capital ?? 0;
  const pl = equity - starting;
  const dd = (gen.survival && gen.survival.drawdown) || 0;
  const plClass = pl >= 0 ? 'kpi pos' : 'kpi neg';
  $('kpis').innerHTML = `
    <div class="panel"><h3>Generation</h3><div class="kpi">${fmt(gen.generation_number ?? '—')}</div>
      <div class="sub">${fmt(gen.status || data.runtime?.state || '')}</div></div>
    <div class="panel"><h3>Equity</h3><div class="kpi">${fmt(equity)}</div><div class="sub">start ${fmt(starting)}</div></div>
    <div class="panel"><h3>Total P/L</h3><div class="${plClass}">${fmt(pl)}</div>
      <div class="sub">${starting ? ((pl / starting) * 100).toFixed(2) + '%' : ''}</div></div>
    <div class="panel"><h3>Drawdown</h3><div class="${dd > 0.15 ? 'kpi neg' : 'kpi'}">${(dd * 100).toFixed(1)}%</div></div>`;
  $('system-summary').innerHTML =
    `<div class="row"><span>Runtime state</span><b>${fmt(data.runtime?.state)}</b></div>
     <div class="row"><span>Cycles completed</span><b>${fmt(data.runtime?.completed_cycles)}</b></div>
     <div class="row"><span>Mode</span><span class="pill ok">PAPER ONLY</span></div>`;
}

function renderSurvival(gen) {
  const s = gen.survival || {};
  const status = s.status || 'HEALTHY';
  const banner = $('survival-banner');
  banner.className = 'survival-banner ' + status;
  banner.textContent = `GENERATION ${gen.generation_number ?? '—'} — ${status}`;
  $('survival-kpis').innerHTML = `
    <div class="panel"><h3>Capital</h3><div class="kpi">${fmt(gen.current_capital)}</div></div>
    <div class="panel"><h3>Equity</h3><div class="kpi">${fmt(s.equity)}</div></div>
    <div class="panel"><h3>Peak Equity</h3><div class="kpi">${fmt(s.peak_equity)}</div></div>
    <div class="panel"><h3>Drawdown</h3><div class="kpi neg">${((s.drawdown || 0) * 100).toFixed(1)}%</div></div>`;
}

// ---------- data loads ----------
async function loadStateOnce() {
  try { renderState(await api('/api/v2/state')); } catch (e) { toast(e.message, true); }
}
async function loadHealth() {
  try {
    const data = await api('/api/v2/health');
    $('health-table').innerHTML = '<tr><th>Component</th><th>Status</th><th>Detail</th></tr>' +
      Object.entries(data.checks).map(([name, c]) =>
        `<tr><td>${name}</td><td><span class="pill ${c.ok ? 'ok' : 'bad'}">${c.ok ? 'OK' : 'FAIL'}</span></td><td class="mono">${fmt(c.detail)}</td></tr>`
      ).join('');
  } catch (e) { toast(e.message, true); }
}
async function loadControls() {
  try {
    const data = await api('/api/v2/controls');
    const c = data.controls;
    const flags = [
      ['fast_loop', 'Fast Loop'], ['research_loop', 'Research Loop'],
      ['paper_trading', 'Paper Trading'], ['learning', 'Learning'],
      ['generation_evolution', 'Generation Evolution'],
    ];
    $('flags').innerHTML = flags.map(([id, label]) => `
      <div class="row"><span>${label}</span>
        <label class="switch"><input type="checkbox" id="flag-${id}" ${c[id] ? 'checked' : ''}
          onchange="setFlag('${id}', this.checked)"><span class="slider"></span></label></div>`).join('');
    const toggles = Object.entries(c.agents || {});
    $('agent-toggles').innerHTML = toggles.map(([agent, on]) => `
      <div class="row"><span>${agent}</span>
        <label class="switch"><input type="checkbox" ${on ? 'checked' : ''}
          onchange="setAgent('${agent}', this.checked)"><span class="slider"></span></label></div>`).join('') +
      (c.protected || []).map(p => `
        <div class="row"><span>${p}</span>
          <label class="switch"><input type="checkbox" disabled checked><span class="slider"></span></label></div>`).join('');
  } catch (e) { toast(e.message, true); }
}
async function setFlag(flag, value) {
  try { await api('/api/v2/controls', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({flag, value})}); toast(flag + ' = ' + value); }
  catch (e) { toast(e.message, true); loadControls(); }
}
async function setAgent(agent, value) {
  try { await api('/api/v2/controls', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({agent, enabled: value})}); toast(agent + ' = ' + value); }
  catch (e) { toast(e.message, true); loadControls(); }
}
async function runtimeControl(action) {
  try { const r = await api('/api/v2/control/' + action, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}); toast(action + ': ' + (r.ok ? 'ok' : 'failed')); }
  catch (e) { toast(e.message, true); }
}
async function emergencyStop(engage) {
  try { await api('/api/v2/control/emergency_stop', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({engage})});
    toast(engage ? 'EMERGENCY STOP ENGAGED — all new paper orders blocked' : 'Emergency stop lifted'); }
  catch (e) { toast(e.message, true); }
}
async function loadModels() {
  try {
    const data = await api('/api/v2/models');
    $('models-table').innerHTML = '<tr><th>Model</th><th>Role</th><th>Version</th><th>Status</th><th>Eval</th><th></th></tr>' +
      (data.models || []).map(m => {
        const evalAgg = m.evaluation_results && m.evaluation_results.aggregate;
        const canActivate = m.status === 'VALIDATED';
        const pill = m.status === 'ACTIVE' ? 'ok' : (m.status === 'FAILED' ? 'bad' : (m.status === 'VALIDATED' ? 'ok' : 'off'));
        return `<tr><td class="mono">${m.model_id}</td><td>${m.role}</td><td>${m.model_version}</td>
          <td><span class="pill ${pill}">${m.status}</span></td>
          <td>${evalAgg !== undefined ? evalAgg : '—'}</td>
          <td>${canActivate ? `<button class="btn secondary" onclick="activateModel('${m.model_id}')">Activate</button>` : ''}</td></tr>`;
      }).join('') || '<tr><td colspan="6" class="sub">no models registered yet</td></tr>';
    $('roles-table').innerHTML = '<tr><th>Role</th><th>Active Model</th></tr>' +
      (data.roles || []).map(r => `<tr><td>${r.display_name}</td><td class="mono">${r.active_model || '— (general model)'}</td></tr>`).join('');
    const roleSel = $('train-role');
    if (roleSel && roleSel.options.length === 0) {
      (data.roles || []).forEach(r => roleSel.add(new Option(r.display_name, r.role)));
    }
  } catch (e) { toast(e.message, true); }
}
async function activateModel(modelId) {
  try {
    const r = await api('/api/v2/models/activate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model_id: modelId})});
    toast(r.ok ? 'Model activated: ' + modelId : 'Activation failed');
    loadModels();
  } catch (e) { toast(e.message, true); }
}
async function loadTraining() {
  try {
    const data = await api('/api/v2/training');
    const q = data.queue || {};
    $('train-queue').innerHTML = (q.items || []).map(i =>
      `<div class="row"><span class="mono">${i.role} ${i.model_version}</span><span class="pill ${i.status === 'DONE' ? 'ok' : i.status === 'FAILED' ? 'bad' : 'warn'}">${i.status}</span></div>`
    ).join('') || 'empty';
    const prog = (q.progress || []).slice(-1)[0];
    if (prog) {
      $('train-progress').textContent = `${prog.stage} ${prog.status || ''} ${prog.step ? prog.step + '/' + prog.max_steps : ''}`;
      if (prog.step && prog.max_steps) $('train-bar').style.width = (100 * prog.step / prog.max_steps) + '%';
    }
    if (data.hardware) {
      const hw = data.hardware;
      $('hw-info').textContent = `GPU: ${hw.gpu_name || 'none'} (${hw.gpu_vram_gb}GB) | RAM: ${hw.ram_gb}GB | CPU threads: ${hw.cpu_threads} | disk free: ${hw.disk_free_gb}GB | torch: ${hw.torch_version || 'not installed'}`;
      const baseSel = $('train-base');
      if (baseSel && baseSel.options.length === 0) {
        const vram = hw.gpu_vram_gb || 0;
        const options = vram >= 10 ? ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']
          : vram >= 6 ? ['Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']
          : ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'];
        options.forEach(o => baseSel.add(new Option(o, o)));
        $('train-estimate').textContent = `Estimated VRAM fits: ${options[0]} via QLoRA/LoRA (hardware-aware profile applied automatically)`;
      }
    }
    $('datasets-list').innerHTML = (data.datasets || []).map(d =>
      `<div class="row"><span class="mono">${d.dataset_id}</span><span>train ${d.stats.train} / val ${d.stats.validation} / test ${d.stats.test}</span></div>`
    ).join('') || 'none built yet';
  } catch (e) { toast(e.message, true); }
}
async function startTraining() {
  const body = {
    role: $('train-role').value, dataset_version: $('train-dataset').value,
    model_version: $('train-model-version').value, base_model: $('train-base').value,
  };
  if (!body.role || !body.base_model) { toast('select a role and base model first', true); return; }
  try {
    await api('/api/v2/training', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    toast('Training job queued for ' + body.role);
    loadTraining();
  } catch (e) { toast(e.message, true); }
}
async function loadAudit() {
  try {
    const data = await api('/api/v2/audit');
    $('audit-table').innerHTML = '<tr><th>Time</th><th>Component</th><th>Task</th><th>Decision</th><th>Risk</th><th>Protection</th><th>Execution</th></tr>' +
      (data.entries || []).map(e =>
        `<tr><td class="mono">${(e.timestamp || '').replace('T', ' ').slice(0, 19)}</td><td>${fmt(e.component)}</td><td class="mono">${fmt(e.task)}</td>
         <td><span class="pill ${e.decision === 'INVEST' ? 'ok' : e.decision === 'DO_NOT_INVEST' ? 'warn' : 'off'}">${fmt(e.decision)}</span></td>
         <td>${fmt(e.risk_result)}</td><td>${fmt(e.capital_protection_result)}</td><td>${fmt(e.execution_result)}</td></tr>`
      ).join('') || '<tr><td colspan="7" class="sub">no audit entries yet</td></tr>';
  } catch (e) { toast(e.message, true); }
}
async function saveSettings() {
  const llm = {type: $('llm-type').value, endpoint: $('llm-endpoint').value, model: $('llm-model').value};
  const alpaca = {api_key: $('alpaca-key').value, api_secret: $('alpaca-secret').value, endpoint: $('alpaca-endpoint').value};
  try {
    await api('/api/v2/providers', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({provider: 'local_llm', settings: llm})});
    if (alpaca.api_key) await api('/api/v2/providers', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({provider: 'alpaca_paper', settings: alpaca})});
    $('settings-result').textContent = 'saved (secrets masked)';
    toast('Settings saved');
  } catch (e) { toast(e.message, true); }
}
async function testProvider(kind) {
  try {
    let settings = {};
    if (kind === 'llm') settings = {type: $('llm-type').value, endpoint: $('llm-endpoint').value, model: $('llm-model').value};
    if (kind === 'alpaca') settings = {api_key: $('alpaca-key').value, api_secret: $('alpaca-secret').value};
    const r = await api('/api/v2/providers/test', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({kind, settings})});
    $('settings-result').textContent = JSON.stringify(r).slice(0, 300);
    toast(r.ok ? 'Connection OK' : 'Connection failed', !r.ok);
  } catch (e) { toast(e.message, true); }
}
async function applyLLM() {
  try {
    const body = {provider: $('llm-type').value, endpoint: $('llm-endpoint').value, model: $('llm-model').value || null};
    const r = await api('/api/v2/llm/configure', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    $('settings-result').textContent = JSON.stringify(r).slice(0, 300);
    toast(r.ok ? 'LLM provider applied: ' + body.provider : 'Apply failed — check keys/connection', !r.ok);
  } catch (e) { toast(e.message, true); }
}
async function loadLLMUsage() {
  try {
    const data = await api('/api/v2/llm/usage');
    const o = (data.summary && data.summary.overall) || {};
    if (!o.calls) { $('llm-usage').textContent = 'no LLM calls recorded yet'; return; }
    const remote = Object.keys(data.available_remote || {});
    $('llm-usage').innerHTML =
      `<div class="row"><span>Calls</span><b>${o.calls}</b></div>` +
      `<div class="row"><span>Tokens (in/out)</span><b>${fmt(o.input_tokens)} / ${fmt(o.output_tokens)}</b></div>` +
      `<div class="row"><span>Estimated cost</span><b>$${(o.estimated_cost_usd || 0).toFixed(4)}</b></div>` +
      `<div class="row"><span>Errors</span><b>${o.errors}</b></div>` +
      (remote.length ? `<div class="sub">remote available: ${remote.join(', ')}</div>` : '<div class="sub">no remote API keys in environment (local/fallback mode)</div>');
  } catch (e) {}
}

// poll controls + indicators lightly (SSE covers the heavy lifting)
setInterval(loadControls, 20000);
setInterval(loadLLMUsage, 30000);
setInterval(async () => {
  try {
    const data = await api('/api/v2/indicators');
    const symbols = Object.keys(data.indicators || {});
    if (!symbols.length) { $('indicator-table').textContent = 'no data yet'; return; }
    $('indicator-table').innerHTML = symbols.map(sym => {
      const l = data.indicators[sym].latest || {};
      return `<div class="row"><span><b>${sym}</b></span>
        <span class="mono">px ${fmt(l.price)} | rsi ${fmt(l.rsi_14)} | sma20 ${fmt(l.sma_20)} | vol ${fmt(l.volatility_20d)}</span></div>`;
    }).join('');
  } catch (e) {}
}, 10000);

// boot
connectStream();
loadStateOnce();
loadControls();
loadLLMUsage();
setInterval(loadStateOnce, 30000);
window.addEventListener('resize', () => { if (lastState) renderState(lastState); });
</script>
</body>
</html>
"""
