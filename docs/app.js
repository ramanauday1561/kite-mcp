/* NIFTY 50 F&O Signal dashboard.
   Zero dependencies: it reads the JSON the pipeline publishes and draws
   everything, charts included, with inline SVG. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);

/* ------------------------------------------------------------- formatting */
const num = (v, d = 2) =>
  (v === null || v === undefined || Number.isNaN(v)) ? '—'
    : Number(v).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
const int = (v) => (v === null || v === undefined) ? '—' : Number(v).toLocaleString('en-IN');
const pct = (v, d = 1) => (v === null || v === undefined) ? '—' : `${Number(v).toFixed(d)}%`;
const signed = (v, d = 2) => (v === null || v === undefined) ? '—'
  : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}`;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function ago(iso) {
  if (!iso) return null;
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
  if (!Number.isFinite(mins)) return null;
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ${mins % 60}m ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

const REGIME_LABEL = { low_vol: 'Low volatility', mid_vol: 'Normal volatility', high_vol: 'High volatility' };
const cls = (v) => v > 0 ? 'up' : v < 0 ? 'down' : '';

/* ------------------------------------------------------------------ theme */
const savedTheme = (() => { try { return localStorage.getItem('theme'); } catch { return null; } })();
if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);
$('#themeBtn').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('theme', next); } catch { /* private mode */ }
});
$('#refreshBtn').addEventListener('click', () => load(true));

/* ------------------------------------------------------------------- data */
async function fetchJSON(path) {
  const res = await fetch(`${path}?t=${Date.now()}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

async function load(manual = false) {
  if (manual) $('#app').innerHTML =
    '<div class="card" style="margin-top:18px"><div class="empty"><span class="spin"></span></div></div>';
  try {
    const latest = await fetchJSON('data/latest.json');
    let history = [];
    try { history = await fetchJSON('data/history.json'); } catch { /* optional */ }
    render(latest, Array.isArray(history) ? history : []);
  } catch (err) {
    $('#app').innerHTML = $('#tpl-empty').innerHTML;
    $('#dataSource').textContent = 'no data';
    $('#freshness').textContent = '';
    console.warn('Could not load signal data:', err);
  }
}

/* ----------------------------------------------------------------- render */
function render(d, history) {
  const rec = d.recommendation || {};
  const action = rec.action || 'NO_TRADE';
  const confidence = (rec.confidence ?? 0) * 100;
  const probUp = (rec.probability_up ?? 0.5) * 100;

  $('#dataSource').textContent = `source: ${d.market_data?.source ?? 'unknown'}`;
  const chainOK = d.data_quality?.option_chain_available;
  $('#dataSource').className = `pill ${chainOK ? 'live' : 'stale'}`;

  // Say plainly how old the data is. The page is static: refreshing re-reads
  // the published file, it does not recompute a signal.
  const age = ago(d.generated_at);
  const ageMins = d.generated_at
    ? (Date.now() - new Date(d.generated_at).getTime()) / 60000 : Infinity;
  $('#freshness').textContent = age ? `updated ${age}` : (d.generated_at_ist ?? '');
  $('#freshness').className = `pill ${ageMins > 180 ? 'stale' : 'live'}`;
  $('#freshness').title = d.generated_at_ist ?? '';

  $('#app').innerHTML = [
    freshnessBar(d),
    verdictCard(d, rec, action, confidence, probUp),
    `<div class="grid cols-2">${tradeCard(d, action)}${marketCard(d)}</div>`,
    `<div class="grid">${strategiesCard(d)}</div>`,
    `<div class="grid cols-2">${performanceCard(d)}${learningCard(d)}</div>`,
    historyCard(history),
    qualityCard(d),
    `<div class="disclaimer"><strong>Not investment advice.</strong> ${esc(d.disclaimer || '')}</div>`,
    footer(d),
  ].join('');
}

function freshnessBar(d) {
  const age = ago(d.generated_at);
  const next = d.next_update?.label;
  const stored = d.storage?.total_archived;
  return `<div class="card freshbar">
    <div>
      <strong>Published ${esc(d.generated_at_ist || '')}</strong>
      ${age ? `<span class="dim"> · ${esc(age)}</span>` : ''}
    </div>
    <div class="dim">
      ${next ? `Next signal due: <strong>${esc(next)}</strong> <span title="GitHub runs scheduled jobs on shared infrastructure, so a run can land late.">(approx)</span>` : ''}
    </div>
    <div class="dim">
      ${stored ? `${int(stored)} signals archived · <a href="data/signals.csv">CSV</a>` : ''}
    </div>
  </div>
  <div class="note refresh-note">
    This page is generated on a schedule, not on page load. Refreshing re-reads
    the latest published file — it does not recompute a signal. New data appears
    pre-open, roughly every 30 minutes through the session, and after the close.
    Scheduled runs are queued on GitHub's shared infrastructure and can arrive
    late or be skipped under load, so treat the time above as the real answer
    rather than assuming a fixed clock.
  </div>`;
}

function verdictCard(d, rec, action, confidence, probUp) {
  const color = action === 'CALL' ? 'var(--call)' : action === 'PUT' ? 'var(--put)' : 'var(--hold)';
  const label = action === 'NO_TRADE' ? 'STAND ASIDE' : action;
  const sub = action === 'NO_TRADE'
    ? `bias leans ${esc(rec.bias || '—')}`
    : `buy ${action === 'CALL' ? 'call' : 'put'} option`;
  const change = d.change_pct;

  return `<div class="card" style="margin-top:18px">
    <h2>Next session verdict · ${esc(d.session_date || '')}</h2>
    <div class="verdict">
      <div class="verdict-badge badge-${esc(action)}">
        <div class="action">${esc(label)}</div>
        <div class="sub">${esc(sub)}</div>
      </div>
      <div class="verdict-body">
        <p class="verdict-reason">${esc(rec.reason || '')}</p>
        ${gauge('Confidence', confidence, color)}
        ${gauge('Probability of an up move', probUp,
          probUp >= 50 ? 'var(--call)' : 'var(--put)')}
        <div class="stat-row" style="margin-top:14px">
          <div class="stat"><div class="k">Spot</div>
            <div class="v">${num(d.spot)}</div></div>
          <div class="stat"><div class="k">Change</div>
            <div class="v ${cls(change)}">${signed(change)}%</div></div>
          <div class="stat"><div class="k">Ensemble score</div>
            <div class="v ${cls(d.ensemble?.score)}">${signed(d.ensemble?.score, 3)}</div></div>
          <div class="stat"><div class="k">Regime</div>
            <div class="v" style="font-size:0.92rem">${esc(REGIME_LABEL[d.regime] || d.regime || '—')}</div></div>
          <div class="stat"><div class="k">Votes</div>
            <div class="v" style="font-size:0.92rem">${int(d.ensemble?.bullish_votes)}▲ ${int(d.ensemble?.bearish_votes)}▼</div></div>
        </div>
        <p style="margin-top:11px;font-size:0.77rem;color:var(--text-faint)">
          Calibrated on ${esc(rec.calibration_basis || 'the model prior')}.</p>
      </div>
    </div>
  </div>`;
}

function gauge(label, value, color) {
  const width = Math.max(0, Math.min(100, value));
  return `<div class="gauge">
    <div class="gauge-label"><span>${esc(label)}</span><span>${value.toFixed(0)}%</span></div>
    <div class="gauge-track"><div class="gauge-fill"
      style="width:${width}%;background:${color}"></div></div>
  </div>`;
}

function tradeCard(d, action) {
  const t = d.trade;
  if (!t) return `<div class="card"><h2>Trade plan</h2>
    <div class="empty">No trade plan for this session.</div></div>`;

  const standAside = action === 'NO_TRADE';
  const spread = t.structure === 'debit_spread';
  const rows = [
    ['Instrument', `NIFTY ${int(t.strike)} ${t.option_type}`],
    ['Expiry', `${esc(t.expiry)} (${int(t.days_to_expiry)}d)`],
    ['Structure', spread ? `Debit spread vs ${int(t.short_leg?.strike)}` : 'Long option'],
    ['Theoretical premium', `₹${num(t.theoretical_premium)}`],
    spread ? ['Net debit', `₹${num(t.net_debit)}`] : null,
    spread ? ['Max profit', `₹${num(t.max_profit)}`] : null,
    ['Stop loss', `₹${num(t.stop_loss)}`],
    ['Target', `₹${num(t.target)}`],
    ['Breakeven', int(t.breakeven)],
    ['Delta / Theta', `${num(t.delta, 2)} / ${num(t.theta_per_day, 1)}`],
    ['Suggested size', `${int(t.suggested_lots)} lot(s) × ${int(t.lot_size)}`],
    ['Premium outlay', `₹${int(t.premium_outlay)}`],
    ['Capital at risk', `₹${int(t.capital_at_risk)}`],
  ].filter(Boolean);

  return `<div class="card">
    <h2>Trade plan ${standAside ? '— reference only' : ''}</h2>
    ${standAside ? `<div class="note" style="margin-bottom:12px">
      The model is standing aside. This plan is shown only to illustrate what the
      ${esc(d.recommendation?.bias)} bias would look like if it strengthened.</div>` : ''}
    <dl class="kv">
      ${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}
    </dl>
    <div class="notes">${(t.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join('')}</div>
  </div>`;
}

function marketCard(d) {
  const m = d.market_data || {};
  const c = d.option_chain;
  const metric = (k, v) => `<div class="metric"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`;

  return `<div class="card">
    <h2>Market context</h2>
    <div class="metric-grid">
      ${metric('India VIX', num(m.vix))}
      ${metric('VIX change', `<span class="${cls(-(m.vix_change_pct ?? 0))}">${signed(m.vix_change_pct)}%</span>`)}
      ${metric('RSI (14)', num(m.rsi, 1))}
      ${metric('ADX', num(m.adx, 1))}
      ${metric('ATR', pct(m.atr_pct))}
      ${metric('Realised vol', pct(m.realized_vol))}
    </div>
    ${c ? `<h2 style="margin-top:18px">Option chain</h2>
    <div class="metric-grid">
      ${metric('PCR (OI)', num(c.pcr_oi))}
      ${metric('Max pain', int(c.max_pain))}
      ${metric('ATM IV', c.atm_iv ? pct(c.atm_iv) : '—')}
      ${metric('Support', int(c.support_strike))}
      ${metric('Resistance', int(c.resistance_strike))}
    </div>` : `<div class="note" style="margin-top:16px">
      Live option chain was not reachable this run, so the PCR and max-pain
      voters abstained. NSE blocks most datacenter IPs — the technical and LLM
      voters carry the signal instead.</div>`}
  </div>`;
}

function strategiesCard(d) {
  const votes = d.votes || [];
  if (!votes.length) return '';
  const maxAbs = Math.max(0.02, ...votes.map((v) => Math.abs(v.score)));

  const rows = votes.map((v) => {
    const width = Math.abs(v.score) / maxAbs * 50;
    // A vote of ~0 is genuinely neutral, so it must not be painted as bearish.
    const color = v.abstained || Math.abs(v.score) < 0.005 ? 'var(--text-faint)'
      : v.score > 0 ? 'var(--call)' : 'var(--put)';
    const side = v.score >= 0 ? `left:50%;width:${width}%` : `right:50%;width:${width}%`;
    const hit = (v.hit_rate !== null && v.hit_rate !== undefined && v.sample > 0)
      ? `hit ${(v.hit_rate * 100).toFixed(0)}% · n=${Number(v.sample).toFixed(0)}`
      : 'no track record yet';
    return `<div class="strat-row">
        <div class="strat-name ${v.abstained ? 'abstained' : ''}">${esc(v.label)}
          <small>weight ${((v.weight ?? 0) * 100).toFixed(1)}% · ${esc(hit)}</small></div>
        <div class="bar-track"><div class="bar-mid"></div>
          <div class="bar-fill" style="${side};background:${color}"></div></div>
        <div class="strat-score" style="color:${color}">
          ${v.abstained ? '—' : signed(v.score, 2)}</div>
      </div>
      <div class="strat-detail">${esc(v.rationale)}</div>`;
  }).join('');

  return `<div class="card">
    <h2>How each strategy voted · ${int(d.ensemble?.active)} active,
      ${pct((d.ensemble?.agreement ?? 0) * 100, 0)} agreement</h2>
    <div class="strat">${rows}</div>
    <p style="margin-top:12px;font-size:0.77rem;color:var(--text-faint)">
      Bars point right for a CALL vote and left for a PUT vote. Bar length is the
      raw conviction; the <em>weight</em> under each name is how much of the final
      verdict that strategy currently controls — earned from its own hit rate.</p>
  </div>`;
}

function performanceCard(d) {
  const p = d.performance || {};
  if (!p.scored) return `<div class="card"><h2>Track record</h2>
    <div class="empty"><p>No resolved signals yet.</p>
    <p style="font-size:0.82rem;margin-top:8px">Accuracy appears once predictions
    have been checked against what the market actually did.</p></div></div>`;

  const metric = (k, v, extra = '') => `<div class="metric"><div class="k">${esc(k)}</div>
    <div class="v" ${extra}>${v}</div></div>`;
  const acc = p.accuracy_all ?? 0;

  return `<div class="card">
    <h2>Track record · ${int(p.scored)} scored signals</h2>
    <div class="metric-grid">
      ${metric('Accuracy', pct(p.accuracy_all), `class="v ${acc >= 55 ? 'up' : acc < 48 ? 'down' : ''}"`)}
      ${metric('Last 20', pct(p.accuracy_last_20))}
      ${metric('Last 50', pct(p.accuracy_last_50))}
      ${metric('Traded only', pct(p.accuracy_traded_only))}
      ${metric('Streak', `${int(p.current_streak)} / ${int(p.best_streak)}`)}
      ${metric('Win rate', pct(p.win_rate_pct))}
    </div>
    ${p.equity_curve && p.equity_curve.length > 2 ? `
      <h2 style="margin-top:18px">Simulated account curve</h2>
      ${equityChart(p.equity_curve)}
      <div class="metric-grid" style="margin-top:12px">
        ${metric('Cumulative', pct(p.cumulative_pnl_pct),
          `class="v ${(p.cumulative_pnl_pct ?? 0) >= 0 ? 'up' : 'down'}"`)}
        ${metric('Avg / trade', pct(p.avg_account_return_pct, 2))}
        ${metric('Max drawdown', pct(p.max_drawdown_pct), 'class="v down"')}
      </div>
      <p style="margin-top:10px;font-size:0.75rem;color:var(--text-faint)">
        Options repriced with Black-Scholes one session later, sized by the risk
        rule in <code>config.json</code>. Excludes brokerage, slippage and the
        bid-ask spread, so live results would be lower.</p>` : ''}
  </div>`;
}

function equityChart(curve) {
  const W = 560, H = 150, pad = { l: 42, r: 10, t: 10, b: 20 };
  const values = curve.map((p) => p.cumulative_pct);
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  if (hi - lo < 1) { hi += 1; lo -= 1; }
  const x = (i) => pad.l + i / Math.max(1, curve.length - 1) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (hi - v) / (hi - lo) * (H - pad.t - pad.b);

  const line = curve.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.cumulative_pct).toFixed(1)}`).join('');
  const area = `${line}L${x(curve.length - 1).toFixed(1)},${y(0).toFixed(1)}L${x(0).toFixed(1)},${y(0).toFixed(1)}Z`;
  const positive = values[values.length - 1] >= 0;
  const color = positive ? 'var(--call)' : 'var(--put)';

  const ticks = [hi, (hi + lo) / 2, lo].map((v) => `
    <line x1="${pad.l}" y1="${y(v).toFixed(1)}" x2="${W - pad.r}" y2="${y(v).toFixed(1)}"
      stroke="var(--border)" stroke-width="1" ${Math.abs(v) < 1e-9 ? '' : 'stroke-dasharray="2,3"'}/>
    <text x="${pad.l - 6}" y="${(y(v) + 3.5).toFixed(1)}" text-anchor="end"
      font-size="9" fill="var(--text-faint)" font-family="ui-monospace,monospace">${v.toFixed(0)}%</text>`).join('');

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Simulated account equity curve, currently ${values[values.length - 1].toFixed(1)} percent">
    <defs><linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity="0.28"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
    </linearGradient></defs>
    ${ticks}
    <path d="${area}" fill="url(#eq)"/>
    <path d="${line}" fill="none" stroke="${color}" stroke-width="1.8"
      stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
}

function learningCard(d) {
  const l = d.learning || {};
  const stats = l.strategy_stats || {};
  const ranked = Object.entries(stats)
    .filter(([, s]) => s.sample > 0)
    .sort((a, b) => b[1].weight - a[1].weight)
    .slice(0, 6);

  const llm = l.llm_enabled
    ? `<div class="note">Open-source LLM panel active: ${(l.llm_models || [])
        .map((m) => esc(m.replace(/^llm:/, ''))).join(', ') || 'none'}. Each model is
        scored like any other voter and keeps only the weight its record earns.</div>`
    : `<div class="note">LLM panel is off. Add an <code>LLM_API_KEY</code> repository
        secret (Groq and OpenRouter both have free tiers) to add open-source model
        voters such as Llama&nbsp;3.3&nbsp;70B or DeepSeek&nbsp;R1.</div>`;

  return `<div class="card">
    <h2>What the model has learned</h2>
    <div class="metric-grid">
      <div class="metric"><div class="k">Runs</div><div class="v">${int(l.runs)}</div></div>
      <div class="metric"><div class="k">Resolved</div><div class="v">${int(l.resolved)}</div></div>
      <div class="metric"><div class="k">This run</div><div class="v">${int(l.resolved_this_run)}</div></div>
    </div>
    ${ranked.length ? `<div class="scroll" style="margin-top:16px"><table>
      <thead><tr><th>Top weighted strategy</th><th>Weight</th><th>Hit rate</th><th>Sample</th></tr></thead>
      <tbody>${ranked.map(([name, s]) => `<tr>
        <td style="font-family:var(--sans)">${esc(name.replace(/^llm:/, 'LLM · '))}</td>
        <td>${pct(s.weight * 100, 1)}</td>
        <td class="${s.hit_rate >= 0.55 ? 'up' : s.hit_rate < 0.45 ? 'down' : ''}">${pct(s.hit_rate * 100, 1)}</td>
        <td>${num(s.sample, 1)}</td></tr>`).join('')}</tbody></table></div>`
      : `<p style="margin-top:14px;font-size:0.83rem;color:var(--text-dim)">
         Weights are still uniform — the model has not resolved enough signals to
         separate the strategies yet.</p>`}
    <div class="notes" style="margin-top:14px">${llm}</div>
  </div>`;
}

function historyCard(history) {
  const resolved = history.filter((r) => r.outcome).slice(-40).reverse();
  if (!resolved.length) return '';

  return `<div class="card" style="margin-top:16px">
    <h2>Signal history · last ${resolved.length}</h2>
    <div class="scroll"><table>
      <thead><tr>
        <th>Session</th><th>Call</th><th>Conf.</th><th>Spot</th>
        <th>Next close</th><th>Move</th><th>Result</th><th>Option P&amp;L</th>
      </tr></thead>
      <tbody>${resolved.map((r) => {
        const o = r.outcome;
        const result = o.flat ? '<span class="tag-flat">flat</span>'
          : o.bias_correct ? '<span class="tag-hit">✔ hit</span>'
            : '<span class="tag-miss">✘ miss</span>';
        const pnl = o.option_pnl_pct === null || o.option_pnl_pct === undefined ? '—'
          : `<span class="${cls(o.option_pnl_pct)}">${signed(o.option_pnl_pct, 1)}%</span>`;
        return `<tr>
          <td>${esc(r.session_date)}</td>
          <td><span class="tag tag-${esc(r.action)}">${esc(r.action.replace('_', ' '))}</span></td>
          <td>${pct((r.confidence ?? 0) * 100, 0)}</td>
          <td>${num(r.spot, 0)}</td>
          <td>${num(o.exit_close, 0)}</td>
          <td class="${cls(o.move_pct)}">${signed(o.move_pct)}%</td>
          <td>${result}</td>
          <td>${pnl}</td></tr>`;
      }).join('')}</tbody>
    </table></div>
  </div>`;
}

function qualityCard(d) {
  const q = d.data_quality || {};
  const notes = q.notes || [];
  return `<div class="card" style="margin-top:16px">
    <h2>Data quality</h2>
    <div class="metric-grid">
      <div class="metric"><div class="k">Sessions</div><div class="v">${int(q.candles)}</div></div>
      <div class="metric"><div class="k">Price source</div>
        <div class="v" style="font-size:0.95rem">${esc(q.price_source || '—')}</div></div>
      <div class="metric"><div class="k">India VIX</div>
        <div class="v ${q.vix_available ? 'up' : 'down'}" style="font-size:0.95rem">
          ${q.vix_available ? 'available' : 'missing'}</div></div>
      <div class="metric"><div class="k">Option chain</div>
        <div class="v ${q.option_chain_available ? 'up' : 'down'}" style="font-size:0.95rem">
          ${q.option_chain_available ? 'live' : 'unavailable'}</div></div>
    </div>
    ${notes.length ? `<div class="notes" style="margin-top:14px">
      ${notes.map((n) => `<div class="note">${esc(n)}</div>`).join('')}</div>` : ''}
  </div>`;
}

function footer(d) {
  return `<footer>
    <span>Generated ${esc(d.generated_at_ist || '')}</span>
    <span>·</span>
    <a href="data/latest.json">latest.json</a>
    <a href="data/history.json">history.json</a>
    <a href="data/performance.json">performance.json</a>
    <span>·</span>
    <span>Updates every trading day, pre-open and post-close.</span>
  </footer>`;
}

load();
setInterval(() => load(false), 10 * 60 * 1000);   // pick up new publishes
