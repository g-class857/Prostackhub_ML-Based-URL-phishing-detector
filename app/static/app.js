const form = document.getElementById('scan-form');
const input = document.getElementById('url-input');
const btn = document.getElementById('scan-btn');
const resultArea = document.getElementById('result-area');
const errorPanel = document.getElementById('error-panel');
const gaugeFill = document.getElementById('gauge-fill');
const gaugeNeedle = document.getElementById('gauge-needle');
const gaugePercent = document.getElementById('gauge-percent');
const verdictBadge = document.getElementById('verdict-badge');
const verdictUrl = document.getElementById('verdict-url');
const verdictDesc = document.getElementById('verdict-desc');
const logList = document.getElementById('log-list');

const GAUGE_CIRCUMFERENCE = 283; // matches stroke-dasharray in CSS

document.querySelectorAll('.hint-fill').forEach(b => {
  b.addEventListener('click', () => { input.value = b.dataset.fill; });
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const url = input.value.trim();
  if (!url) return;

  btn.disabled = true;
  btn.textContent = 'SCANNING…';
  errorPanel.hidden = true;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();

    if (!res.ok) {
      errorPanel.textContent = 'ERROR: ' + (data.error || 'Something went wrong.');
      errorPanel.hidden = false;
      resultArea.hidden = true;
      return;
    }
    renderResult(data);
  } catch (err) {
    errorPanel.textContent = 'ERROR: could not reach the detection service.';
    errorPanel.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = 'RUN SCAN';
  }
});

function renderResult(data) {
  resultArea.hidden = false;
  const pct = Math.round(data.phishing_probability * 100);
  const isPhishing = data.label === 'Phishing';

  // Gauge fill + needle rotation (0% = -90deg, 100% = +90deg across the arc)
  const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * data.phishing_probability);
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = isPhishing ? 'var(--accent-danger)' : 'var(--accent-safe)';
    const angle = -90 + (180 * data.phishing_probability);
    gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  });
  gaugePercent.textContent = pct + '%';

  verdictBadge.textContent = isPhishing ? '⚠ LIKELY PHISHING' : '✓ LIKELY LEGITIMATE';
  verdictBadge.className = 'verdict-badge ' + (isPhishing ? 'danger' : 'safe');
  verdictUrl.textContent = data.url;
  verdictDesc.textContent = isPhishing
    ? `This URL shows structural patterns commonly seen in phishing links (${pct}% phishing likelihood). Avoid entering credentials or personal information.`
    : `This URL's structure looks consistent with legitimate sites (${pct}% phishing likelihood). Always stay alert for context — this is a statistical estimate, not a guarantee.`;

  logList.innerHTML = '';
  data.top_factors.forEach(factor => {
    const li = document.createElement('li');
    const isRisk = looksRisky(factor);
    li.innerHTML = `
      <span class="log-flag ${isRisk ? 'risk' : 'ok'}">${isRisk ? '✕' : '✓'}</span>
      <div class="log-body">
        <span class="log-feature">${factor.feature.toUpperCase()} = ${factor.value} · weight ${factor.importance}</span>
        <span class="log-reason">${factor.reason || 'Contributes to the model\'s overall score.'}</span>
      </div>
    `;
    logList.appendChild(li);
  });

  resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function looksRisky(factor) {
  const riskyOnOne = ['has_at_symbol', 'has_ip_address', 'is_shortened', 'has_suspicious_words', 'has_double_slash_redirect'];
  if (riskyOnOne.includes(factor.feature) && factor.value === 1) return true;
  if (factor.feature === 'has_https' && factor.value === 0) return true;
  if (factor.feature === 'domain_age_days' && factor.value >= 0 && factor.value < 180) return true;
  return false;
}

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    const results = data.results || {};
    const selected = results.selected_model;
    const best = results[selected];
    document.getElementById('model-name').textContent = selected || 'untrained';

    const grid = document.getElementById('status-grid');
    if (!best) {
      grid.innerHTML = '<div class="status-cell"><div class="val">—</div><div class="lbl">run src/train_model.py to populate metrics</div></div>';
      return;
    }
    const cells = [
      ['Accuracy', best.accuracy],
      ['Precision', best.precision],
      ['Recall', best.recall],
      ['F1-score', best.f1_score],
      ['ROC-AUC', best.roc_auc],
    ];
    grid.innerHTML = cells.map(([lbl, val]) => `
      <div class="status-cell">
        <div class="val">${(val * 100).toFixed(1)}%</div>
        <div class="lbl">${lbl}</div>
      </div>`).join('');
  } catch (e) {
    // metrics are optional — fail silently, form still works
  }
}

loadMetrics();
