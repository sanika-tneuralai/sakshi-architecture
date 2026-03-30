/* ═══════════════════════════════════════════════════════════
   GOEC Surveillance Dashboard — Shared JS
   ═══════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────
   CONFIG — Change URLs to match your EC2 setup
───────────────────────────────────────────── */
const CONFIG = {
  orchestrationUrl:   'http://192.168.1.75:8000',
  usecaseUrl:         'http://3.6.160.230:8001',
  alertUrl:           'http://192.168.1.75:8000',  // alerts served from orchestrate DB
  cameraDetectionUrl: 'http://100.123.244.59:8000',
  cameraId:           'camera_01',
  refreshIntervalMs:  12000,
  alertsLimit:        20,
};

/* ─────────────────────────────────────────────
   USECASE DEFINITIONS
───────────────────────────────────────────── */
const USECASES = [
  { id: 'parking_detection',  label: 'Parking Detection',  icon: '🅿️',  color: '#d29922', page: 'parking_detection.html' },
  { id: 'gun_detection',      label: 'Gun Detection',      icon: '🔫',  color: '#da3633', page: 'gun_detection.html' },
  { id: 'parking_compliance', label: 'Parking Compliance', icon: '🚗',  color: '#388bfd', page: 'parking_compliance.html' },
  { id: 'safety_monitoring',  label: 'Safety Monitoring',  icon: '⚠️',  color: '#238636', page: 'safety_monitoring.html' },
];

/* ─────────────────────────────────────────────
   AUTH HELPERS
───────────────────────────────────────────── */
const Auth = (() => {
  const SESSION_KEY = 'goec_session';

  function login(username, password) {
    // Client-side credential check — replace with a real API call if needed
    const VALID_USERS = { 'admin': 'goec@2024', 'operator': 'goec@2024' };
    if (VALID_USERS[username] === password) {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({ username, loginAt: Date.now() }));
      return true;
    }
    return false;
  }

  function logout() {
    sessionStorage.removeItem(SESSION_KEY);
    window.location.href = 'login.html';
  }

  function getUser() {
    const raw = sessionStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  }

  function requireAuth() {
    if (!getUser()) {
      window.location.href = 'login.html';
      return false;
    }
    return true;
  }

  return { login, logout, getUser, requireAuth };
})();

/* ─────────────────────────────────────────────
   FETCH HELPER
───────────────────────────────────────────── */
async function fetchJSON(url, fallback = null) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch {
    return fallback;
  }
}

/* ─────────────────────────────────────────────
   TOAST
───────────────────────────────────────────── */
const Toast = (() => {
  let container;

  function getContainer() {
    if (!container) {
      container = document.getElementById('toast-container');
      if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
      }
    }
    return container;
  }

  function show({ title, message = '', type = 'info', duration = 4000 }) {
    const icons = { info: 'ℹ️', success: '✅', warn: '⚠️', danger: '🚨' };
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `
      <div class="toast-icon">${icons[type] || 'ℹ️'}</div>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        ${message ? `<div class="toast-msg">${message}</div>` : ''}
      </div>
      <button class="toast-close" onclick="this.closest('.toast').remove()">×</button>
    `;
    getContainer().appendChild(el);
    if (duration > 0) {
      setTimeout(() => {
        el.classList.add('fade-out');
        setTimeout(() => el.remove(), 350);
      }, duration);
    }
  }

  return { show };
})();

/* ─────────────────────────────────────────────
   MODAL
───────────────────────────────────────────── */
const Modal = (() => {
  function open(alert) {
    const overlay = document.getElementById('modal-overlay');
    if (!overlay) return;

    document.getElementById('modal-title').textContent  = alert.alert_type || 'Alert Detail';
    document.getElementById('m-usecase').textContent    = alert.usecase_name || '—';
    document.getElementById('m-type').textContent       = alert.alert_type || '—';
    document.getElementById('m-ts').textContent         = alert.timestamp ? new Date(alert.timestamp).toLocaleString() : '—';
    document.getElementById('m-status').textContent     = alert.status || '—';
    document.getElementById('m-camera').textContent     = alert.camera_id || '—';

    const snapEl = document.getElementById('modal-snapshot');
    if (alert.snapshot_b64) {
      snapEl.src = `data:image/jpeg;base64,${alert.snapshot_b64}`;
      snapEl.style.display = 'block';
    } else {
      snapEl.style.display = 'none';
    }

    const extrasSection = document.getElementById('m-extras-section');
    if (alert.extras && Object.keys(alert.extras).length > 0) {
      document.getElementById('m-extras').textContent = JSON.stringify(alert.extras, null, 2);
      extrasSection.style.display = 'block';
    } else {
      extrasSection.style.display = 'none';
    }

    overlay.classList.add('open');
  }

  function close() {
    const overlay = document.getElementById('modal-overlay');
    if (overlay) overlay.classList.remove('open');
  }

  return { open, close };
})();

/* ─────────────────────────────────────────────
   SIDEBAR COLLAPSE
───────────────────────────────────────────── */
function initSidebar() {
  const sidebar   = document.querySelector('.sidebar');
  const mainArea  = document.querySelector('.main-area');
  const toggleBtn = document.getElementById('sidebar-toggle');
  if (!sidebar || !toggleBtn) return;

  const collapsed = localStorage.getItem('sidebar_collapsed') === 'true';
  if (collapsed) {
    sidebar.classList.add('collapsed');
    if (mainArea) mainArea.classList.add('sidebar-collapsed');
  }

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    if (mainArea) mainArea.classList.toggle('sidebar-collapsed');
    localStorage.setItem('sidebar_collapsed', sidebar.classList.contains('collapsed'));
  });
}

/* ─────────────────────────────────────────────
   SERVICE STATUS DOTS (shared across pages)
───────────────────────────────────────────── */
function setDot(id, cls, label) {
  const dot = document.getElementById(id);
  if (dot) dot.className = `dot ${cls}`;
  const lbl = document.getElementById(id.replace('dot-', 'lbl-'));
  if (lbl && label) lbl.textContent = label;
}

async function refreshServiceStatus() {
  const [pipeline, usecaseQ] = await Promise.all([
    fetchJSON(`${CONFIG.orchestrationUrl}/pipeline/status/${CONFIG.cameraId}`),
    fetchJSON(`${CONFIG.usecaseUrl}/usecase/queue-status`),
  ]);

  if (pipeline === null) {
    setDot('dot-orch', 'red', 'Orchestration: Down');
  } else {
    setDot('dot-orch', pipeline.running ? 'green' : 'yellow',
      `Orchestration: ${pipeline.running ? 'Running' : 'Idle'}`);
  }

  if (usecaseQ === null) {
    setDot('dot-usecase', 'red', 'Usecase Service: Down');
    setDot('dot-worker',  'red', 'Workers: Unknown');
  } else {
    setDot('dot-usecase', 'green', 'Usecase Service: Up');
    const wok = (usecaseQ.active_workers ?? 0) > 0;
    setDot('dot-worker', wok ? 'green' : 'yellow',
      `Workers: ${usecaseQ.active_workers ?? 0} active`);
  }

  fetchJSON(`${CONFIG.cameraDetectionUrl}/detection/health`).then(h => {
    setDot('dot-cam', h ? 'green' : 'red',
      h ? 'Camera Detection: Up' : 'Camera Detection: Down');
  });

  return pipeline;
}

/* ─────────────────────────────────────────────
   HELPERS
───────────────────────────────────────────── */
function fmtTs(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return ts; }
}

function fmtTsFull(ts) {
  if (!ts) return '—';
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

function setLastRefresh() {
  const el = document.getElementById('last-refresh');
  if (el) el.textContent = new Date().toLocaleTimeString();
}

function showSpinner(containerId) {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
}

function showEmpty(containerId, message = 'No data available') {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<div class="no-alerts"><div class="no-alerts-icon">📭</div>${message}</div>`;
}

function showError(containerId, message = 'Unable to fetch data. Check service connection.') {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<div class="error-state">⚠️ ${message}</div>`;
}

/* ─────────────────────────────────────────────
   RENDER ALERT LIST (reusable)
───────────────────────────────────────────── */
function renderAlertList(containerId, alerts, prevAlerts) {
  const el = document.getElementById(containerId);
  if (!el) return;

  if (alerts === null) { showError(containerId); return; }
  if (alerts.length === 0) { showEmpty(containerId, 'No violations recorded'); return; }

  // Detect new alerts for toast
  if (prevAlerts && prevAlerts.length > 0) {
    const prevIds = new Set(prevAlerts.map(a => a.alert_id));
    const newOnes = alerts.filter(a => !prevIds.has(a.alert_id));
    newOnes.forEach(a => {
      Toast.show({ title: 'New Alert', message: `${a.usecase_name} — ${a.alert_type}`, type: 'warn' });
    });
  }

  el.innerHTML = alerts.map(a => `
    <div class="alert-item" onclick='Modal.open(${JSON.stringify(a)})'>
      <div class="alert-icon">🔔</div>
      <div class="alert-body">
        <div class="alert-type">${escHtml(a.alert_type || a.usecase_name || '—')}</div>
        <div class="alert-meta">${escHtml(a.camera_id || '')} &mdash; ${escHtml(a.status || '')}</div>
      </div>
      <div class="alert-ts">${fmtTs(a.timestamp)}</div>
    </div>
  `).join('');
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

/* ─────────────────────────────────────────────
   RENDER SNAPSHOT
───────────────────────────────────────────── */
function renderSnapshot(containerId, alerts) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const withSnap = (alerts || []).find(a => a.snapshot_b64);
  if (withSnap) {
    el.innerHTML = `
      <img src="data:image/jpeg;base64,${withSnap.snapshot_b64}" alt="Latest frame" />
      <div class="frame-label">${fmtTs(withSnap.timestamp)}</div>
    `;
  } else {
    el.innerHTML = `<div class="frame-empty"><div class="frame-icon">📷</div>No snapshot available</div>`;
  }
}

/* ─────────────────────────────────────────────
   MODAL HTML TEMPLATE (inject once per page)
───────────────────────────────────────────── */
function injectModalHTML() {
  if (document.getElementById('modal-overlay')) return;
  document.body.insertAdjacentHTML('beforeend', `
    <div id="modal-overlay" class="modal-overlay" onclick="if(event.target===this)Modal.close()">
      <div class="modal">
        <div class="modal-header">
          <h3 id="modal-title">Alert Detail</h3>
          <button class="modal-close" onclick="Modal.close()">×</button>
        </div>
        <div class="modal-body">
          <img id="modal-snapshot" class="modal-snapshot" alt="Alert Snapshot" />
          <div class="modal-field">
            <label>Camera</label>
            <div class="val" id="m-camera">—</div>
          </div>
          <div class="modal-field">
            <label>Usecase</label>
            <div class="val" id="m-usecase">—</div>
          </div>
          <div class="modal-field">
            <label>Alert Type</label>
            <div class="val" id="m-type">—</div>
          </div>
          <div class="modal-field">
            <label>Timestamp</label>
            <div class="val" id="m-ts">—</div>
          </div>
          <div class="modal-field">
            <label>Status</label>
            <div class="val" id="m-status">—</div>
          </div>
          <div class="modal-field" id="m-extras-section" style="display:none">
            <label>Extras</label>
            <pre class="extras" id="m-extras"></pre>
          </div>
        </div>
      </div>
    </div>
  `);
}

/* ─────────────────────────────────────────────
   SIDEBAR HTML TEMPLATE
───────────────────────────────────────────── */
function injectSidebarHTML(activePage) {
  const nav = [
    { id: 'home',               label: 'Home',               icon: '🏠', href: 'home.html' },
    { id: 'parking_detection',  label: 'Parking Detection',  icon: '🅿️', href: 'parking_detection.html' },
    { id: 'gun_detection',      label: 'Gun Detection',      icon: '🔫', href: 'gun_detection.html' },
    { id: 'parking_compliance', label: 'Parking Compliance', icon: '🚗', href: 'parking_compliance.html' },
    { id: 'safety_monitoring',  label: 'Safety Monitoring',  icon: '⚠️', href: 'safety_monitoring.html' },
    { id: 'analytics',          label: 'Analytics',          icon: '📊', href: 'analytics.html' },
  ];

  const user = Auth.getUser();

  const html = `
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-brand">
        <div class="brand-icon">⚡</div>
        <span class="brand-text">GO<span>EC</span></span>
        <button class="sidebar-toggle" id="sidebar-toggle" title="Toggle sidebar">&#9776;</button>
      </div>
      <nav class="sidebar-nav">
        <div class="nav-section-label">Navigation</div>
        ${nav.map(n => `
          <a href="${n.href}" class="nav-item ${activePage === n.id ? 'active' : ''}">
            <span class="nav-icon">${n.icon}</span>
            <span class="nav-label">${n.label}</span>
          </a>
        `).join('')}
      </nav>
      <div class="sidebar-bottom">
        <div class="nav-section-label">Account</div>
        <div class="nav-item" style="cursor:default; color:var(--muted)">
          <span class="nav-icon">👤</span>
          <span class="nav-label">${user ? escHtml(user.username) : 'Guest'}</span>
        </div>
        <button class="nav-item danger" onclick="Auth.logout()">
          <span class="nav-icon">🚪</span>
          <span class="nav-label">Logout</span>
        </button>
      </div>
    </aside>
  `;

  const target = document.getElementById('sidebar-slot');
  if (target) {
    target.outerHTML = html;
  } else {
    document.body.insertAdjacentHTML('afterbegin', html);
  }
}

/* ─────────────────────────────────────────────
   STATUS BAR HTML
───────────────────────────────────────────── */
function statusBarHTML() {
  return `
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px">
      <div class="status-chip"><div class="dot" id="dot-orch"></div><span id="lbl-orch">Orchestration</span></div>
      <div class="status-chip"><div class="dot" id="dot-usecase"></div><span id="lbl-usecase">Usecase Service</span></div>
      <div class="status-chip"><div class="dot" id="dot-alert"></div><span id="lbl-alert">Alert Service</span></div>
      <div class="status-chip"><div class="dot" id="dot-cam"></div><span id="lbl-cam">Camera Detection</span></div>
      <div class="status-chip"><div class="dot" id="dot-worker"></div><span id="lbl-worker">Celery Workers</span></div>
    </div>
  `;
}
