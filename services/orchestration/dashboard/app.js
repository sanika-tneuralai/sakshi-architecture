/* ═══════════════════════════════════════════════════════════
   GOEC Surveillance Dashboard — Shared JS
   ═══════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────
   CONFIG — Change URLs to match your deployment
───────────────────────────────────────────── */
const CONFIG = {
  orchestrationUrl:   'http://13.201.133.171:8000',
  alertUrl:           'http://13.201.133.171:8000',  // alerts served from orchestrate DB
  cameraDetectionUrl: 'http://100.123.244.59:8004',
  cameraId:           'camera_01',
  refreshIntervalMs:  12000,
  alertsLimit:        50,
};

/* ─────────────────────────────────────────────
   USECASE DEFINITIONS
───────────────────────────────────────────── */
const USECASES = [
  { id: 'parking_detection',  label: 'Parking Activity',    icon: '🅿️', color: '#d29922', page: 'parking_detection.html' },
  { id: 'gun_detection',      label: 'Charging Gun',        icon: '⚡', color: '#da3633', page: 'gun_detection.html' },
  { id: 'parking_compliance', label: 'Parking Violations',  icon: '🚫', color: '#388bfd', page: 'parking_compliance.html' },
  { id: 'safety_monitoring',  label: 'Safety Alerts',       icon: '🔥', color: '#da3633', page: 'safety_monitoring.html' },
];

/* ─────────────────────────────────────────────
   AUTH
───────────────────────────────────────────── */
const Auth = (() => {
  const SESSION_KEY = 'goec_session';

  function login(username, password) {
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
   PIPELINE STATUS (monitoring badge only)
───────────────────────────────────────────── */
async function getPipelineStatus() {
  return fetchJSON(`${CONFIG.orchestrationUrl}/pipeline/status/${CONFIG.cameraId}`);
}

function applyMonitoringBadge(badgeEl, pipeline) {
  if (!badgeEl) return;
  if (!pipeline) {
    badgeEl.className = 'badge inactive';
    badgeEl.textContent = 'Offline';
  } else if (pipeline.running) {
    badgeEl.className = 'badge active';
    badgeEl.textContent = 'Monitoring';
  } else {
    badgeEl.className = 'badge inactive';
    badgeEl.textContent = 'Not Running';
  }
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

function timeAgo(ts) {
  if (!ts) return '—';
  const diff = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (diff < 10)  return 'just now';
  if (diff < 60)  return `${diff} seconds ago`;
  if (diff < 120) return '1 minute ago';
  if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
  if (diff < 7200) return '1 hour ago';
  if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
  if (diff < 172800) return 'yesterday';
  return `${Math.floor(diff / 86400)} days ago`;
}

function setLastRefresh() {
  const el = document.getElementById('last-refresh');
  if (el) el.textContent = new Date().toLocaleTimeString();
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function extractAlerts(data) {
  if (!data) return null;
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.alerts)) return data.alerts;
  return null;
}

function renderSnapshot(containerId, alerts) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const withSnap = (alerts || []).find(a => a.snapshot_b64);
  if (withSnap) {
    el.innerHTML = `
      <img src="data:image/jpeg;base64,${withSnap.snapshot_b64}" alt="Latest frame" />
      <div class="frame-label">${fmtTsFull(withSnap.timestamp)}</div>
    `;
  } else {
    el.innerHTML = `<div class="frame-empty"><div class="frame-icon">📷</div>No snapshot available</div>`;
  }
}

/* ─────────────────────────────────────────────
   SIDEBAR HTML TEMPLATE
───────────────────────────────────────────── */
function injectSidebarHTML(activePage) {
  const nav = [
    { id: 'home',               label: 'Overview',            icon: '🏠', href: 'home.html' },
    { id: 'parking_detection',  label: 'Parking Activity',    icon: '🅿️', href: 'parking_detection.html' },
    { id: 'gun_detection',      label: 'Charging Gun',        icon: '⚡', href: 'gun_detection.html' },
    { id: 'parking_compliance', label: 'Parking Violations',  icon: '🚫', href: 'parking_compliance.html' },
    { id: 'safety_monitoring',  label: 'Safety Alerts',       icon: '🔥', href: 'safety_monitoring.html' },
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
        ${nav.map(n => `
          <a href="${n.href}" class="nav-item ${activePage === n.id ? 'active' : ''}">
            <span class="nav-icon">${n.icon}</span>
            <span class="nav-label">${n.label}</span>
          </a>
        `).join('')}
      </nav>
      <div class="sidebar-bottom">
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
