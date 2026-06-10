/* ─────────────────────────────────────────────
   GOEC Surveillance Dashboard — shared client auth
   ─────────────────────────────────────────────
   Client-side login gate for the static dashboard (served by nginx,
   see nginx.conf). This is NOT real security — credentials live in this
   file and the session is just a sessionStorage flag — it only keeps the
   station view behind a sign-in prompt for internal/demo use. If real
   auth is ever needed, move verification into the orchestration service
   (POST /login) and issue a token instead.

   Used by:
     - login.html    → Auth.login() / Auth.getUser()
     - station.html  → Auth.requireAuth() gate in <head>
*/
const Auth = (() => {
  const SESSION_KEY = 'goec_session';

  function login(username, password) {
    const VALID_USERS = { 'admin': 'goec@2026', 'operator': 'goec@2026' };
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

  // Redirect to the login page if there's no active session. Returns false
  // when redirecting so callers can short-circuit before rendering.
  function requireAuth() {
    if (!getUser()) {
      window.location.href = 'login.html';
      return false;
    }
    return true;
  }

  return { login, logout, getUser, requireAuth };
})();
