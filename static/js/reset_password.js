
// Pull token from URL query string
const urlParams = new URLSearchParams(window.location.search);
const TOKEN = urlParams.get('token') || '';

if (!TOKEN) {
  document.getElementById('main-section').innerHTML =
    '<p style="color:#ff4444;font-weight:700;">Invalid or missing reset token.<br>Please request a new password reset link.</p>';
}

function checkStrength(password) {
  const rLen = document.getElementById('r-len');
  const rUpper = document.getElementById('r-upper');
  const rSpecial = document.getElementById('r-special');
  const fill = document.getElementById('fill');
  const mtext = document.getElementById('mtext');

  const hasLen = password.length >= 8;
  const hasUpper = /[A-Z]/.test(password);
  const hasSpecial = /[^a-zA-Z0-9]/.test(password);

  rLen.classList.toggle('ok', hasLen);
  rUpper.classList.toggle('ok', hasUpper);
  rSpecial.classList.toggle('ok', hasSpecial);

  const score = [hasLen, hasUpper, hasSpecial].filter(Boolean).length;
  const cfg = [
    { w: '0%', bg: 'transparent', label: 'Enter a password' },
    { w: '33%', bg: '#ff4444', label: 'Weak' },
    { w: '66%', bg: '#ffaa00', label: 'Fair' },
    { w: '100%', bg: '#00ffcc', label: 'Strong ✓' },
  ][score];

  fill.style.width = cfg.w;
  fill.style.background = cfg.bg;
  mtext.textContent = cfg.label;
  mtext.style.color = cfg.bg === 'transparent' ? 'rgba(255,255,255,0.4)' : cfg.bg;
}

function doReset() {
  const newPwd = document.getElementById('new-password').value;
  const confPwd = document.getElementById('confirm-password').value;
  const errEl = document.getElementById('err-msg');
  const okEl = document.getElementById('ok-msg');
  const btn = document.getElementById('submit-btn');

  errEl.textContent = '';
  okEl.textContent = '';

  if (!newPwd || !confPwd) { errEl.textContent = 'Please fill in both fields.'; return; }
  if (newPwd !== confPwd) { errEl.textContent = 'Passwords do not match.'; return; }

  const hasLen = newPwd.length >= 8;
  const hasUpper = /[A-Z]/.test(newPwd);
  const hasSpecial = /[^a-zA-Z0-9]/.test(newPwd);
  if (!hasLen || !hasUpper || !hasSpecial) {
    errEl.textContent = 'Password does not meet all the requirements above.'; return;
  }

  btn.disabled = true;
  btn.textContent = 'Resetting…';

  fetch('/auth/reset_password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: TOKEN, new_password: newPwd })
  })
    .then(r => r.json())
    .then(data => {

      if (data.access_token) {

        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user_name', data.name);

        launchToast("NEW PASSWORD CREATED!");

        setTimeout(() => {
          window.location.href = "/";
        }, 1500);

      } else {
        errEl.textContent = data.detail || "Something went wrong.";
      }

    })

    .catch(() => {
      errEl.textContent = 'Network error. Please try again.';
      btn.disabled = false;
      btn.textContent = 'Reset Password';
    });
}
function launchToast(message) {
  const toast = document.createElement("div");
  toast.innerText = message;

  toast.style.position = "fixed";
  toast.style.top = "30px";
  toast.style.left = "50%";
  toast.style.transform = "translateX(-50%)";
  toast.style.padding = "14px 24px";
  toast.style.background = "#00ffff";
  toast.style.color = "#000";
  toast.style.fontWeight = "800";
  toast.style.borderRadius = "12px";
  toast.style.boxShadow = "0 0 20px rgba(0,255,255,0.5)";
  toast.style.zIndex = "9999";
  toast.style.opacity = "0";
  toast.style.transition = "opacity 0.3s ease";

  document.body.appendChild(toast);

  setTimeout(() => toast.style.opacity = "1", 50);

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 2000);
}

// Allow Enter key to submit
document.addEventListener('keydown', e => { if (e.key === 'Enter') doReset(); });
