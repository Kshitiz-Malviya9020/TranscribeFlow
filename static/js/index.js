
// ── state ──
let masterTranscript = "";
let masterSummary = "";
let isAutoScroll = false;
let currentDisplayTranscript = "";
let currentFilename = "";

const loadingMessages = [
  "Waking up the AI models...",
  "Listening closely to every frequency...",
  "Whisper AI is translating thoughts to text...",
  "Filtering out background noise...",
  "Extracting core intelligence...",
  "Polishing the final transcript...",
  "Synthesizing your report...",
  "Finalizing the linguistic structure..."
];

// ============================================================
// BOT TRACKING & STAR EYE
// ============================================================
document.addEventListener('mousemove', (e) => {
  document.querySelectorAll('.bot-eye').forEach(eye => {
    if (eye.classList.contains('star-mode')) return;
    const rect = eye.getBoundingClientRect();
    const eyeX = rect.left + rect.width / 2;
    const eyeY = rect.top + rect.height / 2;
    const angle = Math.atan2(e.clientY - eyeY, e.clientX - eyeX);
    const distance = Math.min(8, Math.hypot(e.clientX - eyeX, e.clientY - eyeY));
    eye.style.transform = `translate(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px)`;
  });
});

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.start-btn, .submit-btn').forEach(btn => {
    btn.addEventListener('mouseenter', () => {
      document.querySelectorAll('.bot-eye').forEach(e => { e.classList.add('star-mode'); e.style.transform = 'translate(0,0)'; });
    });
    btn.addEventListener('mouseleave', () => {
      document.querySelectorAll('.bot-eye').forEach(e => e.classList.remove('star-mode'));
    });
  });
});

// ============================================================
// MODAL HELPERS
// ============================================================
function openAuthModal() {
  const modal = document.getElementById('auth-modal');
  modal.style.display = 'flex';
  setTimeout(() => { modal.style.opacity = '1'; document.getElementById('auth-card-inner').style.transform = 'scale(1)'; }, 10);
}
function closeAuthModal() {
  const modal = document.getElementById('auth-modal');
  modal.style.opacity = '0';
  document.getElementById('auth-card-inner').style.transform = 'scale(0.9)';
  setTimeout(() => modal.style.display = 'none', 300);
}
function handleModalBgClick(e) {
  if (e.target === document.getElementById('auth-modal')) closeAuthModal();
}
function openAuth(mode) { openAuthModal(); toggleAuthMode(mode); }

function toggleAuthMode(mode) {
  document.getElementById('login-form').style.display = (mode === 'login') ? 'block' : 'none';
  document.getElementById('register-form').style.display = (mode === 'register') ? 'block' : 'none';
  document.getElementById('forgot-form').style.display = (mode === 'forgot') ? 'block' : 'none';
  clearErrors();
}
function switchToForgot() { toggleAuthMode('forgot'); }
function clearErrors() {
  ['login-error', 'reg-error', 'forgot-error', 'forgot-success'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = '';
  });
}

// ============================================================
// LOGIN TYPE TOGGLE (Email / Phone)
// ============================================================


// ============================================================
// PASSWORD STRENGTH METER
// ============================================================
function checkPasswordStrength(password) {
  const ruleLen = document.getElementById('rule-len');
  const ruleUpper = document.getElementById('rule-upper');
  const ruleSpecial = document.getElementById('rule-special');
  const fill = document.getElementById('pwd-meter-fill');
  const text = document.getElementById('pwd-meter-text');

  const hasLen = password.length >= 8;
  const hasUpper = /[A-Z]/.test(password);
  const hasSpecial = /[^a-zA-Z0-9]/.test(password);

  ruleLen.classList.toggle('ok', hasLen);
  ruleUpper.classList.toggle('ok', hasUpper);
  ruleSpecial.classList.toggle('ok', hasSpecial);

  const score = [hasLen, hasUpper, hasSpecial].filter(Boolean).length;
  const configs = {
    0: { w: '0%', bg: 'transparent', label: 'Enter a password' },
    1: { w: '33%', bg: '#ff4444', label: 'Weak' },
    2: { w: '66%', bg: '#ffaa00', label: 'Fair' },
    3: { w: '100%', bg: '#00ffcc', label: 'Strong ✓' },
  };
  const cfg = configs[score];
  fill.style.width = cfg.w;
  fill.style.background = cfg.bg;
  text.textContent = cfg.label;
  text.style.color = cfg.bg === 'transparent' ? 'rgba(255,255,255,0.4)' : cfg.bg;
}

// ============================================================
// AUTH API CALLS
// ============================================================
function checkLoginStatus() {
  const token = localStorage.getItem('access_token');
  const name = localStorage.getItem('user_name');
  if (token && name) {
    document.getElementById('login-btn').style.display = 'none';
    document.getElementById('signup-btn').style.display = 'none';
    document.getElementById('user-greeting').innerText = "HI, " + name;
    document.getElementById('user-greeting').style.display = 'block';
    document.getElementById('logout-btn').style.display = 'flex';
  }
}

let pendingEmail = "";

function registerUser() {
  const name = document.getElementById('reg-name').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value;
  const errEl = document.getElementById('reg-error');
  const btn = document.getElementById('reg-btn');

  errEl.textContent = '';

  if (!name || !email || !password) {
    errEl.textContent = "Name, email and password are required.";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Sending OTP...";

  fetch('/auth/send_otp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password })
  })
    .then(r => r.json())
    .then(data => {
      if (data.message) {
        pendingEmail = email;
        openOtpModal();
      } else {
        errEl.textContent = data.detail;
      }
    })
    .finally(() => {
      btn.disabled = false;
      btn.textContent = "Create Account";
    });
}

function loginUser() {
  const identifier = document.getElementById('login-identifier').value.trim();
  const password = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');

  errEl.textContent = '';
  if (!identifier || !password) { errEl.textContent = 'Please enter your credentials.'; return; }

  fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ identifier, password })
  })
    .then(r => r.json())
    .then(data => {
      if (data.access_token) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user_name', data.name);
        closeAuthModal();
        launchToast("Logged in successfully!");
        checkLoginStatus();
      } else {
        errEl.textContent = data.detail || "Login failed.";
      }
    })
    .catch(() => { errEl.textContent = "Network error. Please try again."; });
}

function sendForgotPassword() {
  const email = document.getElementById('forgot-email').value.trim();
  const errEl = document.getElementById('forgot-error');
  const succEl = document.getElementById('forgot-success');
  const btn = document.getElementById('forgot-btn');

  errEl.textContent = '';
  succEl.textContent = '';

  if (!email) {
    errEl.textContent = 'Please enter your email address.';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Sending…';

  fetch('/auth/forgot_password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email })
  })
    .then(async (response) => {

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong");
      }

      return data;
    })
    .then(data => {

      succEl.textContent = data.message;

      btn.disabled = false;
      btn.textContent = "Send Reset Link";

    })
    .catch(error => {

      errEl.textContent = error.message;

      btn.disabled = false;
      btn.textContent = "Send Reset Link";

    });
}


function logout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_name');
  location.reload();
}

// ============================================================
// EXISTING FUNCTIONALITY (unchanged)
// ============================================================
function confirmPurge() {
  Swal.fire({
    title: 'Purge All History?',
    text: "This will permanently delete all recordings and transcripts!",
    icon: 'warning',
    showCancelButton: true,
    confirmButtonColor: '#ff8c00',
    cancelButtonColor: '#3085d6',
    confirmButtonText: 'Yes, Purge All',
    background: '#1a1a1a',
    color: '#fff',
    iconColor: '#ff8c00'
  }).then((result) => {
    if (result.isConfirmed) {

      const token = localStorage.getItem('access_token');

      fetch('/clear_all', {
        method: 'POST',
        headers: token ? { 'Authorization': 'Bearer ' + token } : {}
      })

        .then(res => res.json())
        .then(data => {
          if (data.success) {

            // Remove all history items visually
            const container = document.querySelector('.file-list-container');
            container.innerHTML = `
            <p style="text-align: center; opacity: 0.4;">
              No history found.
            </p>
          `;

            launchToast("HISTORY PURGED", "delete");

          } else {
            Swal.fire("Error", "Could not purge files.", "error");
          }
        });
    }
  });
}


function confirmDelete(filename, elementId) {
  Swal.fire({
    title: 'Delete File?', text: `You are about to delete ${filename}`,
    icon: 'warning', showCancelButton: true,
    confirmButtonColor: '#ff3c3c', cancelButtonColor: '#3085d6',
    confirmButtonText: 'Yes, delete it!',
    background: '#1a1a1a', color: '#fff', iconColor: '#ff3c3c'
  }).then((result) => {
    const token = localStorage.getItem('access_token');
    fetch(`/delete/${filename}`, {
      method: 'POST',
      headers: token ? { 'Authorization': 'Bearer ' + token } : {}
    })
      .then(res => {
        if (res.ok) {
          const el = document.getElementById(elementId);
          if (el) el.style.display = 'none';
          launchToast("FILE DELETED", "delete");
        } else if (res.status === 401) {
          openAuth('login');
        }
      });

  });
}

function searchTerm() {
  const query = document.getElementById('transcript-search').value.toLowerCase();
  const box = document.getElementById('dynamic-transcript-box');
  let text = currentDisplayTranscript.replace(/\[(\d{2}:\d{2}(?::\d{2})?\s*-\s*\d{2}:\d{2}(?::\d{2})?)\]/g, '<span class="timestamp">$1</span>');
  if (query.length > 1) {
    const regex = new RegExp(`(${query})`, 'gi');
    text = text.replace(regex, '<mark style="background:#ffff00;color:#000;border-radius:2px;padding:0 2px;">$1</mark>');
  }
  box.innerHTML = text;
}

function toggleAutoScroll() {
  isAutoScroll = !isAutoScroll;
  const btn = document.getElementById('autoscroll-btn');
  btn.innerText = isAutoScroll ? "Auto-Scroll: ON" : "Auto-Scroll: OFF";
  btn.style.background = isAutoScroll ? "#00ffff" : "rgba(255,255,255,0.1)";
  btn.style.color = isAutoScroll ? "#000" : "#fff";
}

function handleAutoScroll() {
  if (isAutoScroll) {
    const box = document.getElementById('dynamic-transcript-box');
    box.scrollTop = box.scrollHeight;
  }
}

window.enterApp = function () {
  const landingView = document.getElementById('landing-view');
  const appView = document.getElementById('app-view');
  landingView.style.opacity = '0';
  setTimeout(() => {
    landingView.style.display = 'none';
    appView.style.display = 'flex';
    setTimeout(() => { appView.style.opacity = '1'; appView.style.transform = 'translateY(0)'; }, 50);
  }, 800);
};

window.onload = function () {
  document.getElementById('forgot-btn')
    .addEventListener('click', sendForgotPassword);
  let wavesurfer = null;
  const ALLOWED_EXTENSIONS = ["mp3", "wav", "m4a", "flac", "aac", "ogg"];

  // Prevent browser default drag behavior
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", e => e.preventDefault());

  // Vanta Background
  if (typeof VANTA !== 'undefined') {
    VANTA.WAVES({
      el: "#vanta-bg",
      mouseControls: true,
      color: 0x070707,
      waveHeight: 15
    });
  }

  checkLoginStatus();

  const input = document.getElementById('audio-input');
  const dropZone = document.querySelector('.drop-zone');
  const submitBtn = document.getElementById('submit-btn');
  const audioElement = document.getElementById('audio-live-preview');

  // ─────────────────────────────
  // FILE PREVIEW FUNCTION
  // ─────────────────────────────
  function handleFile(file) {

    const extension = file.name.split('.').pop().toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      const errorBox = document.getElementById("format-error");
      errorBox.style.display = "block";

      setTimeout(() => {
        errorBox.style.display = "none";
      }, 3000);

      return false;
    }

    // Update UI
    document.getElementById('file-name-display').textContent = "✓ " + file.name;
    document.getElementById('preview-container').style.display = "block";
    document.getElementById('text-wrap').style.display = "none";
    document.getElementById('icon').innerHTML = "✅";

    const audioURL = URL.createObjectURL(file);
    audioElement.src = audioURL;

    // Destroy old waveform
    if (wavesurfer) wavesurfer.destroy();

    // Create waveform
    wavesurfer = WaveSurfer.create({
      container: '#waveform',
      waveColor: '#00ffff',
      progressColor: '#00ff88',
      cursorColor: '#ffffff',
      height: 80,
      barWidth: 2,
      barGap: 2,
      responsive: true
    });

    wavesurfer.load(audioURL);

    // Sync waveform click → audio
    wavesurfer.on('interaction', () => {
      audioElement.currentTime = wavesurfer.getCurrentTime();
    });

    // Sync audio play → waveform
    audioElement.addEventListener('timeupdate', () => {
      if (wavesurfer) {
        wavesurfer.setTime(audioElement.currentTime);
      }
    });

    return true;
  }

  // ─────────────────────────────
  // NORMAL FILE SELECT
  // ─────────────────────────────
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const isValid = handleFile(file);
    if (!isValid) input.value = "";
  };

  // ─────────────────────────────
  // DRAG & DROP
  // ─────────────────────────────
  if (dropZone) {

    ["dragenter", "dragover"].forEach(eventName => {
      dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("drag-active");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("drag-active");
      });
    });

    dropZone.addEventListener("drop", e => {
      const files = e.dataTransfer.files;
      if (files.length === 0) return;

      const file = files[0];

      const isValid = handleFile(file);
      if (!isValid) return;

      // Safely assign file to input
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    });
  }

  // ─────────────────────────────
  // UPLOAD SUBMIT
  // ─────────────────────────────
  const uploadForm = document.getElementById('main-upload-form');

  uploadForm.onsubmit = function (e) {
    e.preventDefault();

    const token = localStorage.getItem('access_token');

    submitBtn.disabled = true;
    submitBtn.innerText = "Uploading...";

    const formData = new FormData(uploadForm);
    const headers = {};
    if (token) headers['Authorization'] = 'Bearer ' + token;
    fetch('/upload', {
      method: 'POST',
      headers: token ? { 'Authorization': 'Bearer ' + token } : {},
      body: formData
    })

      .then(res => {
        if (res.status === 403) {
          return res.json().then(data => {

            submitBtn.disabled = false;
            submitBtn.innerText = "Transcribe Now";

            if (data.trial_ended) {
              Swal.fire({
                title: "Free Trial Ended",
                text: "You’ve reached your free limit. Please sign in to continue.",
                icon: "warning",
                confirmButtonColor: "#00ffff",
                background: "#1a1a1a",
                color: "#fff"
              }).then(() => {
                openAuth('login');
              });
            }
            return;
          });
        }
        return res.json();
      })
      .then(data => {
        if (data.filename) {
          launchToast("UPLOAD SUCCESSFUL");
          document.getElementById('progress-container').style.display = 'block';
          currentFilename = data.filename;
          pollStatus(data.filename);
        } else {
          Swal.fire({
            icon: 'error',
            title: 'Upload Error',
            text: data.error || "Unknown Error"
          });
          submitBtn.disabled = false;
          submitBtn.innerText = "Transcribe Now";
        }
      })
      .catch(err => {
        console.error(err);
        submitBtn.disabled = false;
        submitBtn.innerText = "Transcribe Now";
      });
  };
};


function pollStatus(filename) {
  const interval = setInterval(() => {
    fetch(`/check_status/${encodeURIComponent(filename)}`)
      .then(res => res.json())
      .then(data => {
        const fill = document.getElementById('progress-fill');
        const percentText = document.getElementById('progress-percent');
        const msgText = document.getElementById('progress-message');

        if (data.progress) { fill.style.width = data.progress + '%'; percentText.textContent = data.progress + '%'; }

        if (data.status === 'processing') {
          msgText.textContent = (data.progress > 65 && data.progress < 85)
            ? "AI generating concise summary..."
            : loadingMessages[Math.floor(Math.random() * loadingMessages.length)];
        }

        if (data.status === 'error') {
          clearInterval(interval);
          Swal.fire({ icon: 'error', title: 'Processing Failed', text: data.message });
          document.getElementById('submit-btn').disabled = false;
          document.getElementById('submit-btn').innerText = "Transcribe Now";
          document.getElementById('progress-container').style.display = 'none';
        }

        if (data.status === 'completed') {
          clearInterval(interval);
          masterTranscript = data.transcript;
          masterSummary = data.summary;
          displayResults(data.transcript, data.summary);
          document.getElementById('progress-container').style.display = 'none';
          document.getElementById('submit-btn').innerText = "Transcribe Now";
          document.getElementById('submit-btn').disabled = false;
        }
      })
      .catch(err => console.error("Polling error:", err));
  }, 1500);
}

function translateReport() {
  const targetLang = document.getElementById('report-lang-select').value;
  const translateBtn = document.getElementById('translate-btn');

  if (targetLang === 'en') {
    displayResults(masterTranscript, masterSummary);
    return;
  }

  translateBtn.innerText = "Translating...";
  translateBtn.disabled = true;

  fetch('/translate_on_fly', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transcript: masterTranscript,
      summary: masterSummary,
      target: targetLang
    })
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        displayResults(data.translated_text, data.translated_summary);
        launchToast("Language Updated!");
      }
      translateBtn.innerText = "Update Language";
      translateBtn.disabled = false;
    })
    .catch(() => {
      launchToast("Translation Error", "delete");
      translateBtn.disabled = false;
    });
}


function displayResults(transcript, summary) {
  const resultPanel = document.getElementById('dynamic-result-panel');
  const transcriptBox = document.getElementById('dynamic-transcript-box');
  const summaryBox = document.getElementById('dynamic-summary-box');

  currentDisplayTranscript = transcript;
  document.getElementById('transcript-search').value = "";

  const highlighted = transcript.replace(/\[(\d{2}:\d{2}(?::\d{2})?\s*-\s*\d{2}:\d{2}(?::\d{2})?)\]/g, '<span class="timestamp">$1</span>');
  transcriptBox.innerHTML = highlighted;
  summaryBox.textContent = summary || "No summary available (text might be too short)";

  resultPanel.style.display = 'block';
  if (!isAutoScroll) { resultPanel.scrollIntoView({ behavior: 'smooth' }); }
  else { handleAutoScroll(); }
}

function launchToast(message, type = "success") {
  const toast = document.getElementById('toast-message');
  toast.textContent = message;
  toast.style.background = (type === "delete") ? "#ff3c3c" : "#00ffff";
  toast.style.opacity = '1';
  setTimeout(() => { toast.style.opacity = '0'; }, 3000);
}

// Close modal on Escape key
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAuthModal(); });

function downloadFile(type) {
  const token = localStorage.getItem('access_token');

  if (!token) {
    openAuth('login');
    return;
  }

  const lang = document.getElementById('report-lang-select').value;

  fetch(`/download/${encodeURIComponent(currentFilename)}?type=${type}&lang=${lang}`, {
    headers: {
      'Authorization': 'Bearer ' + token
    }
  })
    .then(res => {
      if (res.status === 401) {
        openAuth('login');
        throw new Error("Login required");
      }

      const disposition = res.headers.get("Content-Disposition");
      return res.blob().then(blob => ({
        blob,
        disposition
      }));
    })
    .then(({ blob, disposition }) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;

      let filename = `${currentFilename}.${type}`;

      if (disposition) {
        const match = disposition.match(/filename="?(.+)"?/);
        if (match && match[1]) {
          filename = match[1];
        }
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    })
    .catch(err => console.error(err));
}


function downloadHistoryFile(filename, type) {
  const token = localStorage.getItem('access_token');

  if (!token) {
    openAuth('register');   // ← Always show Sign-Up popup
    return;
  }

  const lang = document.getElementById('report-lang-select').value;
  fetch(`/download/${encodeURIComponent(filename)}?type=${type}&lang=${lang}`,
    {
      headers: {
        'Authorization': 'Bearer ' + token
      }
    })
    .then(res => {
      if (res.status === 401) {
        openAuth('register');
        throw new Error("Login required");
      }
      return res.blob();
    })
    .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.${type}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    })
    .catch(err => console.error(err));
}

function openOtpModal() {
  const modal = document.getElementById('otp-modal');
  modal.style.display = 'flex';
  setTimeout(() => modal.style.opacity = '1', 10);
}

function closeOtpModal() {
  const modal = document.getElementById('otp-modal');
  modal.style.opacity = '0';
  setTimeout(() => modal.style.display = 'none', 300);
}

function verifyOTP() {
  const otp = document.getElementById('otp-input').value.trim();
  const err = document.getElementById('otp-error');

  err.textContent = "";

  if (!otp) {
    err.textContent = "Please enter OTP.";
    return;
  }

  fetch('/auth/verify_otp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: pendingEmail, otp })
  })
    .then(async (response) => {
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "OTP verification failed.");
      }

      return data;
    })
    .then(data => {

      // ✅ FIX 1: Store token
      localStorage.setItem('access_token', data.access_token);

      // ✅ FIX 2: Store name safely
      localStorage.setItem('user_name', data.name || pendingEmail.split("@")[0]);

      // Clear input
      document.getElementById('otp-input').value = "";

      closeOtpModal();
      closeAuthModal();

      // Update navbar
      checkLoginStatus();

      // ✅ Show toast like delete
      launchToast("ACCOUNT CREATED SUCCESSFULLY!");

      // Go to home screen
      enterApp();

    })
    .catch(error => {
      err.textContent = error.message;
    });
}
