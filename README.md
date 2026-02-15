# 🎧 TranscribeFlow

> **AI-powered audio transcription, summarization & translation — all in one sleek web app.**

TranscribeFlow lets you upload an MP3 or WAV file and instantly get a timestamped transcript, an AI-generated summary, and the option to translate everything into 8 languages. Built with Flask, OpenAI Whisper, and Facebook BART.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ **Transcription** | Powered by OpenAI Whisper (`base` model) with timestamped segments |
| 🤖 **AI Summarization** | Facebook BART large-CNN model condenses long audio into key points |
| 🌍 **Translation** | Translate transcript + summary into 8 languages on the fly |
| 📄 **Export** | Download results as formatted TXT or PDF (with multi-language font support) |
| 🔐 **Auth** | Register & login via **email or phone number** |
| 🔑 **Password Reset** | Secure reset link sent to registered email (30-minute expiry) |
| 💪 **Password Strength** | Live strength meter enforcing length, uppercase & special character rules |
| 📜 **History** | Browse, replay, and delete previously uploaded files |
| 🤖 **Robot Mascot** | Interactive eye-tracking bot with star-eye animation |

---

## 🖥️ Tech Stack

**Backend**
- [Flask](https://flask.palletsprojects.com/) — web framework
- [OpenAI Whisper](https://github.com/openai/whisper) — speech-to-text
- [HuggingFace Transformers](https://huggingface.co/facebook/bart-large-cnn) — BART summarization
- [deep-translator](https://github.com/nidhaloff/deep-translator) — Google Translate wrapper
- [FPDF2](https://py-fpdf2.readthedocs.io/) — PDF generation
- [PassLib + Argon2](https://passlib.readthedocs.io/) — password hashing
- [PyJWT](https://pyjwt.readthedocs.io/) — JWT authentication tokens
- [python-dotenv](https://github.com/theskumar/python-dotenv) — environment variable management

**Frontend**
- Vanilla HTML / CSS / JavaScript
- [Vanta.js](https://www.vantajs.com/) — animated wave background
- [SweetAlert2](https://sweetalert2.github.io/) — styled modals & alerts

**Storage**
- `users.json` — lightweight file-based user database
- `uploads/` — local folder for audio + transcript files

---

## 📁 Project Structure

```
transcribeflow/
│
├── app.py                        # Main Flask application
├── .env                          # 🔒 Secret config (never commit this)
├── .env.example                  # Safe template to share
├── .gitignore
├── requirements.txt
├── users.json                    # Auto-created on first registration
│
├── uploads/                      # Auto-created on first upload
│   ├── recording.mp3
│   ├── recording.mp3.txt         # Transcript
│   └── recording.mp3_summary.txt # Summary
│
└── templates/
    ├── index.html                # Main app UI
    └── reset_password.html       # Password reset page (served via email link)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/transcribeflow.git
cd transcribeflow
```

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Whisper also requires [ffmpeg](https://ffmpeg.org/download.html) to be installed on your system.
> - **Windows:** `winget install ffmpeg` or download from the website
> - **macOS:** `brew install ffmpeg`
> - **Linux:** `sudo apt install ffmpeg`

### 4. Set up environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and edit:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password
SMTP_FROM=your_email@gmail.com
APP_BASE_URL=http://127.0.0.1:5000
```

> **Gmail setup:** Go to **Google Account → Security → 2-Step Verification → App Passwords** and generate a 16-character app password. Use that as `SMTP_PASSWORD` — never your actual Gmail password.

### 5. Run the app

```bash
python app.py
```

Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 📦 Requirements

Create a `requirements.txt` with:

```
flask
openai-whisper
deep-translator
fpdf2
arabic-reshaper
python-bidi
passlib[argon2]
PyJWT
transformers
torch
python-dotenv
```

> First run will automatically download the Whisper `base` model (~150 MB) and BART model (~1.6 GB). Ensure you have a stable internet connection and enough disk space.

---

## 🌍 Supported Translation Languages

| Language | Code |
|---|---|
| English | `en` |
| Hindi | `hi` |
| Spanish | `es` |
| French | `fr` |
| German | `de` |
| Chinese (Simplified) | `zh-CN` |
| Japanese | `ja` |
| Arabic | `ar` |

---

## 🔐 Authentication

### Register
- Requires name, email, and password
- Phone number is optional — but once added, you can log in with it
- Password must have: ≥ 8 characters, 1 uppercase letter, 1 special character

### Login
- Toggle between **Email** and **Phone** login in the modal
- JWT token stored in `localStorage` (6-hour expiry)

### Forgot Password
1. Click "Forgot your password?" in the login modal
2. Enter your registered email
3. Check your inbox for a reset link (valid for 30 minutes)
4. Set a new password that meets the strength requirements

---

## ⚙️ Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `SMTP_HOST` | SMTP server address | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `SMTP_USER` | Your email address | — |
| `SMTP_PASSWORD` | App password (not your login password) | — |
| `SMTP_FROM` | Sender address shown in email | Same as `SMTP_USER` |
| `APP_BASE_URL` | Base URL for reset links | `http://127.0.0.1:5000` |

---

## 🔒 Security Notes

- Passwords are hashed with **Argon2** (resistant to brute-force attacks)
- Password reset tokens are single-use and expire after 30 minutes
- JWT tokens expire after 6 hours
- Never commit your `.env` file — it is listed in `.gitignore`
- The `users.json` file contains hashed passwords — also excluded from git


---

## 🙋 FAQ

**Q: The app loads but transcription never completes.**
A: Make sure `ffmpeg` is installed and accessible in your system PATH.

**Q: I'm not receiving the password reset email.**
A: Double-check your `SMTP_PASSWORD` is a Gmail App Password (not your account password), and that 2-Step Verification is enabled on your Google account.

**Q: Can I use a different email provider?**
A: Yes — update `SMTP_HOST` and `SMTP_PORT` in your `.env` to match your provider (e.g. Outlook uses `smtp.office365.com` on port `587`).

**Q: The BART model takes a long time to load.**
A: The model (~1.6 GB) is loaded into memory at startup. This is a one-time cost per server restart. Consider running on a machine with at least 4 GB RAM.

**Q: How do I deploy this to production?**
A: Use a WSGI server like **Gunicorn** behind **Nginx**, set `APP_BASE_URL` to your domain in `.env`, and use HTTPS so that reset links are secure.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
  <strong>Built with ❤️ using Flask + Whisper + BART</strong>
</div>
