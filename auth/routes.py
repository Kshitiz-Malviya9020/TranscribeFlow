import email
from flask import Blueprint, request, jsonify, session, render_template
from database.db import load_users, save_users
from auth.utils import *
import secrets
import time
import random
from config import Config
import smtplib
from email.mime.text import MIMEText

# Temporary OTP store
otp_store = {}

auth_bp = Blueprint("auth", __name__)

@auth_bp.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()

    identifier = data.get("identifier")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"detail": "Identifier and password required"}), 400

    users_db = load_users()
    user = None
    email_for_token = None

    
    email = identifier.lower()
    user = users_db.get(email)
    email_for_token = email

    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"detail": "Invalid credentials"}), 401

    return jsonify({
        "access_token": create_token(email_for_token),
        "name": user["name"]
    })

reset_tokens = {}  # temporary in-memory storage

@auth_bp.route('/auth/forgot_password', methods=['POST'])
def forgot_password():
    data = request.get_json()

    if not data or not data.get("email"):
        return jsonify({"detail": "Email required"}), 400

    email = data["email"].lower()
    users_db = load_users()

    # ✅ Check if account exists
    if email not in users_db:
        return jsonify({"detail": "No account found with this email"}), 404

    # Generate secure token
    token = secrets.token_urlsafe(32)

    reset_tokens[token] = {
        "email": email,
        "expires": time.time() + 600
    }

    reset_link = f"http://127.0.0.1:5000/reset_password?token={token}"

    try:
        send_reset_email(email, reset_link)
    except Exception as e:
        print("Reset email error:", e)
        return jsonify({"detail": "Failed to send reset email"}), 500

    return jsonify({"message": "Reset link sent successfully"})

@auth_bp.route('/auth/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()

    token = data.get("token")
    new_password = data.get("new_password")

    if not token or not new_password:
        return jsonify({"detail": "Invalid request"}), 400

    if token not in reset_tokens:
        return jsonify({"detail": "Invalid or expired token"}), 400

    token_data = reset_tokens[token]

    if time.time() > token_data["expires"]:
        del reset_tokens[token]
        return jsonify({"detail": "Token expired"}), 400

    email = token_data["email"]

    valid, msg = validate_password(new_password)
    if not valid:
        return jsonify({"detail": msg}), 400

    users_db = load_users()
    users_db[email]["password_hash"] = hash_password(new_password)
    save_users(users_db)

    # Remove used token
    del reset_tokens[token]

    # Create JWT token
    token = create_token(email)
    return jsonify({
        "message": "Password reset successful",
        "access_token": token,
        "name": users_db[email]["name"]
    })


@auth_bp.route('/reset_password')
def serve_reset_page():
    return render_template("reset_password.html")

@auth_bp.route('/auth/send_otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"detail": "All required fields missing"}), 400

    email = email.lower()

    users_db = load_users()
    if email in users_db:
        return jsonify({"detail": "User already exists"}), 400

    valid, msg = validate_password(password)
    if not valid:
        return jsonify({"detail": msg}), 400

    otp_code = str(random.randint(100000, 999999))

    otp_store[email] = {
        "otp": otp_code,
        "name": name,
        "password": password,
        "expires": time.time() + 300,
        "attempts": 0
    }

    try:
        send_otp_email(email, otp_code)
        return jsonify({"message": "OTP sent"})
    except Exception as e:
        print("OTP Error:", e)
        return jsonify({"detail": "Failed to send OTP"}), 500

@auth_bp.route('/auth/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()

    email = data.get("email")
    otp_input = data.get("otp")

    if not email or not otp_input:
        return jsonify({"detail": "Email and OTP required"}), 400

    email = email.lower()
    record = otp_store.get(email)

    if not record:
        return jsonify({"detail": "OTP expired or not found"}), 400

    if time.time() > record["expires"]:
        otp_store.pop(email)
        return jsonify({"detail": "OTP expired"}), 400

    if record["attempts"] >= 3:
        otp_store.pop(email)
        return jsonify({"detail": "Maximum attempts exceeded"}), 400

    if record["otp"] != otp_input:
        record["attempts"] += 1
        return jsonify({
            "detail": f"Invalid OTP. Attempts left: {3 - record['attempts']}"
        }), 400

    users_db = load_users()

    users_db[email] = {
    "name": record["name"],
    "email": email,
    "password_hash": hash_password(record["password"])
    }

    save_users(users_db)
    otp_store.pop(email)

    # Create JWT token
    token = create_token(email)

    return jsonify({
    "message": "Registration successful",
    "access_token": token,
    "name": record.get("name", email.split("@")[0])
})

def send_otp_email(to_email, otp_code):
    subject = "Your TranscribeFlow OTP Code"
    body = f"""
    Hello,

    Your OTP for TranscribeFlow registration is:

    {otp_code}

    This code will expire in 5 minutes.

    If you did not request this, please ignore this email.
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = Config.SMTP_FROM
    msg["To"] = to_email

    with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
        server.starttls()
        server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
        server.send_message(msg)

def send_reset_email(to_email, reset_link):
    subject = "Reset Your TranscribeFlow Password"
    body = f"""
    Hello,
    You requested a password reset.
    Click the link below to reset your password:
    {reset_link}
    This link will expire in 10 minutes.
    If you did not request this, please ignore this email.
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = Config.SMTP_FROM
    msg["To"] = to_email

    with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
        server.starttls()
        server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
        server.send_message(msg)
