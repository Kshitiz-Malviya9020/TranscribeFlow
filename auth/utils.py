import re
import datetime
import jwt
from passlib.context import CryptContext
from config import Config
from functools import wraps
from flask import request, jsonify

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(email):
    payload = {
        "sub": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
    }
    return jwt.encode(payload, Config.SECRET_KEY, algorithm=Config.ALGORITHM)

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r'[^a-zA-Z0-9]', password):
        return False, "Password must contain at least one special character."
    return True, ""

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"detail": "Token missing"}), 401

        try:
            token = auth_header.split()[1]
            payload = jwt.decode(
                token,
                Config.SECRET_KEY,
                algorithms=[Config.ALGORITHM]
            )
            request.user_email = payload["sub"]

        except jwt.ExpiredSignatureError:
            return jsonify({"detail": "Token expired"}), 401
        except Exception:
            return jsonify({"detail": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated