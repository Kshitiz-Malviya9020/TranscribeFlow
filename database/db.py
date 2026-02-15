import os
import json
from config import Config

def load_users():
    if os.path.exists(Config.DB_FILE):
        try:
            with open(Config.DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users_db):
    with open(Config.DB_FILE, 'w') as f:
        json.dump(users_db, f, indent=4)
