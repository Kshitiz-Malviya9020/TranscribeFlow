from flask import Flask, session
from config import Config
from auth.routes import auth_bp
from transcription.routes import transcription_bp
import os

# Ensure upload folder exists
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)

app = Flask(__name__)
app.config.from_object(Config)

app.secret_key = Config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(transcription_bp)

if __name__ == "__main__":
    app.run(debug=False, threaded=True)