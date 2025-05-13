import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "heart_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modeling", "heart_model.pkl")

# Allowed extensions for upload
ALLOWED_EXTENSIONS = {'csv'}

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploaded")

# Logs
LOG_FILE = os.path.join(BASE_DIR, "monitoring", "logs", "predictions.log")
