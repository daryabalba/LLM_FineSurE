from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/bart-large-cnn"
MAX_LENGTH = 1000
MAX_TARGET_LENGTH = 200

GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
CLIENT_SECRET = ''