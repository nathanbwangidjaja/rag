import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

UPLOAD_DIR = "uploads"
DATA_DIR = "data"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
