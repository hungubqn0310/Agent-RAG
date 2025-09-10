import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/agent1_rag")
VECTOR_DIMENSION = 1536

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

# File Upload Configuration
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.png', '.jpg', '.jpeg'}

# Search Configuration
SEARCH_RESULTS_LIMIT = 5
SIMILARITY_THRESHOLD = 0.7

# Voice Configuration
VOICE_ENABLED = True
TTS_LANGUAGE = "vi"
STT_LANGUAGE = "vi-VN"

# External Search Configuration
EXTERNAL_SEARCH_ENABLED = True
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
