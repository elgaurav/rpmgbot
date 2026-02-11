"""
Configuration for RPMG RAG Assistant
Optimized for <30 second response times
UPDATED: Smart query classification to prevent unnecessary source/image display
"""
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "piping" / "test_data"
STORAGE_DIR = BASE_DIR / "storage"
STATIC_DIR = BASE_DIR / "static"
IMAGE_OUTPUT_DIR = BASE_DIR / "static" / "images"
PDF_OUTPUT_DIR = BASE_DIR / "static" / "pdfs"  # For serving source PDFs
QDRANT_PATH = BASE_DIR / "qdrant_db"

# ==================== MODEL CONFIGURATION ====================
# EMBEDDING MODEL - Fast and accurate
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384 dimensions, very fast

# LLM CONFIGURATION
LLM_MODEL = "qwen2.5:1.5b"  # Fast and efficient
# Alternative: "llama3.2:3b" for better reasoning

LLM_REQUEST_TIMEOUT = 120.0
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# ==================== RAG PARAMETERS ====================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 2

# ==================== PERFORMANCE TUNING ====================
USE_GPU = True
ENABLE_STREAMING = True
BATCH_SIZE = 32

# ==================== DOCLING SETTINGS ====================
DOCLING_OCR = True
DOCLING_TABLE_EXTRACTION = True
DOCLING_IMAGE_SCALE = 1.2
DOCLING_GENERATE_PICTURES = True

# ==================== API SETTINGS ====================
API_HOST = "0.0.0.0"
API_PORT = 8000
BASE_URL = f"http://127.0.0.1:{API_PORT}"

# ==================== IMAGE LINKING SETTINGS ====================
IMAGE_PAGE_MATCH_STRICT = True  # Only show images from exact pages
IMAGE_ADJACENT_PAGES = 0  # 0 = exact page only
MAX_IMAGES_PER_QUERY = 10

# ==================== QUERY CLASSIFICATION ====================
# CRITICAL: Prevent showing sources/images for casual queries
ENABLE_QUERY_CLASSIFICATION = True  # Classify if query needs RAG
MIN_RELEVANCE_SCORE = 0.3  # Minimum similarity score to show sources (0.0-1.0)

# Queries matching these patterns will NOT show sources/images
CASUAL_QUERY_PATTERNS = [
    "hello", "hi", "hey", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r u", "sup", "what's up", "whats up",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you", "cya",
    "ok", "okay", "cool", "nice", "great", "ssup", "yo"
]

# ==================== DEBUGGING ====================
VERBOSE = True
LOG_RETRIEVAL = True