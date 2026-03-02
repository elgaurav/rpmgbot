"""
Configuration for RPMG RAG Assistant
UPGRADED: Higher-capacity models for better performance on capable hardware
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
# EMBEDDING MODEL - High-quality embeddings for better retrieval
# UPGRADED: bge-large provides 1024-dim vectors vs 384-dim in bge-small
# Better semantic understanding and retrieval accuracy
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024 dimensions, better accuracy
EMBED_BATCH_SIZE = 32  # Batch size for embedding generation (adjust based on GPU memory)

# LLM CONFIGURATION
# UPGRADED: llama3.1:8b provides significantly better reasoning and coherence
# 4-bit quantized for memory efficiency while maintaining quality
LLM_MODEL = "llama3.1:8b"  # 8B parameter model, 4-bit quantized
# Alternative if memory constrained: "llama3.2:3b"

LLM_REQUEST_TIMEOUT = 180.0  # Increased for larger model inference
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024  # Increased from 512 for more comprehensive answers

# LLM CONTEXT WINDOW
# UPGRADED: Increased context window to utilize model's full capacity
# llama3.1:8b supports up to 128k, we use 8k for practical single-machine use
LLM_CONTEXT_SIZE = 8192  # 8k token context window (was implicit 2048)

# ==================== RAG PARAMETERS ====================
# UPGRADED: Larger chunks for better context preservation
# Larger models can handle more context effectively
CHUNK_SIZE = 1024  # Increased from 512 - better context per chunk
CHUNK_OVERLAP = 150  # Increased from 50 - better continuity between chunks

# UPGRADED: Retrieve more chunks for richer context
# Larger LLM can synthesize information from more sources
SIMILARITY_TOP_K = 4  # Increased from 2 - more comprehensive retrieval

# ==================== PERFORMANCE TUNING ====================
USE_GPU = True  # Enable GPU acceleration when available
ENABLE_STREAMING = True
BATCH_SIZE = 32  # For bulk operations

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
