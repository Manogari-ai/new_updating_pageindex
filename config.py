"""
config.py — Central configuration for PDF Extract Chatbot
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
CHUNKS_DIR   = os.path.join(BASE_DIR, "chunks")
INDEX_DIR    = os.path.join(BASE_DIR, "index")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
IMAGES_DIR   = os.path.join(BASE_DIR, "images")

for d in [UPLOAD_DIR, CHUNKS_DIR, INDEX_DIR, LOG_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# Files
FAISS_INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_JSON_PATH  = os.path.join(CHUNKS_DIR, "chunks.json")
CHAT_HISTORY_PATH = os.path.join(LOG_DIR, "chat_history.txt")

# Chunking
CHUNK_SIZE    = 512          # characters
CHUNK_OVERLAP = 80

# Embedding model (BAAI/bge-m3)
EMBED_MODEL   = "BAAI/bge-m3"
EMBED_DIM     = 1024

# Re-ranker model
RERANKER_MODEL = "BAAI/bge-reranker-large"

# Retrieval
TOP_K_RETRIEVE = 20          # initial FAISS hits
TOP_K_RERANK   = 5           # after re-ranking

# LLM
OLLAMA_MODEL = "qwen3"       # Qwen3 via Ollama
OLLAMA_URL   = "http://localhost:11434"

# Flask
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB

# OCR  (PaddleOCR)
OCR_LANG = "en"
OCR_USE_GPU = False
