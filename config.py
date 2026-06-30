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
# IMPORTANT: v2-m3 is ~4x faster than -large with near-identical accuracy.
# Your previous edit accidentally switched this to -large, which is a
# pure speed regression with no quality benefit — reverted.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval
# IMPORTANT: TOP_K_RERANK=1 was the actual cause of the quality drop you
# saw ("Coimbatore" answer). The correct chunk was ranked #2, not #1 — at
# top_k=1 it never reached the LLM at all, so the model had nothing to
# answer from. 3 is the minimum that reliably avoids this for directory-
# style/cross-referenced content. Don't drop this below 3.
TOP_K_RETRIEVE = 5          # initial FAISS hits
TOP_K_RERANK   = 3          # after re-ranking — DO NOT reduce, causes missed answers

# LLM
OLLAMA_MODEL = "qwen2.5:3b"       # smaller/faster model — keep, this wasn't the problem
OLLAMA_URL   = "http://localhost:11434"

# 3 chunks (≤512 chars each) + system prompt + question needs headroom.
# 1536 was cutting it close enough to risk silent context truncation on
# longer chunks — bumped back up just enough to be safe without paying
# the full cost of the original 2048+.
OLLAMA_NUM_CTX = 1800

# 200 tokens is fine for one-line facts but truncates multi-field answers
# (address + phone + in-charge name in one response). Raised modestly.
OLLAMA_NUM_PREDICT = 300

OLLAMA_TEMPERATURE = 0.0
OLLAMA_KEEP_ALIVE  = "-1"

# Flask
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB

# OCR  (PaddleOCR)
OCR_LANG = "en"
OCR_USE_GPU = False

TABLE_PRECHECK_MIN_ROWS = 2

# ─── Ingestion-time cost gates (used by ingest.py) ─────────────────────────
ENABLE_TABLE_EXTRACTION = True
EXTRACT_EMBEDDED_IMAGES = True
MIN_IMAGE_OCR_AREA = 900

# ─── Query-time cost gates (used by rag.py) ────────────────────────────────
FASTPATH_THRESHOLD = 85

# ==========================
# Cache Settings
# ==========================
ANSWER_CACHE_SIZE = 100
QUERY_CACHE_SIZE  = 500