"""
app.py — Flask backend for PDF Extract Chatbot
Routes:
  GET  /                → chat.html
  GET  /upload          → upload.html
  POST /api/upload      → ingest a PDF
  POST /api/chat        → RAG answer
  GET  /api/status      → index stats
  GET  /api/history     → chat history
  DELETE /api/clear     → clear chat history
  GET  /api/pdfs        → list ingested PDFs
"""

import os
import json
import logging
import threading
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import (Flask, request, jsonify, render_template,
                   send_from_directory, Response, stream_with_context)

import config
import ingest
import rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Thread lock for concurrent ingestion requests
_ingest_lock = threading.Lock()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def list_ingested_pdfs():
    """Return list of PDF names present in the chunk store."""
    if not os.path.exists(config.CHUNKS_JSON_PATH):
        return []
    try:
        with open(config.CHUNKS_JSON_PATH, "r") as f:
            chunks = json.load(f)
        return sorted(set(c.get("pdf", "") for c in chunks if c.get("pdf")))
    except Exception:
        return []

# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

# ─── API: Upload & Ingest ─────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    results = []
    for file in files:
        if not allowed_file(file.filename):
            results.append({"file": file.filename, "error": "Only PDF files are allowed"})
            continue

        filename  = secure_filename(file.filename)
        save_path = os.path.join(config.UPLOAD_DIR, filename)
        file.save(save_path)
        logger.info(f"Saved upload: {save_path}")

        try:
            with _ingest_lock:
                summary = ingest.ingest_pdf(save_path)
            # Reload RAG index after ingestion (also clears the answer cache,
            # so nothing stale from before this PDF existed can be served)
            rag.load_index()
            results.append({"file": filename, "result": summary})
        except Exception as e:
            logger.exception(f"Ingestion error for {filename}")
            results.append({"file": filename, "error": str(e)})

    return jsonify({"uploads": results})

# ─── API: Chat (blocking) ────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    try:
        result = rag.answer(query)
        return jsonify(result)
    except Exception as e:
        logger.exception("RAG error")
        return jsonify({"error": str(e)}), 500

# ─── API: Chat (streaming SSE) ───────────────────────────────────────────────

@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """
    Server-Sent Events endpoint.
    Each event is a JSON line; final event has {"done": true, ...}.
    """
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        try:
            for line in rag.answer_stream(query):
                yield f"data: {line}\n\n"
        except Exception as e:
            logger.exception("Stream RAG error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        }
    )

# ─── API: Status ──────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    pdfs         = list_ingested_pdfs()
    index_exists = os.path.exists(config.FAISS_INDEX_PATH)
    vector_count = 0
    if index_exists and rag._faiss_index is not None:
        vector_count = rag._faiss_index.ntotal

    return jsonify({
        "index_ready":  index_exists,
        "vector_count": vector_count,
        "pdf_count":    len(pdfs),
        "pdfs":         pdfs,
        "embed_model":  config.EMBED_MODEL,
        "llm_model":    config.OLLAMA_MODEL,
        "reranker":     config.RERANKER_MODEL,
        "top_k_retrieve": config.TOP_K_RETRIEVE,
        "top_k_rerank":   config.TOP_K_RERANK,
        # Visibility into the speed-oriented features added on top of the
        # base pipeline — handy when diagnosing why a given query was
        # fast or slow.
        "table_extraction_enabled":  config.ENABLE_TABLE_EXTRACTION,
        "embedded_image_ocr_enabled": config.EXTRACT_EMBEDDED_IMAGES,
        "fastpath_threshold":        config.FASTPATH_THRESHOLD,
        "answer_cache_entries":      len(rag._answer_cache),
        "answer_cache_capacity":     config.ANSWER_CACHE_SIZE,
    })

# ─── API: List PDFs ───────────────────────────────────────────────────────────

@app.route("/api/pdfs", methods=["GET"])
def api_pdfs():
    return jsonify({"pdfs": list_ingested_pdfs()})

# ─── API: Delete PDF ─────────────────────────────────────────────────────────

@app.route("/api/delete-pdf", methods=["DELETE"])
def api_delete_pdf():
    data     = request.get_json(force=True)
    pdf_name = (data.get("pdf") or "").strip()
    if not pdf_name:
        return jsonify({"error": "Missing 'pdf' field"}), 400
    try:
        summary = ingest.delete_pdf_from_index(pdf_name)
        # Reload RAG index so the deleted chunks are no longer searchable
        rag.load_index()
        return jsonify(summary)
    except Exception as e:
        logger.exception(f"Delete error for {pdf_name}")
        return jsonify({"error": str(e)}), 500

# ─── API: Chat History ────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def api_history():
    if not os.path.exists(config.CHAT_HISTORY_PATH):
        return jsonify({"history": ""})
    with open(config.CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"history": content})

@app.route("/api/clear", methods=["DELETE"])
def api_clear_history():
    if os.path.exists(config.CHAT_HISTORY_PATH):
        open(config.CHAT_HISTORY_PATH, "w").close()
    return jsonify({"status": "cleared"})

# ─── Serve Extracted Images ───────────────────────────────────────────────────

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(config.IMAGES_DIR, filename)

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load the embedder, reranker, and ping Ollama once now, in a
    # background thread, instead of letting the first real user request
    # pay for all three cold-starts at once.
    rag.warmup()
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)