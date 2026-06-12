"""
rag.py — Retrieval-Augmented Generation engine
  1. Query embedding  (BAAI/bge-m3, with LRU cache)
  2. FAISS top-K retrieval
  3. Adaptive re-ranking  (bge-reranker-v2-m3 — 4× faster than large)
     skipped when top retrieval score is already very high
  4. Prompt assembly
  5. Qwen3 via Ollama  (streaming → SSE, async chat logging)
"""

import os
import json
import time
import logging
import datetime
import threading
from functools import lru_cache
from typing import List, Dict, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_embed_model  = None
_reranker     = None
_faiss_index  = None
_chunks: List[Dict] = []
_model_lock   = threading.Lock()

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _model_lock:
            if _embed_model is None:
                from FlagEmbedding import FlagModel
                _embed_model = FlagModel(
                    config.EMBED_MODEL,
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                    use_fp16=True
                )
                logger.info(f"Embedding model ready: {config.EMBED_MODEL}")
    return _embed_model

def get_reranker():
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                from FlagEmbedding import FlagReranker
                # bge-reranker-v2-m3 is ~4× faster than bge-reranker-large
                # with nearly identical accuracy on retrieval tasks
                model_name = getattr(config, "RERANKER_MODEL",
                                     "BAAI/bge-reranker-v2-m3")
                _reranker = FlagReranker(model_name, use_fp16=True)
                logger.info(f"Re-ranker ready: {model_name}")
    return _reranker

def load_index():
    """Load (or reload) FAISS index + chunks into module globals."""
    global _faiss_index, _chunks
    if not os.path.exists(config.FAISS_INDEX_PATH):
        logger.warning("FAISS index not found. Ingest PDFs first.")
        _faiss_index = None
        _chunks = []
        return
    import faiss
    _faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    # Bust query embedding cache when index changes
    _embed_query.cache_clear()
    logger.info(f"Index loaded: {_faiss_index.ntotal} vectors, {len(_chunks)} chunks")

# ─── Query embedding with cache ───────────────────────────────────────────────

@lru_cache(maxsize=256)
def _embed_query(query: str) -> bytes:
    """
    Embed a query string and return the vector as raw bytes for caching.
    lru_cache requires hashable args; numpy arrays aren't, so we serialise.
    """
    model = get_embed_model()
    q_raw = model.encode([query]).astype(np.float32)
    norm  = np.linalg.norm(q_raw, axis=1, keepdims=True)
    q_vec = q_raw / np.where(norm == 0, 1.0, norm)
    return q_vec.tobytes()          # serialise → hashable

def embed_query(query: str) -> np.ndarray:
    return np.frombuffer(_embed_query(query), dtype=np.float32).reshape(1, -1)

# Load on import — must be after _embed_query is defined (cache_clear reference)
load_index()

# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = config.TOP_K_RETRIEVE) -> List[Dict]:
    """Embed query (cached) and fetch top-K chunks from FAISS."""
    if _faiss_index is None or not _chunks:
        return []

    q_vec = embed_query(query)
    scores, indices = _faiss_index.search(q_vec, min(top_k, len(_chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results

# ─── Adaptive re-ranking ─────────────────────────────────────────────────────

# Skip reranker when the top FAISS score is already above this threshold
# (the semantic match is already very strong — reranking adds little)
_RERANK_SKIP_THRESHOLD = float(os.environ.get("RERANK_SKIP_THRESHOLD", "0.92"))

def rerank(query: str, candidates: List[Dict],
           top_k: int = config.TOP_K_RERANK) -> List[Dict]:
    """
    Re-rank candidates with the reranker model.
    Skips reranking entirely when the best retrieval score exceeds the threshold
    (saves 200–800 ms on confident matches).
    """
    if not candidates:
        return []

    best_score = candidates[0].get("retrieval_score", 0.0)
    if best_score >= _RERANK_SKIP_THRESHOLD:
        logger.info(f"Rerank skipped (top score={best_score:.3f} ≥ {_RERANK_SKIP_THRESHOLD})")
        for c in candidates:
            c["rerank_score"] = c["retrieval_score"]
        return candidates[:top_k]

    try:
        reranker = get_reranker()
        pairs    = [[query, c["text"]] for c in candidates]
        scores   = reranker.compute_score(pairs, normalize=True)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        logger.warning(f"Re-ranking failed ({e}); using retrieval order.")
        return candidates[:top_k]

# ─── Prompt Builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions strictly based on "
    "the provided document context.\n\n"
    "Rules:\n"
    "- Answer only from the CONTEXT provided below.\n"
    '- If the answer is not in the context, say "I could not find relevant information in the uploaded documents."\n'
    "- Cite page numbers when available (e.g., \"According to page 3…\").\n"
    "- Be concise but complete.\n"
    "- For tables, present data clearly.\n"
    "- Do not hallucinate or add information from outside the context."
)

def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(context_chunks, 1):
        parts.append(
            f"[Source {i} | PDF: {c.get('pdf','document')} | "
            f"Page: {c.get('page','?')} | Type: {c.get('source','text')}]\n{c['text']}"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n" + "\n\n---\n\n".join(parts) +
        f"\n\nQUESTION:\n{query}\n\nANSWER:"
    )

# ─── LLM (Qwen3 via Ollama) — streaming ──────────────────────────────────────

def stream_ollama(prompt: str):
    """
    Generator that yields text tokens from Ollama as they arrive.
    Use this for SSE / streaming responses.
    """
    try:
        import ollama
        for chunk in ollama.generate(
            model=config.OLLAMA_MODEL,
            prompt=prompt,
            stream=True,
            options={
                "temperature": 0.1,
                "top_p":       0.9,
                "num_ctx":     4096,   # 4096 is enough for most RAG prompts; 8192 doubles KV cache
            }
        ):
            token = chunk.get("response", "")
            if token:
                yield token
    except Exception as e:
        logger.error(f"Ollama stream failed: {e}")
        yield f"[LLM Error] Could not reach Ollama ({config.OLLAMA_MODEL}). Error: {e}"

def call_ollama(prompt: str) -> str:
    """Blocking call — collects full streamed response."""
    return "".join(stream_ollama(prompt))

# ─── Async chat logging ───────────────────────────────────────────────────────

def log_chat(query: str, answer: str, sources: List[Dict]):
    """Fire-and-forget: write Q&A log in a background thread."""
    def _write():
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            with open(config.CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*70}\n[{ts}]\nQ: {query}\n\nA: {answer}\n\nSources:\n")
                for s in sources:
                    score = s.get("rerank_score", s.get("retrieval_score", "?"))
                    f.write(f"  - PDF: {s.get('pdf')} | Page: {s.get('page')} | "
                            f"Type: {s.get('source')} | Score: {score:.4f}\n")
        except Exception as e:
            logger.warning(f"Chat logging failed: {e}")
    threading.Thread(target=_write, daemon=True).start()

# ─── Main RAG Entry ───────────────────────────────────────────────────────────

def answer(query: str) -> Dict:
    """
    Full RAG pipeline: retrieve → rerank → generate → log.
    Returns dict with 'answer', 'sources', 'elapsed'.
    """
    t0 = time.time()

    if _faiss_index is None:
        return {
            "answer":  "No documents have been ingested yet. Please upload and ingest a PDF first.",
            "sources": [],
            "elapsed": 0.0
        }

    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    if not top_chunks:
        return {
            "answer":  "No relevant passages found for your query.",
            "sources": [],
            "elapsed": round(time.time() - t0, 2)
        }

    prompt   = build_prompt(query, top_chunks)
    response = call_ollama(prompt)

    sources = [{
        "pdf":     c.get("pdf", ""),
        "page":    c.get("page", ""),
        "source":  c.get("source", ""),
        "score":   round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    log_chat(query, response, top_chunks)   # async — does not block return

    return {
        "answer":  response,
        "sources": sources,
        "elapsed": round(time.time() - t0, 2)
    }


# ─── Streaming answer (for /api/chat/stream) ─────────────────────────────────

def answer_stream(query: str):
    """
    Generator for SSE streaming:
      Yields JSON lines:  {"token": "..."}
      Final line:         {"done": true, "sources": [...], "elapsed": X}
    """
    t0 = time.time()

    if _faiss_index is None:
        yield json.dumps({"token": "No documents have been ingested yet."})
        yield json.dumps({"done": True, "sources": [], "elapsed": 0.0})
        return

    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    if not top_chunks:
        yield json.dumps({"token": "No relevant passages found for your query."})
        yield json.dumps({"done": True, "sources": [], "elapsed": round(time.time()-t0,2)})
        return

    prompt      = build_prompt(query, top_chunks)
    full_answer = []

    for token in stream_ollama(prompt):
        full_answer.append(token)
        yield json.dumps({"token": token})

    sources = [{
        "pdf":     c.get("pdf", ""),
        "page":    c.get("page", ""),
        "source":  c.get("source", ""),
        "score":   round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    log_chat(query, "".join(full_answer), top_chunks)

    yield json.dumps({"done": True, "sources": sources, "elapsed": round(time.time()-t0, 2)})


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    result = answer(q)
    print("\nAnswer:", result["answer"])
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['pdf']} | p.{s['page']}] {s['snippet']}")
    print(f"\nElapsed: {result['elapsed']}s")
