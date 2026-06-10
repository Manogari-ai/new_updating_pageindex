"""
rag.py — Retrieval-Augmented Generation engine
  1. Query embedding (BAAI/bge-m3)
  2. FAISS top-K retrieval
  3. Re-ranking (BAAI/bge-reranker-large)
  4. Prompt assembly
  5. Qwen3 response via Ollama
"""

import os
import json
import time
import logging
import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_embed_model   = None
_reranker      = None
_faiss_index   = None
_chunks: List[Dict] = []

def get_embed_model():
    global _embed_model
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
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(config.RERANKER_MODEL, use_fp16=True)
        logger.info(f"Re-ranker ready: {config.RERANKER_MODEL}")
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
    logger.info(f"Index loaded: {_faiss_index.ntotal} vectors, {len(_chunks)} chunks")

# Load on import
load_index()

# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = config.TOP_K_RETRIEVE) -> List[Dict]:
    """Embed query and fetch top-K chunks from FAISS."""
    if _faiss_index is None or not _chunks:
        return []

    model = get_embed_model()
    q_raw = model.encode([query]).astype(np.float32)
    norm  = np.linalg.norm(q_raw, axis=1, keepdims=True)
    q_vec = q_raw / np.where(norm == 0, 1.0, norm)

    scores, indices = _faiss_index.search(q_vec, min(top_k, len(_chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results

# ─── Re-ranking ───────────────────────────────────────────────────────────────

def rerank(query: str, candidates: List[Dict],
           top_k: int = config.TOP_K_RERANK) -> List[Dict]:
    """Re-rank candidates with bge-reranker-large."""
    if not candidates:
        return []
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

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions strictly based on the provided document context.

Rules:
- Answer only from the CONTEXT provided below.
- If the answer is not in the context, say "I could not find relevant information in the uploaded documents."
- Cite page numbers when available (e.g., "According to page 3…").
- Be concise but complete.
- For tables, present data clearly.
- Do not hallucinate or add information from outside the context.
"""

def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        src  = c.get("source", "text")
        page = c.get("page", "?")
        pdf  = c.get("pdf", "document")
        context_parts.append(
            f"[Source {i} | PDF: {pdf} | Page: {page} | Type: {src}]\n{c['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_str}\n\nQUESTION:\n{query}\n\nANSWER:"

# ─── LLM (Qwen3 via Ollama) ───────────────────────────────────────────────────

def call_ollama(prompt: str, stream: bool = False) -> str:
    """Send prompt to Ollama and return response text."""
    try:
        import ollama
        response = ollama.generate(
            model=config.OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 8192,
            }
        )
        return response.get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"[LLM Error] Could not reach Ollama ({config.OLLAMA_MODEL}). Is Ollama running? Error: {e}"

# ─── Chat History Logging ─────────────────────────────────────────────────────

def log_chat(query: str, answer: str, sources: List[Dict]):
    """Append Q&A + source info to chat_history.txt."""
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        with open(config.CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'='*70}\n")
            f.write(f"[{ts}]\n")
            f.write(f"Q: {query}\n\n")
            f.write(f"A: {answer}\n\n")
            f.write("Sources:\n")
            for s in sources:
                f.write(f"  - PDF: {s.get('pdf')} | Page: {s.get('page')} | "
                        f"Type: {s.get('source')} | Score: {s.get('rerank_score', s.get('retrieval_score', '?')):.4f}\n")
    except Exception as e:
        logger.warning(f"Chat logging failed: {e}")

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

    # 1. Retrieve
    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)

    # 2. Re-rank
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    if not top_chunks:
        return {
            "answer":  "No relevant passages found for your query.",
            "sources": [],
            "elapsed": round(time.time() - t0, 2)
        }

    # 3. Build prompt and call LLM
    prompt   = build_prompt(query, top_chunks)
    response = call_ollama(prompt)

    # 4. Log
    log_chat(query, response, top_chunks)

    # 5. Return
    sources = [{
        "pdf":    c.get("pdf", ""),
        "page":   c.get("page", ""),
        "source": c.get("source", ""),
        "score":  round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    return {
        "answer":  response,
        "sources": sources,
        "elapsed": round(time.time() - t0, 2)
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    result = answer(q)
    print("\nAnswer:", result["answer"])
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['pdf']} | p.{s['page']}] {s['snippet']}")
    print(f"\nElapsed: {result['elapsed']}s")
