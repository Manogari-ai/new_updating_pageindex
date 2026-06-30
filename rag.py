"""
rag.py — Retrieval-Augmented Generation engine
  0. Fast path  — structured directory-entry lookup (e.g. "FRRO Chennai
     phone number") answered straight from chunks.json via fuzzy match,
     bypassing embedding/FAISS/reranking/LLM entirely. Falls through to
     the normal pipeline below if nothing matches confidently.
  0b. Answer cache — exact-repeat queries return the previously computed
     answer instantly. Cache is cleared whenever the index is reloaded.
  1. Query embedding  (BAAI/bge-m3, with LRU cache)
  2. FAISS top-K retrieval
  3. Adaptive re-ranking  (bge-reranker-v2-m3 — 4× faster than large)
     skipped when top retrieval score is already very high
  4. Prompt assembly  (no source/page metadata leaked into the prompt)
  5. Qwen3 via Ollama  (kept warm via keep_alive, streaming → SSE, async chat logging)
  6. Post-processing   (strip any stray citation artifacts, log response time)

All heavy models (embedder, reranker, Ollama) are pre-loaded once via
warmup() so the FIRST real user query doesn't pay a multi-second cold-start
cost — call rag.warmup() once at app startup.
"""

import os
import re
import json
import time
import logging
import datetime
import threading
from functools import lru_cache
from typing import List, Dict, Optional

import numpy as np

import config
import ingest

logger = logging.getLogger(__name__)

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_embed_model  = None
_reranker     = None
_faiss_index  = None
_chunks: List[Dict] = []
_model_lock   = threading.Lock()

# ─── Answer cache ─────────────────────────────────────────────────────────────
# Plain bounded dict, not a strict LRU — good enough for "don't recompute
# an answer we just gave a minute ago" without adding a dependency. Cleared
# wholesale on load_index() so it can never serve a stale answer after the
# corpus changes. Defined here (before load_index() is ever called) since
# load_index() references it directly on module import.

_answer_cache: Dict[str, Dict] = {}
_answer_cache_lock = threading.Lock()

def _cache_key(query: str) -> str:
    return query.strip().lower()

def get_cached_answer(query: str) -> Optional[Dict]:
    with _answer_cache_lock:
        cached = _answer_cache.get(_cache_key(query))
        return dict(cached) if cached else None

def set_cached_answer(query: str, result: Dict):
    with _answer_cache_lock:
        if len(_answer_cache) >= config.ANSWER_CACHE_SIZE:
            _answer_cache.pop(next(iter(_answer_cache)))   # evict oldest entry
        _answer_cache[_cache_key(query)] = {
            "answer":  result["answer"],
            "sources": result["sources"],
        }

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
    # Stale cached answers reference content that may no longer be current
    # (a PDF could have just been deleted/replaced) — clear them too.
    _answer_cache.clear()
    logger.info(f"Index loaded: {_faiss_index.ntotal} vectors, {len(_chunks)} chunks")

# ─── Query embedding with cache ───────────────────────────────────────────────
# Bumped cache size 256 → 512: cheap (a few KB per entry) and raises the hit
# rate for repeated/near-repeated questions, skipping embedding entirely.

@lru_cache(maxsize=512)
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

# ─── Fast path: structured directory lookup ──────────────────────────────────
# Reuses ingest.py's query_location() — the SAME fuzzy matcher and field
# parser used everywhere else — so "FRRO Chennai phone number" or "Cochin
# POE in-charge" gets answered directly from the parsed entry, with zero
# embedding/FAISS/reranking/LLM cost. This is the single biggest lever for
# response time on directory-style content (FRRO/POE/etc.), since it skips
# every expensive step entirely instead of just speeding one of them up.
#
# Returns None (falls through to the normal pipeline) whenever:
#   - there are no directory_entry chunks in the index at all, or
#   - the query doesn't fuzzy-match a location with enough confidence
# so non-directory PDFs and genuinely free-text questions are unaffected.

def try_fast_path(query: str) -> Optional[Dict]:
    if not _chunks:
        return None
    try:
        result = ingest.query_location(query, chunks=_chunks,
                                        threshold=config.FASTPATH_THRESHOLD)
    except Exception as e:
        logger.warning(f"Fast path lookup failed, falling back to full RAG: {e}")
        return None
    if not result:
        return None

    if result.get("field"):
        source_chunk = result["chunk"]
        answer_text  = result["answer"]
    else:
        # A location matched but no specific field was asked for — hand
        # back the whole entry rather than guessing which field they want.
        source_chunk = result["chunks"][0]
        answer_text  = source_chunk["text"]

    # Safety guard: a matched-but-empty field is a sign the parser found
    # the location but not a clean value for it. Returning blank text would
    # be a quality regression — fall through to the full RAG pipeline
    # instead, which can still reason about the surrounding chunk text.
    if not answer_text or not answer_text.strip():
        return None

    return {
        "answer": answer_text,
        "sources": [{
            "pdf":     source_chunk.get("pdf", ""),
            "page":    source_chunk.get("page", ""),
            "source":  source_chunk.get("source", ""),
            "score":   1.0,
            "snippet": source_chunk["text"][:200],
        }],
        "fast_path": True,
    }

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
_RERANK_SKIP_THRESHOLD = float(os.environ.get("RERANK_SKIP_THRESHOLD", "0.88"))

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
# Conversational tone, no citation instructions, key terms highlighted in bold.
# NOTE: source/page/pdf metadata is intentionally NOT included in the prompt
# text below — the model can only cite what it can see, so keeping that
# metadata out of its context is the most reliable way to stop it from
# mentioning pages/sources, and it also shortens the prompt (faster prefill).

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable assistant having a natural conversation "
    "with the user about their documents.\n\n"
    "Rules:\n"
    "- Answer using only the information in the CONTEXT below.\n"
    "- If the answer isn't in the context, say so naturally, e.g. "
    "\"I couldn't find that in the documents you've shared.\"\n"
    "- Write like you're explaining things to a colleague: clear, warm, and "
    "conversational — not robotic, and never mention \"the context\" itself.\n"
    "- Never mention page numbers, source numbers, or which file a fact came "
    "from. Just answer the question directly, as if you simply know it.\n"
    "- Bold the key terms, names, numbers, and important keywords using "
    "**markdown bold** so the most important parts of your answer stand out "
    "at a glance.\n"
    "- Keep it concise but complete — get to the point without padding.\n"
    "- For tables or lists, format them clearly.\n"
    "- Never make anything up or add information that isn't in the context."
)

def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_text}"
        f"\n\nQUESTION:\n{query}\n\nANSWER:"
    )

# ─── Citation / page-reference cleanup ───────────────────────────────────────
# Defense-in-depth: the prompt above already tells the model not to cite
# pages/sources, but models occasionally slip up. This strips common
# citation-style artifacts from the final text before it reaches the user.

_CITATION_PATTERNS = [
    re.compile(r"\(?\s*according to\s+page\s+\d+\s*\)?", re.IGNORECASE),
    re.compile(r"\(?\s*as (?:stated|mentioned|noted) (?:in|on)\s+page\s+\d+\s*\)?", re.IGNORECASE),
    re.compile(r"\(?\s*page\s+\d+\s*\)?", re.IGNORECASE),
    re.compile(r"\[?\s*source\s*\d+\s*\]?\s*:?", re.IGNORECASE),
    re.compile(r"\(?\s*p\.\s?\d+\s*\)?", re.IGNORECASE),
]

def strip_citations(text: str) -> str:
    """Remove any stray page/source references the model may have added."""
    cleaned = text
    for pattern in _CITATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # collapse whitespace / stray punctuation left behind by the removals
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

# ─── LLM (Qwen3 via Ollama) — streaming ──────────────────────────────────────
# keep_alive keeps the model loaded in memory between requests so it doesn't
# have to be reloaded from disk on every query — this is usually the single
# biggest win for perceived response time on a local Ollama setup.

_OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

def stream_ollama(prompt: str):
    """
    Generator that yields raw text tokens from Ollama as they arrive.
    """
    try:
        import ollama
        for chunk in ollama.generate(
            model=config.OLLAMA_MODEL,
            prompt=prompt,
            stream=True,
            keep_alive=_OLLAMA_KEEP_ALIVE,
            options={
                "temperature": 0.1,
                "top_p":       0.9,
                # Smaller context window than the previous 4096 default —
                # less attention compute per token = faster prefill on
                # every request. Raise via config if you see truncation.
                "num_ctx":     config.OLLAMA_NUM_CTX,
                # Hard cap on generated tokens so a single verbose answer
                # can't dominate response time. Raise if answers are
                # getting cut off for your content.
                "num_predict": config.OLLAMA_NUM_PREDICT,
            }
        ):
            token = chunk.get("response", "")
            if token:
                yield token
    except Exception as e:
        logger.error(f"Ollama stream failed: {e}")
        yield f"[LLM Error] Could not reach Ollama ({config.OLLAMA_MODEL}). Error: {e}"


_SENTENCE_END = re.compile(r"([.!?])(\s+)")

def stream_ollama_clean(prompt: str):
    """
    Wraps stream_ollama: buffers tokens into sentence-sized chunks so that
    strip_citations() can see a whole phrase before it's sent to the client.
    Preserves original whitespace (including newlines before bullets/lists)
    instead of collapsing everything to single spaces.
    """
    buffer = ""
    while True:
        match = _SENTENCE_END.search(buffer)
        if not match:
            break
        end = match.end()
        sentence  = buffer[:match.start() + 1]   # text up to and including . ! ?
        separator = match.group(2)               # the actual whitespace that followed
        cleaned = strip_citations(sentence)
        if cleaned:
            yield cleaned + separator
        buffer = buffer[end:]
        continue

    for token in stream_ollama(prompt):
        buffer += token
        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break
            sentence  = buffer[:match.start() + 1]
            separator = match.group(2)
            cleaned = strip_citations(sentence)
            if cleaned:
                yield cleaned + separator
            buffer = buffer[match.end():]

    if buffer:
        cleaned = strip_citations(buffer)
        if cleaned:
            yield cleaned

def call_ollama(prompt: str) -> str:
    """Blocking call — collects the full streamed response, then sanitizes it."""
    raw = "".join(stream_ollama(prompt))
    return strip_citations(raw)

# ─── Warmup ───────────────────────────────────────────────────────────────────
# Every model here is normally lazy-loaded on first use, which means
# whoever sends the FIRST real query pays for loading the embedder, the
# reranker, AND a cold Ollama model load all at once — often several
# seconds. Calling this once at app startup moves that entire cost off
# the request path so it never shows up as "response time" to a user.

def warmup():
    def _warm():
        try:
            t0 = time.time()
            get_embed_model()
            get_reranker()
            import ollama
            ollama.generate(
                model=config.OLLAMA_MODEL,
                prompt="hi",
                stream=False,
                keep_alive=_OLLAMA_KEEP_ALIVE,
                options={"num_predict": 1},
            )
            logger.info(f"Warmup complete in {time.time() - t0:.2f}s "
                        f"(embed model, reranker, Ollama all loaded)")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal — models will lazy-load instead): {e}")
    threading.Thread(target=_warm, daemon=True).start()

# ─── Async chat logging ───────────────────────────────────────────────────────

def _format_duration(elapsed: float) -> str:
    """e.g. 136.03 -> '136.03 seconds = 2 minutes 16.03 seconds'"""
    minutes, seconds = divmod(elapsed, 60)
    return f"{elapsed:.2f} seconds = {int(minutes)} minutes {seconds:.2f} seconds"

def log_chat(query: str, answer: str, sources: List[Dict], elapsed: float):
    """Fire-and-forget: write Q&A log (with response time) in a background thread."""
    def _write():
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration = _format_duration(elapsed)
            with open(config.CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(
                    f"\n{'='*70}\n"
                    f"[{ts}] (response time: {duration})\n"
                    f"Q: {query}\n\nA: {answer}\n\nSources:\n"
                )
                for s in sources:
                    score = s.get("rerank_score", s.get("retrieval_score", "?"))
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                    f.write(f"  - PDF: {s.get('pdf')} | Page: {s.get('page')} | "
                            f"Type: {s.get('source')} | Score: {score_str}\n")
        except Exception as e:
            logger.warning(f"Chat logging failed: {e}")
    threading.Thread(target=_write, daemon=True).start()

# ─── Main RAG Entry ───────────────────────────────────────────────────────────

def answer(query: str) -> Dict:
    """
    Full pipeline: fast-path → cache → retrieve → rerank → generate → clean → log.
    Returns dict with 'answer', 'sources', 'elapsed'.
    """
    t0 = time.time()

    if _faiss_index is None:
        return {
            "answer":  "No documents have been ingested yet. Please upload and ingest a PDF first.",
            "sources": [],
            "elapsed": 0.0
        }

    # 0. Fast path — structured directory lookup, no LLM involved at all.
    fast = try_fast_path(query)
    if fast:
        elapsed = round(time.time() - t0, 4)
        log_chat(query, fast["answer"], [], elapsed)
        fast["elapsed"] = elapsed
        return fast

    # 0b. Exact-repeat question already answered before.
    cached = get_cached_answer(query)
    if cached:
        cached["elapsed"] = round(time.time() - t0, 4)
        cached["cached"]  = True
        return cached

    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    if not top_chunks:
        elapsed = round(time.time() - t0, 2)
        return {
            "answer":  "No relevant passages found for your query.",
            "sources": [],
            "elapsed": elapsed
        }

    prompt   = build_prompt(query, top_chunks)
    response = call_ollama(prompt)   # already cleaned of citation artifacts

    sources = [{
        "pdf":     c.get("pdf", ""),
        "page":    c.get("page", ""),
        "source":  c.get("source", ""),
        "score":   round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    elapsed = round(time.time() - t0, 2)
    log_chat(query, response, top_chunks, elapsed)   # async — does not block return

    result = {
        "answer":  response,
        "sources": sources,
        "elapsed": elapsed
    }
    set_cached_answer(query, result)
    return result


# ─── Streaming answer (for /api/chat/stream) ─────────────────────────────────

def answer_stream(query: str):
    """
    Generator for SSE streaming:
      Yields JSON lines:  {"token": "..."}
      Final line:         {"done": true, "sources": [...], "elapsed": X}
    Tokens are pre-cleaned of citation artifacts via stream_ollama_clean().
    Fast-path and cache hits skip straight to a single token + done event.
    """
    t0 = time.time()

    if _faiss_index is None:
        yield json.dumps({"token": "No documents have been ingested yet."})
        yield json.dumps({"done": True, "sources": [], "elapsed": 0.0})
        return

    fast = try_fast_path(query)
    if fast:
        elapsed = round(time.time() - t0, 4)
        log_chat(query, fast["answer"], [], elapsed)
        yield json.dumps({"token": fast["answer"]})
        yield json.dumps({"done": True, "sources": fast["sources"],
                           "elapsed": elapsed, "fast_path": True})
        return

    cached = get_cached_answer(query)
    if cached:
        elapsed = round(time.time() - t0, 4)
        yield json.dumps({"token": cached["answer"]})
        yield json.dumps({"done": True, "sources": cached["sources"],
                           "elapsed": elapsed, "cached": True})
        return

    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    if not top_chunks:
        elapsed = round(time.time() - t0, 2)
        yield json.dumps({"token": "No relevant passages found for your query."})
        yield json.dumps({"done": True, "sources": [], "elapsed": elapsed})
        return

    prompt      = build_prompt(query, top_chunks)
    full_answer = []

    for token in stream_ollama_clean(prompt):
        full_answer.append(token)
        yield json.dumps({"token": token})

    sources = [{
        "pdf":     c.get("pdf", ""),
        "page":    c.get("page", ""),
        "source":  c.get("source", ""),
        "score":   round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    elapsed     = round(time.time() - t0, 2)
    final_text  = "".join(full_answer)
    log_chat(query, final_text, top_chunks, elapsed)
    set_cached_answer(query, {"answer": final_text, "sources": sources})

    yield json.dumps({"done": True, "sources": sources, "elapsed": elapsed})


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    result = answer(q)
    print("\nAnswer:", result["answer"])
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['pdf']} | p.{s['page']}] {s['snippet']}")
    print(f"\nElapsed: {result['elapsed']}s")