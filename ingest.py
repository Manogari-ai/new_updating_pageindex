"""
ingest.py — PDF ingestion pipeline
  1. Text extraction via PyMuPDF  (single doc open, reused for images)
  2. Table extraction via Docling  (singleton converter)
  3. OCR for scanned/image pages via PaddleOCR
  4. Image extraction via PyMuPDF
  5. Chunking with overlap
  6. FAISS index — incremental add (never re-embeds existing chunks)
"""

import os
import json
import time
import hashlib
import logging
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Any

import fitz                          # PyMuPDF
import numpy as np
from PIL import Image
import io

import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_ocr_engine      = None
_embed_model     = None
_docling_conv    = None          # reuse across calls — init is expensive
_singleton_lock  = threading.Lock()

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        with _singleton_lock:
            if _ocr_engine is None:
                from paddleocr import PaddleOCR
                _ocr_engine = PaddleOCR(use_angle_cls=True, lang=config.OCR_LANG,
                                        use_gpu=config.OCR_USE_GPU, show_log=False)
                logger.info("PaddleOCR initialised")
    return _ocr_engine

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _singleton_lock:
            if _embed_model is None:
                from FlagEmbedding import FlagModel
                _embed_model = FlagModel(
                    config.EMBED_MODEL,
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                    use_fp16=True
                )
                logger.info(f"Embedding model loaded: {config.EMBED_MODEL}")
    return _embed_model

def get_docling():
    global _docling_conv
    if _docling_conv is None:
        with _singleton_lock:
            if _docling_conv is None:
                try:
                    from docling.document_converter import DocumentConverter
                    _docling_conv = DocumentConverter()
                    logger.info("Docling DocumentConverter initialised")
                except ImportError:
                    logger.warning("Docling not installed; table extraction disabled")
                    _docling_conv = "unavailable"
    return _docling_conv if _docling_conv != "unavailable" else None

# ─── Text + Image Extraction (single doc open) ────────────────────────────────

def extract_text_and_images(pdf_path: str, img_out: str) -> tuple:
    """
    Open the PDF ONCE and extract:
      - text pages (with OCR fallback for scanned pages)
      - embedded images with OCR text
    Returns (page_records, image_records).
    """
    os.makedirs(img_out, exist_ok=True)
    pdf_stem    = Path(pdf_path).stem
    page_records  = []
    image_records = []

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, 1):
        # ── Text ──────────────────────────────────────────────────────────────
        text   = page.get_text("text").strip()
        source = "pymupdf"

        if len(text) < 30:                          # scanned page → OCR
            logger.info(f"  OCR fallback p{page_num}")
            mat = fitz.Matrix(1.5, 1.5)            # 1.5× is enough, faster than 2×
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            arr = np.array(img)
            try:
                result = get_ocr().ocr(arr, cls=True)
                if result and result[0]:
                    text   = " ".join(l[1][0] for l in result[0] if l and l[1])
                    source = "paddleocr"
            except Exception as e:
                logger.warning(f"OCR p{page_num}: {e}")

        page_records.append({"page": page_num, "text": text, "source": source})

        # ── Embedded images ───────────────────────────────────────────────────
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                ext        = base_image["ext"]
                img_path   = os.path.join(img_out, f"{pdf_stem}_p{page_num}_img{img_idx}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                result  = get_ocr().ocr(img_arr, cls=True)
                ocr_txt = ""
                if result and result[0]:
                    ocr_txt = " ".join(l[1][0] for l in result[0] if l and l[1])
                if ocr_txt.strip():
                    image_records.append({
                        "page":       page_num,
                        "text":       f"[IMAGE OCR p{page_num}] {ocr_txt}",
                        "source":     "image_ocr",
                        "image_path": img_path
                    })
            except Exception as e:
                logger.warning(f"Image p{page_num} img{img_idx}: {e}")

    doc.close()
    return page_records, image_records

# ─── Table Extraction ─────────────────────────────────────────────────────────

def extract_tables_docling(pdf_path: str) -> List[Dict]:
    """Extract tables via Docling singleton; skip gracefully if unavailable."""
    converter = get_docling()
    if converter is None:
        return []
    table_chunks = []
    try:
        result = converter.convert(pdf_path)
        for table_idx, table in enumerate(result.document.tables):
            try:
                df = table.export_to_dataframe()
                if df is None or df.empty:
                    continue
                md_table = df.to_markdown(index=False)
                page_no  = getattr(table, "page_no", "?")
                table_chunks.append({
                    "page":   page_no,
                    "text":   f"[TABLE {table_idx+1}]\n{md_table}",
                    "source": "docling_table"
                })
            except Exception as e:
                logger.warning(f"Table {table_idx} export: {e}")
    except Exception as e:
        logger.warning(f"Docling failed: {e}")
    return table_chunks

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, page: int, source: str,
               pdf_name: str,
               chunk_size: int = config.CHUNK_SIZE,
               overlap: int    = config.CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks, start, idx = [], 0, 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append({
                "chunk_id": hashlib.md5(f"{pdf_name}_{page}_{idx}".encode()).hexdigest()[:12],
                "pdf":    pdf_name,
                "page":   page,
                "source": source,
                "text":   piece
            })
            idx += 1
        start += chunk_size - overlap
    return chunks

# ─── Embedding + FAISS ────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts in large batches; L2-normalise for cosine via IndexFlatIP."""
    model      = get_embed_model()
    embeddings = model.encode(texts, batch_size=64)   # 64 saturates faster than 32
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms      = np.where(norms == 0, 1.0, norms)
    return (embeddings / norms).astype(np.float32)

def _load_existing() -> tuple:
    """Load existing chunks + FAISS index, or return empty structures."""
    import faiss
    if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.CHUNKS_JSON_PATH):
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks
    return None, []

def save_index(index, chunks: List[Dict]):
    """Persist FAISS index and chunk metadata (compact JSON — no indent)."""
    import faiss
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)   # no indent → 3× smaller/faster
    logger.info(f"Saved {index.ntotal} vectors, {len(chunks)} chunks")

def load_index():
    """Load existing FAISS index and chunks (used by rag.py)."""
    import faiss
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

# ─── Delete PDF from Index ────────────────────────────────────────────────────

def delete_pdf_from_index(pdf_name: str) -> Dict:
    """
    Remove all chunks for pdf_name, re-embed remaining, rebuild index.
    """
    if not os.path.exists(config.CHUNKS_JSON_PATH):
        return {"status": "error", "message": "No index found"}

    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    target    = pdf_name.replace(".pdf", "")
    remaining = [c for c in all_chunks if c.get("pdf", "").replace(".pdf", "") != target]
    removed   = len(all_chunks) - len(remaining)

    if removed == 0:
        return {"status": "not_found", "message": f"No chunks for '{pdf_name}'"}

    if not remaining:
        for path in (config.FAISS_INDEX_PATH, config.CHUNKS_JSON_PATH):
            if os.path.exists(path):
                os.remove(path)
        return {"status": "ok", "pdf": pdf_name, "removed_chunks": removed, "remaining_chunks": 0}

    embeddings = embed_texts([c["text"] for c in remaining])
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    save_index(index, remaining)
    logger.info(f"Deleted '{pdf_name}': {removed} chunks removed, {len(remaining)} remain")
    return {"status": "ok", "pdf": pdf_name, "removed_chunks": removed, "remaining_chunks": len(remaining)}

# ─── Main Ingestion Entry ─────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> Dict:
    """
    Incremental ingestion:
      - Extracts + embeds ONLY the new PDF's chunks
      - Adds them directly to the existing FAISS index (no re-embedding old chunks)
    """
    t0       = time.time()
    pdf_name = Path(pdf_path).stem
    img_out  = os.path.join(config.IMAGES_DIR, pdf_name)
    logger.info(f"=== Ingesting: {pdf_path} ===")

    # 1+4. Text extraction + image OCR in ONE doc pass
    page_records, image_records = extract_text_and_images(pdf_path, img_out)

    # 2. Table extraction (Docling singleton, no re-init cost)
    table_records = extract_tables_docling(pdf_path)

    # 3. Chunk everything
    all_records = page_records + table_records + image_records
    new_chunks  = []
    for rec in all_records:
        new_chunks.extend(chunk_text(rec["text"], rec["page"], rec["source"], pdf_name))

    if not new_chunks:
        logger.warning("No chunks produced!")
        return {"status": "empty", "chunks": 0}

    # 4. Load existing index; strip any old version of this PDF
    import faiss
    existing_index, existing_chunks = _load_existing()
    existing_chunks = [c for c in existing_chunks if c.get("pdf") != pdf_name]

    # 5. Embed ONLY the new chunks
    logger.info(f"Embedding {len(new_chunks)} new chunks (existing: {len(existing_chunks)}) …")
    new_embeddings = embed_texts([c["text"] for c in new_chunks])

    # 6. Build/extend index
    dim = new_embeddings.shape[1]
    if existing_index is not None and existing_index.ntotal == len(existing_chunks):
        # Incremental: add new vectors to the existing flat index
        existing_index.add(new_embeddings)
        final_index  = existing_index
        final_chunks = existing_chunks + new_chunks
    else:
        # First PDF or mismatch → rebuild from scratch (still only new chunks here)
        final_chunks = existing_chunks + new_chunks
        if existing_chunks:
            # Need to re-embed the kept existing chunks too (rare: mismatch case)
            all_emb = embed_texts([c["text"] for c in final_chunks])
        else:
            all_emb = new_embeddings
        final_index = faiss.IndexFlatIP(dim)
        final_index.add(all_emb)

    save_index(final_index, final_chunks)

    elapsed = round(time.time() - t0, 2)
    summary = {
        "status":        "ok",
        "pdf":           pdf_name,
        "pages":         len(page_records),
        "tables":        len(table_records),
        "images":        len(image_records),
        "total_chunks":  len(new_chunks),
        "total_indexed": len(final_chunks),
        "elapsed_sec":   elapsed
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/file.pdf>")
        sys.exit(1)
    result = ingest_pdf(sys.argv[1])
    print(json.dumps(result, indent=2))
