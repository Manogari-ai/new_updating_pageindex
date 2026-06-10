"""
ingest.py — PDF ingestion pipeline
  1. Text extraction via PyMuPDF
  2. Table extraction via Docling
  3. OCR for scanned/image pages via PaddleOCR
  4. Image extraction via PyMuPDF
  5. Chunking with overlap
  6. FAISS index creation with BAAI/bge-m3 embeddings
"""

import os
import json
import time
import hashlib
import logging
import warnings
import base64
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

_ocr_engine   = None
_embed_model  = None
_faiss_index  = None

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang=config.OCR_LANG,
                                 use_gpu=config.OCR_USE_GPU, show_log=False)
        logger.info("PaddleOCR initialised")
    return _ocr_engine

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import FlagModel
        _embed_model = FlagModel(config.EMBED_MODEL,
                                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                                  use_fp16=True)
        logger.info(f"Embedding model loaded: {config.EMBED_MODEL}")
    return _embed_model

# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_pymupdf(pdf_path: str) -> List[Dict]:
    """Extract text page-by-page using PyMuPDF."""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        pages.append({
            "page": page_num,
            "text": text,
            "source": "pymupdf"
        })
    doc.close()
    return pages

def is_scanned_page(page_text: str, threshold: int = 30) -> bool:
    """Heuristic: page is likely scanned if extracted text is very short."""
    return len(page_text.strip()) < threshold

def ocr_page(pdf_path: str, page_num: int) -> str:
    """Run PaddleOCR on a single PDF page rendered as image."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)          # 2× zoom for better OCR
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        # PaddleOCR accepts numpy array
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        ocr = get_ocr()
        result = ocr.ocr(img_array, cls=True)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for line in result[0] if line and line[1]]
        return " ".join(lines)
    except Exception as e:
        logger.warning(f"OCR failed on page {page_num}: {e}")
        return ""

# ─── Table Extraction ─────────────────────────────────────────────────────────

def extract_tables_docling(pdf_path: str) -> List[Dict]:
    """Extract tables using Docling; returns list of table dicts with page info."""
    table_chunks = []
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        for table_idx, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe()
                if df is None or df.empty:
                    continue
                # Markdown representation for retrieval
                md_table = df.to_markdown(index=False)
                page_no = getattr(table, "page_no", "?")
                table_chunks.append({
                    "page": page_no,
                    "text": f"[TABLE {table_idx+1}]\n{md_table}",
                    "source": "docling_table"
                })
            except Exception as e:
                logger.warning(f"Table {table_idx} export failed: {e}")
    except ImportError:
        logger.warning("Docling not installed; skipping table extraction.")
    except Exception as e:
        logger.warning(f"Docling extraction failed: {e}")
    return table_chunks

# ─── Image Extraction ─────────────────────────────────────────────────────────

def extract_images_pymupdf(pdf_path: str, output_dir: str) -> List[Dict]:
    """Extract embedded images and run OCR on them; save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    image_records = []
    doc = fitz.open(pdf_path)
    pdf_stem = Path(pdf_path).stem

    for page_num, page in enumerate(doc, 1):
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                ext        = base_image["ext"]
                img_name   = f"{pdf_stem}_p{page_num}_img{img_idx}.{ext}"
                img_path   = os.path.join(output_dir, img_name)

                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # OCR the extracted image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_array = np.array(img)
                ocr = get_ocr()
                result = ocr.ocr(img_array, cls=True)
                ocr_text = ""
                if result and result[0]:
                    ocr_text = " ".join([line[1][0] for line in result[0] if line and line[1]])

                if ocr_text.strip():
                    image_records.append({
                        "page": page_num,
                        "text": f"[IMAGE OCR p{page_num}] {ocr_text}",
                        "source": "image_ocr",
                        "image_path": img_path
                    })
            except Exception as e:
                logger.warning(f"Image extraction failed (page {page_num}, img {img_idx}): {e}")

    doc.close()
    return image_records

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, page: int, source: str,
               pdf_name: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "chunk_id": hashlib.md5(f"{pdf_name}_{page}_{chunk_idx}".encode()).hexdigest()[:12],
                "pdf":      pdf_name,
                "page":     page,
                "source":   source,
                "text":     chunk_text_str
            })
            chunk_idx += 1
        start += chunk_size - overlap
    return chunks

# ─── Embedding + FAISS ────────────────────────────────────────────────────────

def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    """Embed all chunk texts using BAAI/bge-m3."""
    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks …")
    embeddings = model.encode(texts, batch_size=32)
    # Manual L2 normalisation for cosine similarity via IndexFlatIP
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    return embeddings.astype(np.float32)

def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build flat L2 FAISS index."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (cosine after normalisation)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index

def save_index(index, chunks: List[Dict]):
    """Persist FAISS index and chunk metadata."""
    import faiss
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved index → {config.FAISS_INDEX_PATH}")
    logger.info(f"Saved chunks → {config.CHUNKS_JSON_PATH}")

def load_index():
    """Load existing FAISS index and chunks."""
    import faiss
    index  = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

# ─── Main Ingestion Entry ─────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> Dict:
    """
    Full ingestion pipeline for a single PDF.
    Returns summary dict.
    """
    t0 = time.time()
    pdf_name = Path(pdf_path).stem
    img_out  = os.path.join(config.IMAGES_DIR, pdf_name)
    logger.info(f"=== Ingesting: {pdf_path} ===")

    # 1. PyMuPDF text extraction
    pages = extract_text_pymupdf(pdf_path)

    # 2. OCR fallback for scanned pages
    for p in pages:
        if is_scanned_page(p["text"]):
            logger.info(f"  OCR fallback for page {p['page']}")
            ocr_text = ocr_page(pdf_path, p["page"])
            if ocr_text:
                p["text"] = ocr_text
                p["source"] = "paddleocr"

    # 3. Table extraction via Docling
    table_pages = extract_tables_docling(pdf_path)

    # 4. Image extraction + OCR
    image_pages = extract_images_pymupdf(pdf_path, img_out)

    # 5. Combine all page records
    all_records = pages + table_pages + image_pages

    # 6. Chunk
    all_chunks = []
    for rec in all_records:
        all_chunks.extend(chunk_text(rec["text"], rec["page"], rec["source"], pdf_name))

    if not all_chunks:
        logger.warning("No chunks produced!")
        return {"status": "empty", "chunks": 0}

    # 7. Load existing chunks and merge
    existing_chunks = []
    if os.path.exists(config.CHUNKS_JSON_PATH):
        with open(config.CHUNKS_JSON_PATH, "r") as f:
            existing_chunks = json.load(f)
        # Remove old chunks for this PDF
        existing_chunks = [c for c in existing_chunks if c.get("pdf") != pdf_name]

    merged_chunks = existing_chunks + all_chunks

    # 8. Re-embed ALL chunks and rebuild index
    embeddings = embed_chunks(merged_chunks)
    index = build_faiss_index(embeddings)
    save_index(index, merged_chunks)

    elapsed = round(time.time() - t0, 2)
    summary = {
        "status":       "ok",
        "pdf":          pdf_name,
        "pages":        len(pages),
        "tables":       len(table_pages),
        "images":       len(image_pages),
        "total_chunks": len(all_chunks),
        "total_indexed":len(merged_chunks),
        "elapsed_sec":  elapsed
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
