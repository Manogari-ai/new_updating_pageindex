"""
ingest.py — PDF ingestion pipeline
  1. Text extraction via PyMuPDF  (single doc open, reused for images)
     + Table extraction via Docling, running CONCURRENTLY in a worker
       thread (they read the same file independently, so wall time is
       max(text, tables) instead of text + tables)
  1b. Docling itself is gated behind a near-free PyMuPDF structural
      pre-check (quick_table_scan) — the expensive ML conversion only
      runs if that heuristic actually sees a table-shaped region
  2. OCR for scanned/image pages via PaddleOCR (already gated — only
     runs when a page's native text layer is too short to be real text)
  3. Image extraction via PyMuPDF — SKIPPED ENTIRELY if the PDF has no
     embedded images at all (cheap upfront scan, no decoding), and
     OCR skipped for tiny embedded images (icons/logos/bullets) below
     a configurable pixel-area floor
  4. Directory-style listing detection (dynamic — FRRO, POE, or any other
     "N. Location / Label: value" block, detected by structure, not by
     hardcoded names)
  5. Chunking with overlap (for everything that ISN'T a directory listing)
  6. FAISS index — incremental add (never re-embeds existing chunks)
"""

import os
import re
import json
import time
import bisect
import hashlib
import logging
import threading
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

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

def pdf_has_images(doc) -> bool:
    """
    Cheap upfront check across all pages — only reads each page's image
    XObject list (metadata), never decodes pixel data. Lets the caller
    skip the entire embedded-image branch (no OCR engine load, no
    extract_image calls, no disk writes) for PDFs that simply don't
    contain any embedded images, which is common for plain text-only
    directories/notices.
    """
    for page in doc:
        if page.get_images(full=True):
            return True
    return False


def extract_text_and_images(pdf_path: str, img_out: str) -> tuple:
    """
    Open the PDF ONCE and extract:
      - text pages (with OCR fallback for scanned pages)
      - embedded images with OCR text (skipped entirely if the PDF has
        no embedded images at all)
    Returns (page_records, image_records).
    """
    os.makedirs(img_out, exist_ok=True)
    pdf_stem    = Path(pdf_path).stem
    page_records  = []
    image_records = []

    doc = fitz.open(pdf_path)

    # Decide ONCE whether the image-extraction branch is even relevant
    # for this PDF, instead of paying for get_images()/extract_image()
    # bookkeeping on every page when there's nothing there to find.
    do_image_extraction = config.EXTRACT_EMBEDDED_IMAGES and pdf_has_images(doc)
    if config.EXTRACT_EMBEDDED_IMAGES and not do_image_extraction:
        logger.info("No embedded images found — skipping image extraction entirely")

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
        if not do_image_extraction:
            continue

        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                width  = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip OCR (and the disk write) for tiny images — logos,
                # icons, and decorative bullets almost never carry text
                # worth indexing, and the OCR call is the expensive part
                # of this whole loop.
                if (width * height) < config.MIN_IMAGE_OCR_AREA:
                    continue

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

def quick_table_scan(pdf_path: str, min_rows: int = config.TABLE_PRECHECK_MIN_ROWS) -> bool:
    """
    Cheap structural pre-check using PyMuPDF's built-in table finder
    (line/rect heuristics, no model weights to load) so we only pay for
    the much heavier Docling ML conversion when there's actually
    something table-shaped on the page.

    Requires PyMuPDF >= 1.23 for Page.find_tables(); on older versions
    (or any other failure) this fails OPEN — i.e. assumes a table might
    exist — so a missing table is never silently dropped, only a wasted
    Docling call in the worst case.
    """
    try:
        doc = fitz.open(pdf_path)
        found = False
        for page in doc:
            tabs = page.find_tables()
            for t in tabs.tables:
                if len(t.rows) >= min_rows:
                    found = True
                    break
            if found:
                break
        doc.close()
        return found
    except Exception as e:
        logger.warning(f"Quick table scan unavailable, defaulting to 'run Docling': {e}")
        return True

def extract_tables_docling(pdf_path: str) -> List[Dict]:
    """Extract tables via Docling singleton; skip gracefully if unavailable
    or if nothing table-shaped was found by the cheap pre-check."""
    if not config.ENABLE_TABLE_EXTRACTION:
        return []

    if not quick_table_scan(pdf_path):
        logger.info("Quick scan found no table-like structures — skipping Docling")
        return []

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

# ─── Directory-Style Listing Detection ────────────────────────────────────────
# Dynamic / structural — NOT hardcoded to FRRO, POE, or any specific list.
# It only recognises the SHAPE of a directory: a numbered header
# ("1. Ahmedabad") followed by labeled fields ("In-Charge: ...",
# "Address: ...", "Phone: ...", "Email: ..."). The same logic is what
# picks up a 15-entry FRRO directory and a 12-entry POE directory — there
# is no per-directory list to maintain, nothing to keep in sync by hand.

# A numbered entry header, on its own line, possibly wrapping to the next
# line for the location name (matches how PyMuPDF extracts these lists):
#   "1.\nAhmedabad"   "12. Cochin"   "3.  Bangalore"
_ENTRY_HEADER_RE = re.compile(
    r'(?m)^\s*(\d{1,3})\s*\.\s*\n?\s*([A-Za-z][A-Za-z .\-/()&]{1,60})\s*$'
)

# Vocabulary of field labels seen across directory-style PDFs. If a new
# directory type uses different wording than FRRO/POE, this tuple is the
# ONLY place that would ever need a one-line addition — the detection and
# chunking logic itself never changes.
_FIELD_LABELS = (
    "in-?charge", "officer[- ]?in[- ]?charge", "designation", "contact person",
    "address", "phone", "telephone", "fax", "email", "e-?mail", "contact",
)
_FIELD_LABEL_LINE_RE = re.compile(
    r'(?i)^\s*o?\s*(' + "|".join(_FIELD_LABELS) + r')\s*:\s*(.*)$'
)

# Heading line just above the first entry — purely cosmetic, used only to
# label the chunk's "directory_title", e.g. "FRRO CONTACT DIRECTORY" or
# "POE DIRECTORY". Never used to decide what counts as a directory.
_TITLE_RE = re.compile(r'(?m)^([A-Z][A-Z \-]{4,60}(?:DIRECTORY|LIST|CONTACTS?))\s*$')


def _parse_entry_fields(block: str) -> Dict[str, str]:
    """
    Line-by-line field parser (handles values that wrap onto a second
    line, e.g. a long Address) and silently skips bare bullet markers
    ("o" alone on its own line) that PyMuPDF often leaves behind.
    """
    fields: Dict[str, str] = {}
    current_label: Optional[str] = None
    for raw_line in block.split("\n"):
        line = raw_line.strip()
        if not line or line == "o":
            continue
        m = _FIELD_LABEL_LINE_RE.match(line)
        if m:
            current_label = m.group(1).strip().lower()
            fields[current_label] = m.group(2).strip()
        elif current_label:
            # continuation of the previous field's value (wrapped line)
            fields[current_label] = (fields[current_label] + " " + line).strip()
    return fields


def detect_directory_entries(full_text: str, min_entries: int = 4) -> Tuple[List[Dict], str]:
    """
    Returns ([], "") if full_text doesn't look like a numbered directory
    listing, so the caller falls back to normal chunk_text() for
    everything else. A numbered header only counts as a real entry if it
    actually contains at least one labeled field — this is what stops a
    plain numbered list (steps, FAQ, etc.) from being misread as a
    directory.
    """
    headers = list(_ENTRY_HEADER_RE.finditer(full_text))
    if not headers:
        return [], ""

    raw_entries = []
    for i, m in enumerate(headers):
        start = m.start()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        block = full_text[start:end]

        fields = _parse_entry_fields(block)
        if fields:
            raw_entries.append({
                "number":        m.group(1),
                "location_name": m.group(2).strip(),
                "fields":        fields,
                "raw_text":      block.strip(),
                "char_start":    start,
            })

    if len(raw_entries) < min_entries:
        return [], ""

    title_match     = _TITLE_RE.search(full_text[:raw_entries[0]["char_start"]])
    directory_title = title_match.group(1).strip() if title_match else ""
    return raw_entries, directory_title


def entries_to_chunks(entries: List[Dict], pdf_name: str,
                       page_offsets: List[Tuple[int, int]],
                       directory_title: str = "") -> List[Dict]:
    """
    Converts detected entries into chunk dicts using the SAME schema your
    chunks.json already uses (chunk_id / pdf / page / source / text), plus
    a few extra metadata keys that uniquely identify each location:
      - entry_number    -> the original directory numbering (1-15, 1-12, ...)
      - location_name   -> the entry's title, used for exact/fuzzy lookups
      - directory_title -> which listing this came from (FRRO/POE/etc.)
    These chunks still get embedded and added to FAISS like everything
    else — nothing about storage changes, only how this content is split.
    """
    offsets = [o for o, _ in page_offsets]
    chunks  = []
    for e in entries:
        pos  = bisect.bisect_right(offsets, e["char_start"]) - 1
        page = page_offsets[max(pos, 0)][1]

        chunks.append({
            "chunk_id": hashlib.md5(
                f"{pdf_name}_dir_{e['number']}_{e['location_name']}".encode()
            ).hexdigest()[:12],
            "pdf":             pdf_name,
            "page":            page,
            "source":          "directory_entry",
            "directory_title": directory_title,
            "entry_number":    e["number"],
            "location_name":   e["location_name"],
            "text":            e["raw_text"],
        })
    return chunks

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

# ─── Location Lookup (dynamic — built fresh from chunks.json every time) ──────
# No static location list is stored or maintained anywhere. Every time
# this is called it scans whatever is currently in chunks.json for
# source == "directory_entry" and builds the lookup on the fly. Ingest a
# brand-new directory tomorrow (any number of locations, any title) and it
# shows up here automatically on the next call — nothing to edit.
#
# rag.py's fast-path uses exactly this lookup to answer directory-style
# questions (e.g. "FRRO Chennai phone number") with zero embedding, FAISS,
# reranking, or LLM cost.

def build_location_index(chunks: List[Dict]) -> Dict[str, List[int]]:
    """Maps normalized location name -> indices of matching chunks."""
    index: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        if c.get("source") == "directory_entry":
            key = c["location_name"].strip().lower()
            index.setdefault(key, []).append(i)
    return index


_FIELD_QUERY_HINTS = {
    "in-charge": ["in-charge", "incharge", "in charge", "officer"],
    "address":   ["address"],
    "phone":     ["phone", "contact number", "telephone"],
    "email":     ["email", "e-mail"],
}

def _requested_field(query: str) -> Optional[str]:
    q = query.lower()
    for field, hints in _FIELD_QUERY_HINTS.items():
        if any(h in q for h in hints):
            return field
    return None


def query_location(query: str, chunks: Optional[List[Dict]] = None,
                    threshold: int = 80) -> Optional[Dict]:
    """
    Resolve a query like "FRRO Chennai Directory" or
    "Chennai FRRO In-Charge Only" against whatever directory entries
    currently exist in chunks.json — works the same for FRRO, POE, or
    anything ingested later, since matching is by location_name, not by
    directory type.

    If `chunks` isn't passed in, it's loaded fresh via load_index().
    Returns None if no location in the query matches closely enough.
    """
    from rapidfuzz import fuzz, process   # lazy import, same pattern as other deps

    if chunks is None:
        _, chunks = load_index()

    loc_index = build_location_index(chunks)
    if not loc_index:
        return None

    match = process.extractOne(query.lower(), list(loc_index.keys()),
                                scorer=fuzz.partial_ratio)
    if not match or match[1] < threshold:
        return None

    matched_key    = match[0]
    matched_chunks = [chunks[i] for i in loc_index[matched_key]]

    field = _requested_field(query)
    if field:
        # Reuse the SAME line-aware parser used at ingestion time, so a
        # wrapped value (e.g. a long Address spanning two lines) comes
        # back complete here too, not just the first line of it.
        for c in matched_chunks:
            parsed = _parse_entry_fields(c["text"])
            if field in parsed:
                return {
                    "location": matched_key,
                    "field":    field,
                    "answer":   parsed[field],
                    "chunk":    c,
                }
        # field requested but not found in the entry -> fall through and
        # return the full entry instead of nothing

    return {"location": matched_key, "field": None, "chunks": matched_chunks}

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
      - Any directory-style listing (FRRO, POE, etc.) is split one chunk
        per location instead of by raw character count
      - Text/image extraction and table extraction run CONCURRENTLY: they
        read the same file independently, so wall-clock time becomes
        max(text_extraction, table_extraction) instead of the sum of the two
      - Docling itself only runs at all if quick_table_scan() saw something
        table-shaped; otherwise extract_tables_docling() returns instantly
      - Embedded-image OCR only runs at all if the PDF actually contains
        embedded images; otherwise that branch is skipped entirely
    """
    t0       = time.time()
    pdf_name = Path(pdf_path).stem
    img_out  = os.path.join(config.IMAGES_DIR, pdf_name)
    logger.info(f"=== Ingesting: {pdf_path} ===")

    # 1+2+3. Text extraction + image OCR, and table extraction, in parallel.
    # Each opens its own independent fitz.Document handle, so there's no
    # shared-state hazard between the two threads. Each branch internally
    # skips its own expensive work (image OCR / Docling) when that element
    # type isn't present in this particular PDF.
    with ThreadPoolExecutor(max_workers=2) as pool:
        text_future  = pool.submit(extract_text_and_images, pdf_path, img_out)
        table_future = pool.submit(extract_tables_docling, pdf_path)
        page_records, image_records = text_future.result()
        table_records               = table_future.result()

    # 1b. Reconstruct full document text (with page offsets) so directory
    #     entries that straddle a page break are still detected as one
    #     unbroken block, and can still be mapped back to a page number.
    full_text, page_offsets = "", []
    for rec in page_records:
        page_offsets.append((len(full_text), rec["page"]))
        full_text += rec["text"] + "\n"

    directory_entries, directory_title = detect_directory_entries(full_text)
    directory_chunks: List[Dict] = []
    if directory_entries:
        directory_chunks = entries_to_chunks(
            directory_entries, pdf_name, page_offsets, directory_title
        )
        covered_pages = {c["page"] for c in directory_chunks}
        # Pages fully covered by a clean directory entry are dropped from
        # the generic per-page text so they don't ALSO get fragmented by
        # chunk_text() below — one clean chunk per location, not both.
        page_records = [r for r in page_records if r["page"] not in covered_pages]
        logger.info(
            f"Detected directory listing '{directory_title or pdf_name}': "
            f"{len(directory_chunks)} locations"
        )

    # 4. Chunk everything that ISN'T a directory entry
    all_records = page_records + table_records + image_records
    new_chunks  = list(directory_chunks)
    for rec in all_records:
        new_chunks.extend(chunk_text(rec["text"], rec["page"], rec["source"], pdf_name))

    if not new_chunks:
        logger.warning("No chunks produced!")
        return {"status": "empty", "chunks": 0}

    # 5. Load existing index; strip any old version of this PDF
    import faiss
    existing_index, existing_chunks = _load_existing()
    existing_chunks = [c for c in existing_chunks if c.get("pdf") != pdf_name]

    # 6. Embed ONLY the new chunks
    logger.info(f"Embedding {len(new_chunks)} new chunks (existing: {len(existing_chunks)}) …")
    new_embeddings = embed_texts([c["text"] for c in new_chunks])

    # 7. Build/extend index
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
        "status":           "ok",
        "pdf":              pdf_name,
        "pages":            len(page_records),
        "tables":           len(table_records),
        "images":           len(image_records),
        "directory_title":  directory_title,
        "directory_entries": len(directory_chunks),
        "total_chunks":     len(new_chunks),
        "total_indexed":    len(final_chunks),
        "elapsed_sec":      elapsed
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