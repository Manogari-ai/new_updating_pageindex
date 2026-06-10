# PDF Extract Chatbot

A local RAG chatbot that extracts and understands text, tables, scanned pages, and images from PDFs, then answers questions using Qwen3 via Ollama.

---

## Technology Stack

| Purpose | Technology |
|---|---|
| Text Extraction | PyMuPDF |
| Tables | Docling |
| OCR | PaddleOCR |
| Image Extraction | PyMuPDF |
| Embeddings | BAAI/bge-m3 |
| Vector Search | FAISS |
| Re-ranking | bge-reranker-large |
| LLM | Qwen3 via Ollama |
| Backend | Flask |
| Frontend | HTML + Bootstrap 5 |

---

## Project Structure

```
pdf_chatbot/
├── app.py              ← Flask server (routes + API)
├── ingest.py           ← PDF extraction + FAISS indexing
├── rag.py              ← Retrieval + re-ranking + Qwen3 response
├── config.py           ← All configuration constants
├── requirements.txt    ← Python dependencies
├── templates/
│   ├── upload.html     ← PDF upload UI
│   └── chat.html       ← Chatbot UI + Architecture view
├── uploads/            ← Uploaded PDFs
├── chunks/             ← chunks.json (extracted chunk metadata)
├── index/              ← faiss.index (FAISS binary)
├── logs/               ← chat_history.txt
└── images/             ← Extracted PDF images
```

---

## Setup

### 1. Install Ollama and pull Qwen3

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull qwen3
```

### 2. Create Python environment

```bash
cd pdf_chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

PaddlePaddle note — choose the right build for your machine:
```bash
# CPU only
pip install paddlepaddle

# GPU (CUDA 11.8)
pip install paddlepaddle-gpu==2.6.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 4. Run the server

```bash
python app.py
```

Open http://localhost:5000

---

## Usage

### Upload PDFs
1. Go to http://localhost:5000/upload
2. Drag & drop one or more PDFs
3. Click **Ingest & Index** — extraction, OCR, and indexing runs automatically
4. You'll see chunk counts and elapsed time per file

### Chat
1. Go to http://localhost:5000 (or click "Go to Chat")
2. Type your question and press Enter
3. The chatbot will:
   - Embed your query with BAAI/bge-m3
   - Retrieve top 20 chunks from FAISS
   - Re-rank to top 5 with bge-reranker-large
   - Generate a grounded answer with Qwen3
   - Show source citations (PDF name, page, type, score)

### Architecture View
Click the **Architecture** tab in the chat UI to see the full pipeline diagram and live index stats.

### Chat History
All Q&A pairs are logged to `logs/chat_history.txt`. View via the **History** tab or open the file directly.

---

## Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 80 | Overlap between chunks |
| `EMBED_MODEL` | BAAI/bge-m3 | Embedding model |
| `RERANKER_MODEL` | BAAI/bge-reranker-large | Re-ranker model |
| `TOP_K_RETRIEVE` | 20 | Initial FAISS hits |
| `TOP_K_RERANK` | 5 | Chunks passed to LLM |
| `OLLAMA_MODEL` | qwen3 | Ollama model name |
| `OCR_USE_GPU` | False | Enable GPU OCR |

---

## CLI Usage

```bash
# Ingest a PDF directly
python ingest.py path/to/document.pdf

# Query from command line
python rag.py What are the visa requirements?
```

---

## Troubleshooting

**Ollama not responding**: Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull qwen3`).

**PaddleOCR install fails**: Try installing PaddlePaddle first, then PaddleOCR. GPU builds require matching CUDA version.

**Docling not available**: Install separately with `pip install docling`. Table extraction will be skipped gracefully if unavailable.

**Out of memory on embeddings**: Reduce batch size in `ingest.py` → `embed_chunks()` function.
