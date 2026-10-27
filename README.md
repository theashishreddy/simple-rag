# 🧠 Simple RAG: Lightweight & Modular Retrieval-Augmented Generation

An educational, extensible, production-ready Retrieval-Augmented Generation (RAG) project written in modern Python. Designed with zero bloated abstractions, complete transparency, and native offline fallback capabilities.

---

## 📐 How RAG Works

```text
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│ Documents (.md, │ ────> │  Text Splitter   │ ────> │ Embedding Model │
│   .txt, .json)  │       │ (Chunk & Overlap)│       │ (OpenAI / Local)│
└─────────────────┘       └──────────────────┘       └────────┬────────┘
                                                              │
                                                              ▼
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│ User Question   │ ────> │ Vector Similarity│ <──── │  Vector Store   │
│                 │       │ (Cosine Ranking) │       │ (NumPy / JSON)  │
└────────┬────────┘       └────────┬─────────┘       └─────────────────┘
         │                         │ Top-K Chunks
         ▼                         ▼
┌────────────────────────────────────────────┐
│      Context-Augmented Prompt Template     │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│          Large Language Model (LLM)        │ ────> Grounded Answer + Citations
└────────────────────────────────────────────┘
```

---

## ✨ Features

- **No Heavy Black-Box Frameworks**: Built from first principles so you can understand and customize every component (Loader, Chunking, Embeddings, Vector Indexing, Prompt Synthesis).
- **Dual Engine (API + Offline Fallback)**:
  - **OpenAI Engine**: `text-embedding-3-small` + `gpt-4o-mini` with real-time streaming and citation generation.
  - **Offline Local Engine**: Built-in TF-IDF sub-linear vectorizer + heuristic synthesis that works 100% offline with zero external API keys!
- **Fast In-Memory Vector Store**: Vectorized NumPy cosine similarity with instant JSON/file serialization and loading.
- **Multiple Interfaces**:
  1. **CLI Commands**: `simple-rag ingest`, `simple-rag query`
  2. **Interactive Terminal REPL**: `simple-rag chat` with live streaming & `/sources` inspection
  3. **Web UI Studio**: Full-featured FastAPI & Tailwind web application with document quick-ingest, live similarity matches, and markdown rendering.
- **Python Library API**: Clean, typed interfaces (`RAGPipeline`, `DocumentLoader`, `SimpleVectorStore`) ready for embed into any Python application.
- **Comprehensive Test Suite**: Pre-configured with `pytest` unit and integration tests.

---

## 📁 Project Structure

```text
simple-rag/
├── knowledge_base/               # Sample document knowledge base
│   ├── company_policies.md       # Remote work, PTO, health benefits
│   ├── engineering_handbook.md   # PR reviews, release cycle, SEV levels
│   └── travel_and_expenses.md    # Per diems, flights, expense filing
├── src/
│   └── simple_rag/
│       ├── __init__.py           # Library public exports & .env auto-loader
│       ├── models.py             # Document, Chunk, SearchResult, RAGResponse models
│       ├── document_loader.py    # Recursive multi-format loader (.txt, .md, .json)
│       ├── text_splitter.py      # Recursive splitter with sliding-window overlap
│       ├── embeddings.py         # OpenAI & Local TF-IDF embedding implementations
│       ├── vector_store.py       # NumPy cosine similarity search & persistence
│       ├── llm.py                # OpenAI & Local heuristic LLM wrappers
│       ├── pipeline.py           # Unified RAGPipeline orchestrator
│       ├── cli.py                # Rich interactive terminal interface
│       └── web.py                # FastAPI server & responsive browser UI
├── tests/                        # Automated unit and integration tests
│   ├── test_splitter.py
│   ├── test_vector_store.py
│   └── test_pipeline.py
├── pyproject.toml                # Project metadata and dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

This project is managed with [`uv`](https://docs.astral.sh/uv/) (or standard `pip`):

```bash
cd simple-rag

# Install all dependencies into virtual environment
uv sync
```

*(Optional)* If you wish to use OpenAI models, copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```
> **Note**: If `OPENAI_API_KEY` is not provided, Simple RAG automatically runs in **Offline Local Mode** using normalized TF-IDF embeddings and heuristic extraction.

---

### 2. Ingest Knowledge Base

Index files or folders into the vector database:

```bash
uv run simple-rag ingest knowledge_base
```

Output:
```text
+-------------------------------+
| Simple RAG Document Ingestion |
+-------------------------------+
                               Ingestion Summary                               
+-----------------------------------------------------------------------------+
| Metric                | Value                                               |
|-----------------------+-----------------------------------------------------|
| Source Path           | knowledge_base                                      |
| Documents Ingested    | 3                                                   |
| Chunks Created        | 15                                                  |
| Total Chunks in Index | 15                                                  |
| Saved Index File      | .../simple-rag/rag_index.json                       |
+-----------------------------------------------------------------------------+
```

---

### 3. Ask Questions via CLI

Run a single query:

```bash
uv run simple-rag query "What is the policy for deploying code on Fridays?"
```

Or query flight reimbursement rules:

```bash
uv run simple-rag query "What flights qualify for business class?"
```

---

### 4. Interactive Terminal Chat

Launch an interactive conversation with token streaming:

```bash
uv run simple-rag chat
```

- Type your questions naturally.
- Type `/sources` to view the raw retrieved chunks, similarity percentages, and source files from the previous answer.
- Type `exit` or `quit` to leave.

---

### 5. Web UI Studio

Launch the built-in browser interface:

```bash
uv run simple-rag serve
```

Then open your browser at **`http://127.0.0.1:8000`** to:
- Chat with the RAG assistant with formatted Markdown rendering.
- Inspect retrieved sources and relevance percentages in real time in the right-hand panel.
- Ingest new snippets or FAQs directly on the fly.

---

### 6. Running Tests

Execute the automated test suite:

```bash
uv run pytest
```

---

## 🐍 Python Library Usage

You can embed `simple_rag` directly in your own Python scripts:

```python
from simple_rag import RAGPipeline

# 1. Initialize pipeline (auto-detects OpenAI or fallback)
pipeline = RAGPipeline()

# 2. Ingest text or documents
pipeline.ingest_text(
    text="Project Aurora is scheduled for deployment on Mars in Q3 2028.",
    source_name="aurora_mission.txt"
)

# 3. Query
response = pipeline.query("When will Project Aurora launch?")

print("Answer:", response.answer)
print("Model:", response.model_used)
for source in response.sources:
    print(f"Source: {source.chunk.metadata['filename']} (Score: {source.score:.2f})")
```

---

## 🛠️ Configuration Options

| Option | Flag / Arg | Default | Description |
|---|---|---|---|
| Provider | `--provider` | `auto` | Choose between `auto`, `openai`, or `local` |
| Index File | `--index` | `rag_index.json` | Path where vectors and chunks are stored |
| Top-K | `--top-k` | `3` | Number of chunks retrieved per question |
| Web Host | `--host` | `127.0.0.1` | Host for web server |
| Web Port | `--port` | `8000` | Port for web server |
