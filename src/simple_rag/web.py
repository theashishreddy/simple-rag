"""FastAPI Web UI and REST API for Simple RAG."""

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from simple_rag.pipeline import RAGPipeline
from simple_rag.embeddings import get_embedding_model
from simple_rag.llm import get_llm
from simple_rag.vector_store import SimpleVectorStore

app = FastAPI(title="Simple RAG UI", description="Minimalist & Powerful RAG Web Interface")

pipeline: Optional[RAGPipeline] = None
current_index_path: str = "rag_index.json"


def init_web_pipeline(index_path: str = "rag_index.json", provider: str = "auto"):
    global pipeline, current_index_path
    current_index_path = index_path
    embedder = get_embedding_model(provider=provider)
    llm = get_llm(provider=provider)
    store = SimpleVectorStore()

    p = Path(index_path)
    if p.exists():
        store.load(p)

    pipeline = RAGPipeline(embedding_model=embedder, llm=llm, vector_store=store)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    min_similarity: float = 0.0


class IngestTextRequest(BaseModel):
    title: str
    content: str


class SourceItem(BaseModel):
    filename: str
    score: float
    content: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    retrieval_ms: float
    generation_ms: float
    model: str


@app.on_event("startup")
def startup_event():
    global pipeline
    if pipeline is None:
        init_web_pipeline()


@app.get("/api/status")
def get_status():
    global pipeline
    if not pipeline:
        return {"status": "not initialized"}
    return {
        "status": "ready",
        "chunks_indexed": pipeline.vector_store.count,
        "model": pipeline.llm.model_name,
        "embedding_dim": pipeline.embedding_model.dimension,
    }


@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    if pipeline.vector_store.count == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Ingest documents first via CLI or web UI.",
        )

    resp = pipeline.query(req.query, top_k=req.top_k, min_similarity=req.min_similarity)

    sources = [
        SourceItem(
            filename=s.chunk.metadata.get("filename", "snippet"),
            score=round(s.score, 4),
            content=s.chunk.content,
        )
        for s in resp.sources
    ]

    return QueryResponse(
        query=resp.query,
        answer=resp.answer,
        sources=sources,
        retrieval_ms=round(resp.retrieval_time_ms, 2),
        generation_ms=round(resp.generation_time_ms, 2),
        model=resp.model_used,
    )


@app.post("/api/ingest-text")
def ingest_text(req: IngestTextRequest):
    global pipeline, current_index_path
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    stats = pipeline.ingest_text(req.content, source_name=req.title)
    pipeline.save_index(current_index_path)
    return {"message": "Text ingested successfully", "stats": stats}


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Simple RAG Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .chat-bubble-assistant { background-color: #161b22; border: 1px solid #30363d; }
    .chat-bubble-user { background-color: #1f6feb; color: #ffffff; }
    .card { background-color: #161b22; border: 1px solid #30363d; }
  </style>
</head>
<body class="h-screen flex flex-col">

  <!-- Header -->
  <header class="border-b border-gray-800 px-6 py-4 flex items-center justify-between bg-[#161b22]">
    <div class="flex items-center space-x-3">
      <div class="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-lg">
        <i class="fa-solid fa-brain"></i>
      </div>
      <div>
        <h1 class="text-lg font-bold text-white tracking-wide">Simple RAG Studio</h1>
        <p class="text-xs text-gray-400">Retrieval-Augmented Generation Playground</p>
      </div>
    </div>
    <div class="flex items-center space-x-4 text-xs">
      <div class="flex items-center space-x-2 bg-gray-900 px-3 py-1.5 rounded-full border border-gray-700">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="model-badge" class="text-gray-300 font-mono">Loading model...</span>
      </div>
      <div class="bg-gray-900 px-3 py-1.5 rounded-full border border-gray-700">
        <span id="chunks-badge" class="text-emerald-400 font-semibold font-mono">0</span>
        <span class="text-gray-400"> chunks indexed</span>
      </div>
    </div>
  </header>

  <!-- Main Body -->
  <div class="flex-1 flex overflow-hidden">
    
    <!-- Left Chat Panel -->
    <div class="flex-1 flex flex-col bg-[#0d1117]">
      <!-- Messages List -->
      <div id="messages" class="flex-1 overflow-y-auto p-6 space-y-6">
        <div class="flex items-start space-x-3">
          <div class="w-8 h-8 rounded-full bg-indigo-600/30 text-indigo-400 flex items-center justify-center text-sm shrink-0">
            <i class="fa-solid fa-robot"></i>
          </div>
          <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-sm leading-relaxed shadow-sm">
            <p class="font-semibold text-white mb-1">Welcome to Simple RAG!</p>
            <p class="text-gray-300">I can answer questions based on your indexed knowledge base with exact citations. Try asking:</p>
            <div class="mt-3 flex flex-wrap gap-2">
              <button onclick="setQuery('What is the company policy on remote work and home office stipends?')" class="bg-gray-800 hover:bg-gray-700 text-xs px-3 py-1.5 rounded-lg border border-gray-700 transition">
                🏠 Remote Work Policy
              </button>
              <button onclick="setQuery('What is the daily meal per diem for domestic and international travel?')" class="bg-gray-800 hover:bg-gray-700 text-xs px-3 py-1.5 rounded-lg border border-gray-700 transition">
                ✈️ Travel & Per Diem
              </button>
              <button onclick="setQuery('What are the rules for deploying code on Fridays and code review approvals?')" class="bg-gray-800 hover:bg-gray-700 text-xs px-3 py-1.5 rounded-lg border border-gray-700 transition">
                💻 Deployment Cadence
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="p-4 border-t border-gray-800 bg-[#161b22]/50">
        <form id="chat-form" onsubmit="handleSend(event)" class="max-w-4xl mx-auto flex items-center space-x-3">
          <input 
            id="query-input" 
            type="text" 
            placeholder="Ask a question about the indexed documents..." 
            class="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 transition"
            required
          />
          <button 
            type="submit" 
            id="send-btn"
            class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-5 py-3 rounded-xl text-sm flex items-center space-x-2 transition disabled:opacity-50">
            <span>Ask</span>
            <i class="fa-solid fa-paper-plane text-xs"></i>
          </button>
        </form>
      </div>
    </div>

    <!-- Right Sidebar: Context & Sources -->
    <div class="w-96 border-l border-gray-800 bg-[#161b22] flex flex-col">
      <div class="p-4 border-b border-gray-800 flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <i class="fa-solid fa-layer-group text-indigo-400"></i>
          <h2 class="font-bold text-sm text-white">Retrieved Context</h2>
        </div>
        <span id="timing-badge" class="text-xs text-gray-500 font-mono">0ms</span>
      </div>

      <div id="sources-container" class="flex-1 overflow-y-auto p-4 space-y-4">
        <div class="text-center py-16 text-gray-500 text-xs">
          <i class="fa-regular fa-folder-open text-3xl mb-3 block opacity-40"></i>
          Retrieved source chunks with relevance scores will appear here when you query.
        </div>
      </div>

      <!-- Quick Document Ingestion Form -->
      <div class="p-4 border-t border-gray-800 bg-gray-900/60">
        <details class="text-xs">
          <summary class="cursor-pointer font-semibold text-indigo-400 hover:text-indigo-300">
            + Quick Ingest Snippet
          </summary>
          <div class="mt-3 space-y-2">
            <input id="new-doc-title" type="text" placeholder="Title (e.g. FAQ.txt)" class="w-full bg-gray-950 border border-gray-700 rounded p-2 text-white text-xs">
            <textarea id="new-doc-content" rows="3" placeholder="Paste document content..." class="w-full bg-gray-950 border border-gray-700 rounded p-2 text-white text-xs"></textarea>
            <button onclick="handleQuickIngest()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-1.5 rounded transition text-xs">
              Index Into Knowledge Base
            </button>
          </div>
        </details>
      </div>
    </div>

  </div>

  <script>
    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('model-badge').textContent = data.model;
        document.getElementById('chunks-badge').textContent = data.chunks_indexed;
      } catch (e) {
        console.error(e);
      }
    }

    function setQuery(text) {
      document.getElementById('query-input').value = text;
      document.getElementById('query-input').focus();
    }

    async function handleSend(e) {
      e.preventDefault();
      const input = document.getElementById('query-input');
      const query = input.value.trim();
      if (!query) return;

      input.value = '';
      appendUserMessage(query);

      const sendBtn = document.getElementById('send-btn');
      sendBtn.disabled = true;

      const assistantMsgId = appendAssistantSkeleton();

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: query, top_k: 3})
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Failed to query');
        }

        const data = await res.json();
        renderAssistantResponse(assistantMsgId, data.answer, data.sources);
        renderSources(data.sources, data.retrieval_ms, data.generation_ms);
        updateStatus();
      } catch (err) {
        renderAssistantError(assistantMsgId, err.message);
      } finally {
        sendBtn.disabled = false;
      }
    }

    function appendUserMessage(text) {
      const container = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = 'flex items-start justify-end space-x-3';
      div.innerHTML = `
        <div class="chat-bubble-user p-4 rounded-2xl max-w-2xl text-sm leading-relaxed shadow-sm">
          <p>${escapeHtml(text)}</p>
        </div>
        <div class="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm shrink-0">
          <i class="fa-solid fa-user"></i>
        </div>
      `;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    }

    function appendAssistantSkeleton() {
      const container = document.getElementById('messages');
      const id = 'msg-' + Date.now();
      const div = document.createElement('div');
      div.id = id;
      div.className = 'flex items-start space-x-3';
      div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-indigo-600/30 text-indigo-400 flex items-center justify-center text-sm shrink-0">
          <i class="fa-solid fa-robot"></i>
        </div>
        <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-sm leading-relaxed text-gray-400 flex items-center space-x-2">
          <i class="fa-solid fa-circle-notch fa-spin text-indigo-400"></i>
          <span>Retrieving relevant context and formulating grounded answer...</span>
        </div>
      `;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
      return id;
    }

    function renderAssistantResponse(id, markdownAnswer, sources) {
      const elem = document.getElementById(id);
      if (!elem) return;
      
      const citations = sources.map(s => `<span class="bg-indigo-950 text-indigo-300 border border-indigo-800 text-[11px] px-2 py-0.5 rounded-full mr-1.5"><i class="fa-solid fa-file-lines mr-1"></i>${escapeHtml(s.filename)} (${Math.round(s.score*100)}%)</span>`).join('');

      elem.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-indigo-600/30 text-indigo-400 flex items-center justify-center text-sm shrink-0">
          <i class="fa-solid fa-robot"></i>
        </div>
        <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-sm leading-relaxed text-gray-200">
          <div class="prose prose-invert text-sm max-w-none">
            ${marked.parse(markdownAnswer)}
          </div>
          ${sources.length > 0 ? `
            <div class="mt-4 pt-3 border-t border-gray-800 flex flex-wrap items-center gap-1">
              <span class="text-xs text-gray-500 mr-2 font-medium">Sources:</span>
              ${citations}
            </div>
          ` : ''}
        </div>
      `;
      const container = document.getElementById('messages');
      container.scrollTop = container.scrollHeight;
    }

    function renderAssistantError(id, msg) {
      const elem = document.getElementById(id);
      if (!elem) return;
      elem.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-red-600/30 text-red-400 flex items-center justify-center text-sm shrink-0">
          <i class="fa-solid fa-triangle-exclamation"></i>
        </div>
        <div class="chat-bubble-assistant border-red-800 p-4 rounded-2xl max-w-2xl text-sm text-red-300">
          <strong>Error:</strong> ${escapeHtml(msg)}
        </div>
      `;
    }

    function renderSources(sources, retrievalMs, generationMs) {
      const container = document.getElementById('sources-container');
      document.getElementById('timing-badge').textContent = `${retrievalMs}ms retrieve | ${generationMs}ms generate`;

      if (!sources || sources.length === 0) {
        container.innerHTML = '<div class="text-center py-10 text-gray-500 text-xs">No matching sources found.</div>';
        return;
      }

      container.innerHTML = sources.map((s, idx) => {
        const pct = Math.round(s.score * 100);
        const color = pct >= 80 ? 'text-emerald-400' : pct >= 60 ? 'text-amber-400' : 'text-blue-400';
        return `
          <div class="card p-3 rounded-xl text-xs space-y-2">
            <div class="flex items-center justify-between">
              <span class="font-bold text-gray-200 truncate max-w-[180px]">${idx + 1}. ${escapeHtml(s.filename)}</span>
              <span class="font-mono ${color} font-bold text-[11px]">${pct}% match</span>
            </div>
            <p class="text-gray-400 leading-relaxed font-mono text-[11px] bg-gray-950 p-2 rounded border border-gray-800 max-h-36 overflow-y-auto whitespace-pre-wrap">${escapeHtml(s.content)}</p>
          </div>
        `;
      }).join('');
    }

    async function handleQuickIngest() {
      const title = document.getElementById('new-doc-title').value.trim() || 'note.txt';
      const content = document.getElementById('new-doc-content').value.trim();
      if (!content) return alert('Please enter some content');

      try {
        const res = await fetch('/api/ingest-text', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({title, content})
        });
        const data = await res.json();
        alert(`Ingested! Created ${data.stats.chunks_created} chunks.`);
        document.getElementById('new-doc-content').value = '';
        updateStatus();
      } catch (e) {
        alert('Failed to ingest: ' + e.message);
      }
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    updateStatus();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT
