from pathlib import Path
from typing import List, Optional, Dict

import traceback
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import StorageContext, load_index_from_storage

from app.config import configure_llamaindex

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

STORE_DIR = Path("store")
STATIC_DIR = Path("static")

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(title="Lab Knowledge Base Chatbot")
chat_engine = None

# Serve static files (for Alpine.js UI)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None


class SourceItem(BaseModel):
    dataset_id: str
    source_type: str
    section_title: Optional[str] = None
    sample_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ---------------------------------------------------------------------
# Startup: configure LlamaIndex and load index
# ---------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    global chat_engine

    print("[startup] Configuring LlamaIndex...")
    configure_llamaindex()

    print(f"[startup] Loading index from {STORE_DIR}...")
    storage_context = StorageContext.from_defaults(
        persist_dir=str(STORE_DIR)
    )
    index = load_index_from_storage(storage_context)

    # Use condense_plus_context so we can keep a system prompt
    chat_engine_local = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=8,
        system_prompt=(
            "You are a lab assistant for internal biology and genomics datasets. "
            "Answer strictly using the retrieved context. "
            "If the information is not present, say you do not know."
        ),
    )

    chat_engine = chat_engine_local
    print("[startup] Chat engine initialized.")


# ---------------------------------------------------------------------
# Root: serve Alpine.js UI
# ---------------------------------------------------------------------

@app.get("/")
async def root() -> FileResponse:
    """
    Serve the static Alpine.js chat UI.
    """
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat endpoint used by the Alpine.js frontend.
    Expects: { "message": "..." }
    Returns: { "answer": "...", "sources": [...] }
    """
    global chat_engine

    if chat_engine is None:
        return ChatResponse(
            answer="Error: chat engine not initialized.",
            sources=[],
        )

    try:
        print(f"[chat] Incoming question: {req.message!r}")
        result = chat_engine.chat(req.message)
        print("[chat] LLM response obtained.")

        # Collect unique sources
        sources: Dict[str, SourceItem] = {}
        for node in result.source_nodes:
            meta = node.metadata or {}
            key = (
                f"{meta.get('dataset_id')}|"
                f"{meta.get('source_type')}|"
                f"{meta.get('section_title')}|"
                f"{meta.get('sample_id')}"
            )
            if key not in sources:
                sources[key] = SourceItem(
                    dataset_id=meta.get("dataset_id", "unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    section_title=meta.get("section_title"),
                    sample_id=meta.get("sample_id"),
                )

        return ChatResponse(
            answer=result.response,
            sources=list(sources.values()),
        )

    except Exception as e:
        print("[chat] ERROR during chat:")
        traceback.print_exc()
        return ChatResponse(
            answer=f"Error during chat: {e}",
            sources=[],
        )
