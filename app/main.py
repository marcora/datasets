# app/main.py
from pathlib import Path
from typing import List, Optional, Dict, Any

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import StorageContext, load_index_from_storage, Settings

from app.config import configure_llamaindex
from app.build_index import DOCS_DIR, parse_qmd  # parse_qmd reads README.qmd YAML + body

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

STORE_DIR = Path("store")
STATIC_DIR = Path("static")

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(title="Chat with the Goate lab datasets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
index = None
chat_engine = None


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None


class SourceItem(BaseModel):
    dataset_id: str
    source_type: Optional[str] = None
    section_title: Optional[str] = None
    sample_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class DatasetDetail(BaseModel):
    dataset_id: str
    metadata: Dict[str, Any]
    body_markdown: str


# ---------------------------------------------------------------------
# Startup: configure LlamaIndex and load index
# ---------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    global index, chat_engine

    print("[startup] Configuring LlamaIndex...")
    configure_llamaindex()

    print(f"[startup] Loading index from {STORE_DIR}...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(STORE_DIR))
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print("[startup] Failed to load index:", e)
        index = None

    if index is not None:
        try:
            # default chat engine (conversational)
            chat_engine = index.as_chat_engine(
                chat_mode="condense_plus_context",
                similarity_top_k=8,
                system_prompt=(
                    "You are a helpful lab assistant for internal biology and genomics datasets "
                    "from the Goate lab. Answer using only the provided retrieved context. "
                    "If the information is not present, say you do not know.\n\n"
                    "When you refer to a specific dataset in your answer, format it as:\n"
                    "**<dataset title>** (`<dataset_id>`)\n"
                    "For example: **PBMC 10x RNA-seq** (`pbmc_rnaseq_10x`). "
                    "Use the dataset title and dataset_id that you see in the context or metadata "
                    "(for example, lines like 'Dataset: <title> (id: <dataset_id>)' or 'dataset_id' fields)."
                ),
            )
            print("[startup] Chat engine initialized.")
        except Exception as e:
            print("[startup] Failed to initialize chat engine:", e)
            chat_engine = None
    else:
        print("[startup] No index loaded; chat_engine unavailable.")


# ---------------------------------------------------------------------
# Root: serve Alpine/Tailwind UI
# ---------------------------------------------------------------------

@app.get("/")
async def root() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat endpoint used by the front-end.
    Returns { answer: str, sources: [{dataset_id, source_type, ...}, ...] }
    Sources are deduplicated by dataset_id to keep the UI tidy.
    """
    global index, chat_engine

    if chat_engine is None:
        return ChatResponse(answer="Error: chat engine not initialized.", sources=[])

    q = req.message
    print(f"[chat] Incoming question: {q!r}")

    try:
        result = chat_engine.chat(q)
        # result may be a ChatResponse-like object from LlamaIndex; adapt
        answer_text = (
            getattr(result, "response", None)
            or getattr(result, "answer", None)
            or getattr(result, "text", "")
        )
        if answer_text is None:
            answer_text = ""

        # collect unique dataset sources (dedupe by dataset_id)
        per_dataset: Dict[str, SourceItem] = {}
        for node in getattr(result, "source_nodes", []) or []:
            meta = node.metadata or {}
            dataset_id = meta.get("dataset_id") or meta.get("dataset") or "unknown"
            # skip empty keys
            if not dataset_id:
                continue
            # keep first seen source_type / section_title for the dataset
            if dataset_id not in per_dataset:
                per_dataset[dataset_id] = SourceItem(
                    dataset_id=dataset_id,
                    source_type=meta.get("source_type"),
                    section_title=meta.get("section_title"),
                    sample_id=meta.get("sample_id"),
                )

        sources = list(per_dataset.values())

        print(f"[chat] Returning answer ({len(sources)} sources).")
        return ChatResponse(answer=str(answer_text), sources=sources)

    except Exception as e:
        print("[chat] ERROR during chat:")
        traceback.print_exc()
        return ChatResponse(answer=f"Error during chat: {e}", sources=[])


# ---------------------------------------------------------------------
# Dataset detail endpoint
# ---------------------------------------------------------------------

@app.get("/dataset/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str) -> DatasetDetail:
    """
    Return the README YAML metadata and full README markdown body for a dataset.
    """
    dataset_dir = DOCS_DIR / dataset_id
    readme_path = dataset_dir / "README.qmd"

    if not readme_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"README.qmd not found for dataset {dataset_id!r}",
        )

    meta, body = parse_qmd(readme_path)
    return DatasetDetail(
        dataset_id=dataset_id,
        metadata=meta or {},
        body_markdown=body or "",
    )


# ---------------------------------------------------------------------
# Simple health endpoint
# ---------------------------------------------------------------------

@app.get("/health")
async def health():
  return JSONResponse(
      content={
          "ok": True,
          "index_loaded": index is not None,
          "chat_engine": chat_engine is not None,
      }
  )
