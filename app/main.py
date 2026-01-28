import traceback
from pathlib import Path
from typing import Dict, List, Optional

from app.config import configure_llamaindex
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from llama_index.core import StorageContext, load_index_from_storage
from pydantic import BaseModel

STORE_DIR = Path("store")

app = FastAPI(title="Lab Knowledge Base Chatbot")
chat_engine = None


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


@app.on_event("startup")
def startup():
    global chat_engine

    print("[startup] Configuring LlamaIndex and loading index...")
    configure_llamaindex()

    storage_context = StorageContext.from_defaults(persist_dir=str(STORE_DIR))
    index = load_index_from_storage(storage_context)
    print("[startup] Index loaded from", STORE_DIR)

    # use condense_plus_context so we can set system_prompt
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=8,
        system_prompt=(
            "You are a lab assistant for internal biology and genomics datasets. "
            "Answer strictly using the retrieved context. "
            "If the information is not present, say you do not know."
        ),
    )
    print("[startup] Chat engine initialized.")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
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


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Lab Knowledge Base Chatbot</title>
<style>
body {
  font-family: system-ui, sans-serif;
  max-width: 900px;
  margin: 2rem auto;
}
#chat {
  border: 1px solid #ccc;
  padding: 1rem;
  height: 60vh;
  overflow-y: auto;
}
.msg {
  margin: 0.5rem 0;
}
.user {
  text-align: right;
}
.assistant {
  text-align: left;
}
.bubble {
  display: inline-block;
  padding: 0.4rem 0.7rem;
  border-radius: 8px;
  max-width: 80%;
  white-space: pre-wrap;
}
.user.bubble {
  background: #d9edf7;
}
.assistant.bubble {
  background: #f5f5f5;
}
#sources {
  margin-top: 0.5rem;
  font-size: 0.9em;
  color: #555;
  white-space: pre-wrap;
}
</style>
</head>
<body>
<h1>Lab Knowledge Base Chatbot</h1>

<div id="chat"></div>
<div id="sources"></div>

<form id="form">
  <input
    id="input"
    style="width:80%"
    placeholder="Ask about datasets, samples, QC, protocols…"
  />
  <button>Send</button>
</form>

<script>
const chat = document.getElementById("chat");
const sourcesDiv = document.getElementById("sources");
const input = document.getElementById("input");

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble " + role;
  bubble.textContent = text;
  div.appendChild(bubble);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

document.getElementById("form").onsubmit = async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;

  addMessage("user", q);
  input.value = "";
  sourcesDiv.textContent = "";

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q })
    });

    const data = await resp.json();
    addMessage("assistant", data.answer || "[no answer]");

    if (data.sources && data.sources.length > 0) {
      const lines = data.sources.map(s =>
        `• dataset: ${s.dataset_id}` +
        ` (type: ${s.source_type}` +
        (s.section_title ? `, section: ${s.section_title}` : "") +
        (s.sample_id ? `, sample: ${s.sample_id}` : "") +
        `)`
      );
      sourcesDiv.textContent = "Sources:\\n" + lines.join("\\n");
    }
  } catch (err) {
    addMessage("assistant", "Error calling /chat: " + err);
  }
};
</script>
</body>
</html>
"""
