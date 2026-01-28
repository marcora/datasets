from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

LLM_MODEL_NAME = "llama3"
EMBED_MODEL_NAME = "nomic-embed-text"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # explicit


def configure_llamaindex():
    llm = Ollama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 800
    Settings.chunk_overlap = 100

    print("[config] LlamaIndex configured with Ollama at", OLLAMA_BASE_URL)
