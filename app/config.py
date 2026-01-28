from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ---------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3"

# ---------------------------------------------------------------------
# LlamaIndex global configuration
# ---------------------------------------------------------------------

def configure_llamaindex() -> None:
    """
    Configure global LlamaIndex settings to use local Ollama models
    for both generation and embeddings.

    This function is intentionally idempotent and safe to call
    from both build_index.py and main.py.
    """

    # Initialize LLM
    llm = Ollama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=180.0,   # generous for local inference
    )

    # Initialize embedding model
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )

    # -----------------------------------------------------------------
    # Fail-fast health check
    # -----------------------------------------------------------------
    # This prevents silent failures if Ollama isn't running or a model
    # hasn't been pulled yet.
    try:
        _ = embed_model.get_text_embedding("ollama health check")
    except Exception as e:
        raise RuntimeError(
            "\n[config] Failed to contact Ollama embedding model.\n"
            "Make sure Ollama is running and required models are pulled:\n\n"
            f"  ollama pull {EMBED_MODEL_NAME}\n"
            f"  ollama pull {LLM_MODEL_NAME}\n\n"
            f"Underlying error: {e}\n"
        )

    # -----------------------------------------------------------------
    # Apply global LlamaIndex settings
    # -----------------------------------------------------------------

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Chunking tuned for README.qmd + technical prose
    Settings.chunk_size = 800
    Settings.chunk_overlap = 100

    # Optional but helpful for debugging
    print(
        "[config] LlamaIndex configured with:\n"
        f"  Ollama URL: {OLLAMA_BASE_URL}\n"
        f"  LLM model:  {LLM_MODEL_NAME}\n"
        f"  Embedder:   {EMBED_MODEL_NAME}\n"
        f"  Chunk size: {Settings.chunk_size}\n"
        f"  Overlap:    {Settings.chunk_overlap}"
    )
