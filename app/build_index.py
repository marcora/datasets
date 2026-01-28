from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from app.config import configure_llamaindex
from llama_index.core import Document, VectorStoreIndex

DOCS_DIR = Path("docs")
STORE_DIR = Path("store")


def parse_qmd(path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Parse a Quarto/Markdown file with YAML frontmatter.
    Returns (meta_dict, body_text).
    """
    text = path.read_text(encoding="utf-8")
    meta: Dict[str, Any] = {}
    body = text

    if text.startswith("---"):
        try:
            _, yaml_part, body = text.split("---", 2)
            meta = yaml.safe_load(yaml_part) or {}
        except ValueError:
            # no proper YAML frontmatter; treat whole file as body
            body = text

    return meta, body.strip()


def make_documents_for_dataset(dataset_dir: Path) -> List[Document]:
    """
    For a dataset directory, produce Documents for:
      - the full README.qmd body (single document, no section splitting)
      - each row of SAMPLES.xlsx (if present)
    Attach dataset_id and all YAML metadata to each Document.metadata.
    """
    docs: List[Document] = []

    dataset_id = dataset_dir.name
    readme_path = dataset_dir / "README.qmd"
    if not readme_path.exists():
        return docs

    raw_meta, body = parse_qmd(readme_path)

    # 1) README as a single document
    if body.strip():
        meta = {
            "dataset_id": dataset_id,
            "source_type": "readme",
            "source_path": str(readme_path),
            # include all YAML fields as-is so we can filter on them later
            **(raw_meta or {}),
        }
        docs.append(Document(text=body.strip(), metadata=meta))

    # 2) SAMPLES.xlsx -> one Document per row, inheriting YAML metadata
    samples_path = dataset_dir / "SAMPLES.xlsx"
    if samples_path.exists():
        try:
            df = pd.read_excel(samples_path)
        except Exception as e:
            print(f"[build_index] Warning: failed to read {samples_path}: {e}")
            df = None

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        parts.append(f"{col}: {val}")
                if not parts:
                    continue

                text = "Sample: " + "; ".join(parts)
                meta = {
                    "dataset_id": dataset_id,
                    "source_type": "sample",
                    "source_path": str(samples_path),
                    **(raw_meta or {}),
                }

                # If the sheet has a sample identifier column, include it
                for possible_id in ("sample_id", "sample", "id"):
                    if possible_id in df.columns:
                        meta["sample_id"] = row.get(possible_id)
                        break

                docs.append(Document(text=text, metadata=meta))

    return docs


def build_index() -> None:
    """
    Build a vector index from docs/ and persist it to store/.
    Each README is a single document; YAML metadata is preserved.
    """
    configure_llamaindex()

    all_docs: List[Document] = []
    for dataset_dir in sorted(DOCS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        docs = make_documents_for_dataset(dataset_dir)
        if docs:
            all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("No documents found in docs/ — nothing to index")

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    index = VectorStoreIndex.from_documents(all_docs)
    index.storage_context.persist(persist_dir=str(STORE_DIR))
    print(f"[build_index] Indexed {len(all_docs)} documents into {STORE_DIR}/")


if __name__ == "__main__":
    build_index()
