from pathlib import Path
from typing import Dict, Any, List

import yaml
import pandas as pd

from llama_index.core import Document, VectorStoreIndex
from app.config import configure_llamaindex

DOCS_DIR = Path("docs")
STORE_DIR = Path("store")


def parse_qmd(path: Path) -> tuple[Dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    meta = {}
    body = text

    if text.startswith("---"):
        try:
            _, yaml_part, body = text.split("---", 2)
            meta = yaml.safe_load(yaml_part) or {}
        except ValueError:
            pass

    return meta, body.strip()


def make_documents(dataset_dir: Path) -> List[Document]:
    docs: List[Document] = []

    readme = dataset_dir / "README.qmd"
    if not readme.exists():
        return docs

    meta, body = parse_qmd(readme)
    dataset_id = dataset_dir.name

    base_meta = {
        "dataset_id": dataset_id,
        "source_type": "readme",
        "source_path": str(readme),
    }
    base_meta.update(meta or {})

    # Split by markdown headings
    sections = []
    title = "General"
    lines = []

    for line in body.splitlines():
        if line.startswith("#"):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
                lines = []
            title = line.lstrip("#").strip() or "Section"
        else:
            lines.append(line)

    if lines:
        sections.append((title, "\n".join(lines).strip()))

    for title, text in sections:
        if text:
            docs.append(
                Document(
                    text=text,
                    metadata={**base_meta, "section_title": title},
                )
            )

    # Samples
    samples = dataset_dir / "SAMPLES.xlsx"
    if samples.exists():
        df = pd.read_excel(samples)
        for _, row in df.iterrows():
            parts = [
                f"{col}: {val}"
                for col, val in row.items()
                if pd.notna(val)
            ]
            docs.append(
                Document(
                    text="Sample info. " + "; ".join(parts),
                    metadata={
                        "dataset_id": dataset_id,
                        "source_type": "sample",
                        "source_path": str(samples),
                        "sample_id": row.get("sample_id"),
                    },
                )
            )

    return docs


def main():
    configure_llamaindex()

    all_docs: List[Document] = []

    for dataset_dir in DOCS_DIR.iterdir():
        if dataset_dir.is_dir():
            all_docs.extend(make_documents(dataset_dir))

    if not all_docs:
        raise RuntimeError("No documents found in docs/")

    STORE_DIR.mkdir(exist_ok=True)
    index = VectorStoreIndex.from_documents(all_docs)
    index.storage_context.persist(persist_dir=str(STORE_DIR))

    print(f"Indexed {len(all_docs)} documents into {STORE_DIR}/")


if __name__ == "__main__":
    main()
