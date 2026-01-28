from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import pandas as pd

from llama_index.core import Document, VectorStoreIndex
from app.config import configure_llamaindex

DOCS_DIR = Path("docs")
STORE_DIR = Path("store")


# ---------- YAML / Quarto parsing ----------

def parse_qmd(path: Path) -> Tuple[Dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    meta: Dict[str, Any] = {}
    body = text

    if text.startswith("---"):
        try:
            _, yaml_part, body = text.split("---", 2)
            meta = yaml.safe_load(yaml_part) or {}
        except ValueError:
            # no proper YAML frontmatter
            pass

    return meta, body.strip()


def to_list(value) -> List[str]:
    """Normalize YAML scalars / lists into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def normalize_metadata(raw_meta: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    """
    Take raw YAML metadata and normalize key fields based on observed structure.

    Example fields in headers:
      assay, organism, tissue, cell_type, platform, technology,
      tags, categories, title, ...
    """
    meta: Dict[str, Any] = dict(raw_meta or {})

    # Required dataset-level identifier
    meta["dataset_id"] = dataset_id

    # Normalize key fields
    meta["assay"] = to_list(meta.get("assay"))
    meta["organism"] = meta.get("organism")  # usually scalar
    meta["tissue"] = meta.get("tissue")
    meta["cell_type"] = to_list(meta.get("cell_type"))
    meta["platform"] = meta.get("platform")
    meta["technology"] = meta.get("technology")
    meta["tags"] = [t.lower() for t in to_list(meta.get("tags"))]
    meta["categories"] = [c.lower() for c in to_list(meta.get("categories"))]

    # Convenience flags you *might* want for later routing/filtering
    tech = (meta.get("technology") or "").lower()
    assay_list = meta.get("assay", [])
    # crude heuristics, adjust as you like
    meta["is_single_cell"] = any("10x" in (meta.get("platform") or "").lower()
                                 or "single-cell" in tech
                                 or "multiome" in tech
                                 for _ in [0])
    meta["has_rna"] = "rna-seq" in assay_list or "rna" in tech
    meta["has_atac"] = "atac-seq" in assay_list or "atac" in tech
    meta["is_spatial"] = "spatial" in tech

    return meta


# ---------- README / SAMPLES to Documents ----------

def split_readme_sections(body: str) -> List[Tuple[str, str]]:
    """
    Split README body into (section_title, text) by markdown headings.
    """
    sections: List[Tuple[str, str]] = []
    current_title = "General"
    lines: List[str] = []

    for line in body.splitlines():
        if line.startswith("#"):
            if lines:
                sections.append((current_title, "\n".join(lines).strip()))
                lines = []
            current_title = line.lstrip("#").strip() or "Section"
        else:
            lines.append(line)

    if lines:
        sections.append((current_title, "\n".join(lines).strip()))

    return sections


def make_dataset_overview_text(meta: Dict[str, Any]) -> str:
    """
    Turn YAML header fields into a short natural language overview.
    This is great for retrieval when users describe datasets by modality / tissue / platform.
    """
    title = meta.get("title", "Untitled dataset")
    assay = ", ".join(meta.get("assay", [])) or "unspecified assay"
    organism = meta.get("organism", "unspecified organism")
    tissue = meta.get("tissue", "unspecified tissue")
    cell_types = ", ".join(meta.get("cell_type", [])) or "unspecified cell types"
    platform = meta.get("platform", "unspecified platform")
    technology = meta.get("technology", "unspecified technology")
    tags = ", ".join(meta.get("tags", []))
    categories = ", ".join(meta.get("categories", []))

    lines = [
        f"Dataset title: {title}.",
        f"Assay: {assay}.",
        f"Organism: {organism}.",
        f"Tissue: {tissue}.",
        f"Cell types: {cell_types}.",
        f"Platform: {platform}.",
        f"Technology: {technology}.",
    ]
    if tags:
        lines.append(f"Tags: {tags}.")
    if categories:
        lines.append(f"Categories: {categories}.")

    return " ".join(lines)


def make_documents_for_dataset(dataset_dir: Path) -> List[Document]:
    docs: List[Document] = []

    dataset_id = dataset_dir.name
    readme_path = dataset_dir / "README.qmd"
    if not readme_path.exists():
        return docs

    raw_meta, body = parse_qmd(readme_path)
    base_meta = normalize_metadata(raw_meta, dataset_id)

    # 1) Dataset overview document (from YAML header only)
    overview_text = make_dataset_overview_text(base_meta)
    overview_meta = {
        **base_meta,
        "source_type": "dataset_overview",
        "source_path": str(readme_path),
    }
    docs.append(Document(text=overview_text, metadata=overview_meta))

    # 2) README section documents
    sections = split_readme_sections(body)
    for title, section_text in sections:
        if not section_text.strip():
            continue
        section_meta = {
            **base_meta,
            "source_type": "readme",
            "source_path": str(readme_path),
            "section_title": title,
        }
        docs.append(Document(text=section_text.strip(), metadata=section_meta))

    # 3) Sample row documents (inherit dataset metadata)
    samples_path = dataset_dir / "SAMPLES.xlsx"
    if samples_path.exists():
        df = pd.read_excel(samples_path)
        for _, row in df.iterrows():
            parts = []
            for col, val in row.items():
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
            if not parts:
                continue
            text = "Sample info. " + "; ".join(parts)
            sample_meta = {
                **base_meta,
                "source_type": "sample",
                "source_path": str(samples_path),
            }
            if "sample_id" in df.columns:
                sample_meta["sample_id"] = row.get("sample_id")
            docs.append(Document(text=text, metadata=sample_meta))

    return docs


# ---------- Build index ----------

def build_index():
    configure_llamaindex()

    all_docs: List[Document] = []
    for dataset_dir in DOCS_DIR.iterdir():
        if dataset_dir.is_dir():
            all_docs.extend(make_documents_for_dataset(dataset_dir))

    if not all_docs:
        raise RuntimeError("No documents found in docs/")

    STORE_DIR.mkdir(exist_ok=True)
    index = VectorStoreIndex.from_documents(all_docs)
    index.storage_context.persist(persist_dir=str(STORE_DIR))
    print(f"[build_index] Indexed {len(all_docs)} documents into {STORE_DIR}/")


if __name__ == "__main__":
    build_index()
