"""
Module B: Data Processing & Vector Ingestion
=============================================
Reads the scraped JSON produced by crawler.py, chunks the text with
RecursiveCharacterTextSplitter, embeds chunks via Google's
text-embedding-004 model, and stores them in a persistent ChromaDB
collection.

Usage:
    python ingest.py                                    # defaults
    python ingest.py --input scraped_data.json --collection gov_services
    GOOGLE_API_KEY=xxx python ingest.py
"""

import json
import os
import logging
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_FILE = "scraped_data.json"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION = "gov_services_bd"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/gemini-embedding-001"


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def load_scraped_data(filepath: str) -> list[dict]:
    """Load the JSON array produced by the crawler."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Scraped data file not found: {filepath}. Run crawler.py first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d documents from %s", len(data), filepath)
    return data


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[list[str], list[dict]]:
    """
    Split each document's content into overlapping chunks.
    Returns (texts, metadatas) suitable for ChromaDB ingestion.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "।", ".", "।", " ", ""],  # Bengali period (daari) aware
        length_function=len,
    )

    all_texts: list[str] = []
    all_metadatas: list[dict] = []

    for doc in documents:
        url = doc.get("url", "")
        title = doc.get("title", "")
        content = doc.get("content", "")

        if not content.strip():
            continue

        chunks = splitter.split_text(content)

        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "source_url": url,
                    "title": title,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            )

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(all_texts),
        chunk_size,
        chunk_overlap,
    )
    return all_texts, all_metadatas


def build_vector_store(
    texts: list[str],
    metadatas: list[dict],
    api_key: str,
    persist_directory: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 10,
    batch_delay: float = 5.0,
) -> Chroma:
    """
    Embed *texts* using Google gemini-embedding-001 and persist them in
    ChromaDB under *collection_name*.  Processes in small batches with
    delays to respect free-tier rate limits.
    """
    import time

    logger.info("Initialising embedding model: %s", EMBEDDING_MODEL)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    # Ensure persist directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Embedding %d chunks in batches of %d (delay=%.1fs) into ChromaDB at %s …",
        len(texts),
        batch_size,
        batch_delay,
        persist_directory,
    )

    # Create initial vector store with the first batch
    first_batch_texts = texts[:batch_size]
    first_batch_meta = metadatas[:batch_size]

    vector_store = Chroma.from_texts(
        texts=first_batch_texts,
        metadatas=first_batch_meta,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    logger.info("  ✓ Batch 1/%d done (%d chunks)", -(-len(texts) // batch_size), len(first_batch_texts))

    # Add remaining batches with delays
    for i in range(batch_size, len(texts), batch_size):
        time.sleep(batch_delay)
        batch_texts = texts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = -(-len(texts) // batch_size)

        try:
            vector_store.add_texts(texts=batch_texts, metadatas=batch_meta)
            logger.info("  ✓ Batch %d/%d done (%d chunks)", batch_num, total_batches, len(batch_texts))
        except Exception as exc:
            logger.warning(
                "  ⚠ Batch %d failed: %s — retrying after 30s…", batch_num, exc
            )
            time.sleep(30)
            vector_store.add_texts(texts=batch_texts, metadatas=batch_meta)
            logger.info("  ✓ Batch %d/%d done (retry succeeded)", batch_num, total_batches)

    logger.info(
        "✓ Vector store ready. Collection '%s' has %d vectors.",
        collection_name,
        vector_store._collection.count(),
    )
    return vector_store


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_ingestion(
    input_file: str = DEFAULT_INPUT_FILE,
    persist_directory: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    api_key: str | None = None,
) -> Chroma:
    """End-to-end pipeline: load → chunk → embed → persist."""
    api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY must be set as an environment variable or passed "
            "directly. Get one at https://aistudio.google.com/app/apikey"
        )

    documents = load_scraped_data(input_file)
    texts, metadatas = chunk_documents(documents)
    vector_store = build_vector_store(
        texts=texts,
        metadatas=metadatas,
        api_key=api_key,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vector_store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk and embed scraped data into ChromaDB."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to scraped JSON (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help=f"ChromaDB persist directory (default: {DEFAULT_CHROMA_DIR}).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google API key (overrides GOOGLE_API_KEY env var).",
    )
    args = parser.parse_args()

    run_ingestion(
        input_file=args.input,
        persist_directory=args.chroma_dir,
        collection_name=args.collection,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
