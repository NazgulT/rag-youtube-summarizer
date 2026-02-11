"""Central configuration for embedding model, LLM, ChromaDB, and chunk params."""
import os
from pathlib import Path

# Embedding model (Hugging Face sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM for summary and Q&A (Hugging Face)
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "google/flan-t5-base")

# ChromaDB persist directory (relative to project root or absolute)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_data"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# RAG retriever
RAG_RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "4"))

# Optional: Hugging Face API token (for Inference API; leave unset for local pipeline)
HF_TOKEN = os.getenv("HF_TOKEN")
