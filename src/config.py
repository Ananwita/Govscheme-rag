"""Central configuration for the Multilingual Government Scheme RAG project.

Every setting can be overridden with an environment variable or a `.env`
file in the project root (see `.env.example`).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- paths
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "schemes"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))
EVAL_DIR = ROOT / "eval"
RULES_FILE = ROOT / "data" / "scheme_rules.json"

# ---------------------------------------------------------------- embeddings
# Multilingual sentence-embedding model from Hugging Face.
#   intfloat/multilingual-e5-small  (default, 118M params, 100+ languages incl. Hindi)
#   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (lighter alternative)
#   BAAI/bge-m3  (stronger but ~2.2 GB)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# ---------------------------------------------------------------- chunking
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", 200))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 30))

# ---------------------------------------------------------------- retrieval
TOP_K = int(os.getenv("TOP_K", 5))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.6))   # weight of dense ranking in fusion
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))     # weight of keyword ranking in fusion
RRF_K = 60                                             # reciprocal-rank-fusion constant
SAME_LANGUAGE_BOOST = float(os.getenv("SAME_LANGUAGE_BOOST", 0.15))
SECTION_INTENT_BOOST = float(os.getenv("SECTION_INTENT_BOOST", 0.15))

# ---------------------------------------------------------------- generation (optional LLM)
# Any OpenAI-compatible chat endpoint works:
#   Ollama (local, free):  LLM_BASE_URL=http://localhost:11434/v1   LLM_MODEL=qwen2.5:3b
#   Groq (free tier):      LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=llama-3.1-8b-instant
# Leave LLM_BASE_URL empty to use the built-in extractive answerer (works fully offline).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 120))
