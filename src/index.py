"""Build, save and load the vector index (chunks + embeddings)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config
from .embeddings import Embedder
from .ingest import Chunk, chunk_documents, load_documents, passage_for_embedding


def build_index(index_dir: Path = config.INDEX_DIR, embedder: Embedder | None = None,
                data_dir: Path = config.DATA_DIR) -> dict:
    t0 = time.time()
    index_dir.mkdir(parents=True, exist_ok=True)
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    if not chunks:
        raise SystemExit(f"No documents found in {data_dir}")
    embedder = embedder or Embedder()
    passages = [passage_for_embedding(c) for c in chunks]
    embedder.fit(passages)
    vecs = embedder.encode_passages(passages)

    np.save(index_dir / "embeddings.npy", vecs)
    (index_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
    embedder.save(index_dir)
    meta = {
        "embedder": embedder.name,
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "dim": int(vecs.shape[1]),
        "languages": sorted({c.language for c in chunks}),
        "schemes": sorted({c.scheme_id for c in chunks}),
        "build_seconds": round(time.time() - t0, 2),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_index(index_dir: Path = config.INDEX_DIR) -> tuple[list[Chunk], np.ndarray, dict]:
    chunks_raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    vecs = np.load(index_dir / "embeddings.npy")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return [Chunk(**c) for c in chunks_raw], vecs, meta


def index_exists(index_dir: Path = config.INDEX_DIR) -> bool:
    return all((index_dir / f).exists() for f in ("chunks.json", "embeddings.npy", "meta.json"))
