"""Hybrid retriever: multilingual dense search + BM25, merged with weighted
Reciprocal Rank Fusion (RRF), then re-scored with two light heuristics:

  * same-language boost  - prefer passages in the language of the question
  * section-intent boost - "who is eligible" questions prefer Eligibility sections
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .bm25 import BM25
from .embeddings import Embedder
from .index import load_index
from .ingest import Chunk, passage_for_embedding
from .text_utils import INTENT_SECTIONS, detect_intents, detect_language


@dataclass
class Hit:
    chunk: Chunk
    score: float          # final fused score (0..~1.3)
    dense: float          # cosine similarity
    bm25: float           # raw BM25 score
    rank: int = 0


class HybridRetriever:
    def __init__(self, embedder: Embedder | None = None, index_dir=config.INDEX_DIR):
        self.chunks, self.vecs, self.meta = load_index(index_dir)
        self.embedder = embedder or Embedder()
        self.embedder.load(index_dir)
        if self.embedder.name != self.meta["embedder"]:
            raise RuntimeError(
                f"Index was built with {self.meta['embedder']} but current embedder is "
                f"{self.embedder.name}. Re-run: python build_index.py")
        self.bm25 = BM25([passage_for_embedding(c) for c in self.chunks])

    def search(self, query: str, top_k: int = config.TOP_K,
               languages: list[str] | None = None,
               schemes: list[str] | None = None,
               prefer_language: str | None = None) -> list[Hit]:
        n = len(self.chunks)
        mask = np.ones(n, dtype=bool)
        if languages:
            mask &= np.array([c.language in languages for c in self.chunks])
        if schemes:
            mask &= np.array([c.scheme_id in schemes for c in self.chunks])
        if not mask.any():
            return []

        q_vec = self.embedder.encode_queries([query])[0]
        dense = self.vecs @ q_vec
        bm = self.bm25.scores(query)
        dense_m = np.where(mask, dense, -np.inf)
        bm_m = np.where(mask, bm, -np.inf)

        dense_rank = np.empty(n, dtype=np.int64)
        dense_rank[np.argsort(-dense_m)] = np.arange(1, n + 1)
        bm_rank = np.empty(n, dtype=np.int64)
        bm_rank[np.argsort(-bm_m)] = np.arange(1, n + 1)

        k = config.RRF_K
        fused = config.DENSE_WEIGHT / (k + dense_rank)
        fused += np.where(bm > 0, config.BM25_WEIGHT / (k + bm_rank), 0.0)
        fused /= (config.DENSE_WEIGHT + config.BM25_WEIGHT) / (k + 1)   # scale to 0..1

        prefer_language = prefer_language or detect_language(query)
        intents = detect_intents(query)
        wanted_sections = {s for i in intents for s in INTENT_SECTIONS[i]}
        for i, c in enumerate(self.chunks):
            if c.language == prefer_language:
                fused[i] += config.SAME_LANGUAGE_BOOST
            if wanted_sections and any(ws in c.section.lower() for ws in wanted_sections):
                fused[i] += config.SECTION_INTENT_BOOST
        fused = np.where(mask, fused, -np.inf)

        order = np.argsort(-fused)[:top_k]
        return [Hit(self.chunks[i], float(fused[i]), float(dense[i]), float(bm[i]), r + 1)
                for r, i in enumerate(order) if np.isfinite(fused[i])]

    def scheme_chunks(self, scheme_id: str, language: str, sections: list[str]) -> list[Chunk]:
        """All chunks of one scheme whose section matches one of `sections`."""
        return [c for c in self.chunks
                if c.scheme_id == scheme_id and c.language == language
                and any(s in c.section.lower() for s in sections)]
