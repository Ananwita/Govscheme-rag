"""End-to-end RAG pipeline: question -> language detection -> hybrid retrieval
-> cited answer generation."""
from __future__ import annotations

import time

from . import config
from .embeddings import Embedder
from .generator import ExtractiveGenerator, LLMGenerator, build_citations, llm_from_config
from .index import build_index, index_exists
from .retriever import HybridRetriever
from .text_utils import detect_intents, detect_language


class RAGPipeline:
    def __init__(self, rebuild: bool = False):
        self.embedder = Embedder()
        if rebuild or not index_exists():
            print("[pipeline] Building index ...")
            print("[pipeline]", build_index(embedder=self.embedder))
        try:
            self.retriever = HybridRetriever(embedder=self.embedder)
        except RuntimeError as exc:          # index built with a different embedder
            print(f"[pipeline] {exc} -> rebuilding")
            build_index(embedder=self.embedder)
            self.retriever = HybridRetriever(embedder=self.embedder)
        self.extractive = ExtractiveGenerator(self.embedder)
        self.llm = llm_from_config()

    def configure_llm(self, base_url: str, model: str, api_key: str) -> tuple[bool, str]:
        """Connect an LLM at runtime (used by the app's sidebar). Returns (ok, message)."""
        llm = LLMGenerator(base_url=base_url, model=model, api_key=api_key)
        ok, msg = llm.check()
        if ok:
            self.llm = llm
        return ok, msg

    def disable_llm(self) -> None:
        self.llm = None

    @property
    def info(self) -> dict:
        return {**self.retriever.meta,
                "embedder_active": self.embedder.name,
                "llm": self.llm.name if self.llm else "not configured (extractive mode)"}

    def ask(self, query: str, top_k: int = config.TOP_K, use_llm: bool = True,
            languages: list[str] | None = None, schemes: list[str] | None = None,
            answer_language: str = "auto") -> dict:
        t0 = time.time()
        query = query.strip()
        lang = detect_language(query) if answer_language == "auto" else answer_language
        hits = self.retriever.search(query, top_k=top_k, languages=languages,
                                     schemes=schemes, prefer_language=lang)
        t1 = time.time()

        generator = self.llm if (use_llm and self.llm) else self.extractive
        try:
            answer = generator.generate(query, hits, lang)
        except Exception as exc:              # LLM down -> graceful fallback
            answer = self.extractive.generate(query, hits, lang)
            answer += f"\n\n_(LLM unavailable: {exc.__class__.__name__}; showing extractive answer)_"
            generator = self.extractive
        t2 = time.time()

        return {
            "query": query,
            "language": lang,
            "intents": detect_intents(query),
            "answer": answer,
            "citations": build_citations(hits),
            "generator": generator.name,
            "timings_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000)},
        }
