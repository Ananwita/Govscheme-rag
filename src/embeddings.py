"""Multilingual sentence embeddings.

Primary: a Hugging Face sentence-transformers model (default
`intfloat/multilingual-e5-small`) that maps English and Hindi text into the
SAME vector space, so a Hindi question can retrieve an English passage and
vice-versa (cross-lingual retrieval).

Fallback: if sentence-transformers / torch are not installed, a character
n-gram TF-IDF model is used so that the app still runs (lower quality and
NOT cross-lingual). The UI shows which backend is active.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.backend = "sentence-transformers"
        self._model = None
        self._tfidf = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # ImportError, network error, etc.
            print(f"[embeddings] Could not load '{model_name}' ({exc.__class__.__name__}: {exc}).")
            print("[embeddings] Falling back to char n-gram TF-IDF (not cross-lingual).")
            self.backend = "tfidf-fallback"
            self.model_name = "char-ngram-tfidf"

    # E5 models expect "query: " / "passage: " prefixes.
    @property
    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.model_name}"

    # ------------------------------------------------------------ fitting
    def fit(self, passages: list[str]) -> None:
        """Only needed for the TF-IDF fallback."""
        if self.backend == "tfidf-fallback":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                          sublinear_tf=True, min_df=1)
            self._tfidf.fit(passages)

    def save(self, index_dir: Path) -> None:
        if self._tfidf is not None:
            with open(index_dir / "tfidf.pkl", "wb") as fh:
                pickle.dump(self._tfidf, fh)

    def load(self, index_dir: Path) -> None:
        p = index_dir / "tfidf.pkl"
        if self.backend == "tfidf-fallback" and p.exists():
            with open(p, "rb") as fh:
                self._tfidf = pickle.load(fh)

    # ------------------------------------------------------------ encoding
    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, batch_size=32, show_progress_bar=len(texts) > 64,
                                      normalize_embeddings=True, convert_to_numpy=True)
            return vecs.astype(np.float32)
        if self._tfidf is None:
            raise RuntimeError("TF-IDF fallback not fitted; build the index first.")
        mat = self._tfidf.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-9, None)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        if self._is_e5 and self._model is not None:
            passages = [f"passage: {p}" for p in passages]
        return self._encode(passages)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if self._is_e5 and self._model is not None:
            queries = [f"query: {q}" for q in queries]
        return self._encode(queries)
