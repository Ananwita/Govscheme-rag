"""Okapi BM25 keyword retrieval (pure NumPy, works for English and Hindi tokens).

Keyword search complements dense embeddings: it is very good at exact matches
such as scheme names ("PMJJBY"), numbers ("Rs 436") and rare terms.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text_utils import tokenize


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in corpus]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.docs) else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_toks = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            denom_const = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            s = 0.0
            for t in q_toks:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + denom_const)
            out[i] = s
        return out
