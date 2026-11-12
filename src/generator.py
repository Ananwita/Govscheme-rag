"""Answer generation with citations.

Two interchangeable generators:

1. ExtractiveGenerator (default, offline, no GPU / API key needed)
   Picks the most relevant sentences from the retrieved passages using the
   multilingual embedder + keyword overlap and stitches them into an answer.
   Every sentence carries a [n] citation, so nothing is "made up".

2. LLMGenerator (optional)
   Sends the numbered passages to any OpenAI-compatible chat endpoint
   (Ollama locally, Groq, OpenAI, ...) with strict instructions to answer ONLY
   from the passages, in the user's language, with [n] citations.
"""
from __future__ import annotations

import numpy as np
import requests

from . import config
from .embeddings import Embedder
from .retriever import Hit
from .text_utils import INTENT_SECTIONS, detect_intents, split_sentences, tokenize

NOT_FOUND = {
    "en": "I could not find this information in the indexed government scheme documents.",
    "hi": "यह जानकारी उपलब्ध सरकारी योजना दस्तावेज़ों में नहीं मिली।",
}
DISCLAIMER = {
    "en": "Information is from public scheme documents; please verify on the official portal before applying.",
    "hi": "जानकारी सार्वजनिक योजना दस्तावेज़ों से ली गई है; आवेदन से पहले आधिकारिक पोर्टल पर पुष्टि करें।",
}
SOURCE_LANG_NOTE = {
    "en": "(Matching passages were found in Hindi.)",
    "hi": "(संबंधित जानकारी अंग्रेज़ी दस्तावेज़ में मिली है।)",
}


def build_citations(hits: list[Hit]) -> list[dict]:
    return [{
        "n": i + 1,
        "title": h.chunk.title,
        "section": h.chunk.section,
        "language": h.chunk.language,
        "source": h.chunk.source,
        "file": h.chunk.file,
        "score": round(h.score, 3),
        "dense": round(h.dense, 3),
        "bm25": round(h.bm25, 2),
        "text": h.chunk.text,
    } for i, h in enumerate(hits)]


class ExtractiveGenerator:
    """Offline answerer.

    * Focuses on the best-matching scheme (rank-1 passage).
    * If the question has an intent (eligibility / benefits / documents / apply)
      and the retrieved passages contain that section, the whole section is
      returned verbatim with its citation - ideal for "who is eligible" questions.
    * Otherwise the most relevant sentences are selected with the embedder
      plus keyword overlap.
    """
    name = "extractive (offline)"

    def __init__(self, embedder: Embedder, max_sentences: int = 5, max_blocks: int = 2):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.max_blocks = max_blocks

    @staticmethod
    def _render_block(text: str, n: int) -> list[str]:
        lines = []
        for part in text.split("\n"):
            part = part.strip()
            if not part:
                continue
            lines.append(f"- {part[2:]}" if part.startswith("- ") else part)
        lines[-1] += f" [{n}]"
        return lines

    def generate(self, query: str, hits: list[Hit], lang: str) -> str:
        if not hits:
            return NOT_FOUND[lang]
        primary = hits[0].chunk.scheme_id
        prim = [(n, h) for n, h in enumerate(hits, start=1) if h.chunk.scheme_id == primary]
        # prefer passages in the answer language when the scheme has them
        same_lang = [(n, h) for n, h in prim if h.chunk.language == lang]
        prim = same_lang or prim
        title = prim[0][1].chunk.title

        wanted = {s for i in detect_intents(query) for s in INTENT_SECTIONS[i]}
        blocks = [(n, h) for n, h in prim
                  if wanted and any(w in h.chunk.section.lower() for w in wanted)][: self.max_blocks]

        lines = [f"**{title}**", ""]
        if blocks:
            blocks.sort(key=lambda nh: nh[1].chunk.chunk_id)        # document order
            for n, h in blocks:
                lines.append(f"*{h.chunk.section}*")
                lines += self._render_block(h.chunk.text, n)
                lines.append("")
        else:
            lines += self._select_sentences(query, prim)

        out = "\n".join(lines).strip()
        if prim[0][1].chunk.language != lang:
            out += f"\n\n_{SOURCE_LANG_NOTE[lang]}_"
        return out + f"\n\n_{DISCLAIMER[lang]}_"

    def _select_sentences(self, query: str, hits: list[tuple[int, Hit]]) -> list[str]:
        cands = []   # (n, position, sentence)
        for n, h in hits:
            for pos, s in enumerate(split_sentences(h.chunk.text)):
                cands.append((n, pos, s))
        if not cands:
            return []
        q_tokens = set(tokenize(query))
        q_vec = self.embedder.encode_queries([query])[0]
        sims = self.embedder.encode_passages([c[2] for c in cands]) @ q_vec
        scores = np.array([float(cos) + 0.3 * len(q_tokens & set(tokenize(s))) / (len(q_tokens) or 1)
                           + 0.05 / n for (n, _, s), cos in zip(cands, sims)])
        best = scores.max()
        chosen = [i for i in np.argsort(-scores) if scores[i] >= best - 0.2][: self.max_sentences]
        # a sentence ending with ':' introduces a list -> pull in the following items
        extra = []
        for i in chosen:
            if cands[i][2].rstrip().endswith(":"):
                j = i + 1
                while j < len(cands) and cands[j][0] == cands[i][0] and len(extra) < 8:
                    extra.append(j)
                    j += 1
        chosen = sorted(set(chosen) | set(extra), key=lambda i: (cands[i][0], cands[i][1]))
        return [f"- {cands[i][2].rstrip()} [{cands[i][0]}]" for i in chosen]


# Ready-made OpenAI-compatible endpoints. Groq and Gemini need only a free API key
# (no big download); Ollama runs a model locally on your laptop.
PROVIDERS = {
    "Groq (free API key)": {"base_url": "https://api.groq.com/openai/v1",
                            "model": "llama-3.3-70b-versatile",
                            "key_url": "https://console.groq.com/keys"},
    "Google Gemini (free API key)": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                                     "model": "gemini-2.5-flash",
                                     "key_url": "https://aistudio.google.com/apikey"},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:3b",
                       "key_url": "https://ollama.com"},
}

# When the configured model is not offered by the endpoint, pick the first match.
_PREFERRED = ["llama-3.3-70b", "llama-3.1-70b", "llama-3.1-8b", "llama", "gemini-2.5-flash",
              "gemini", "qwen", "gpt"]
_SKIP = ["guard", "whisper", "embed", "tts", "vision", "image", "audio", "aqa"]


class LLMGenerator:
    """Generation step of RAG: the LLM writes the answer, but ONLY from the
    retrieved passages, in the user's language, with [n] citations."""

    SYSTEM = (
        "You are a helpful assistant for Indian government welfare schemes. "
        "Answer ONLY using the numbered context passages. After every fact add the "
        "passage number in square brackets, e.g. [1] or [2][3]. When the question is about "
        "eligibility, list who is eligible and who is excluded as bullet points. "
        "If the answer is not in the passages, say you could not find it. "
        "Never invent amounts, dates or criteria. Reply in {language}. Be concise."
    )

    def __init__(self, base_url: str = config.LLM_BASE_URL, model: str = config.LLM_MODEL,
                 api_key: str = config.LLM_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "none"

    @property
    def name(self) -> str:
        return f"LLM ({self.model})"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def check(self) -> tuple[bool, str]:
        """Test the connection and choose an available model. Returns (ok, message)."""
        if not self.base_url:
            return False, "No LLM endpoint configured."
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=8)
        except requests.RequestException as exc:
            return False, f"Cannot reach {self.base_url} ({exc.__class__.__name__})."
        if r.status_code in (401, 403):
            return False, "API key was rejected - check that it is copied correctly."
        if not r.ok:
            return False, f"Endpoint returned HTTP {r.status_code}."
        try:
            ids = [m["id"].removeprefix("models/") for m in r.json().get("data", [])]
        except ValueError:
            ids = []
        if ids and self.model not in ids:
            usable = [i for i in ids if not any(s in i.lower() for s in _SKIP)]
            for pref in _PREFERRED:
                match = next((i for i in usable if pref in i.lower()), None)
                if match:
                    self.model = match
                    break
            else:
                if usable:
                    self.model = usable[0]
        return True, f"Connected - using {self.model}."

    def generate(self, query: str, hits: list[Hit], lang: str) -> str:
        if not hits:
            return NOT_FOUND[lang]
        context = "\n\n".join(
            f"[{i + 1}] ({h.chunk.title} — {h.chunk.section})\n{h.chunk.text}"
            for i, h in enumerate(hits))
        language = "Hindi (Devanagari script)" if lang == "hi" else "English"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": self.SYSTEM.format(language=language)},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        }
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                             headers=self._headers(), timeout=config.LLM_TIMEOUT)
        if not resp.ok:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text + f"\n\n_{DISCLAIMER[lang]}_"


def llm_from_config() -> LLMGenerator | None:
    """LLM configured in .env, if it is reachable; otherwise None (extractive mode)."""
    if not config.LLM_BASE_URL:
        return None
    llm = LLMGenerator()
    ok, msg = llm.check()
    print(f"[generator] LLM: {msg}")
    return llm if ok else None
