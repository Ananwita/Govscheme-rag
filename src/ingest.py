"""Document loading and section-aware chunking.

Supported inputs in data/schemes/** :
  * .md  - with optional YAML-style front matter (scheme_id, title, language, ministry, source)
  * .txt - plain text (metadata taken from file name)
  * .pdf - official scheme PDFs/guidelines (text extracted with pypdf)

Each chunk keeps its scheme, section heading, language and source URL so that
every answer can be cited back to an exact place in a public document.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .text_utils import detect_language

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.M)


@dataclass
class Chunk:
    chunk_id: str
    scheme_id: str
    title: str
    section: str
    language: str
    source: str
    file: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation_label(self) -> str:
        return f"{self.title} — {self.section}"


# ------------------------------------------------------------------ loaders
def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily: only needed for PDFs

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(f"## Page {i}\n{txt}")
    return "\n\n".join(pages)


def load_documents(data_dir: Path = config.DATA_DIR) -> list[dict]:
    """Return a list of {meta, body, file} dicts for every supported file."""
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"} or not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            meta, body = {}, _read_pdf(path)
        else:
            meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        if not body.strip():
            print(f"[ingest] skipped empty file: {path}")
            continue
        meta.setdefault("scheme_id", path.stem)
        meta.setdefault("title", path.stem.replace("_", " ").title())
        meta.setdefault("language", detect_language(body))
        meta.setdefault("source", path.name)
        docs.append({"meta": meta, "body": body, "file": str(path.relative_to(config.ROOT))
                     if path.is_relative_to(config.ROOT) else str(path)})
    return docs


# ------------------------------------------------------------------ chunking
def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, text) pairs using ## / ### headings."""
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("Document", body.strip())]
    sections = []
    if body[: matches[0].start()].strip():
        sections.append(("Introduction", body[: matches[0].start()].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[m.end():end].strip()
        if len(m.group(1)) == 1 and not text:      # '# Title' line with no body
            continue
        if text:
            sections.append((m.group(2).strip(), text))
    return sections


def _window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_documents(docs: list[dict],
                    max_words: int = config.CHUNK_MAX_WORDS,
                    overlap: int = config.CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        meta = doc["meta"]
        for s_idx, (heading, text) in enumerate(_split_sections(doc["body"])):
            for w_idx, words in enumerate(_window(text.split(), max_words, overlap)):
                piece = " ".join(words)
                # Keep line breaks of bullet lists readable
                piece = re.sub(r"\s+-\s+", "\n- ", piece)
                chunks.append(Chunk(
                    chunk_id=f"{meta['scheme_id']}_{meta['language']}_{s_idx:02d}_{w_idx}",
                    scheme_id=meta["scheme_id"],
                    title=meta["title"],
                    section=heading,
                    language=meta["language"],
                    source=meta["source"],
                    file=doc["file"],
                    text=piece,
                ))
    return chunks


def passage_for_embedding(chunk: Chunk) -> str:
    """Text that is embedded: title + section give the model useful context."""
    return f"{chunk.title} | {chunk.section}: {chunk.text}"
