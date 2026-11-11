"""Draws docs/pipeline_diagram.png - the processing-pipeline block diagram."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, GREEN, ORANGE, PURPLE, GREY = "#dbe9f7", "#dcf1e0", "#fde7cf", "#e9e1f6", "#eeeeee"
EDGE = "#33475b"

fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=EDGE, lw=1.3))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=10.5, weight="bold", color="#1b2733")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.3, color="#2b3a48", linespacing=1.35)


def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=EDGE))
    if label:
        ax.text((x1 + x2) / 2 + 1.6, (y1 + y2) / 2 + 0.05, label, ha="center", fontsize=7.5, color="#556")


ax.text(7, 7.0, "Multilingual RAG for Government Schemes - Processing Pipeline",
        ha="center", fontsize=13.5, weight="bold", color="#1b2733")
ax.text(0.2, 6.45, "OFFLINE  (indexing:  python build_index.py)", fontsize=9, weight="bold", color="#7a5a00")
ax.text(0.2, 3.25, "ONLINE  (query time:  streamlit run app.py)", fontsize=9, weight="bold", color="#1f5f2f")

# ---------------- offline row
box(0.2, 4.0, 2.5, 2.2, "1. Scheme documents",
    "Public scheme docs\n(PM-KISAN, PM-JAY, PMUY,\nAPY, SSY, MUDRA, ...)\nEnglish + Hindi\n.md / .pdf / .txt", GREY)
box(3.1, 4.0, 2.5, 2.2, "2. Ingestion",
    "Front-matter metadata:\nscheme, language,\nofficial source URL\nPDF text via pypdf", ORANGE)
box(6.0, 4.0, 2.5, 2.2, "3. Section chunking",
    "Split on headings\n(Eligibility, Benefits,\nDocuments, How to apply)\n<=200 words, 30 overlap", ORANGE)
box(8.9, 4.0, 2.5, 2.2, "4. Embedding",
    "multilingual-e5-small\n(Hugging Face)\n384-dim, normalised\n'passage:' prefix", PURPLE)
box(11.8, 4.0, 2.0, 2.2, "5. Index",
    "embeddings.npy\nchunks.json\n+ BM25 keyword\nindex", BLUE)
for x in (2.7, 5.6, 8.5, 11.4):
    arrow(x, 5.1, x + 0.4, 5.1)

# ---------------- online row
box(0.2, 0.3, 2.3, 2.6, "A. User query",
    "English or Hindi\n\"PM Kisan ke liye\nkaun patra hai?\"\nStreamlit UI / CLI", GREY)
box(2.85, 0.3, 2.3, 2.6, "B. Query analysis",
    "Language detection\n(Devanagari ratio)\nIntent detection:\neligibility / benefits /\ndocuments / apply", ORANGE)
box(5.5, 0.3, 2.8, 2.6, "C. Hybrid retrieval",
    "Dense cosine (e5, 'query:')\n+ BM25 keywords\nWeighted RRF fusion\n+ same-language boost\n+ section-intent boost\n-> top-k chunks", PURPLE)
box(8.65, 0.3, 2.7, 2.6, "D. Answer generation",
    "Extractive (offline):\nsection / sentence selection\nOR LLM (Ollama / Groq)\ngrounded on passages\n[n] citations", GREEN)
box(11.7, 0.3, 2.1, 2.6, "E. Output",
    "Answer in query\nlanguage\nCited sources +\nofficial links\nEligibility checker", BLUE)
for x1, x2 in ((2.5, 2.85), (5.15, 5.5), (8.3, 8.65), (11.35, 11.7)):
    arrow(x1, 1.6, x2, 1.6)

arrow(12.8, 4.0, 6.9, 2.9, "vector + BM25 lookup")

fig.savefig("docs/pipeline_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved docs/pipeline_diagram.png")
