"""Runs every question in samples/sample_queries.txt through the pipeline and
saves the answers + citations to samples/sample_outputs.md and .json
(use these in the report's "Results and Sample Outputs" section).

Run:  python make_samples.py
"""
import json
import sys

from src import config
from src.pipeline import RAGPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

rag = RAGPipeline()
queries = [q.strip() for q in (config.ROOT / "samples" / "sample_queries.txt")
           .read_text(encoding="utf-8").splitlines() if q.strip()]
results = [rag.ask(q) for q in queries]

md = [f"# Sample inputs and outputs\n\nEmbedder: `{rag.embedder.name}` · Generator: "
      f"`{results[0]['generator']}`\n"]
for i, r in enumerate(results, 1):
    md.append(f"---\n\n## {i}. {r['query']}\n")
    md.append(f"*Language: {r['language']} · intent: {', '.join(r['intents']) or 'general'} · "
              f"retrieval {r['timings_ms']['retrieval']} ms*\n")
    md.append(r["answer"] + "\n")
    md.append("**Sources**\n")
    for c in r["citations"]:
        md.append(f"- [{c['n']}] {c['title']} — {c['section']} ({c['language']}), "
                  f"score {c['score']} — {c['source']}")
    md.append("")
out = config.ROOT / "samples"
(out / "sample_outputs.md").write_text("\n".join(md), encoding="utf-8")
(out / "sample_outputs.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved {len(results)} answers to samples/sample_outputs.md and samples/sample_outputs.json")
