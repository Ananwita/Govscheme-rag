"""Evaluate the RAG system on a labelled set of English + Hindi questions.

Metrics (scheme-level, top-5 retrieved chunks):
  Hit@1, Hit@3   - correct scheme appears at rank 1 / within top 3
  MRR            - mean reciprocal rank of the first correct chunk
  Section@1      - top chunk comes from the section the question asks about
                   (e.g. an Eligibility question -> Eligibility/Exclusions section)
  FactRecall     - expected key fact (amount, age, ...) appears in the final answer

Configurations compared:
  hybrid         - dense + BM25 + boosts (the system as deployed)
  dense_only     - multilingual embeddings only
  bm25_only      - keyword search only
  cross_lingual  - each question searched ONLY against documents in the OTHER
                   language (Hindi question -> English docs and vice-versa).
                   This isolates the benefit of multilingual embeddings.

Run:  python evaluate.py
Outputs: eval/results.json, eval/results.md, eval/metrics_chart.png
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.text_utils import INTENT_SECTIONS  # noqa: E402

TOP_K = 5


def run_config(rag: RAGPipeline, questions: list[dict], name: str, answers: bool = False) -> dict:
    rows = []
    for q in questions:
        languages = None
        if name == "cross_lingual":
            languages = ["en"] if q["lang"] == "hi" else ["hi"]
        hits = rag.retriever.search(q["q"], top_k=TOP_K, languages=languages,
                                    prefer_language=q["lang"])
        rank = next((h.rank for h in hits if h.chunk.scheme_id == q["scheme"]), None)
        sec_ok = bool(hits) and hits[0].chunk.scheme_id == q["scheme"] and any(
            s in hits[0].chunk.section.lower() for s in INTENT_SECTIONS[q["intent"]])
        row = {"id": q["id"], "lang": q["lang"], "rank": rank, "section_ok": sec_ok,
               "top1": f"{hits[0].chunk.scheme_id}/{hits[0].chunk.section}" if hits else None}
        if answers:
            ans = rag.ask(q["q"], top_k=TOP_K, use_llm=False)["answer"]
            row["fact_ok"] = all(f.lower() in ans.lower() for f in q["facts"])
        rows.append(row)
    return {"name": name, "rows": rows}


def summarise(rows: list[dict]) -> dict:
    n = len(rows) or 1
    out = {
        "n": len(rows),
        "Hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "Hit@3": sum(r["rank"] is not None and r["rank"] <= 3 for r in rows) / n,
        "MRR": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "Section@1": sum(r["section_ok"] for r in rows) / n,
    }
    if rows and "fact_ok" in rows[0]:
        out["FactRecall"] = sum(r["fact_ok"] for r in rows) / n
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    questions = json.loads((config.EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    rag = RAGPipeline()
    print("Embedder:", rag.embedder.name)

    results = {}
    dw, bw = config.DENSE_WEIGHT, config.BM25_WEIGHT
    for name, weights, answers in [("hybrid", (dw, bw), True), ("dense_only", (1.0, 0.0), False),
                                   ("bm25_only", (0.0, 1.0), False), ("cross_lingual", (dw, bw), False)]:
        config.DENSE_WEIGHT, config.BM25_WEIGHT = weights
        res = run_config(rag, questions, name, answers)
        results[name] = {
            "all": summarise(res["rows"]),
            "en": summarise([r for r in res["rows"] if r["lang"] == "en"]),
            "hi": summarise([r for r in res["rows"] if r["lang"] == "hi"]),
            "rows": res["rows"],
        }
    config.DENSE_WEIGHT, config.BM25_WEIGHT = dw, bw

    # -------- save JSON
    (config.EVAL_DIR / "results.json").write_text(
        json.dumps({"embedder": rag.embedder.name, "results": results}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    # -------- markdown table
    metrics = ["Hit@1", "Hit@3", "MRR", "Section@1", "FactRecall"]
    lines = [f"Embedder: `{rag.embedder.name}`  |  questions: {len(questions)} "
             f"({sum(q['lang'] == 'en' for q in questions)} EN, {sum(q['lang'] == 'hi' for q in questions)} HI)\n",
             "| Configuration | Subset | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for name, r in results.items():
        for subset in ("all", "en", "hi"):
            vals = [f"{r[subset].get(m, '-')}" for m in metrics]
            lines.append(f"| {name} | {subset} | " + " | ".join(vals) + " |")
    failures = [f"- `{row['id']}` expected **{q['scheme']}** got rank {row['rank']} (top1: {row['top1']})"
                for q, row in zip(questions, results["hybrid"]["rows"]) if row["rank"] != 1]
    lines += ["", "### Hybrid misses at rank 1", *(failures or ["- none"])]
    md = "\n".join(lines)
    (config.EVAL_DIR / "results.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)

    # -------- chart
    names = list(results)
    show = ["Hit@1", "Hit@3", "MRR", "Section@1"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    for i, m in enumerate(show):
        vals = [results[n]["all"][m] for n in names]
        bars = ax.bar([x + i * width for x in range(len(names))], vals, width, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks([x + 1.5 * width for x in range(len(names))])
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title(f"Retrieval quality on {len(questions)} EN+HI questions\n({rag.embedder.name})", fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.EVAL_DIR / "metrics_chart.png", dpi=160)
    print(f"\nSaved eval/results.json, eval/results.md, eval/metrics_chart.png")


if __name__ == "__main__":
    main()
