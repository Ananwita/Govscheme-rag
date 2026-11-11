"""Terminal question-answering (handy for a quick demo or debugging).

Run:  python cli.py                 -> interactive mode
      python cli.py "PM Kisan ke liye kaun patra hai?"   -> single question
"""
import sys

from src.pipeline import RAGPipeline


def show(res: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Q [{res['language']}]: {res['query']}")
    print(f"intents: {res['intents'] or '-'} | generator: {res['generator']} | "
          f"time: {res['timings_ms']}")
    print("-" * 78)
    print(res["answer"])
    print("-" * 78 + "\nSources:")
    for c in res["citations"]:
        print(f"  [{c['n']}] {c['title']} — {c['section']} ({c['language']}) "
              f"score={c['score']}  {c['source']}")
    print("=" * 78)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Hindi output on Windows terminals
    rag = RAGPipeline()
    print("Loaded:", rag.info)
    if len(sys.argv) > 1:
        show(rag.ask(" ".join(sys.argv[1:])))
        return
    print("\nAsk about Indian government schemes in English or Hindi. Type 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            show(rag.ask(q))


if __name__ == "__main__":
    main()
