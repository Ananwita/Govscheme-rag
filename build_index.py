"""Step 1: read all scheme documents, chunk them, embed them and save the index.

Run:  python build_index.py
Re-run whenever you add or edit files in data/schemes/.
"""
from src.index import build_index

if __name__ == "__main__":
    meta = build_index()
    print("\nIndex built successfully")
    for k, v in meta.items():
        print(f"  {k:14s}: {v}")
