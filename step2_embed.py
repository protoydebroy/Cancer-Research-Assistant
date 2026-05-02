"""
CANCER CHATBOT — STEP 2: GENERATE EMBEDDINGS (FREE, LOCAL)
============================================================
Reads chunks from data/chunks.jsonl (produced by Step 1),
runs them through a local sentence-transformer model,
and saves the resulting vectors to data/embeddings.npz.

Why this model?
  all-MiniLM-L6-v2:
    - 22 MB (downloads once, runs offline forever)
    - 384-dim embeddings (fast similarity search)
    - Strong on medical/scientific text
    - Runs on CPU in ~30 seconds for hundreds of chunks
    - 100% free, no API key, no rate limits

Usage:
  pip install sentence-transformers numpy
  python step2_embed.py

Output:
  data/embeddings.npz  — vectors + metadata, ready for Step 3 (ChromaDB)
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILE      = Path("data/chunks.jsonl")
OUTPUT_FILE     = Path("data/embeddings.npz")
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 32           # chunks per forward pass — adjust if low RAM
SHOW_PROGRESS   = True
# ─────────────────────────────────────────────────────────────────────────────


def load_chunks(path: Path) -> list:
    """Read JSONL file produced by Step 1."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n  {path} not found.\n"
            f"  Run Step 1 first:  python step1_ingest.py\n"
        )
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def run_embedding():
    print("=" * 60)
    print("  STEP 2 — GENERATE EMBEDDINGS (local, free)")
    print("=" * 60)

    # ── 1. Load chunks ──
    print("\n  Loading chunks from", INPUT_FILE, "...")
    chunks = load_chunks(INPUT_FILE)
    print(f"  Loaded {len(chunks)} chunks")

    if not chunks:
        print("\n  No chunks to embed. Exiting.\n")
        return

    # ── 2. Load embedding model ──
    print(f"\n  Loading model: {MODEL_NAME}")
    print("  (first run downloads ~22 MB; subsequent runs are instant)")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── 3. Generate embeddings ──
    texts = [c["text"] for c in chunks]
    print(f"\n  Embedding {len(texts)} chunks (batch size {BATCH_SIZE})...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=SHOW_PROGRESS,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → cosine sim = dot product
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}  size: {embeddings.nbytes / 1024:.1f} KB")

    # ── 4. Build parallel metadata arrays ──
    chunk_ids    = np.array([c["chunk_id"]    for c in chunks])
    sources      = np.array([c["source"]      for c in chunks])
    cancer_types = np.array([c["cancer_type"] for c in chunks])
    page_nums    = np.array([c["page_num"]    for c in chunks], dtype=np.int32)
    token_counts = np.array([c["token_count"] for c in chunks], dtype=np.int32)
    texts_arr    = np.array(texts, dtype=object)   # variable-length strings

    # ── 5. Save everything ──
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings   = embeddings.astype(np.float32),
        chunk_ids    = chunk_ids,
        sources      = sources,
        cancer_types = cancer_types,
        page_nums    = page_nums,
        token_counts = token_counts,
        texts        = texts_arr,
        model_name   = np.array(MODEL_NAME),
        embed_dim    = np.array(embeddings.shape[1]),
    )

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    # ── 6. Summary ──
    print("\n" + "=" * 60)
    print("  EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Embedding model : {MODEL_NAME}")
    print(f"  Vector dim      : {embeddings.shape[1]}")
    print(f"  Output file     : {OUTPUT_FILE}  ({file_size_mb:.2f} MB)")

    # By cancer type
    cancer_counts = {}
    for ct in cancer_types:
        cancer_counts[str(ct)] = cancer_counts.get(str(ct), 0) + 1
    print(f"\n  Embeddings by cancer type:")
    for ct, cnt in sorted(cancer_counts.items(), key=lambda x: -x[1]):
        print(f"    {ct:<18} {cnt:>4}")

    # ── 7. Sanity check — similarity between first 2 chunks ──
    if len(embeddings) >= 2:
        sim = float(np.dot(embeddings[0], embeddings[1]))
        print(f"\n  Sanity check:")
        print(f"    Cosine sim (chunk 0 vs chunk 1) = {sim:.4f}")
        print(f"    (1.0 = identical, 0.0 = unrelated, ~0.3-0.7 typical for related text)")

    print("\n  Next: Step 3 — load these into ChromaDB for fast retrieval.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_embedding()
