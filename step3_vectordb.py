"""
CANCER CHATBOT — STEP 3: STORE EMBEDDINGS IN CHROMADB
========================================================
Reads data/embeddings.npz (produced by Step 2),
loads vectors + metadata into a persistent ChromaDB
collection on disk, and runs a smoke-test query to
verify retrieval works.

ChromaDB is:
  - 100% free, runs locally
  - Persists to disk → no re-embedding needed across runs
  - Supports metadata filtering (e.g. cancer_type == "breast")
  - Uses HNSW index → similarity search in <100ms

Usage:
  pip install chromadb numpy
  python step3_vectordb.py

Output:
  data/chroma_db/  → persistent vector store directory
"""

import time
from pathlib import Path

import numpy as np
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = Path("data/embeddings.npz")
CHROMA_DIR      = Path("data/chroma_db")
COLLECTION_NAME = "cancer_research"
BATCH_SIZE      = 500      # rows per insert — Chroma is happy with 100–5000
# ─────────────────────────────────────────────────────────────────────────────


def load_embeddings(path: Path):
    """Load the NPZ archive saved by Step 2."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n  {path} not found.\n"
            f"  Run Step 2 first:  python step2_embed.py\n"
        )
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings":   data["embeddings"],
        "chunk_ids":    data["chunk_ids"],
        "sources":      data["sources"],
        "cancer_types": data["cancer_types"],
        "page_nums":    data["page_nums"],
        "token_counts": data["token_counts"],
        "texts":        data["texts"],
        "model_name":   str(data["model_name"]),
        "embed_dim":    int(data["embed_dim"]),
    }


def build_metadatas(d: dict) -> list:
    """Construct the per-chunk metadata dicts ChromaDB needs."""
    n = len(d["chunk_ids"])
    return [
        {
            "source":      str(d["sources"][i]),
            "cancer_type": str(d["cancer_types"][i]),
            "page_num":    int(d["page_nums"][i]),
            "token_count": int(d["token_counts"][i]),
        }
        for i in range(n)
    ]


def run_vectordb_load():
    print("=" * 60)
    print("  STEP 3 — STORE EMBEDDINGS IN CHROMADB")
    print("=" * 60)

    # ── 1. Load embeddings from Step 2 ──
    print(f"\n  Loading embeddings from {EMBEDDINGS_FILE} ...")
    d = load_embeddings(EMBEDDINGS_FILE)
    n_chunks = len(d["chunk_ids"])
    print(f"  Loaded {n_chunks} chunks")
    print(f"  Vector dim     : {d['embed_dim']}")
    print(f"  Embedding model: {d['model_name']}")

    # ── 2. Initialise persistent ChromaDB client ──
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Initialising ChromaDB at {CHROMA_DIR}/")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # If a previous run created the collection, delete it so we start clean.
    # (Comment this out if you want incremental updates instead of replace.)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Existing collection '{COLLECTION_NAME}' found — replacing")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space":      "cosine",   # cosine similarity (best for embeddings)
            "embedding_model": d["model_name"],
            "embed_dim":       d["embed_dim"],
        },
    )
    print(f"  Created collection: '{COLLECTION_NAME}'")

    # ── 3. Insert in batches ──
    metadatas    = build_metadatas(d)
    ids          = [str(x) for x in d["chunk_ids"]]
    documents    = [str(x) for x in d["texts"]]
    embeddings_l = d["embeddings"].tolist()

    print(f"\n  Inserting {n_chunks} chunks in batches of {BATCH_SIZE}...")
    t0 = time.time()
    for start in range(0, n_chunks, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_chunks)
        collection.add(
            ids        = ids[start:end],
            embeddings = embeddings_l[start:end],
            documents  = documents[start:end],
            metadatas  = metadatas[start:end],
        )
        print(f"    [{end:>5}/{n_chunks}] inserted")
    elapsed = time.time() - t0
    print(f"  Insertion done in {elapsed:.1f}s")

    # ── 4. Verify count ──
    final_count = collection.count()
    print(f"\n  Collection count: {final_count}")
    assert final_count == n_chunks, "Mismatch — some chunks failed to insert"

    # ── 5. Smoke-test query ──
    # Query with the first chunk's own embedding — best match should be itself
    print("\n  Smoke test: querying with chunk[0]'s own embedding")
    test_emb = d["embeddings"][0].tolist()
    results = collection.query(
        query_embeddings=[test_emb],
        n_results=3,
    )
    print(f"  Top-3 results (cosine distance, lower = better):")
    for i, (rid, dist, meta) in enumerate(zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        marker = "  ← self" if rid == ids[0] else ""
        print(f"    {i+1}. dist={dist:.4f}  id={rid}  "
              f"cancer={meta['cancer_type']}  page={meta['page_num']}{marker}")

    # ── 6. Filtered-query demo ──
    cancer_types_present = set(str(c) for c in d["cancer_types"])
    if len(cancer_types_present) > 1:
        sample_cancer = sorted(cancer_types_present)[0]
        print(f"\n  Filtered-query demo: top-3 chunks where cancer_type='{sample_cancer}'")
        filtered = collection.query(
            query_embeddings=[test_emb],
            n_results=3,
            where={"cancer_type": sample_cancer},
        )
        for i, (rid, dist, meta) in enumerate(zip(
            filtered["ids"][0],
            filtered["distances"][0],
            filtered["metadatas"][0],
        )):
            print(f"    {i+1}. dist={dist:.4f}  id={rid}  "
                  f"page={meta['page_num']}  source={meta['source']}")

    # ── 7. Summary ──
    print("\n" + "=" * 60)
    print("  VECTOR STORE READY")
    print("=" * 60)
    print(f"  Collection name : {COLLECTION_NAME}")
    print(f"  Chunks indexed  : {final_count}")
    print(f"  Persisted to    : {CHROMA_DIR}/")
    print(f"  Distance metric : cosine")

    db_size_mb = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file())
    print(f"  On-disk size    : {db_size_mb / (1024*1024):.2f} MB")

    print("\n  Cancer types in index:")
    cancer_counts = {}
    for ct in d["cancer_types"]:
        cancer_counts[str(ct)] = cancer_counts.get(str(ct), 0) + 1
    for ct, cnt in sorted(cancer_counts.items(), key=lambda x: -x[1]):
        print(f"    {ct:<18} {cnt:>4} chunks")

    print("\n  Next: Step 4 — embed user queries and retrieve top-k chunks.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_vectordb_load()
