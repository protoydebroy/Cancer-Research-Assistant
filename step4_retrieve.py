"""
CANCER CHATBOT — STEP 4: RETRIEVAL ENGINE
============================================
Embeds a doctor's natural-language query using the SAME
sentence-transformer model from Step 2, then runs a
similarity search over the ChromaDB collection from Step 3.

Two ways to use this file:

  1) Run it standalone to test queries from the terminal:
       python step4_retrieve.py

  2) Import the Retriever class from another script (Step 5
     will use it to feed context to the LLM):
       from step4_retrieve import Retriever
       r = Retriever()
       hits = r.retrieve("Stage 3 breast cancer treatment?", top_k=5)

Why use the same embedding model as Step 2?
  Embeddings only make sense within their own model's vector
  space. all-MiniLM-L6-v2 generates 384-dim vectors with a
  specific geometry — using a different model would give
  garbage similarity scores even with identical text.
"""

import time
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config (must match Step 2 & Step 3) ──────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db")
COLLECTION_NAME = "cancer_research"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 5
# ─────────────────────────────────────────────────────────────────────────────


class Retriever:
    """Encapsulates query embedding + similarity search + filtering."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = MODEL_NAME,
    ):
        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"\n  ChromaDB folder {chroma_dir} not found.\n"
                f"  Run Step 3 first:  python step3_vectordb.py\n"
            )

        print(f"  Loading embedding model: {model_name}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        print(f"  Connecting to ChromaDB at {chroma_dir}/")
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(name=collection_name)
        print(f"  Collection '{collection_name}' has {self.collection.count()} chunks\n")

    # ── Core method ─────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        cancer_type: Optional[str] = None,
        cancer_types: Optional[list] = None,
    ) -> list:
        """
        Embed `query` and return the top_k most similar chunks.

        Args:
            query:        Doctor's question, e.g. "Stage 3 breast cancer Tx?"
            top_k:        Number of chunks to return.
            cancer_type:  If set, only search within this cancer type.
            cancer_types: If set, only search within this list (uses $or).

        Returns:
            List of dicts ordered best→worst:
              { rank, similarity, distance, chunk_id, source,
                cancer_type, page_num, text }
        """
        # 1. Embed the query — must use the SAME model + normalisation as Step 2
        query_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # 2. Build optional metadata filter
        where = None
        if cancer_type:
            where = {"cancer_type": cancer_type}
        elif cancer_types:
            if len(cancer_types) == 1:
                where = {"cancer_type": cancer_types[0]}
            else:
                where = {"$or": [{"cancer_type": c} for c in cancer_types]}

        # 3. Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where,
        )

        # 4. Re-shape into a clean list of dicts
        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for rank, (rid, dist, meta, doc) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ), start=1):
            hits.append({
                "rank":        rank,
                "similarity":  1.0 - dist,    # cosine sim = 1 - cosine dist
                "distance":    dist,
                "chunk_id":    rid,
                "source":      meta["source"],
                "cancer_type": meta["cancer_type"],
                "page_num":    meta["page_num"],
                "text":        doc,
            })
        return hits

    # ── Helper: pretty-print results ─────────────────────────────────────────
    @staticmethod
    def display(query: str, hits: list, max_text_chars: int = 220) -> None:
        print("─" * 70)
        print(f"  QUERY: {query}")
        print("─" * 70)
        if not hits:
            print("  No matching chunks. Try a different query or remove filters.\n")
            return

        for h in hits:
            print(f"\n  [{h['rank']}] similarity={h['similarity']:.3f}  "
                  f"cancer={h['cancer_type']}  page={h['page_num']}")
            print(f"      source: {h['source']}")
            text_preview = h["text"][:max_text_chars].replace("\n", " ")
            print(f"      text  : {text_preview}...")
        print()


# ── Demo / smoke-test mode ───────────────────────────────────────────────────
DEMO_QUERIES = [
    # (query_text, cancer_type_filter_or_None)
    ("What is the first-line treatment for HER2+ breast cancer?", None),
    ("How is stage 3 lung cancer staged and treated?",            None),
    ("Explain BRCA mutation testing in breast cancer",            "breast"),
    ("What are common metastatic sites in colorectal cancer?",    "colorectal"),
    ("Side effects of cisplatin chemotherapy",                    None),
]


def run_demo():
    print("=" * 70)
    print("  STEP 4 — RETRIEVAL ENGINE")
    print("=" * 70 + "\n")

    retriever = Retriever()

    for query, cancer_filter in DEMO_QUERIES:
        t0 = time.time()
        hits = retriever.retrieve(query, top_k=3, cancer_type=cancer_filter)
        elapsed_ms = (time.time() - t0) * 1000

        if cancer_filter:
            print(f"\n  [filtered to cancer_type='{cancer_filter}']")
        retriever.display(query, hits)
        print(f"  → retrieved in {elapsed_ms:.0f} ms\n")

    # Interactive mode
    print("=" * 70)
    print("  INTERACTIVE MODE — type your own queries (blank line to exit)")
    print("=" * 70)
    while True:
        try:
            q = input("\n  Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.\n")
            break
        if not q:
            print("  Bye.\n")
            break
        hits = retriever.retrieve(q, top_k=5)
        retriever.display(q, hits)


if __name__ == "__main__":
    run_demo()
