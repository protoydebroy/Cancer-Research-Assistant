"""
CANCER CHATBOT — STEP 5: LLM ANSWER GENERATION (FREE, via Groq)
==================================================================
Combines retrieval (Step 4) with a real LLM call to produce
grounded, cited answers. Uses Groq's free Llama 3.3 70B —
no credit card, generous rate limits, ~500 tokens/sec.

The full RAG flow on a query:

   doctor's question
        │
        ▼
   Retriever (Step 4)  ──►  top-k cancer document chunks
        │
        ▼
   build prompt: system + chunks + question
        │
        ▼
   Groq Llama 3.3 70B  ──►  grounded answer with citations

Setup (one time):
  1. pip install groq python-dotenv
  2. Create account at https://console.groq.com  (free, no card)
  3. Settings → API Keys → Create
  4. In your .env file, add:
       GROQ_API_KEY=gsk_yourActualKeyHere

Usage:
  python step5_chat.py               # interactive REPL
  python step5_chat.py "your question here"   # one-shot
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from step4_retrieve import Retriever

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME    = "llama-3.3-70b-versatile"   # free, fast, 128K context
TEMPERATURE   = 0.2          # low → factual, deterministic answers
MAX_TOKENS    = 1024         # cap response length
TOP_K         = 5            # chunks to retrieve per query
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an oncology research assistant helping doctors with cancer-related research questions. You answer based on excerpts from the Indian Cancer Society's clinical reference documents.

GROUND RULES — these are non-negotiable:

1. ONLY answer using the provided <context> chunks. If the context does not contain enough information, say so clearly: "The provided documents do not cover this in detail." Do not invent facts.

2. CITE every clinical claim using the format [source.pdf, p.N]. Multiple citations are fine: [breast.pdf, p.3][breast.pdf, p.7].

3. STRUCTURE answers for clinicians:
   - Lead with a direct one-sentence answer.
   - Then expand with relevant detail: staging, biomarkers, line of treatment, key metrics.
   - Use bullet points sparingly, only for genuinely list-like content (drug regimens, contraindications).

4. CLINICAL TONE — assume the reader is a physician. Use proper terminology (e.g., "neoadjuvant chemotherapy", "ECOG PS", "5-year OS"). Do not over-explain basics.

5. SAFETY — always end with: "Note: This is a research aid, not clinical decision support. Verify with current guidelines and patient-specific factors."

6. If asked something outside oncology / the provided documents, politely decline and stay on-topic."""


def build_user_prompt(question: str, hits: list) -> str:
    """Assemble the prompt that includes retrieved context + the question."""
    if not hits:
        return (
            f"<context>\n  (No matching documents were retrieved.)\n</context>\n\n"
            f"<question>\n{question}\n</question>"
        )

    context_blocks = []
    for h in hits:
        block = (
            f"[chunk {h['rank']}] source={h['source']} | page={h['page_num']} | "
            f"cancer_type={h['cancer_type']} | similarity={h['similarity']:.3f}\n"
            f"{h['text']}"
        )
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    return (
        f"<context>\n{context_str}\n</context>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"Answer the question using only the context above. "
        f"Cite each clinical claim as [source, p.N]."
    )


class CancerChatbot:
    """Wraps retrieval + Groq LLM call in one clean interface."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "\n  GROQ_API_KEY not found.\n"
                "  Add it to your .env file:\n"
                "      GROQ_API_KEY=gsk_yourKeyHere\n"
                "  Get a free key at: https://console.groq.com\n"
            )

        print("  Initialising chatbot...")
        self.retriever = Retriever()
        self.llm       = Groq(api_key=api_key)
        print(f"  LLM model: {MODEL_NAME}")
        print("  Ready.\n")

    def ask(
        self,
        question: str,
        top_k: int = TOP_K,
        cancer_type: str = None,
        stream: bool = True,
        show_sources: bool = True,
    ) -> dict:
        """
        Run the full RAG pipeline on `question`.

        Returns dict with: answer, hits, latency_ms.
        """
        # 1. Retrieve relevant chunks
        t_retrieve = time.time()
        hits = self.retriever.retrieve(question, top_k=top_k, cancer_type=cancer_type)
        retrieve_ms = (time.time() - t_retrieve) * 1000

        # 2. Build prompt
        user_prompt = build_user_prompt(question, hits)

        # 3. Call Groq
        t_llm = time.time()
        if stream:
            print("\n  Answer:\n")
            answer_chunks = []
            response = self.llm.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    answer_chunks.append(delta)
            print()  # newline after streamed output
            answer = "".join(answer_chunks)
        else:
            response = self.llm.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            answer = response.choices[0].message.content

        llm_ms = (time.time() - t_llm) * 1000

        # 4. Show source list
        if show_sources and hits:
            print("\n  ── Sources used ──")
            seen = set()
            for h in hits:
                key = (h["source"], h["page_num"])
                if key in seen:
                    continue
                seen.add(key)
                print(f"    • {h['source']}, page {h['page_num']}  "
                      f"(similarity {h['similarity']:.2f})")

        # 5. Latency report
        print(f"\n  ── Latency ──")
        print(f"    retrieval : {retrieve_ms:>6.0f} ms")
        print(f"    LLM call  : {llm_ms:>6.0f} ms")
        print(f"    total     : {retrieve_ms + llm_ms:>6.0f} ms")

        return {
            "answer":      answer,
            "hits":        hits,
            "retrieve_ms": retrieve_ms,
            "llm_ms":      llm_ms,
        }


# ── REPL / one-shot driver ────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  STEP 5 — CANCER RESEARCH CHATBOT (RAG + Groq Llama 3.3)")
    print("=" * 70 + "\n")

    bot = CancerChatbot()

    # One-shot mode: passed a question on the command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"  Question: {question}")
        bot.ask(question)
        print()
        return

    # Interactive REPL
    print("  Type your medical question. Blank line to exit.")
    print("  Tip: prefix with 'breast:' / 'lung:' / etc. to filter by cancer type.\n")
    print("  " + "─" * 66)

    while True:
        try:
            q = input("\n  Doctor > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Bye.\n")
            break

        if not q:
            print("  Bye.\n")
            break

        # Optional cancer-type prefix:  "breast: what's first-line therapy?"
        cancer_filter = None
        if ":" in q[:20]:
            prefix, rest = q.split(":", 1)
            prefix = prefix.strip().lower()
            # match against retriever's known cancer types
            known = {"breast","cervical","oral","bladder","eye","pancreas",
                     "colorectal","esophageal","kidney","laryngeal","liver",
                     "lung","ovarian","prostate","skin","stomach","testicular",
                     "thyroid","uterine"}
            if prefix in known:
                cancer_filter = prefix
                q = rest.strip()
                print(f"  [filter: cancer_type='{cancer_filter}']")

        bot.ask(q, cancer_type=cancer_filter)


if __name__ == "__main__":
    main()
