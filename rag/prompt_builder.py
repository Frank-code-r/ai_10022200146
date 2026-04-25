# prompt_builder.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# Part C: Prompt Engineering & Generation

"""
PROMPT DESIGN ITERATIONS
=========================

v1 — Basic prompt (no hallucination guard):
  Just passed context and asked a question.
  Problem: LLM invented facts not in the context.

v2 — Added hallucination guard:
  Told the LLM to only answer from context.
  If not found, say "I don't have that information."
  Result: Hallucination eliminated.

v3 — Added numbered passages + source labels (current):
  Each chunk labelled [P1], [P2] with its source.
  LLM now cites which passage it used.
  Result: Answers are traceable and more trustworthy.

CONTEXT WINDOW MANAGEMENT
==========================
- Max context: 3000 characters (~750 tokens)
- Chunks are already ranked by retrieval score
- We include highest ranked chunks first
- Stop adding chunks when we hit the limit
- Chunks with very low scores are discarded (noise filter)
"""

MAX_CONTEXT_CHARS = 3000
SCORE_THRESHOLD = 0.001  # discard near-irrelevant chunks

SYSTEM_PROMPT = """You are ACity AI, an intelligent assistant for Academic City University, Ghana.
You answer questions about Ghana election results and the 2025 Ghana Budget Statement.

Rules:
- Answer ONLY using the context passages provided below.
- If the context does not contain the answer, say: "I don't have enough information in my knowledge base to answer that accurately."
- Do NOT invent figures, names, or facts.
- Cite the passage number you used, e.g. [P1] or [P2].
- Be concise and factual.
- If the context contains regional vote data for multiple regions, use it to identify the candidate with the most votes overall."""


def _select_chunks(retrieved):
    """
    Filter and truncate chunks to fit the context window.
    - Drop chunks below score threshold
    - Add chunks greedily until MAX_CONTEXT_CHARS is reached
    """
    filtered = [(c, s) for c, s in retrieved if s >= SCORE_THRESHOLD]

    selected = []
    total_chars = 0
    for chunk, score in filtered:
        if total_chars + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        selected.append((chunk, score))
        total_chars += len(chunk.text)

    return selected


def build_prompt(query, retrieved):
    """
    Build the final prompt to send to the LLM.

    Args:
        query:     The user's question
        retrieved: List of (Chunk, score) from the retriever

    Returns:
        system_prompt  — instructions for the LLM
        user_message   — context + question
        selected       — chunks that were included (for display in UI)
    """
    selected = _select_chunks(retrieved)

    # Build numbered context block
    context_lines = []
    for i, (chunk, score) in enumerate(selected, 1):
        source_label = "Election Data" if chunk.source == "csv" else "Budget Document"
        context_lines.append(
            f"[P{i}] ({source_label} | score: {score:.4f})\n{chunk.text}"
        )

    if context_lines:
        context_block = "\n\n".join(context_lines)
    else:
        context_block = "No relevant context found."

    user_message = f"""Context passages:
{context_block}

Question: {query}

Answer:"""

    return SYSTEM_PROMPT, user_message, selected