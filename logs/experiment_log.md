# Experiment Log
# Author: Frank Afelete Kofi Dogli | Index: 10022200146

---

# Experiment 1 — Chunking Strategy(Part A)

- CSV loaded 615 rows, each became one chunk (615 total)
- DF first produced 2822 chunks but many were garbled (scanned table pages with characters like o f S ta ff d G o v e rn a n c e)
- After adding the _is_readable filter, PDF chunks dropped to 1294 — only clean readable text kept
- Total clean chunks: 1909
- Chose chunk size 400 characters with 80 character overlap so sentences don't get cut at boundaries

--- 

## Experiment 2 — Retrieval Failure Case (Part B)

Query tested: "Who won the 2020 Ghana presidential election?"

Vector-only top result: Ivor Kobina Greenstreet (CPP) with score 0.49 — a minor candidate, wrong answer

Hybrid top results: Still showed Greenstreet first but John Mahama (NDC) appeared at position 3 with score 0.016

Why it failed: The CSV has many rows per candidate across different regions. All 2020 rows are semantically similar so the vector search can't distinguish main candidates from minor ones based on meaning alone.

Fix applied: Added _is_readable to remove garbled PDF chunks. The keyword component of hybrid search also helped surface Mahama because his name appears more frequently across chunks.

---

## Experiment 3 — Prompt Iterations (Part C)

v1: No rules given to LLM — it invented answers

v2: Added "only answer from context" rule — hallucination stopped

v3: Added numbered passages [P1][P2] and source labels — answers became traceable

---

### Experiment 4 — Chunk size impact (Part A/D)

Chunk size 400 chars: LLM couldn't answer — chunks too short, only got section headers

Chunk size 800 chars: LLM answered correctly — chunks now contain full policy sentences

Lesson: Larger chunks work better for dense policy documents like the budget

---

# Adversarial Query 1 — "What happened in the election?" (Ambiguous)

Both RAG versions attempted an answer but returned regional data for minor candidates
Neither could determine a winner — correct behaviour, the query is too vague
Hybrid RAG gave more detail than vector-only
Key finding: RAG handles ambiguity by returning what it finds but admits it can't determine an overall winner — better than a pure LLM which would confidently make something up

# Adversarial Query 2 — "Ghana GDP in 2030 and who is president?" (Misleading/Future)

Both versions correctly refused to answer
Hybrid RAG gave a better explanation — it told you exactly why it couldn't answer (future year, not in context)
Vector-only just said "no passage mentions it"
Key finding: Hallucination guard worked perfectly — LLM did not invent a GDP figure or a president's name

