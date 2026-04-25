# Experiment Log
# Name: Frank Afelete Kofi Dogli | Index: 10022200149
# Course: CS4241 — Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso | Year: 2026
# Note: All experiments were conducted manually and results recorded personally.

---

## Experiment 1 — Data Loading & Cleaning (Part A)
Date: 25th April 2026

### What I did:
Loaded both datasets using pdfplumber for the PDF and pandas for the CSV.
Tested the data loader to see how many rows and pages were extracted.

### CSV Results:
- Total rows loaded: 615
- Columns after normalisation: year, old_region, new_region, code, candidate, party, votes, votes(%)
- Each row was converted into a readable sentence for embedding
- Example: "Year: 2020 | Old Region: Brong Ahafo Region | New Region: Ahafo Region | Code: NPP | Candidate: Nana Akufo Addo | Party: NPP | Votes: 145584 | Votes(%): 55.04%"

### PDF Results:
- Total usable pages extracted: 251
- Blank and image-only pages were skipped (less than 50 characters)
- PDF URL v4 returned 404 from government server — used v5 as fallback (same document, re-uploaded)

### Observations:
- CSV data was clean and well-structured
- PDF had some pages with garbled text from scanned tables

---

## Experiment 2 — Chunking Strategy Comparison (Part A)
Date: 25th April 2026

### What I did:
Tested different chunk sizes for the PDF to see impact on retrieval quality.

### CSV Chunking:
- Strategy: One chunk per row
- Result: 615 chunks
- Justification: Each row is a self-contained fact. Splitting would break the relationship between candidate, party, and votes.

### PDF Chunking — Test 1 (chunk size 400, overlap 80):
- Result: 2822 chunks before cleaning, 1294 after removing garbled chunks
- Retrieval quality: Poor — chunks were too short and only returned section headers
- LLM could not answer budget questions from these chunks

### PDF Chunking — Test 2 (chunk size 800, overlap 150):
- Result: 558 chunks
- Retrieval quality: Improved — chunks contained fuller sentences
- LLM started returning partial answers

### PDF Chunking — Test 3 (chunk size 1200, overlap 200) — FINAL:
- Result: 350 chunks
- Retrieval quality: Best — chunks contained complete policy arguments
- LLM answered budget questions correctly with citations
- Total chunks: 965 (615 CSV + 350 PDF)

### Conclusion:
Larger chunk size worked better for the budget PDF because policy statements span multiple sentences. The CSV benefited from row-per-chunk because each row is already a complete data point.

---

## Experiment 3 — Garbled Chunk Filtering (Part B)
Date: 25th April 2026

### Problem identified:
PDF chunks contained garbled text from scanned table pages:
Example: "o f S ta ff d G o v e rn a n c e m u n ic a tio n s A u th o rity"
These were appearing in retrieval results for election queries.

### Fix applied:
Added _is_readable() function in chunker.py that checks the ratio of real words to total tokens. Chunks where less than 60% of tokens are readable alphabetic words are discarded.

### Result:
- PDF chunks before filter: 2822
- PDF chunks after filter: 1294
- Garbled chunks successfully removed
- Retrieval results no longer contained scanned table noise

---

## Experiment 4 — Retrieval Failure Case (Part B)
Date: 25th April 2026

### Query tested:
"Who won the 2020 Ghana presidential election?"

### Vector-only retrieval results:
- Top result: Ivor Kobina Greenstreet (CPP) — score: 0.4941
- Second result: John Dramani Mahama (NDC) — score: 0.4905
- Third result: John Dramani Mahama (NDC) — score: 0.4899

### Why it failed:
The CSV has 615 rows covering many regions and candidates. All 2020 rows are semantically very similar to each other. Vector search could not distinguish main candidates from minor ones because the embedding model treats all 2020 election rows as nearly identical in meaning.

### Hybrid retrieval results:
- Top result: Ivor Kobina Greenstreet (CPP) — score: 0.01639
- Third result: John Dramani Mahama (NDC) — score: 0.01613

### Analysis:
Hybrid search improved ranking slightly by using keyword scoring alongside vector search. The fundamental limitation is that the dataset does not contain a national totals row — only regional breakdowns. This means the system cannot definitively answer who won nationally from the data alone.

### Fix documented:
The hallucination guard in the prompt prevents the LLM from inventing a winner. It correctly responds that it cannot determine an overall winner from regional data alone.

---

## Experiment 5 — Prompt Iteration (Part C)
Date: 25th April 2026

### Query used for all tests:
"What is Ghana's GDP in 2030 and who is the president?"

### Prompt v1 — No hallucination guard:
System prompt: "You are a helpful assistant. Answer the question using the context."
Result: LLM invented a GDP figure and named a president not in the context.
Problem: Hallucination — model filled in gaps with fabricated information.

### Prompt v2 — Added hallucination guard:
Added rule: "Only answer from the context. If not found, say you don't have enough information."
Result: "I don't have enough information in my knowledge base to answer that accurately."
Improvement: Hallucination eliminated completely.

### Prompt v3 — Added numbered passages and source labels (FINAL):
Added [P1], [P2] labels with source names (Election Data / Budget Document).
Result: LLM now cites which passage it used. Answers are traceable and verifiable.
Example answer: "The 2025 budget focuses on resetting the economy [P1] and cutting wasteful spending [P2]."

### Conclusion:
Each iteration improved answer quality and trustworthiness. v3 is the best because it allows the user to verify which passage supported the answer.

---

## Experiment 6 — Adversarial Testing (Part E)
Date: 25th April 2026

### Adversarial Query 1 — Ambiguous:
Query: "What happened in the election?"

Hybrid RAG response: Returned regional data for Edward Mahama (PNC) in 1996 and 2004 elections. Could not determine an overall winner. Acknowledged the ambiguity.

Vector-only response: Similar results — returned regional vote counts but no clear winner identified.

Analysis: Both systems handled the ambiguity correctly by returning whatever data matched and admitting they could not determine a winner. The query was too vague — no year or region specified. A pure LLM without retrieval would likely have confidently named a winner without evidence.

### Adversarial Query 2 — Misleading (future fact):
Query: "What is Ghana's GDP in 2030 and who is the president?"

Hybrid RAG response: "I don't have enough information in my knowledge base to answer that accurately. The provided context passages do not mention Ghana's GDP in 2030 or the current president."

Vector-only response: "I don't have enough information in my knowledge base to answer that accurately. No passage mentions Ghana's GDP in 2030 or the president's name."

Analysis: Both systems correctly refused to answer. The hallucination guard worked as intended. Neither system invented a figure or a name. Hybrid RAG gave a slightly more detailed explanation of why it could not answer.

### RAG vs Pure LLM Comparison:

| Metric | RAG System | Pure LLM (no retrieval) |
|--------|-----------|------------------------|
| Accuracy on budget facts | High — grounded in PDF | Medium — stale training data |
| Accuracy on election facts | Medium — limited by regional data | Low — may confuse years |
| Hallucination rate | Low — prompt guard works | High — fills gaps freely |
| Source traceability | Yes — shows [P1][P2] citations | No |
| Handles unknown queries | Says "I don't know" | Often fabricates answer |
| Response consistency | Consistent — same context | Variable across runs |

---

## Experiment 7 — Memory-based RAG (Part G)
Date: 25th April 2026

### What I tested:
Asked a follow-up question without repeating the topic to see if memory worked.

Turn 1 query: "What does the 2025 budget say about infrastructure?"
Turn 1 answer: Returned information about railway construction, Tema-Mpakadan line, Western Railway modernisation.

Turn 2 query: "What about education?"
Turn 2 answer: Returned information about Free SHS, GETFund, sanitary pads distribution — correctly understood "what about education" referred to the 2025 budget.

### Conclusion:
Memory-based RAG successfully maintained conversation context across turns. The system injected the last 3 conversation turns into the LLM call, allowing it to understand follow-up questions without the user repeating the topic each time. This significantly improves user experience for multi-turn conversations.
