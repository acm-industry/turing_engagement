# Week 3 Evaluation: Agentic RAG for Maintenance Classification

### 1. When the LLM decides not to retrieve — was that the right call?
The agent chose to retrieve for **100% of the calls**. This was the correct decision because the system was tasked with mapping transcripts to a specific database of 21 technical codes, which seemed to be just right for the 10 transcripts that were given. 


### 2. Did the document grading step ever reject a relevant code or keep an irrelevant one?
Not for the 10-transcript test. However, I did a "stress test" by inserting a very vague and confusing transcript, and the document grading system did step in. The `rewrite_question` node was triggered and a more targeted search was attempted.


### 3. Compare your accuracy this week (extraction + RAG + classification) to last week (extraction only). What changed?
* **Last Week (Extraction Only):** Accuracy was high (~100%) but relied entirely on knowledge where the LLM had to guess the problem category based on its training data and prompt instructions.
* **This Week (Extraction + RAG):** Accuracy remained at **100%** for the test set, but the system's **robustness and scalability** improved significantly.

**What Changed?**
The fundamental change is the shift from **Instruction-Based** reasoning to **Retrieval-Based** reasoning. The system now makes "smarter" decisions because of the reflection grading and rewriting.