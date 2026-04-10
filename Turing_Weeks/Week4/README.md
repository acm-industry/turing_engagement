# Week 4: Human-in-the-Loop Review Interface

## Running the app

From the project root:

```bash
streamlit run weeks/week4/hitl_app.py
```

Make sure you have a `.env` file with your `OPENAI_API_KEY` set.

## What this demonstrates

1. The full pipeline from Week 2 + 3 (extraction → RAG → classification) now also includes a **routing decision** node
2. A **HITL check** flags calls for human review when:
   - Severity = Critical (always needs human eyes)
   - Confidence < 0.80 (the AI isn't sure enough)
3. The Streamlit UI lets a "reviewer" see the full analysis and either **approve** or **override** the routing

## Things to discuss with students

- Why is the confidence threshold set at 0.80? What happens if you raise/lower it?
- What percentage of calls get flagged for review? Is that sustainable for a real team?
- How would you store the human overrides to improve the system over time?
