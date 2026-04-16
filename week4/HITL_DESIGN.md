# HITL Design Document: CBRE Maintenance Call Classification Pipeline

## Where Each HITL Type Lives

My pipeline processes maintenance calls through six stages: Extract, Retrieve, Grade/Rewrite (the agentic loop from Week 3), Classify, Review & Route, and Log Result. I placed the three HITL patterns at specific points in this flow based on where human judgment adds the most value.

The **Validator Gate** sits in the `review_and_route` node, positioned between classification and logging. I chose this location because it lets the AI finish its work first, but still gives a human the chance to approve or reject the result before anything gets routed to a maintenance team. In practice, the node calls LangGraph's `interrupt()` function to freeze the pipeline state, and only resumes when a human submits their decision through `Command(resume=...)`.

The **Override Mechanism** works through two separate pathways. The first is built into the interrupt flow itself: when the validator gate pauses execution, the reviewer can supply an `override_code` and `override_reason`, which replaces the AI's classification while keeping the original prediction stored in state for auditing purposes. The second pathway uses `pipeline.update_state()` for situations where a supervisor needs to correct a call that already finished processing and was auto-routed. This modifies the checkpoint directly without re-running the pipeline.

The **Trainer Log** lives in the `log_result` node, which is the last step before the graph terminates. Every single decision flows through this node, whether the call was auto-routed, approved by a human, or overridden. Each entry captures the timestamp, the final selected code, the model's confidence score, its reasoning, and a flag indicating whether the original prediction was changed. Over time, this data accumulates into a feedback dataset that can inform prompt adjustments or model fine-tuning.

## What Triggers the Validator

The validator gate activates under two conditions: when the extracted severity is Critical, or when the classification confidence falls below 0.7. I picked these triggers because they cover the two most important failure modes. Critical severity captures life-safety situations like gas leaks, fires, and trapped occupants, where routing to the wrong team could have serious consequences. The confidence threshold catches ambiguous transcripts where the retrieval step returned problem codes that did not clearly match the caller's issue.

One important detail is that both conditions are evaluated after the agentic retrieval loop has already had a chance to improve the search results. The pipeline grades the initial retrieval, rewrites the query if needed, and re-retrieves up to two times before classifying. So by the time the validator checks these conditions, the system has already done its best to find relevant codes. This means the gate only fires on genuinely uncertain or high-stakes calls, not on cases where a simple query rewrite would have resolved the ambiguity.

## What the Human Sees

When the pipeline pauses at the validator gate, the console displays a structured review screen with all the context a dispatcher would need to make a fast decision. This includes the extracted severity level, the type of maintenance problem, the building name and specific location within the building, a one-sentence summary of the issue, the AI's selected problem code, the confidence score, the model's reasoning for its choice, and the full list of candidate codes that were retrieved from the vector store.

I designed this display to be actionable within about 30 seconds per call. The reviewer sees the AI's best guess alongside the alternatives, so they can either confirm the classification with a simple approval or select a different code and provide a brief explanation for the correction. That explanation gets stored in the trainer log alongside the original prediction, creating a paired training signal for future improvements.

## Scaling to 10,000 Calls per Day

At a volume of 10,000 calls per day, I would expect the validator gate to trigger on roughly 5 to 15 percent of incoming calls, based on the proportion that are either Critical severity or produce low confidence scores. That translates to somewhere between 500 and 1,500 calls requiring human review each day, with the remaining 85 to 95 percent flowing through automatically.

To handle this volume, the architecture would need several changes from the current prototype. First, the `InMemorySaver` checkpointer would need to be replaced with a persistent store like `PostgresSaver`, so that paused pipeline states survive process restarts and can be distributed across multiple worker instances. Second, the interactive review flow would move from inline console prompts to an asynchronous queue backed by a web interface similar to the Streamlit app included in this project. Critical-severity calls would sit at the top of the queue, followed by low-confidence calls, so that dispatchers handle the most urgent items first.

Third, the confidence threshold itself should be treated as a tunable parameter rather than a fixed constant. As the trainer log grows, we can analyze override rates by category and adjust thresholds accordingly: tightening the gate for problem types that get corrected frequently, and loosening it for categories where the model consistently gets it right. This optimizes the human review budget by focusing attention where it matters most.

Finally, the trainer log enables a weekly batch analysis workflow. By examining which codes get overridden most often and what patterns appear in the corrected examples, the team can make targeted updates to extraction prompts, retrieval keywords, or classification instructions. Once the log contains 500 or more override entries for a given category, there is enough signal to consider fine-tuning the underlying model on human-corrected examples, further reducing the fraction of calls that need manual review over time.
