# Week 2 Written Assignment

---

## Pipeline Decomposition

To process the CBRE maintenance call transcript from arrival to final routing, this system follows a 5-step agentic pipeline:

1. **Entity & Severity Extraction (LLM)**
   * **Action:** The system ingests the unstructured transcript and extracts key structured entities (problem type, location, caller role, urgency indicators, etc.) and assigns a baseline severity level.
   * **Why:** Requires language understanding to parse context and tone from free text.

2. **Severity Reflection & Critique (LLM)**
   * **Action:** A second more rigorous review step where the LLM evaluates the initial extraction to check for missed compound hazards (e.g., water near a server room) or understated urgency, revising the severity score if necessary.
   * **Why:** Reflection prevents critical failures by catching nuances the first pass missed. There is no room for error if the situation is urgent, which is why we need to double check.

3. **Problem Code & Context Retrieval (Tool)**
   * **Action:** The system takes the extracted problem type and location to query a vector database for standard problem codes or past similar issues in the building.
   * **Why:** This requires querying an external database/API, not reasoning.

4. **Confidence Check & Routing Logic (Code)**
   * **Action:** The system evaluates the final severity level and the LLM's confidence score against predefined thresholds. If the severity is "Critical" or the confidence is below a certain threshold, it flags the ticket for human review; otherwise, it assigns it to the appropriate maintenance queue.
   * **Why:** This is simple, deterministic if/else logic that does not require an LLM.

5. **Ticket Creation & Notification (Tool)**
   * **Action:** The system makes an API call to the building management software to officially create the work order and notify the assigned vendor or emergency services.
   * **Why:** Involves executing an external action via an API.

---

## Human-in-the-Loop

Based on my proposed pipeline, two critical points where a human-in-the-loop (HITL) must review the system's output before an action is executed are as follows. 

1. **Before Triggering Emergency / Life-Safety Dispatch**
   * **Where:** Between Step 4 (Routing Logic) and Step 5 (Ticket Creation), specifically for tickets graded as "Critical" or "High".
   * **Why:** If the system decides a situation is a life-safety hazard (like a gas leak or a or something similar), a human dispatcher must verify it before 911 or emergency vendors are automatically deployed. If the AI acted alone, a false positive could cause unnecessary panic and waste emergency resources, while a misunderstood false negative could delay life-saving help.

2. **Low Confidence / Ambiguous Edge Cases**
   * **Where:** During Step 4 (Routing Logic), if the LLM's confidence score falls below a designated threshold (e.g., < 80%).
   * **Why:** Transcripts will inevitably include contradictory statements or multiple overlapping problems that the AI struggles to parse cleanly. Problems may also be very vague. If the AI acted alone on a low-confidence ticket, it might guess the wrong problem type, resulting in a a plumber being sent to fix an electrical issue, for instance. A human should intercept these edge cases to manually confirm the problem type and route them.

---

## Tricky Transcript

**Transcript:**
"Hey, this is Yash from apartment 402. I'mjust putting in a quick ticket for Suite 402. The coffee machine in the breakroom is acting up again and making this weird hissing noise, plus it smells kinda like rotten eggs in here, which is super annoying because we have a huge client pitch in ten minutes. Also, I think the machine might be leaking a little bit of water near the wall outlet, but no big deal, just wanted to get it on the schedule for whenever someone is free today."

**Why this is hard for an AI to classify:**
* **Misdirection & Tone:** The caller dismisses the issue as "super annoying" and "no big deal," explicitly requesting service "whenever someone is free." A naive AI will read this tone and classify it as `Low` severity (appliance repair).
* **Compound Life-Safety Hazards:** The "rotten egg" smell and hissing indicate a potential natural gas leak. Furthermore, water leaking near a wall outlet is an immediate electrical and fire hazard. 
* **The Trap:** The AI has to ignore the caller's casual assessment and explicitly extract the latent hazards to correctly upgrade this to `Critical`.


