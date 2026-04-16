import json
import time
import warnings
import os
from typing import Optional
from enum import Enum
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

# Suppress harmless Pydantic serialization warnings from LangChain's structured output
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Load OPENAI_API_KEY from .env file
load_dotenv()

# --- Part 1: Setup the Data & Vector Store ---
with open("problem_codes.json") as f:
    problem_codes = json.load(f)

documents = []
for pc in problem_codes:
    text = f"{pc['code']}: {pc['category']} — {pc['subcategory']}\n{pc['description']}\nKeywords: {', '.join(pc['keywords'])}"
    doc = Document(page_content=text, metadata={"code": pc["code"], "category": pc["category"]})
    documents.append(doc)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings, collection_name="problem_codes")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Part 2: Setup Models & State ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class MaintenanceEntity(BaseModel):
    problem_type: str = Field(description="Brief description of the maintenance problem")
    location_building: str = Field(description="Name of the building")
    location_detail: Optional[str] = Field(description="Specific location within the building")
    severity: Severity = Field(description="Critical / High / Medium / Low")
    caller_role: Optional[str] = Field(description="Role of the caller")
    urgency_indicators: list[str] = Field(description="Phrases indicating urgency")
    summary: str = Field(description="One-sentence summary")

class PipelineState(TypedDict):
    transcript: str
    entities: Optional[dict]
    current_query: Optional[str]
    retrieved_codes: Optional[list[dict]]
    classification: Optional[dict]
    retries: int
    routing_decision: Optional[str]
    trainer_log: Optional[list[dict]]

# --- Part 3: Nodes ---
EXTRACTION_PROMPT = """
You are an expert building maintenance call analyst. Extract key information from this transcript.

Severity guidelines:
- Critical: Immediate danger to life/safety (gas leak, fire, trapped persons, flooding)
- High: Significant disruption or escalation risk (major leak, broken security glass, HVAC failure)
- Medium: Needs attention soon, not emergency (broken door, minor plumbing, elevator malfunction)
- Low: Minor/cosmetic (flickering light, carpet stain, empty soap dispenser)

Transcript:
{transcript}
"""

def extract_entities(state: PipelineState) -> dict:
    extractor = llm.with_structured_output(MaintenanceEntity)
    result = extractor.invoke(EXTRACTION_PROMPT.format(transcript=state["transcript"]))
    entities_dict = result.model_dump()
    query = f"{entities_dict['problem_type']} {entities_dict['summary']}"
    return {"entities": entities_dict, "current_query": query, "retries": 0}

def retrieve_codes(state: PipelineState) -> dict:
    docs = retriever.invoke(state["current_query"])
    codes = [{"code": d.metadata["code"], "content": d.page_content} for d in docs]
    return {"retrieved_codes": codes}

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: PipelineState) -> str:
    grader_model = llm.with_structured_output(GradeDocuments)
    
    query = state["current_query"]
    candidates = "\n\n".join(c["content"] for c in state["retrieved_codes"])
    
    prompt = (
        "You are a grader assessing relevance of retrieved problem codes to a maintenance issue.\n"
        f"Issue Description / Query:\n{query}\n\n"
        f"Retrieved Problem Codes:\n{candidates}\n\n"
        "If the problem codes contain a relevant match or clearly related category for the issue, grade it as 'yes'.\n"
        "If none of them match the issue well, grade it as 'no'."
    )
    
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    
    # If the score is yes, or we've retried too many times, proceed to classification
    if response.binary_score.lower() == "yes" or state.get("retries", 0) >= 2:
        return "classify_problem"
    else:
        return "rewrite_query"

def rewrite_query(state: PipelineState) -> dict:
    prompt = (
        "You are a query rewriting assistant for a maintenance problem classification system.\n"
        "The previous query did not return any closely related problem codes.\n"
        "Here is the transcript of the maintenance call:\n"
        "-------\n"
        f"{state['transcript']}\n"
        "-------\n"
        "Here is the previously extracted problem summary:\n"
        f"{state['entities'].get('summary', '')}\n\n"
        "Here is the previous query that failed:\n"
        f"{state['current_query']}\n\n"
        "Please formulate a new, improved, and distinct search query to find the correct problem code. "
        "Focus on core issue keywords, systems, or broader categories rather than highly specific details. "
        "Return just the rewritten query text."
    )
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {
        "current_query": response.content.strip(), 
        "retries": state.get("retries", 0) + 1
    }

class Classification(BaseModel):
    selected_code: str = Field(description="The problem code that best matches (e.g., PLUMB-001)")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of why this code was selected")

CLASSIFICATION_PROMPT = """
You are classifying a building maintenance issue. Based on the extracted 
information and the candidate problem codes retrieved from our database, 
select the best matching code.

Extracted information:
- Problem: {problem_type}
- Severity: {severity}
- Summary: {summary}

Candidate problem codes:
{candidates}

Respond with your classification.
"""

def classify_problem(state: PipelineState) -> dict:
    entities = state["entities"]
    codes = state["retrieved_codes"]
    candidates = "\n\n".join(c["content"] for c in codes)

    classifier = llm.with_structured_output(Classification)
    result = classifier.invoke(CLASSIFICATION_PROMPT.format(
        problem_type=entities["problem_type"],
        severity=entities["severity"],
        summary=entities["summary"],
        candidates=candidates,
    ))
    return {"classification": result.model_dump()}

# --- Part 4: HITL Nodes (Validator Gate + Trainer Log) ---

# Global trainer log — in production this would be a database.
trainer_log = []

CONFIDENCE_THRESHOLD = 0.7

def review_and_route(state: PipelineState) -> dict:
    """Node: Validator gate — conditionally pauses for human review.

    If severity is Critical OR confidence < 0.7:
        → interrupt() pauses the pipeline
        → human reviews and either approves or provides an override
        → pipeline resumes with the human's decision

    Otherwise:
        → auto-route (no human needed)
    """
    classification = state["classification"]
    entities = state["entities"]

    severity = entities.get("severity", "Medium")
    # Handle "Critical", "Severity.CRITICAL", and Severity.CRITICAL enum formats from Pydantic
    # Use .lower() for case-insensitive comparison since split on enum gives "CRITICAL"
    severity_str = str(severity).split(".")[-1].strip()
    confidence = classification.get("confidence", 0.5)

    needs_review = severity_str.lower() == "critical" or confidence < CONFIDENCE_THRESHOLD

    if needs_review:
        reason = []
        if severity_str.lower() == "critical":
            reason.append("severity is Critical")
        if confidence < CONFIDENCE_THRESHOLD:
            reason.append(f"confidence is low ({confidence:.2f})")

        print(f"\n{'='*60}")
        print(f"  HUMAN REVIEW REQUIRED — {' and '.join(reason)}")
        print(f"  Code:       {classification['selected_code']}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Severity:   {severity_str}")
        print(f"  Reasoning:  {classification['reasoning']}")
        print(f"{'='*60}")

        # interrupt() PAUSES the pipeline here.
        # It returns whatever value the human passes via Command(resume=...).
        decision = interrupt({
            "type": "review_required",
            "reason": " and ".join(reason),
            "severity": severity_str,
            "classification": classification,
        })

        # --- Handle the human's decision ---
        if decision.get("override_code"):
            # OVERRIDE: human changed the classification
            new_classification = {
                **classification,
                "selected_code": decision["override_code"],
                "reasoning": f"Human override: {decision.get('override_reason', 'N/A')}",
                "original_code": classification["selected_code"],
                "human_reviewed": True,
            }
            return {
                "classification": new_classification,
                "routing_decision": f"Human override → {decision['override_code']}",
            }
        else:
            # APPROVE: human confirmed the AI's decision
            return {
                "routing_decision": f"Human approved → {classification['selected_code']}",
            }

    # No review needed — auto-route
    return {
        "routing_decision": f"Auto-routed → {classification['selected_code']} (confidence {confidence:.2f})",
    }


def log_result(state: PipelineState) -> dict:
    """Node: Trainer pattern — log the final decision for future improvement.

    Captures the original AI prediction alongside any human corrections.
    Over time, these logs reveal patterns that inform prompt updates or fine-tuning.
    """
    classification = state["classification"]
    routing = state.get("routing_decision", "unknown")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "selected_code": classification["selected_code"],
        "confidence": classification.get("confidence"),
        "reasoning": classification.get("reasoning"),
        "routing_decision": routing,
        "was_overridden": classification.get("original_code") is not None,
        "original_code": classification.get("original_code"),
    }
    trainer_log.append(entry)

    action = "OVERRIDE" if entry["was_overridden"] else "LOGGED"
    print(f"  [TRAINER LOG] {action}: {entry['selected_code']} (conf: {entry['confidence']:.2f})")
    if entry["was_overridden"]:
        print(f"               Original: {entry['original_code']} → Corrected: {entry['selected_code']}")

    return {"trainer_log": trainer_log.copy()}


# --- Part 5: Build the Graph ---
workflow = StateGraph(PipelineState)

workflow.add_node("extract_entities", extract_entities)
workflow.add_node("retrieve_codes", retrieve_codes)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("classify_problem", classify_problem)
workflow.add_node("review_and_route", review_and_route)
workflow.add_node("log_result", log_result)

workflow.add_edge(START, "extract_entities")
workflow.add_edge("extract_entities", "retrieve_codes")
workflow.add_conditional_edges(
    "retrieve_codes",
    grade_documents,
)
workflow.add_edge("rewrite_query", "retrieve_codes")
workflow.add_edge("classify_problem", "review_and_route")
workflow.add_edge("review_and_route", "log_result")
workflow.add_edge("log_result", END)

# InMemorySaver stores state at every node — required for interrupt() to work.
# In production, use a persistent checkpointer (e.g., SqliteSaver, PostgresSaver).
memory = InMemorySaver()
pipeline = workflow.compile(checkpointer=memory)

# --- Part 6: Execute Evaluation (Optional/Verification) ---
if __name__ == "__main__":
    def classify_one(t):
        config = {"configurable": {"thread_id": f"eval-{t['id']}"}}
        output = pipeline.invoke({"transcript": t["transcript"], "retries": 0}, config)

        # If interrupted (Critical severity or low confidence), auto-approve for eval
        state = pipeline.get_state(config)
        if state.next:  # Pipeline is paused at an interrupt
            output = pipeline.invoke(Command(resume={"approved": True}), config)

        predicted = output["classification"]["selected_code"]
        actual = t["true_category"]
        match = predicted == actual
        return {
            "id": t["id"],
            "predicted": predicted,
            "actual": actual,
            "confidence": output["classification"]["confidence"],
            "match": match,
            "retries_used": output.get("retries", 0),
            "routing": output.get("routing_decision", "unknown"),
        }

    # Load test transcripts
    try:
        with open("transcripts.json") as f:
            transcripts = json.load(f)

        print("Starting HITL-enabled pipeline evaluation across all transcripts...")
        print(f"Validator triggers: severity=Critical OR confidence < {CONFIDENCE_THRESHOLD}")
        print(f"(Interrupted calls are auto-approved for evaluation purposes)\n")
        start = time.time()
        all_results = []

        # Sequential processing — checkpointer requires unique thread_id per run
        for t in transcripts:
            r = classify_one(t)
            status = "CORRECT" if r["match"] else "WRONG"
            print(f"  {r['id']}: predicted={r['predicted']:10s} actual={r['actual']:10s} "
                  f"conf={r['confidence']:.2f} retries={r['retries_used']} "
                  f"routing={r['routing'][:40]:40s} {status}")
            all_results.append(r)

        elapsed = time.time() - start
        correct = sum(1 for r in all_results if r["match"])
        print(f"\nClassification accuracy: {correct}/{len(all_results)} ({100*correct/len(all_results):.0f}%)")
        avg_conf = sum(r["confidence"] for r in all_results) / len(all_results)
        print(f"Average confidence: {avg_conf:.2f}")
        total_retries = sum(r["retries_used"] for r in all_results)
        print(f"Total query rewrites performed: {total_retries}")
        print(f"Wall-clock time: {elapsed:.1f}s")

        # --- Trainer Log Summary ---
        print(f"\n{'='*60}")
        print(f"  TRAINER LOG — {len(trainer_log)} entries")
        print(f"{'='*60}")
        print(f"{'#':<4} {'Code':<12} {'Conf':<6} {'Overridden?':<12} {'Original':<12} {'Routing'}")
        print("-" * 80)
        for i, entry in enumerate(trainer_log):
            override_str = "YES" if entry["was_overridden"] else "no"
            original = entry.get("original_code") or "—"
            routing_short = entry["routing_decision"][:40] if entry["routing_decision"] else "unknown"
            print(f"{i+1:<4} {entry['selected_code']:<12} {entry['confidence']:.2f}  "
                  f"{override_str:<12} {original:<12} {routing_short}")

        auto_routed = sum(1 for e in trainer_log if "Auto-routed" in e.get("routing_decision", ""))
        human_reviewed = sum(1 for e in trainer_log if "Human" in e.get("routing_decision", ""))
        overrides = sum(1 for e in trainer_log if e["was_overridden"])
        print(f"\n  Auto-routed:          {auto_routed}")
        print(f"  Human-reviewed:       {human_reviewed} (auto-approved in eval)")
        print(f"  Overrides:            {overrides}")

        # --- Part 7: Post-Execution Override Demo (update_state) ---
        print(f"\n{'='*60}")
        print("  POST-EXECUTION OVERRIDE DEMO — update_state()")
        print(f"{'='*60}")

        # Pick the first auto-routed transcript for the demo
        demo_target = None
        for r in all_results:
            if "Auto-routed" in r["routing"]:
                demo_target = r
                break

        if demo_target:
            demo_config = {"configurable": {"thread_id": f"eval-{demo_target['id']}"}}
            state_before = pipeline.get_state(demo_config)
            before_code = state_before.values["classification"]["selected_code"]
            before_routing = state_before.values.get("routing_decision", "N/A")

            print(f"\n  Target transcript:  {demo_target['id']}")
            print(f"  Current code:       {before_code}")
            print(f"  Current routing:    {before_routing}")

            # Simulate a supervisor overriding the classification after the fact
            override_code = "SAFE-002"  # Arbitrary override for demo
            pipeline.update_state(
                demo_config,
                {
                    "classification": {
                        **state_before.values["classification"],
                        "selected_code": override_code,
                        "reasoning": "Supervisor override: reclassified during post-shift review",
                        "original_code": before_code,
                    },
                    "routing_decision": f"Post-execution override → {override_code}",
                },
            )

            state_after = pipeline.get_state(demo_config)
            after_code = state_after.values["classification"]["selected_code"]
            print(f"\n  BEFORE: {before_code}")
            print(f"  AFTER:  {after_code}")
            print(f"  Checkpoint modified via update_state().")
            print(f"  In production, this triggers a re-routing notification.")
        else:
            print("\n  No auto-routed transcript found for demo.")

    except FileNotFoundError:
        print("transcripts.json not found, skipping evaluation loop.")
