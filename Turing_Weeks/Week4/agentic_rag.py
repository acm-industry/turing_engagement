import json
import os
import time
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

warnings.filterwarnings("ignore", message="Deserializing unregistered type")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# LLM, doc loading, chunking, embedding

def init_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4.1", temperature=0, api_key=OPENAI_API_KEY)


def load_docs() -> list[Document]:
    loader = JSONLoader(
        file_path="problem_codes.json",
        jq_schema=".[]",
        text_content=False,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    return docs


def chunking(docs) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    return text_splitter.split_documents(docs)


def embed_and_store(chunks: list[Document]) -> InMemoryVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def make_retriever_tool(vector_store: InMemoryVectorStore):
    @tool
    def retrieve_problem_codes(query: str) -> str:
        """Search and return information about problem codes,
        including their descriptions, categories, and resolutions."""
        docs = vector_store.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_problem_codes



#Structured extraction models
class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class MaintenanceEntity(BaseModel):
    """Structured extraction from a problem description."""
    problem_type: str = Field(description="Specific description of the problem (e.g., 'VPN login error', 'printer offline', 'database connection timeout') — not vague terms like 'issue' or 'problem'")
    affected_system: Optional[str] = Field(description="The exact device, software name, or system affected (e.g., 'Cisco AnyConnect', 'HP LaserJet 400', 'PostgreSQL 14') — not just 'computer' or 'software'")
    location_detail: Optional[str] = Field(description="Full location including building name AND city (e.g., 'Tower B, Austin TX') — a floor or room number alone is not sufficient")
    severity: Severity = Field(description="Critical (blocking all work / data loss risk), High (significant disruption), Medium (needs attention soon), Low (minor / cosmetic)")
    urgency_indicators: list[str] = Field(description="Specific phrases from the description that indicate urgency or timeline (e.g., 'been down since this morning', 'client demo in 1 hour')")
    summary: str = Field(description="One-sentence summary of the issue")


class EntityQualityCheck(BaseModel):
    """Validate that the problem description is specific enough to drive accurate retrieval."""
    problem_type_ok: bool = Field(description="True only if problem_type describes a specific symptom or failure (e.g. 'water leaking from ceiling pipe', 'elevator stuck between floors') — not vague words like 'issue', 'problem', or 'something wrong'")
    problem_type_followup: str = Field(description="Question to get a more specific problem description. Empty string if problem_type_ok is True.")


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")
    confidence: float = Field(description="Confidence in this relevance decision, from 0.0 to 1.0. Use < 0.7 for borderline cases where the document is partially relevant.")


class Classification(BaseModel):
    """Final problem code classification from retrieved context."""
    selected_code: str = Field(description="The problem code that best matches (e.g., PLUMB-001, HVAC-001)")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of why this code was selected over the others")



# Prompts
ENTITY_EXTRACTION_PROMPT = (
    "You are a building maintenance analyst. A caller has described a problem.\n"
    "Extract the key information from their description.\n"
    "If a field cannot be determined from the description, leave it as null.\n\n"
    "Severity guidelines:\n"
    "- Critical: Immediate danger to life/safety (gas leak, fire, trapped persons, flooding near electrical/IT equipment)\n"
    "- High: Significant disruption or escalation risk (major water leak, broken security, HVAC failure, power outage)\n"
    "- Medium: Needs attention soon but not an emergency (broken door latch, minor plumbing, non-life-safety HVAC)\n"
    "- Low: Minor/cosmetic (flickering light, carpet stain, empty dispenser, aesthetic repairs)\n\n"
    "Key compound hazard rules:\n"
    "- Water near electrical equipment or servers → Critical\n"
    "- Any gas smell in an enclosed space → Critical\n"
    "- People trapped anywhere → Critical\n\n"
    "Caller description: {description}"
)

QUALITY_CHECK_PROMPT = (
    "You are validating whether a maintenance ticket has a specific enough problem description to search accurately.\n\n"
    "Extracted problem_type: {problem_type}\n\n"
    "Is this a specific symptom or failure (e.g. 'water leaking from ceiling pipe', 'elevator stuck between floors')? "
    "Or is it too vague (e.g. 'issue', 'problem', 'something wrong')? "
    "If vague, generate one targeted follow-up question to get a concrete description."
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n ------- \n{question}\n ------- \n"
    "Formulate an improved question:"
)

CLASSIFY_PROMPT = (
    "You are classifying a building maintenance issue to a problem code.\n"
    "Based on the extracted information and the retrieved candidate codes, "
    "select the single best matching code. Do not be overconfident.\n\n"
    "Extracted information:\n"
    "- Problem: {problem_type}\n"
    "- Severity: {severity}\n"
    "- Summary: {summary}\n\n"
    "Candidate problem codes:\n{candidates}\n\n"

    "- Output confidence > 0.85 ONLY if you are certain no other code could apply.\n"
    "- Output confidence < 0.8 if two or more codes are plausible given the description.\n"
    # "- When in doubt, lower your confidence to below 0.7 so a human reviewer can verify.\n"
)


# Extended graph state adds entity-extraction fields to MessagesState

class WorkflowState(MessagesState):
    """Full workflow state: chat messages + entity extraction tracking."""
    accumulated_description: str          # builds up as follow-ups are added
    entity: Optional[dict]                # serialised MaintenanceEntity
    follow_ups: list[str]                 # pending follow-up questions
    enriched_query: str                   # final query sent to retrieval
    extraction_complete: bool             # gate: entity extraction done?
    classification: Optional[dict]        # final Classification result
    routing_decision: Optional[str]       # outcome of the validator gate
    trainer_log: Optional[list]           # accumulated log of all decisions



# Node factories — each returns a closure over the shared LLM / tools

def make_extract_entities(llm: ChatOpenAI):
    """Node: run structured entity extraction on the accumulated description."""
    extractor = llm.with_structured_output(MaintenanceEntity)

    def extract_entities(state: WorkflowState):
        # print("\n[NODE] extract_entities")
        description = state.get("accumulated_description", "")
        if not description:
            description = state["messages"][0].content

        entity = extractor.invoke(
            ENTITY_EXTRACTION_PROMPT.format(description=description)
        )
        # print(f"  problem_type:   {entity.problem_type}")
        # print(f"  affected_system:{entity.affected_system}")
        # print(f"  location_detail:{entity.location_detail}")
        # print(f"  severity:       {entity.severity.value}")
        # print(f"  urgency:        {entity.urgency_indicators}")
        return {
            "entity": entity.model_dump(mode="json"),
            "accumulated_description": description,
        }

    return extract_entities


def make_check_quality(llm: ChatOpenAI):
    """Node: validate entity specificity, populate follow_ups list.
    Skips validation if extraction_complete is already True (e.g. transcript eval).
    """
    checker = llm.with_structured_output(EntityQualityCheck)

    def check_quality(state: WorkflowState):
        # print("\n[NODE] check_quality")
        if state.get("extraction_complete"):
            # Transcript text is already complete — skip clarification loop
            return {"follow_ups": []}

        ent = state["entity"]
        quality = checker.invoke(
            QUALITY_CHECK_PROMPT.format(problem_type=ent.get("problem_type") or "null")
        )
        follow_ups = [quality.problem_type_followup] if not quality.problem_type_ok else []
        # print(f"  problem_type_ok: {quality.problem_type_ok}")
        # if follow_ups:
        #     print(f"  follow-up: {follow_ups[0]}")
        return {
            "follow_ups": follow_ups,
            "extraction_complete": len(follow_ups) == 0,
        }

    return check_quality


def gather_followup(state: WorkflowState):
    """Node: prompt the user for the first outstanding follow-up, append to description."""
    # print("\n[NODE] gather_followup")
    question = state["follow_ups"][0]
    answer = input(f"\n{question}\n> ").strip()
    new_description = f"{state['accumulated_description']}. {answer}"
    return {
        "accumulated_description": new_description,
        "follow_ups": state["follow_ups"][1:],
    }


def build_enriched_query(state: WorkflowState):
    """Node: convert the validated entity into a rich query string and inject it as a message."""
    # print("\n[NODE] build_enriched_query")
    ent = state["entity"]
    enriched = (
        f"Problem: {ent['problem_type']}."
        + (f" System: {ent['affected_system']}." if ent.get("affected_system") else "")
        + (f" Location: {ent['location_detail']}." if ent.get("location_detail") else "")
        + f" Severity: {ent['severity']}."
        + f" {ent['summary']}"
    )
    # print(f"  {enriched}")
    return {
        "enriched_query": enriched,
        "messages": [HumanMessage(content=enriched)],
    }


def make_generate_query_or_respond(llm: ChatOpenAI, retriever_tool):
    """Node: let the LLM decide whether to call the retriever tool or answer directly."""
    def generate_query_or_respond(state: WorkflowState):
        # print("\n[NODE] generate_query_or_respond")
        response = (
            llm
            .bind_tools([retriever_tool])
            .invoke(state["messages"])
        )
        # if response.tool_calls:
        #     print(f"  decision: call retriever tool")
        # else:
        #     print(f"  decision: respond directly (no retrieval needed)")
        return {"messages": [response]}

    return generate_query_or_respond


def make_rewrite_question(llm: ChatOpenAI):
    def rewrite_question(state: WorkflowState):
        # print("\n[NODE] rewrite_question")
        question = state.get("enriched_query", state["messages"][0].content)
        prompt = REWRITE_PROMPT.format(question=question)
        response = llm.invoke([{"role": "user", "content": prompt}])
        # print(f"  rewritten: {response.content[:120]}")
        return {"messages": [HumanMessage(content=response.content)]}

    return rewrite_question


def make_grade_documents(llm: ChatOpenAI):
    grader = llm.with_structured_output(GradeDocuments)

    def grade_documents(state: WorkflowState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade retrieved docs for relevance — route to answer or rewrite."""
        # print("\n[NODE] grade_documents")
        question = state.get("enriched_query", state["messages"][0].content)
        context = state["messages"][-1].content

        response = grader.invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}]
        )
        # print(f"  relevant:   {response.binary_score}")
        # print(f"  confidence: {response.confidence:.2f}")
        # Borderline "no" (low confidence) → still try to answer rather than rewrite
        if response.binary_score == "yes" or response.confidence < 0.8:
            decision = "generate_answer"
        else:
            decision = "rewrite_question"
        # print(f"[ROUTE] retrieve → {decision}")
        return decision

    return grade_documents


def make_generate_answer(llm: ChatOpenAI):
    classifier = llm.with_structured_output(Classification)

    def generate_answer(state: WorkflowState):
        # print("\n[NODE] generate_answer")
        ent = state.get("entity") or {}
        context = state["messages"][-1].content
        prompt = CLASSIFY_PROMPT.format(
            problem_type=ent.get("problem_type", "unknown"),
            severity=ent.get("severity", "unknown"),
            summary=ent.get("summary", state.get("enriched_query", "")),
            candidates=context,
        )
        result = classifier.invoke([{"role": "user", "content": prompt}])
        # print(f"  selected_code: {result.selected_code}")
        # print(f"  confidence:    {result.confidence:.2f}")
        # print(f"  reasoning:     {result.reasoning}")
        return {
            "classification": result.model_dump(),
            "messages": [HumanMessage(content=f"[{result.selected_code}] {result.reasoning}")],
        }

    return generate_answer



def review_and_route(state: WorkflowState) -> dict:
    """Node: Validator gate — pauses for human review on Critical severity or low confidence.
    Otherwise auto-routes. Supports approve and override.
    """
    classification = state["classification"]
    entity = state.get("entity") or {}

    severity = entity.get("severity", "Medium")
    # Handle both "Critical" and "Severity.CRITICAL" formats from Pydantic enum
    severity_str = severity.split(".")[-1] if "." in str(severity) else str(severity)
    confidence = classification.get("confidence", 0.5)

    needs_review = severity_str == "Critical" or confidence < 0.7

    if needs_review:
        reason = []
        if severity_str == "Critical":
            reason.append("severity is Critical")
        if confidence < 0.7:
            reason.append(f"confidence is low ({confidence:.2f})")

        decision = interrupt({
            "type": "review_required",
            "reason": " and ".join(reason),
            "severity": severity_str,
            "classification": classification,
        })

        if decision.get("override_code"):
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
            return {
                "routing_decision": f"Human approved → {classification['selected_code']}",
            }

    return {
        "routing_decision": f"Auto-routed → {classification['selected_code']} (confidence {confidence:.2f})",
    }


#Conditional edge functions
def needs_followup(state: WorkflowState) -> Literal["gather_followup", "build_enriched_query"]:
    """Route after quality check: loop back for more info or proceed."""
    decision = "build_enriched_query" if state.get("extraction_complete") else "gather_followup"
    # print(f"\n[ROUTE] check_quality → {decision}")
    return decision



#Pipeline evaluation across Week 2 transcripts using the actual graph
def run_pipeline_evaluation(graph, transcripts_path: str):
    """
    Run all transcripts through the actual graph.invoke() in parallel.
    Compares predicted problem code AND extracted severity vs ground truth.
    Transcript runs pass extraction_complete=True to skip the clarification loop.
    """
    with open(transcripts_path) as f:
        transcripts = json.load(f)

    def evaluate_one(t):
        config = {"configurable": {"thread_id": f"eval-{t['id']}"}}
        result = graph.invoke({
            "messages": [HumanMessage(content=t["transcript"])],
            "accumulated_description": "",
            "entity": None,
            "follow_ups": [],
            "enriched_query": "",
            "extraction_complete": True,   # skip clarification loop for complete transcripts
            "classification": None,
            "routing_decision": None,
            "trainer_log": None,
        }, config)
        # Auto-approve if the validator gate interrupted (Critical severity)
        if graph.get_state(config).next:
            result = graph.invoke(Command(resume={"approved": True}), config)
        clf = result.get("classification") or {}
        ent = result.get("entity") or {}

        predicted_code = clf.get("selected_code", "UNKNOWN")
        confidence = clf.get("confidence", 0.0)
        predicted_severity = ent.get("severity", "Unknown")
        # Severity enum may serialize as dict with a 'value' key
        if isinstance(predicted_severity, dict):
            predicted_severity = predicted_severity.get("value", str(predicted_severity))

        return {
            "id": t["id"],
            "predicted_code": predicted_code,
            "confidence": confidence,
            "true_code": t["true_category"],
            "code_match": predicted_code == t["true_category"],
            "predicted_severity": predicted_severity,
            "true_severity": t["true_severity"],
            "severity_match": predicted_severity == t["true_severity"],
        }

    print(f"\n[Evaluation] running {len(transcripts)} transcripts in parallel...\n")
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(evaluate_one, t): t["id"] for t in transcripts}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["id"])
    for r in results:
        status = "CORRECT" if r["code_match"] else "WRONG"
        print(
            f"  {r['id']}: predicted={r['predicted_code']:<10}  actual={r['true_code']:<10}"
            f"  conf={r['confidence']:.2f}  {status}"
        )

    elapsed = time.time() - start
    code_correct = sum(1 for r in results if r["code_match"])
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    n = len(results)
    print(f"\n  Classification accuracy: {code_correct}/{n} ({100*code_correct/n:.0f}%)")
    print(f"  Average confidence:      {avg_conf:.2f}")
    print(f"  Wall-clock time:         {elapsed:.1f}s (parallel across {n} transcripts)")


# Global trainer log — in production this would be a database
trainer_log = []


def log_result(state: WorkflowState) -> dict:
    """Node: Trainer pattern — log the final decision for future improvement.
    Captures the original AI prediction alongside any human corrections.
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
    # print(f"  [TRAINER] {action}: {entry['selected_code']} (conf: {entry['confidence']:.2f})")
    if entry["was_overridden"]:
        print(f"            Original: {entry['original_code']} → Corrected: {entry['selected_code']}")

    return {"trainer_log": trainer_log.copy()}


#Graph builder defining nodes and edges

def build_graph(llm: ChatOpenAI, retriever_tool, checkpointer=None):
    workflow = StateGraph(WorkflowState)

    #Entity extraction & clarification nodes
    workflow.add_node("extract_entities", make_extract_entities(llm))
    workflow.add_node("check_quality", make_check_quality(llm))
    workflow.add_node("gather_followup", gather_followup)
    workflow.add_node("build_enriched_query", build_enriched_query)

    #Retrieval & answering nodes
    workflow.add_node("generate_query_or_respond", make_generate_query_or_respond(llm, retriever_tool))
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", make_rewrite_question(llm))
    workflow.add_node("generate_answer", make_generate_answer(llm))

    #Validator gate
    workflow.add_node("review_and_route", review_and_route)

    #Edges
    #Entity extraction loop
    workflow.add_edge(START, "extract_entities")
    workflow.add_edge("extract_entities", "check_quality")
    workflow.add_conditional_edges("check_quality", needs_followup)
    workflow.add_edge("gather_followup", "extract_entities")       # loop back

    #Transition from extraction to retrieval
    workflow.add_edge("build_enriched_query", "generate_query_or_respond")

    #Retrieval loop
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", make_grade_documents(llm))
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    #Validator gate and trainer log sit between classification and END
    workflow.add_node("log_result", log_result)
    workflow.add_edge("generate_answer", "review_and_route")
    workflow.add_edge("review_and_route", "log_result")
    workflow.add_edge("log_result", END)

    return workflow.compile(checkpointer=checkpointer)


def vis_graph(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        out = Path(__file__).parent / "graph.png"
        out.write_bytes(png_bytes)
        # print(f"\n[Graph] saved to {out}")
    except Exception:
        print("\n[Graph]")
        print(graph.get_graph().draw_ascii())



#Entry point


if __name__ == "__main__":
    llm = init_llm()
    docs = load_docs()
    chunks = chunking(docs)
    vector_db = embed_and_store(chunks)
    retriever_tool = make_retriever_tool(vector_db)

    # Load problem codes dict for severity lookup
    problem_codes_path = Path(__file__).parent / "problem_codes.json"
    with open(problem_codes_path) as f:
        codes_list = json.load(f)
    problem_codes_dict = {c["code"]: c for c in codes_list}

    memory = InMemorySaver()
    graph = build_graph(llm=llm, retriever_tool=retriever_tool, checkpointer=memory)
    vis_graph(graph)

    #Evaluate all transcripts through the graph

    transcripts_path = Path(__file__).parent / "transcripts.json"
    run_pipeline_evaluation(graph, str(transcripts_path))

    #Describe Issue
    print("\n" + "="*60)
    user_query = input("\nDescribe the issue: ").strip()
    config = {"configurable": {"thread_id": "interactive-001"}}
    result = graph.invoke({
        "messages": [HumanMessage(content=user_query)],
        "accumulated_description": "",
        "entity": None,
        "follow_ups": [],
        "enriched_query": "",
        "extraction_complete": False,
        "classification": None,
        "routing_decision": None,
    }, config)

    # Handle validator gate interrupt if it fired
    if graph.get_state(config).next:
        paused_state = graph.get_state(config).values
        paused_clf = paused_state.get("classification", {})
        paused_ent = paused_state.get("entity") or {}
        paused_sev = paused_ent.get("severity", "N/A")
        if isinstance(paused_sev, dict):
            paused_sev = paused_sev.get("value", str(paused_sev))
        print(f"\n{'='*60}")
        print(f"  HUMAN REVIEW REQUIRED")
        print(f"  Code:       {paused_clf.get('selected_code')}")
        print(f"  Confidence: {paused_clf.get('confidence', 0):.2f}")
        print(f"  Severity:   {paused_sev}")
        print(f"  Reasoning:  {paused_clf.get('reasoning')}")
        print(f"{'='*60}")
        decision = input("\nYour decision — approve / override: ").strip().lower()
        if decision == "override":
            print("\nAvailable codes: PLUMB-001, PLUMB-002, ELEC-001, ELEC-002, ELEC-003,")
            print("  HVAC-001, HVAC-002, HVAC-003, ELEV-001, ELEV-002, DOOR-001, DOOR-002,")
            print("  DOOR-003, SAFE-001, SAFE-002, SAFE-003, JANI-001, JANI-002, PEST-001, GRND-001")
            new_code = input("Enter correct code: ").strip().upper()
            reason = input("Reason for override: ").strip()
            result = graph.invoke(
                Command(resume={"override_code": new_code, "override_reason": reason}),
                config,
            )
        else:
            result = graph.invoke(Command(resume={"approved": True}), config)

    clf = result.get("classification")
    ent = result.get("entity") or {}
    if clf:
        code = clf["selected_code"]
        entity_sev = ent.get("severity", "N/A")
        if isinstance(entity_sev, dict):
            entity_sev = entity_sev.get("value", str(entity_sev))
        official_sev = problem_codes_dict.get(code, {}).get("typical_severity", "N/A")
        print(f"\n[Classification] {code}  (confidence: {clf['confidence']:.2f})")
        print(f"  Reasoning:         {clf['reasoning']}")
        print(f"  Entity severity:   {entity_sev}")
        print(f"  Official severity: {official_sev}")
        print(f"  Routing:           {result.get('routing_decision', 'N/A')}")
    else:
        print(f"\nAnswer: {result['messages'][-1].content}")

    # Post-execution override — reviews after the graph has finished
    print(f"\n{'='*60}")
    print("  POST-EXECUTION OVERRIDE — update_state()")
    print(f"{'='*60}")
    state_before = graph.get_state(config)
    current_clf = state_before.values.get("classification") or {}
    current_code = current_clf.get("selected_code", "N/A")
    current_routing = state_before.values.get("routing_decision", "N/A")
    print(f"  Current code:    {current_code}")
    print(f"  Current routing: {current_routing}")

    change = input("\nPost-execution override? (yes/no): ").strip().lower()
    if change in ("yes", "y"):
        print("\nAvailable codes: PLUMB-001, PLUMB-002, ELEC-001, ELEC-002, ELEC-003,")
        print("  HVAC-001, HVAC-002, HVAC-003, ELEV-001, ELEV-002, DOOR-001, DOOR-002,")
        print("  DOOR-003, SAFE-001, SAFE-002, SAFE-003, JANI-001, JANI-002, PEST-001, GRND-001")
        new_code = input("Enter correct code: ").strip().upper()
        reason = input("Reason: ").strip()
        graph.update_state(
            config,
            {
                "classification": {
                    **current_clf,
                    "selected_code": new_code,
                    "reasoning": f"Supervisor override: {reason}",
                    "original_code": current_code,
                },
                "routing_decision": f"Post-execution override → {new_code}",
            },
        )
        state_after = graph.get_state(config)
        print(f"\n  BEFORE: {current_code}")
        print(f"  AFTER:  {state_after.values['classification']['selected_code']}")
        print(f"  Checkpoint updated.")
    else:
        print("  No changes. Classification remains as-is.")
