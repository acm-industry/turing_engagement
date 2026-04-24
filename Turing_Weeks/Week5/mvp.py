import json
import os
import sqlite3
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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

warnings.filterwarnings("ignore", message="Deserializing unregistered type")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path(__file__).parent.parent / "Week4"
DB_PATH = Path(__file__).parent / "mvp.db"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")


_node_timings: dict[str, list[float]] = {}


def timed_node(node_name: str, fn):
    """Wrapper that records wall-clock time for each node invocation."""
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        _node_timings.setdefault(node_name, []).append(elapsed)
        return result
    wrapper.__name__ = fn.__name__ if hasattr(fn, "__name__") else node_name
    return wrapper


def print_timing_report():
    """Print a summary of node latencies collected during this session."""
    if not _node_timings:
        return
    print(f"\n{'='*60}")
    print("  NODE LATENCY REPORT")
    print(f"{'='*60}")
    for name, times in sorted(_node_timings.items()):
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"  {name:<30} avg={avg:.3f}s  total={total:.2f}s  calls={len(times)}")
    total_all = sum(sum(t) for t in _node_timings.values())
    print(f"  {'TOTAL':<30} {total_all:.2f}s")
    print(f"{'='*60}")


# LLM, doc loading, chunking, embedding
def init_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4.1", temperature=0, api_key=OPENAI_API_KEY)


def load_docs() -> list[Document]:
    loader = JSONLoader(
        file_path=str(DATA_DIR / "problem_codes.json"),
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


def embed_and_store(chunks: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name="problem_codes_mvp",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    if db._collection.count() == 0:
        db.add_documents(chunks)
        print(f"Embedded {len(chunks)} chunks into Chroma.")
    else:
        print(f"Chroma collection already has {db._collection.count()} documents — skipping re-embed.")
    return db


def make_retriever_tool(vector_store: Chroma):
    @tool
    def retrieve_problem_codes(query: str) -> str:
        """Search and return information about problem codes,
        including their descriptions, categories, and resolutions."""
        docs = vector_store.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_problem_codes


def load_problem_codes_dict() -> dict:
    """Load problem_codes.json into a dict keyed by code."""
    with open(DATA_DIR / "problem_codes.json") as f:
        codes_list = json.load(f)
    return {c["code"]: c for c in codes_list}


def init_trainer_db(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trainer_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            thread_id TEXT,
            selected_code TEXT,
            confidence REAL,
            reasoning TEXT,
            routing_decision TEXT,
            was_overridden INTEGER,
            original_code TEXT,
            risk_score REAL,
            department TEXT,
            final_human_action TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_trainer_log(db_path: str | Path, entry: dict) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO trainer_log (timestamp, thread_id, selected_code, confidence, "
        "reasoning, routing_decision, was_overridden, original_code, risk_score, "
        "department, final_human_action) "
        "VALUES (:timestamp, :thread_id, :selected_code, :confidence, :reasoning, "
        ":routing_decision, :was_overridden, :original_code, :risk_score, "
        ":department, :final_human_action)",
        entry,
    )
    conn.commit()
    conn.close()


def get_historical_agreement_rate(db_path: str | Path, code: str) -> float:
    """Fraction of times this code was NOT overridden historically."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT COUNT(*) as total, "
        "SUM(CASE WHEN was_overridden=0 THEN 1 ELSE 0 END) as agreed "
        "FROM trainer_log WHERE selected_code=?",
        (code,),
    ).fetchone()
    conn.close()
    total, agreed = row
    if total == 0:
        return 0.8
    return agreed / total



#Structured extraction models
class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class MaintenanceEntity(BaseModel):
    """Structured extraction from a caller's problem description."""
    problem_category: str = Field(
        description="Specific description of the problem (e.g., 'water leaking from ceiling pipe', "
                    "'elevator stuck between floors') — not vague terms like 'issue' or 'problem'"
    )
    affected_system: Optional[str] = Field(
        description="The exact device, software name, or system affected "
                    "(e.g., 'ceiling pipe in hallway', 'elevator car B') — not just 'equipment'"
    )
    address: Optional[str] = Field(
        description="Full street address including building name, street, and city "
                    "(e.g., 'Westfield Office Building, 200 Main Street, Austin TX'). "
                    "Combine any building name, street number, and city mentioned."
    )
    location_detail: Optional[str] = Field(
        description="Specific location within the building (floor, suite, room, wing) "
                    "(e.g., '3rd floor, suite 310, hallway near stairwell')"
    )
    severity: Severity = Field(
        description="Critical (blocking all work / data loss risk), High (significant disruption), "
                    "Medium (needs attention soon), Low (minor / cosmetic)"
    )
    urgency_indicators: list[str] = Field(
        description="Specific phrases from the description that indicate urgency or timeline "
                    "(e.g., 'been down since this morning', 'client demo in 1 hour')"
    )
    summary: str = Field(description="One-sentence summary of the issue")


class EntityQualityCheck(BaseModel):
    """Validate that the problem description is specific enough to drive accurate retrieval."""
    problem_type_ok: bool = Field(
        description="True only if the problem describes a specific symptom or failure — "
                    "not vague words like 'issue', 'problem', or 'something wrong'"
    )
    problem_type_followup: str = Field(
        description="Question to get a more specific problem description. Empty string if ok."
    )


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")
    confidence: float = Field(
        description="Confidence in this relevance decision, from 0.0 to 1.0. "
                    "Use < 0.7 for borderline cases."
    )


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
    "Extracted problem: {problem_category}\n\n"
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
    "- Problem: {problem_category}\n"
    "- Severity: {severity}\n"
    "- Summary: {summary}\n\n"
    "Candidate problem codes:\n{candidates}\n\n"
    "- Output confidence > 0.85 ONLY if you are certain no other code could apply.\n"
    "- Output confidence < 0.8 if two or more codes are plausible given the description.\n"
)

# Extended graph state adds entity-extraction fields to MessagesState

class WorkflowState(MessagesState):
    """Full workflow state: chat messages + entity extraction tracking."""
    accumulated_description: str
    entity: Optional[dict]
    follow_ups: list[str]
    enriched_query: str
    extraction_complete: bool
    classification: Optional[dict]
    risk_score: Optional[float]
    risk_details: Optional[dict]
    routing_decision: Optional[str]
    trainer_log: Optional[list]



# Node factories — each returns a closure over the shared LLM / tools

def make_extract_entities(llm: ChatOpenAI):
    extractor = llm.with_structured_output(MaintenanceEntity)

    def extract_entities(state: WorkflowState):
        description = state.get("accumulated_description", "")
        if not description:
            description = state["messages"][0].content

        entity = extractor.invoke(
            ENTITY_EXTRACTION_PROMPT.format(description=description)
        )
        return {
            "entity": entity.model_dump(mode="json"),
            "accumulated_description": description,
        }

    return extract_entities


def make_check_quality(llm: ChatOpenAI):
    checker = llm.with_structured_output(EntityQualityCheck)

    def check_quality(state: WorkflowState):
        if state.get("extraction_complete"):
            return {"follow_ups": []}

        ent = state["entity"]
        quality = checker.invoke(
            QUALITY_CHECK_PROMPT.format(
                problem_category=ent.get("problem_category") or "null"
            )
        )
        follow_ups = [quality.problem_type_followup] if not quality.problem_type_ok else []
        return {
            "follow_ups": follow_ups,
            "extraction_complete": len(follow_ups) == 0,
        }

    return check_quality


def gather_followup(state: WorkflowState):
    """Node: pause for caller's follow-up answer via interrupt()."""
    question = state["follow_ups"][0]

    answer = interrupt({
        "type": "followup_question",
        "question": question,
        "accumulated_so_far": state["accumulated_description"],
    })

    new_description = f"{state['accumulated_description']}. {answer}"
    return {
        "accumulated_description": new_description,
        "follow_ups": state["follow_ups"][1:],
    }


def build_enriched_query(state: WorkflowState):
    ent = state["entity"]
    enriched = (
        f"Problem: {ent['problem_category']}."
        + (f" System: {ent['affected_system']}." if ent.get("affected_system") else "")
        + (f" Address: {ent['address']}." if ent.get("address") else "")
        + (f" Location: {ent['location_detail']}." if ent.get("location_detail") else "")
        + f" Severity: {ent['severity']}."
        + f" {ent['summary']}"
    )
    return {
        "enriched_query": enriched,
        "messages": [HumanMessage(content=enriched)],
    }


def make_generate_query_or_respond(llm: ChatOpenAI, retriever_tool):
    def generate_query_or_respond(state: WorkflowState):
        response = (
            llm
            .bind_tools([retriever_tool])
            .invoke(state["messages"])
        )
        return {"messages": [response]}

    return generate_query_or_respond


def make_rewrite_question(llm: ChatOpenAI):
    def rewrite_question(state: WorkflowState):
        question = state.get("enriched_query", state["messages"][0].content)
        prompt = REWRITE_PROMPT.format(question=question)
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    return rewrite_question


def make_grade_documents(llm: ChatOpenAI):
    grader = llm.with_structured_output(GradeDocuments)

    def grade_documents(state: WorkflowState) -> Literal["generate_answer", "rewrite_question"]:
        question = state.get("enriched_query", state["messages"][0].content)
        context = state["messages"][-1].content

        response = grader.invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}]
        )
        if response.binary_score == "yes" or response.confidence < 0.8:
            return "generate_answer"
        return "rewrite_question"

    return grade_documents


def make_generate_answer(llm: ChatOpenAI):
    classifier = llm.with_structured_output(Classification)

    def generate_answer(state: WorkflowState):
        ent = state.get("entity") or {}
        context = state["messages"][-1].content
        prompt = CLASSIFY_PROMPT.format(
            problem_category=ent.get("problem_category", "unknown"),
            severity=ent.get("severity", "unknown"),
            summary=ent.get("summary", state.get("enriched_query", "")),
            candidates=context,
        )
        result = classifier.invoke([{"role": "user", "content": prompt}])
        return {
            "classification": result.model_dump(),
            "messages": [HumanMessage(content=f"[{result.selected_code}] {result.reasoning}")],
        }

    return generate_answer



SEVERITY_WEIGHTS = {
    "Critical": 1.0,
    "High": 0.75,
    "Medium": 0.4,
    "Low": 0.15,
}


def make_compute_risk_score(db_path: str | Path):

    def compute_risk_score(state: WorkflowState):
        classification = state["classification"]
        entity = state.get("entity") or {}

        severity = entity.get("severity", "Medium")
        severity_str = severity.split(".")[-1] if "." in str(severity) else str(severity)

        severity_weight = SEVERITY_WEIGHTS.get(severity_str, 0.4)
        confidence = classification.get("confidence", 0.5)
        urgency_count = len(entity.get("urgency_indicators", []))
        urgency_factor = min(1.0, 0.3 + (urgency_count * 0.15))

        code = classification["selected_code"]
        agreement_rate = get_historical_agreement_rate(db_path, code)

        risk = (
            severity_weight * 0.35
            + (1 - confidence) * 0.30
            + urgency_factor * 0.15
            + (1 - agreement_rate) * 0.20
        )
        risk = round(min(1.0, max(0.0, risk)), 4)

        needs_review = risk > 0.55

        reason_parts = []
        if severity_str == "Critical":
            reason_parts.append("Critical severity")
        if confidence < 0.7:
            reason_parts.append(f"low confidence ({confidence:.2f})")
        if urgency_count >= 3:
            reason_parts.append(f"high urgency ({urgency_count} indicators)")
        if agreement_rate < 0.7:
            reason_parts.append(f"low historical agreement ({agreement_rate:.2f})")

        return {
            "risk_score": risk,
            "risk_details": {
                "severity_weight": severity_weight,
                "confidence": confidence,
                "urgency_count": urgency_count,
                "urgency_factor": urgency_factor,
                "historical_agreement_rate": agreement_rate,
                "needs_review": needs_review,
                "reason": " and ".join(reason_parts) if reason_parts else "within normal parameters",
            },
        }

    return compute_risk_score


def review_and_route(state: WorkflowState) -> dict:
    """Validator gate — always triggers on Critical severity or high risk score."""
    classification = state["classification"]
    entity = state.get("entity") or {}
    risk_details = state.get("risk_details") or {}
    risk_score = state.get("risk_score", 0.0)

    severity = entity.get("severity", "Medium")
    severity_str = severity.split(".")[-1] if "." in str(severity) else str(severity)

    is_critical = severity_str == "Critical"
    risk_triggered = risk_details.get("needs_review", False)
    needs_review = is_critical or risk_triggered

    if needs_review:
        reason_parts = []
        if is_critical:
            reason_parts.append("Critical severity")
        if risk_triggered and not is_critical:
            reason_parts.append(risk_details.get("reason", "elevated risk score"))
        reason = " and ".join(reason_parts)

        decision = interrupt({
            "type": "review_required",
            "reason": reason,
            "risk_score": risk_score,
            "classification": classification,
            "risk_details": risk_details,
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
        "routing_decision": f"Auto-routed → {classification['selected_code']} (risk {risk_score:.2f})",
    }


def make_log_result(db_path: str | Path):

    def log_result(state: WorkflowState):
        classification = state["classification"]
        routing_decision = state.get("routing_decision", "unknown")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "thread_id": None,
            "selected_code": classification["selected_code"],
            "confidence": classification.get("confidence"),
            "reasoning": classification.get("reasoning"),
            "routing_decision": routing_decision,
            "was_overridden": 1 if classification.get("original_code") else 0,
            "original_code": classification.get("original_code"),
            "risk_score": state.get("risk_score"),
            "department": None,
            "final_human_action": routing_decision,
        }
        insert_trainer_log(db_path, entry)

        if entry["was_overridden"]:
            print(f"  [TRAINER] OVERRIDE: {entry['original_code']} → {entry['selected_code']}")

        return {"trainer_log": [entry]}

    return log_result


#Conditional edge functions
def needs_followup(state: WorkflowState) -> Literal["gather_followup", "build_enriched_query"]:
    return "build_enriched_query" if state.get("extraction_complete") else "gather_followup"


#Graph builder defining nodes and edges

def build_graph(
    llm: ChatOpenAI,
    retriever_tool,
    db_path: str | Path = DB_PATH,
    checkpointer=None,
):
    workflow = StateGraph(WorkflowState)

    #Entity extraction & clarification nodes
    workflow.add_node("extract_entities", timed_node("extract_entities", make_extract_entities(llm)))
    workflow.add_node("check_quality", timed_node("check_quality", make_check_quality(llm)))
    workflow.add_node("gather_followup", gather_followup)
    workflow.add_node("build_enriched_query", timed_node("build_enriched_query", build_enriched_query))

    #Retrieval & answering nodes
    workflow.add_node("generate_query_or_respond", timed_node("generate_query_or_respond", make_generate_query_or_respond(llm, retriever_tool)))
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", timed_node("rewrite_question", make_rewrite_question(llm)))
    workflow.add_node("generate_answer", timed_node("generate_answer", make_generate_answer(llm)))

    workflow.add_node("compute_risk_score", timed_node("compute_risk_score", make_compute_risk_score(db_path)))

    #Validator gate
    workflow.add_node("review_and_route", review_and_route)
    workflow.add_node("log_result", make_log_result(db_path))

    #Edges
    #Entity extraction loop
    workflow.add_edge(START, "extract_entities")
    workflow.add_edge("extract_entities", "check_quality")
    workflow.add_conditional_edges("check_quality", needs_followup)
    workflow.add_edge("gather_followup", "extract_entities")

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
    workflow.add_edge("generate_answer", "compute_risk_score")
    workflow.add_edge("compute_risk_score", "review_and_route")
    workflow.add_edge("review_and_route", "log_result")
    workflow.add_edge("log_result", END)

    return workflow.compile(checkpointer=checkpointer)


def vis_graph(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        out = Path(__file__).parent / "graph.png"
        out.write_bytes(png_bytes)
    except Exception:
        print("\n[Graph]")
        print(graph.get_graph().draw_ascii())



#Pipeline evaluation across transcripts using the actual graph
def run_pipeline_evaluation(graph, transcripts_path: str):
    """
    Run all transcripts through the actual graph.invoke() sequentially.
    Compares predicted problem code AND extracted severity vs ground truth.
    Transcript runs pass extraction_complete=True to skip the clarification loop.
    """
    with open(transcripts_path) as f:
        transcripts = json.load(f)

    def evaluate_one(t):
        run_id = datetime.now().strftime('%Y%m%d%H%M%S')
        config = {"configurable": {"thread_id": f"eval-{t['id']}-{run_id}"}}
        result = graph.invoke({
            "messages": [HumanMessage(content=t["transcript"])],
            "accumulated_description": "",
            "entity": None,
            "follow_ups": [],
            "enriched_query": "",
            "extraction_complete": True,
            "classification": None,
            "risk_score": None,
            "risk_details": None,
            "routing_decision": None,
            "trainer_log": None,
        }, config)
        # Auto-approve if the validator gate interrupted (Critical severity)
        while graph.get_state(config).next:
            result = graph.invoke(Command(resume={"approved": True}), config)
        clf = result.get("classification") or {}
        ent = result.get("entity") or {}

        predicted_code = clf.get("selected_code", "UNKNOWN")
        confidence = clf.get("confidence", 0.0)
        predicted_severity = ent.get("severity", "Unknown")
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
            "risk_score": result.get("risk_score") or 0.0,
        }

    print(f"\n[Evaluation] running {len(transcripts)} transcripts...\n")
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(evaluate_one, t): t["id"] for t in transcripts}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["id"])
    for r in results:
        status = "CORRECT" if r["code_match"] else "WRONG"
        print(
            f"  {r['id']}: predicted={r['predicted_code']:<10}  actual={r['true_code']:<10}"
            f"  conf={r['confidence']:.2f}  risk={r['risk_score']:.2f}  {status}"
        )

    elapsed = time.time() - start
    code_correct = sum(1 for r in results if r["code_match"])
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    n = len(results)
    print(f"\n  Classification accuracy: {code_correct}/{n} ({100*code_correct/n:.0f}%)")
    print(f"  Average confidence:      {avg_conf:.2f}")
    print(f"  Wall-clock time:         {elapsed:.1f}s ({n} transcripts)")

    print_timing_report()


def setup(db_path: str | Path = DB_PATH):
    """Initialize all components. Returns (graph, problem_codes_dict)."""
    llm = init_llm()
    docs = load_docs()
    chunks = chunking(docs)
    vector_db = embed_and_store(chunks)
    retriever_tool = make_retriever_tool(vector_db)
    problem_codes_dict = load_problem_codes_dict()

    init_trainer_db(db_path)

    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = build_graph(
        llm=llm,
        retriever_tool=retriever_tool,
        db_path=db_path,
        checkpointer=checkpointer,
    )
    return graph, problem_codes_dict


#Entry point

if __name__ == "__main__":
    graph, problem_codes_dict = setup()
    vis_graph(graph)

    #Evaluate all transcripts through the graph
    transcripts_path = DATA_DIR / "transcripts.json"
    run_pipeline_evaluation(graph, str(transcripts_path))

    #Describe Issue
    print("\n" + "=" * 60)
    user_query = input("\nDescribe the issue: ").strip()
    config = {"configurable": {"thread_id": f"interactive-{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "accumulated_description": "",
            "entity": None,
            "follow_ups": [],
            "enriched_query": "",
            "extraction_complete": False,
            "classification": None,
            "risk_score": None,
            "risk_details": None,
            "routing_decision": None,
            "trainer_log": None,
        },
        config,
    )

    # Handle validator gate interrupt if it fired
    while graph.get_state(config).next:
        snapshot = graph.get_state(config)
        tasks = snapshot.tasks
        interrupt_data = {}
        if tasks and tasks[0].interrupts:
            interrupt_data = tasks[0].interrupts[0].value

        if interrupt_data.get("type") == "followup_question":
            question = interrupt_data["question"]
            answer = input(f"\n{question}\n> ").strip()
            result = graph.invoke(Command(resume=answer), config)

        elif interrupt_data.get("type") == "review_required":
            paused_clf = interrupt_data.get("classification", {})
            risk = interrupt_data.get("risk_score", 0)
            print(f"\n{'=' * 60}")
            print(f"  HUMAN REVIEW REQUIRED (risk score: {risk:.2f})")
            print(f"  Reason:     {interrupt_data.get('reason')}")
            print(f"  Code:       {paused_clf.get('selected_code')}")
            print(f"  Confidence: {paused_clf.get('confidence', 0):.2f}")
            print(f"{'=' * 60}")
            decision = input("\nApprove / override: ").strip().lower()
            if decision == "override":
                print("\nAvailable codes: PLUMB-001, PLUMB-002, PLUMB-003, ELEC-001, ELEC-002, ELEC-003,")
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
        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULT")
        print(f"{'=' * 60}")
        print(f"  Classification:  {code} (confidence: {clf['confidence']:.2f})")
        print(f"  Reasoning:       {clf['reasoning']}")
        print(f"  Entity severity: {entity_sev}")
        print(f"  Official sev:    {official_sev}")
        print(f"  Address:         {ent.get('address', 'N/A')}")
        print(f"  Location:        {ent.get('location_detail', 'N/A')}")
        print(f"  Risk score:      {result.get('risk_score', 'N/A')}")
        print(f"  Decision:        {result.get('routing_decision', 'N/A')}")
        print(f"{'=' * 60}")
    else:
        print(f"\nAnswer: {result['messages'][-1].content}")
