import json
import time
import warnings
import os
from typing import Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
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

# --- Part 4: Build the Graph ---
workflow = StateGraph(PipelineState)

workflow.add_node("extract_entities", extract_entities)
workflow.add_node("retrieve_codes", retrieve_codes)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("classify_problem", classify_problem)

workflow.add_edge(START, "extract_entities")
workflow.add_edge("extract_entities", "retrieve_codes")
workflow.add_conditional_edges(
    "retrieve_codes",
    grade_documents,
)
workflow.add_edge("rewrite_query", "retrieve_codes")
workflow.add_edge("classify_problem", END)

pipeline = workflow.compile()

# --- Part 5: Execute Evaluation (Optional/Verification) ---
if __name__ == "__main__":
    def classify_one(t):
        output = pipeline.invoke({"transcript": t["transcript"], "retries": 0})
        predicted = output["classification"]["selected_code"]
        actual = t["true_category"]
        match = predicted == actual
        return {
            "id": t["id"],
            "predicted": predicted,
            "actual": actual,
            "confidence": output["classification"]["confidence"],
            "match": match,
            "retries_used": output["retries"]
        }

    # Load test transcripts
    try:
        with open("transcripts.json") as f:
            transcripts = json.load(f)
            
        print("Starting full pipeline evaluation across 10 transcripts...")
        start = time.time()
        all_results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(classify_one, t): t for t in transcripts}
            for future in as_completed(futures):
                r = future.result()
                status = "CORRECT" if r["match"] else "WRONG"
                print(f"{r['id']}: predicted={r['predicted']:10s} actual={r['actual']:10s} conf={r['confidence']:.2f} retries={r['retries_used']} {status}")
                all_results.append(r)

        elapsed = time.time() - start
        correct = sum(1 for r in all_results if r["match"])
        print(f"\nClassification accuracy: {correct}/{len(all_results)} ({100*correct/len(all_results):.0f}%)")
        avg_conf = sum(r["confidence"] for r in all_results) / len(all_results)
        print(f"Average confidence: {avg_conf:.2f}")
        total_retries = sum(r["retries_used"] for r in all_results)
        print(f"Total query rewrites performed: {total_retries}")
        print(f"Wall-clock time: {elapsed:.1f}s")
    except FileNotFoundError:
        print("transcripts.json not found, skipping evaluation loop.")
