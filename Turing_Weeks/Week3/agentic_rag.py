import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

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
    """Validate that each extracted entity is specific enough to be actionable."""
    problem_type_ok: bool = Field(description="True only if problem_type describes a specific error or symptom, not a vague word like 'issue' or 'problem'")
    affected_system_ok: bool = Field(description="True only if affected_system names a specific product, tool, or device — not generic terms like 'computer' or 'system'")
    location_ok: bool = Field(description="True only if location_detail includes both a building name AND a city — a floor number or room alone is NOT sufficient")
    urgency_ok: bool = Field(description="True only if urgency_indicators contains at least one specific phrase about timing or impact from the description")
    problem_type_followup: str = Field(description="Question to get a more specific problem description. Empty string if problem_type_ok is True.")
    affected_system_followup: str = Field(description="Question to get the exact device or software name. Empty string if affected_system_ok is True.")
    location_followup: str = Field(description="Question to get building name and city. Empty string if location_ok is True.")
    urgency_followup: str = Field(description="Question to understand timing or business impact. Empty string if urgency_ok is True.")


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")
    confidence: float = Field(description="Confidence in this relevance decision, from 0.0 to 1.0")


class Classification(BaseModel):
    """Final problem code classification from retrieved context."""
    selected_code: str = Field(description="The problem code that best matches (e.g., PLUMB-001, HVAC-001)")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of why this code was selected over the others")



# Prompts

ENTITY_EXTRACTION_PROMPT = (
    "You are a tech support analyst. A user has described a problem.\n"
    "Extract the key information from their description.\n"
    "If a field cannot be determined from the description, leave it as null.\n\n"
    "Severity guidelines:\n"
    "- Critical: Data loss risk, full outage, security breach, or all work blocked\n"
    "- High: Significant disruption, multiple users affected, or escalating quickly\n"
    "- Medium: One user affected, workaround exists, or intermittent issue\n"
    "- Low: Minor inconvenience, cosmetic, or low-priority request\n\n"
    "User description: {description}"
)

QUALITY_CHECK_PROMPT = (
    "You are validating whether a tech support ticket has enough specific information.\n\n"
    "Extracted entity:\n"
    "- problem_type: {problem_type}\n"
    "- affected_system: {affected_system}\n"
    "- location_detail: {location_detail}\n"
    "- urgency_indicators: {urgency_indicators}\n\n"
    "Check each field for specificity and generate a targeted follow-up question "
    "for any that are vague, generic, or incomplete."
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

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\nContext: {context}"
)

CLASSIFY_PROMPT = (
    "You are classifying a building maintenance issue to a problem code.\n"
    "Based on the extracted information and the retrieved candidate codes, "
    "select the single best matching code.\n\n"
    "Extracted information:\n"
    "- Problem: {problem_type}\n"
    "- Severity: {severity}\n"
    "- Summary: {summary}\n\n"
    "Candidate problem codes:\n{candidates}"
)


# Extended graph state — adds entity-extraction fields to MessagesState

class WorkflowState(MessagesState):
    """Full workflow state: chat messages + entity extraction tracking."""
    accumulated_description: str          # builds up as follow-ups are added
    entity: Optional[dict]                # serialised MaintenanceEntity
    follow_ups: list[str]                 # pending follow-up questions
    enriched_query: str                   # final query sent to retrieval
    extraction_complete: bool             # gate: entity extraction done?
    classification: Optional[dict]        # final Classification result



# Node factories — each returns a closure over the shared LLM / tools

def make_extract_entities(llm: ChatOpenAI):
    """Node: run structured entity extraction on the accumulated description."""
    extractor = llm.with_structured_output(MaintenanceEntity)

    def extract_entities(state: WorkflowState):
        print("\n[NODE] extract_entities")
        description = state.get("accumulated_description", "")
        if not description:
            description = state["messages"][0].content

        entity = extractor.invoke(
            ENTITY_EXTRACTION_PROMPT.format(description=description)
        )
        print(f"  problem_type:   {entity.problem_type}")
        print(f"  affected_system:{entity.affected_system}")
        print(f"  location_detail:{entity.location_detail}")
        print(f"  severity:       {entity.severity.value}")
        print(f"  urgency:        {entity.urgency_indicators}")
        return {
            "entity": entity.model_dump(),
            "accumulated_description": description,
        }

    return extract_entities


def make_check_quality(llm: ChatOpenAI):
    """Node: validate entity specificity, populate follow_ups list."""
    checker = llm.with_structured_output(EntityQualityCheck)

    def check_quality(state: WorkflowState):
        print("\n[NODE] check_quality")
        ent = state["entity"]
        quality = checker.invoke(
            QUALITY_CHECK_PROMPT.format(
                problem_type=ent.get("problem_type") or "null",
                affected_system=ent.get("affected_system") or "null",
                location_detail=ent.get("location_detail") or "null",
                urgency_indicators=ent.get("urgency_indicators") or [],
            )
        )
        follow_ups = [
            q for q in [
                quality.problem_type_followup if not quality.problem_type_ok else "",
                quality.affected_system_followup if not quality.affected_system_ok else "",
                quality.location_followup if not quality.location_ok else "",
                quality.urgency_followup if not quality.urgency_ok else "",
            ]
            if q
        ]
        print(f"  problem_type_ok:   {quality.problem_type_ok}")
        print(f"  affected_system_ok:{quality.affected_system_ok}")
        print(f"  location_ok:       {quality.location_ok}")
        print(f"  urgency_ok:        {quality.urgency_ok}")
        if follow_ups:
            print(f"  pending follow-ups: {len(follow_ups)}")
        else:
            print("  all fields satisfied — proceeding")
        return {
            "follow_ups": follow_ups,
            "extraction_complete": len(follow_ups) == 0,
        }

    return check_quality


def gather_followup(state: WorkflowState):
    """Node: prompt the user for the first outstanding follow-up, append to description."""
    print("\n[NODE] gather_followup")
    question = state["follow_ups"][0]
    answer = input(f"\n{question}\n> ").strip()
    new_description = f"{state['accumulated_description']}. {answer}"
    return {
        "accumulated_description": new_description,
        "follow_ups": state["follow_ups"][1:],
    }


def build_enriched_query(state: WorkflowState):
    """Node: convert the validated entity into a rich query string and inject it as a message."""
    print("\n[NODE] build_enriched_query")
    ent = state["entity"]
    enriched = (
        f"Problem: {ent['problem_type']}."
        + (f" System: {ent['affected_system']}." if ent.get("affected_system") else "")
        + (f" Location: {ent['location_detail']}." if ent.get("location_detail") else "")
        + f" Severity: {ent['severity']}."
        + f" {ent['summary']}"
    )
    print(f"  {enriched}")
    return {
        "enriched_query": enriched,
        "messages": [HumanMessage(content=enriched)],
    }


def make_generate_query_or_respond(llm: ChatOpenAI, retriever_tool):
    """Node: let the LLM decide whether to call the retriever tool or answer directly."""
    def generate_query_or_respond(state: WorkflowState):
        print("\n[NODE] generate_query_or_respond")
        response = (
            llm
            .bind_tools([retriever_tool])
            .invoke(state["messages"])
        )
        if response.tool_calls:
            print(f"  decision: call retriever tool")
        else:
            print(f"  decision: respond directly (no retrieval needed)")
        return {"messages": [response]}

    return generate_query_or_respond


def make_rewrite_question(llm: ChatOpenAI):
    def rewrite_question(state: WorkflowState):
        print("\n[NODE] rewrite_question")
        question = state.get("enriched_query", state["messages"][0].content)
        prompt = REWRITE_PROMPT.format(question=question)
        response = llm.invoke([{"role": "user", "content": prompt}])
        print(f"  rewritten: {response.content[:120]}")
        return {"messages": [HumanMessage(content=response.content)]}

    return rewrite_question


def make_generate_answer(llm: ChatOpenAI):
    classifier = llm.with_structured_output(Classification)

    def generate_answer(state: WorkflowState):
        print("\n[NODE] generate_answer")
        ent = state.get("entity") or {}
        context = state["messages"][-1].content
        prompt = CLASSIFY_PROMPT.format(
            problem_type=ent.get("problem_type", "unknown"),
            severity=ent.get("severity", "unknown"),
            summary=ent.get("summary", state.get("enriched_query", "")),
            candidates=context,
        )
        result = classifier.invoke([{"role": "user", "content": prompt}])
        print(f"  selected_code: {result.selected_code}")
        print(f"  confidence:    {result.confidence:.2f}")
        print(f"  reasoning:     {result.reasoning}")
        return {
            "classification": result.model_dump(),
            "messages": [HumanMessage(content=f"[{result.selected_code}] {result.reasoning}")],
        }

    return generate_answer



#Conditional edge functions

def needs_followup(state: WorkflowState) -> Literal["gather_followup", "build_enriched_query"]:
    """Route after quality check: loop back for more info or proceed."""
    decision = "build_enriched_query" if state.get("extraction_complete") else "gather_followup"
    print(f"\n[ROUTE] check_quality → {decision}")
    return decision


def make_grade_documents(llm: ChatOpenAI):
    grader = llm.with_structured_output(GradeDocuments)

    def grade_documents(state: WorkflowState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade retrieved docs for relevance — route to answer or rewrite."""
        print("\n[NODE] grade_documents")
        question = state.get("enriched_query", state["messages"][0].content)
        context = state["messages"][-1].content

        response = grader.invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}]
        )
        print(f"  relevant:   {response.binary_score}")
        print(f"  confidence: {response.confidence:.2f}")
        decision = "generate_answer" if response.binary_score == "yes" else "rewrite_question"
        print(f"[ROUTE] retrieve → {decision}")
        return decision

    return grade_documents



#Graph builder defining nodes and edges

def build_graph(llm: ChatOpenAI, retriever_tool):
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
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


def vis_graph(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        out = Path(__file__).parent / "graph.png"
        out.write_bytes(png_bytes)
        print(f"\n[Graph] saved to {out}")
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

    graph = build_graph(llm=llm, retriever_tool=retriever_tool)
    vis_graph(graph)

    user_query = input("\nDescribe the issue: ").strip()
    result = graph.invoke({
        "messages": [HumanMessage(content=user_query)],
        "accumulated_description": "",
        "entity": None,
        "follow_ups": [],
        "enriched_query": "",
        "extraction_complete": False,
        "classification": None,
    })

    clf = result.get("classification")
    if clf:
        print(f"\n[Classification] {clf['selected_code']}  (confidence: {clf['confidence']:.2f})")
        print(f"  Reasoning: {clf['reasoning']}")
    else:
        print(f"\nAnswer: {result['messages'][-1].content}")


















