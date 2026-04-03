#!/usr/bin/env python3
"""
cbre_agentic_rag.py

Agentic RAG pipeline for CBRE building maintenance call classification.
Adapted from the official LangChain "Build a Custom RAG Agent with LangGraph"
tutorial, replacing the default blog-post Q&A use case with maintenance
transcript classification.

Graph architecture:

    START
      │
      ▼
  ┌─────────────────────────────────────────────────────┐
  │  agent  (generate_query_or_respond)                 │
  │  LLM decides: call retriever tool OR answer direct  │
  └─────────────────────────────────────────────────────┘
          │                            │
    [tool call]                  [no tool call]
          │                            │
          ▼                            ▼
  ┌──────────────┐            ┌─────────────────┐
  │   retrieve   │            │  generate_answer │──► END
  │  (ToolNode)  │            └─────────────────┘
  └──────────────┘
          │
   grade_documents  ◄── conditional edge: grades retrieved codes
          │                        │
     [relevant]               [not relevant]
          │                        │
          ▼                        ▼
  ┌──────────────────┐    ┌───────────────────┐
  │  generate_answer │    │  rewrite_question │
  └──────────────────┘    └───────────────────┘
          │                        │
         END               [loops back to agent]
"""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# Resolve paths relative to this script first so load_dotenv can find .env
# regardless of which directory the script is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

# =============================================================================
# 1. DATA INGESTION
#    Load problem_codes.json and convert each entry into a LangChain Document.
#    page_content concatenates all descriptive fields so the embedder sees the
#    full context; metadata keeps the code/category for downstream filtering.
# =============================================================================

with open(os.path.join(_HERE, "problem_codes.json")) as f:
    _raw_codes = json.load(f)

documents: list[Document] = []
for entry in _raw_codes:
    page_content = (
        f"Code: {entry['code']}\n"
        f"Category: {entry['category']}\n"
        f"Subcategory: {entry['subcategory']}\n"
        f"Description: {entry['description']}\n"
        f"Keywords: {', '.join(entry['keywords'])}"
    )
    documents.append(
        Document(
            page_content=page_content,
            metadata={"code": entry["code"], "category": entry["category"]},
        )
    )

print(f"[Setup] Ingested {len(documents)} problem code documents.")

# =============================================================================
# 2. VECTOR STORE & RETRIEVER TOOL
#    Index the documents with OpenAI embeddings inside a Chroma collection.
#    Wrap the retriever as a named LangChain tool the agent can invoke.
# =============================================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
    collection_name="cbre_problem_codes",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_problem_codes",
    description=(
        "Search the CBRE maintenance problem code database. "
        "Use this tool to look up the correct problem code, category, and "
        "routing procedure for any building maintenance or safety issue. "
        "Input should be a concise description of the maintenance problem."
    ),
)

tools = [retriever_tool]
print("[Setup] Retriever tool ready.")

# =============================================================================
# 3. LLM
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent LLM has the retriever tool bound so it can emit structured tool calls
llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# 4. GRAPH NODES
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — agent (generate_query_or_respond)
#
# The LLM receives the full message history and decides:
#   a) Emit a tool call to search_problem_codes  →  LangGraph routes to retrieve
#   b) Generate a text response with no tool call →  LangGraph routes to generate_answer
#
# The routing after this node is handled by the built-in `tools_condition`
# helper, which inspects the last AIMessage for tool_calls.
# ─────────────────────────────────────────────────────────────────────────────
def generate_query_or_respond(state: MessagesState) -> dict:
    """
    Agent node: the LLM (with retriever tool bound) decides whether to search
    the problem code database or respond directly.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — retrieve
#
# A standard LangGraph ToolNode. It reads the last AIMessage, executes any
# tool_calls found in it (i.e., calls search_problem_codes), and appends the
# results as ToolMessage(s) to the state. No custom code needed here.
# ─────────────────────────────────────────────────────────────────────────────
retrieve = ToolNode(tools)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge — grade_documents
#
# Called after `retrieve`. Uses an LLM with structured output to judge whether
# the returned problem codes are actually relevant to the transcript's issue.
#
# Returns:
#   "generate_answer"  → relevant codes found; proceed to classification
#   "rewrite_question" → codes are off-target; loop back through rewriter
# ─────────────────────────────────────────────────────────────────────────────
class GradeDocuments(BaseModel):
    """Structured output schema for the document relevance grader."""
    binary_score: str = Field(
        description=(
            "'yes' if the retrieved problem codes are relevant to the maintenance "
            "issue in the transcript, 'no' if they are not."
        )
    )


grader_llm = llm.with_structured_output(GradeDocuments)

_grade_prompt = PromptTemplate(
    template=(
        "You are a grader assessing whether retrieved CBRE maintenance problem "
        "codes are relevant to a building maintenance call transcript.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Retrieved problem codes:\n{documents}\n\n"
        "Score 'yes' if at least one retrieved code directly addresses the "
        "maintenance issue described in the transcript. Score 'no' if the codes "
        "are unrelated or describe a different type of problem entirely."
    ),
    input_variables=["transcript", "documents"],
)
_grader_chain = _grade_prompt | grader_llm


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Conditional edge function. After retrieval, checks whether the returned
    codes are useful. Routes to generation or triggers a query rewrite loop.
    """
    # After ToolNode runs, the last message is a ToolMessage with retrieved docs
    retrieved_docs = state["messages"][-1].content

    # The original transcript is the first HumanMessage in the conversation
    transcript = next(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )

    result = _grader_chain.invoke({"transcript": transcript, "documents": retrieved_docs})
    score = result.binary_score.lower().strip()

    print(f"  [Grader]   Relevance = {score!r}")
    return "generate_answer" if score == "yes" else "rewrite_question"


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — rewrite_question
#
# Called when the grader finds the retrieved codes irrelevant. Reformulates
# the search query as a concise keyword string and injects it back into the
# message history as a new HumanMessage. The agent will pick it up on the
# next loop iteration and emit a new (better) tool call.
# ─────────────────────────────────────────────────────────────────────────────
_rewrite_prompt = PromptTemplate(
    template=(
        "You are optimizing a search query for a CBRE maintenance problem code "
        "database.\n\n"
        "Original maintenance transcript:\n{transcript}\n\n"
        "The previous search did not return relevant codes. Write a short, "
        "keyword-focused search query (10 words or fewer) that will better match "
        "the correct maintenance category in the database.\n\n"
        "Examples of good queries:\n"
        "  - 'burst pipe water pouring ceiling flooding'\n"
        "  - 'elevator stuck between floors trapped passengers'\n"
        "  - 'rotten egg gas smell hazardous odor'\n\n"
        "Return ONLY the query string, nothing else."
    ),
    input_variables=["transcript"],
)


def rewrite_question(state: MessagesState) -> dict:
    # 1. Get the original transcript
    transcript = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    
    # 2. Get all previous search queries attempted in this session
    # We look for HumanMessages that AREN'T the first transcript
    past_queries = [m.content for m in state["messages"][2:] if isinstance(m, HumanMessage)]
    past_queries_str = ", ".join([f"'{q}'" for q in past_queries]) if past_queries else "None"

    _rewrite_prompt_with_memory = PromptTemplate(
        template=(
            "You are optimizing a search query for a maintenance database.\n"
            "Original Transcript: {transcript}\n"
            "FAILED QUERIES ALREADY TRIED: {past_queries}\n\n"
            "CRITICAL: Do not repeat any of the failed queries. They returned no results. "
            "Try a completely different strategy. If you tried 'thumping', try 'vibration' or 'structural'. "
            "Return ONLY the new query string."
        ),
        input_variables=["transcript", "past_queries"],
    )
    
    chain = _rewrite_prompt_with_memory | llm
    new_query = chain.invoke({"transcript": transcript, "past_queries": past_queries_str}).content.strip()
    
    print(f"  [Rewriter] New unique query: {new_query!r}")
    return {"messages": [HumanMessage(content=new_query)]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — generate_answer
#
# Terminal node. Combines the original transcript with all retrieved problem
# codes (accumulated across any retry loops) to produce a final structured
# classification: Problem Code, Severity, Confidence, and Reasoning.
#
# NOTE: Severity is assessed on actual safety risk, not the caller's tone.
# This is the key design decision that handles the "Tricky Transcript" case
# where a caller downplays a life-safety hazard.
# ─────────────────────────────────────────────────────────────────────────────
_generation_prompt = PromptTemplate(
    template=(
        "You are a CBRE building maintenance dispatch AI.\n\n"
        "Using the transcript and the retrieved problem codes below, produce a "
        "final structured maintenance classification.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Retrieved Problem Codes:\n{context}\n\n"
        "CRITICAL INSTRUCTION: Base severity on the ACTUAL SAFETY RISK described, "
        "not the caller's tone or urgency language. A caller saying 'no big deal' "
        "or 'whenever you're free' may still be describing a Critical hazard "
        "(e.g., gas smell + water near outlet = life-safety emergency).\n\n"
        "Output your answer in this exact format:\n"
        "Problem Code: <code>\n"
        "Severity: <Critical | High | Medium | Low>\n"
        "Confidence: <0-100>%\n"
        "Reasoning: <1-2 sentences explaining the code choice and severity>"
    ),
    input_variables=["transcript", "context"],
)


def generate_answer(state: MessagesState) -> dict:
    """
    Generator node: produces the final classification by combining the original
    transcript with all retrieved problem codes from the conversation history.
    """
    # Collect content from all ToolMessages (may span multiple retrieval passes)
    retrieved_context = "\n\n---\n\n".join(
        m.content for m in state["messages"] if isinstance(m, ToolMessage)
    )
    if not retrieved_context:
        retrieved_context = (
            "No problem codes retrieved. Classify from transcript context alone."
        )

    # Original transcript = first HumanMessage in the conversation
    transcript = next(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )

    chain = _generation_prompt | llm
    answer = chain.invoke({"transcript": transcript, "context": retrieved_context}).content

    return {"messages": [AIMessage(content=answer)]}


# =============================================================================
# 5. GRAPH WIRING
#    Assemble the StateGraph with conditional routing that mirrors the official
#    LangGraph Agentic RAG tutorial structure.
# =============================================================================

graph_builder = StateGraph(MessagesState)

# Register nodes
graph_builder.add_node("agent",            generate_query_or_respond)
graph_builder.add_node("retrieve",         retrieve)
graph_builder.add_node("rewrite_question", rewrite_question)
graph_builder.add_node("generate_answer",  generate_answer)

# ── START → agent ─────────────────────────────────────────────────────────────
graph_builder.add_edge(START, "agent")

# ── agent → retrieve OR generate_answer ───────────────────────────────────────
# tools_condition(state) inspects the last AIMessage:
#   returns "tools"  if the LLM emitted a tool call  →  remap to "retrieve"
#   returns END      if the LLM responded directly   →  remap to "generate_answer"
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",        # tool call present → execute retrieval
        END:     "generate_answer", # no tool call → skip retrieval, generate directly
    },
)

# ── retrieve → generate_answer OR rewrite_question ────────────────────────────
# grade_documents(state) grades the ToolMessage content:
#   returns "generate_answer"  if codes are relevant
#   returns "rewrite_question" if codes are off-target
# No path_map needed because the function already returns node name strings.
graph_builder.add_conditional_edges("retrieve", grade_documents)

# ── rewrite_question → agent (retry loop) ─────────────────────────────────────
# After injecting the new query, hand control back to the agent so it can
# issue an improved tool call with the reformulated keywords.
graph_builder.add_edge("rewrite_question", "agent")

# ── generate_answer → END ─────────────────────────────────────────────────────
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()
print("[Setup] LangGraph pipeline compiled successfully.\n")


# =============================================================================
# 6. EVALUATION
#    Load transcripts.json and run the first 3 transcripts through the graph.
#    Print the node-by-node execution trace and the final classification.
# =============================================================================

# System prompt primes the agent to always search before answering.
# This is important: without it, an overconfident LLM might skip the tool.
_SYSTEM_PROMPT = (
    "You are a CBRE building maintenance dispatch AI. You will receive a "
    "maintenance call transcript. You MUST call the search_problem_codes tool "
    "to look up relevant problem codes before making any classification. "
    "Do not guess the problem code — always search first."
)


def run_transcript(entry: dict) -> str:
    """Run a single transcript through the compiled LangGraph pipeline."""
    tx_id      = entry["id"]
    transcript = entry["transcript"]
    true_code  = entry["true_category"]
    true_sev   = entry["true_severity"]

    print(f"{'═'*65}")
    print(f"  {tx_id}  |  Ground Truth: {true_code}  ({true_sev})")
    print(f"{'─'*65}")
    print(f"  \"{transcript[:110]}...\"")
    print(f"{'─'*65}")
    print("  [Graph Execution]")

    initial_state = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": transcript},
        ]
    }

    final_answer = None

    # stream_mode="updates" yields {node_name: state_delta} for each graph step,
    # letting us trace the execution path in real time.
    for step in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(step.keys())[0]
        print(f"    → {node_name}")

        # Capture the AIMessage produced by generate_answer
        if node_name == "generate_answer":
            msgs = step["generate_answer"].get("messages", [])
            if msgs:
                final_answer = msgs[-1].content

    print(f"\n  [Final Classification]")
    if final_answer:
        for line in final_answer.strip().splitlines():
            print(f"    {line}")
    else:
        print("    (no classification output captured)")

    print()
    return final_answer or ""


if __name__ == "__main__":
    with open(os.path.join(_HERE, "transcripts.json")) as f:
        all_transcripts = json.load(f)


    # chaos_entry = {
    #         "id": "TX-CHAOS",
    #         "transcript": "Hey, there's a persistent, low-frequency thumping coming from somewhere inside the walls... [insert full text above]",
    #         "true_category": "STRUCT-001", # Or whatever fits best
    #         "true_severity": "Medium"
    #     }

    # print("\n--- TRIGGERING THE REWRITER TEST ---")
    # run_transcript(chaos_entry)
    print(f"\n{'═'*65}")
    print("  CBRE Agentic RAG — Evaluation (first 3 transcripts)")
    print(f"{'═'*65}\n")

    for entry in all_transcripts:
        run_transcript(entry)

    print(f"{'═'*65}")
    print("  Evaluation complete.")
    print(f"{'═'*65}\n")
