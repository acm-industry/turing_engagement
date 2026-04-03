import json
import warnings
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
load_dotenv()

with open("problem_codes.json") as f:
    problem_codes = json.load(f)

docs = []
for pc in problem_codes:
    text = (
        f"{pc['code']}: {pc['category']} — {pc['subcategory']}\n"
        f"{pc['description']}\n"
        f"Keywords: {', '.join(pc['keywords'])}"
    )
    docs.append(Document(page_content=text, metadata={"code": pc["code"]}))

embeddings = OpenAIEmbeddings()
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="problem_codes")
retriever = vectorstore.as_retriever()


@tool
def retrieve_problem_codes(query: str) -> str:
    results = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in results)


retriever_tool = retrieve_problem_codes

response_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def generate_query_or_respond(state: MessagesState):
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved CBRE problem codes to a maintenance call.\n\n"
    "Retrieved codes (context):\n{context}\n\n"
    "Maintenance call (question):\n{question}\n\n"
    "If the retrieved codes contain a problem code that correctly matches the call, "
    "grade it as 'yes'. Otherwise grade it as 'no'.\n"
    "Return only 'yes' or 'no'."
)

grader_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    return "generate_answer" if result.binary_score.strip().lower() == "yes" else "rewrite_question"


from langchain_core.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the maintenance call description below and reason about the underlying issue.\n"
    "Here is the initial call:\n"
    "-------\n"
    "{question}\n"
    "-------\n"
    "Formulate an improved retrieval query that will help find the correct CBRE problem codes.\n"
    "Focus on the core problem type and key details (e.g., plumbing leak, HVAC failure, elevator entrapment).\n"
)


def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


GENERATE_PROMPT = (
    "You are an assistant for CBRE maintenance call classification.\n"
    "Use the retrieved CBRE problem codes (context) to answer the question.\n"
    "If you don't know the answer, say you don't know.\n"
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Question: {question}\n"
    "Context (CBRE problem codes):\n{context}\n"
)


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()


def demo_stream() -> None:
    question = (
        "There's water leaking from the ceiling near suite 310 and the carpet is soaked. "
        "How should this be classified using the CBRE problem codes?"
    )
    for chunk in graph.stream({"messages": [{"role": "user", "content": question}]}):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n")


if __name__ == "__main__":
    demo_stream()
