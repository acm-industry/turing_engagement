"""
Week 4: Human-in-the-Loop Review Interface

Run with: streamlit run weeks/week4/hitl_app.py
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

DATA_DIR = Path(__file__).parent
CONFIDENCE_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

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


class Classification(BaseModel):
    selected_code: str = Field(description="The problem code that best matches")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of why this code was selected")


class RoutingDecision(BaseModel):
    route_to: str = Field(description="Who/what team to route this to")
    action: str = Field(description="Recommended immediate action")
    requires_911: bool = Field(description="Whether 911 should be called")
    reasoning: str = Field(description="Brief explanation of the routing decision")


class PipelineState(TypedDict):
    transcript: str
    entities: Optional[dict]
    retrieved_codes: Optional[list[dict]]
    classification: Optional[dict]
    routing: Optional[dict]
    needs_human_review: Optional[bool]
    review_reason: Optional[str]


# ---------------------------------------------------------------------------
# Pipeline (cached)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
You are an expert building maintenance call analyst. Extract key information.

Severity guidelines:
- Critical: Immediate danger to life/safety (gas leak, fire, trapped persons, flooding)
- High: Significant disruption or escalation risk (major leak, broken security glass, HVAC failure)
- Medium: Needs attention soon, not emergency (broken door, minor plumbing, elevator malfunction)
- Low: Minor/cosmetic (flickering light, carpet stain, empty soap dispenser)

Transcript:
{transcript}
"""

CLASSIFICATION_PROMPT = """
You are classifying a building maintenance issue. Select the best matching code.

Extracted information:
- Problem: {problem_type}
- Severity: {severity}
- Summary: {summary}

Candidate problem codes:
{candidates}

Respond with your classification.
"""

ROUTING_PROMPT = """
You are a building maintenance routing specialist. Based on the problem classification,
decide where to route this request.

Problem code: {code}
Severity: {severity}
Problem: {problem_type}
Summary: {summary}

Decide: who should handle this, what immediate action to take, and whether 911 is needed.
Only recommend 911 for genuine life-safety emergencies (active fire, gas leak, people trapped
in immediate danger, structural collapse).
"""


@st.cache_resource
def build_vectorstore():
    with open(DATA_DIR / "problem_codes.json") as f:
        problem_codes = json.load(f)

    documents = []
    for pc in problem_codes:
        text = (
            f"{pc['code']}: {pc['category']} — {pc['subcategory']}\n"
            f"{pc['description']}\nKeywords: {', '.join(pc['keywords'])}"
        )
        doc = Document(page_content=text, metadata={"code": pc["code"], "category": pc["category"]})
        documents.append(doc)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents, embeddings, collection_name="problem_codes_hitl")


@st.cache_resource
def build_pipeline():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    vs = build_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    def extract_entities(state: PipelineState) -> dict:
        extractor = llm.with_structured_output(MaintenanceEntity)
        result = extractor.invoke(EXTRACTION_PROMPT.format(transcript=state["transcript"]))
        return {"entities": result.model_dump()}

    def retrieve_codes(state: PipelineState) -> dict:
        entities = state["entities"]
        query = f"{entities['problem_type']} {entities['summary']}"
        docs = retriever.invoke(query)
        codes = [{"code": d.metadata["code"], "content": d.page_content} for d in docs]
        return {"retrieved_codes": codes}

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

    def check_hitl(state: PipelineState) -> dict:
        classification = state["classification"]
        entities = state["entities"]
        severity = entities["severity"]
        confidence = classification["confidence"]

        needs_review = False
        reason = ""

        if severity == "Critical":
            needs_review = True
            reason = "Severity is CRITICAL — requires human verification before routing"
        elif confidence < CONFIDENCE_THRESHOLD:
            needs_review = True
            reason = f"Low confidence ({confidence:.2f}) — below threshold of {CONFIDENCE_THRESHOLD}"

        return {"needs_human_review": needs_review, "review_reason": reason}

    def route_decision(state: PipelineState) -> dict:
        classification = state["classification"]
        entities = state["entities"]
        router = llm.with_structured_output(RoutingDecision)
        result = router.invoke(ROUTING_PROMPT.format(
            code=classification["selected_code"],
            severity=entities["severity"],
            problem_type=entities["problem_type"],
            summary=entities["summary"],
        ))
        return {"routing": result.model_dump()}

    graph = StateGraph(PipelineState)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("retrieve_codes", retrieve_codes)
    graph.add_node("classify_problem", classify_problem)
    graph.add_node("check_hitl", check_hitl)
    graph.add_node("route_decision", route_decision)

    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", "retrieve_codes")
    graph.add_edge("retrieve_codes", "classify_problem")
    graph.add_edge("classify_problem", "check_hitl")
    graph.add_edge("check_hitl", "route_decision")
    graph.add_edge("route_decision", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "result": None,
        "selected_idx": 0,
        "review_status": None,
        "override_data": None,
        "review_history": [],
        "pipeline_step": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

with open(DATA_DIR / "transcripts.json") as f:
    transcripts = json.load(f)

with open(DATA_DIR / "problem_codes.json") as f:
    all_codes = json.load(f)

code_options = [pc["code"] for pc in all_codes]
code_labels = {pc["code"]: f"{pc['code']} — {pc['category']} / {pc['subcategory']}" for pc in all_codes}

# ---------------------------------------------------------------------------
# Page config & styles
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CBRE Maintenance — HITL Review", layout="wide", page_icon="🏢")

st.markdown("""
<style>
    /* Pipeline step tracker */
    .pipeline-container {
        display: flex; align-items: center; justify-content: center;
        gap: 0; margin: 0.8rem 0 1.2rem 0; flex-wrap: wrap;
    }
    .pipeline-step {
        display: flex; align-items: center; gap: 0.35rem;
        padding: 0.4rem 0.7rem; border-radius: 8px;
        font-size: 0.82rem; font-weight: 600; white-space: nowrap;
    }
    .step-done { background: rgba(40,167,69,0.2); color: #4caf50; border: 1px solid rgba(40,167,69,0.4); }
    .step-active { background: rgba(255,193,7,0.2); color: #ffc107; border: 2px solid rgba(255,193,7,0.5); }
    .step-pending { background: rgba(150,150,150,0.1); color: #888; border: 1px solid rgba(150,150,150,0.2); }
    .step-paused { background: rgba(220,53,69,0.2); color: #ff6b6b; border: 2px solid rgba(220,53,69,0.5); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
    .pipeline-arrow { font-size: 1.1rem; color: #666; margin: 0 0.15rem; }

    /* Severity badges */
    .sev-badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;
    }
    .sev-critical { background: #ff4b4b; color: #fff; }
    .sev-high { background: #ff8c00; color: #fff; }
    .sev-medium { background: #1f77b4; color: #fff; }
    .sev-low { background: #2ca02c; color: #fff; }

    /* Concept labels — semi-transparent for dark mode */
    .concept-tag {
        display: inline-block; padding: 0.2rem 0.55rem; border-radius: 4px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; margin-right: 0.3rem;
    }
    .tag-validator { background: rgba(255,193,7,0.2); color: #ffc107; border: 1px solid rgba(255,193,7,0.4); }
    .tag-override { background: rgba(0,123,255,0.2); color: #5dade2; border: 1px solid rgba(0,123,255,0.4); }
    .tag-trainer { background: rgba(40,167,69,0.2); color: #4caf50; border: 1px solid rgba(40,167,69,0.4); }
    .tag-rag { background: rgba(149,117,205,0.2); color: #b39ddb; border: 1px solid rgba(149,117,205,0.4); }

    /* Card containers — transparent backgrounds for dark mode */
    .info-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .review-card {
        border: 2px solid rgba(255,193,7,0.6); border-radius: 10px;
        padding: 1.2rem; background: rgba(255,193,7,0.08); margin-top: 0.5rem;
    }
    .approved-card {
        border: 2px solid rgba(40,167,69,0.6); border-radius: 10px;
        padding: 1rem 1.2rem; background: rgba(40,167,69,0.08);
    }
    .override-card {
        border: 2px solid rgba(0,123,255,0.6); border-radius: 10px;
        padding: 1rem 1.2rem; background: rgba(0,123,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏢 CBRE Maintenance Call Router")
st.caption("Week 4 — Human-in-the-Loop Review Interface")


# ---------------------------------------------------------------------------
# Helper: pipeline step tracker
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    ("📝", "Extract"),
    ("🔍", "Retrieve"),
    ("🏷️", "Classify"),
    ("⚖️", "HITL Check"),
    ("🚀", "Route"),
]


def render_pipeline_tracker(completed: int, paused_at: int | None = None):
    """Render a horizontal pipeline progress bar. completed = number of steps done."""
    parts = []
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        if paused_at is not None and i == paused_at:
            cls = "step-paused"
        elif i < completed:
            cls = "step-done"
        elif i == completed:
            cls = "step-active"
        else:
            cls = "step-pending"
        parts.append(f'<span class="pipeline-step {cls}">{icon} {label}</span>')
        if i < len(PIPELINE_STEPS) - 1:
            parts.append('<span class="pipeline-arrow">→</span>')
    st.markdown(f'<div class="pipeline-container">{"".join(parts)}</div>', unsafe_allow_html=True)


def severity_badge(sev: str) -> str:
    cls = {"Critical": "sev-critical", "High": "sev-high", "Medium": "sev-medium", "Low": "sev-low"}
    return f'<span class="sev-badge {cls.get(sev, "")}">{sev}</span>'


def concept_tag(name: str) -> str:
    tag_map = {"Validator": "tag-validator", "Override": "tag-override", "Trainer": "tag-trainer", "RAG": "tag-rag"}
    return f'<span class="concept-tag {tag_map.get(name, "")}">{name}</span>'


# ---------------------------------------------------------------------------
# Sidebar: Trainer Log
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### {concept_tag('Trainer')} Trainer Log", unsafe_allow_html=True)
    st.caption("Every human decision is captured for model improvement.")

    history = st.session_state["review_history"]

    if history:
        total = len(history)
        approved = sum(1 for e in history if e["action"] == "approved")
        overridden = sum(1 for e in history if e["action"] == "overridden")
        auto = sum(1 for e in history if e["action"] == "auto-routed")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("Auto", auto)
        c3.metric("Approved", approved)
        c4.metric("Override", overridden)

        if overridden > 0:
            st.markdown(
                f"**Override rate:** {overridden}/{total} ({overridden/total:.0%}) "
                f"— *these corrections feed back into prompt tuning*"
            )

        st.divider()

        for i, entry in enumerate(reversed(history)):
            icons = {"approved": "✅", "overridden": "🔄", "auto-routed": "⚡"}
            icon = icons.get(entry["action"], "❓")
            with st.expander(f"{icon} {entry['transcript_id']} → `{entry['code']}`", expanded=(i == 0)):
                st.markdown(f"**Action:** {entry['action'].title()}")
                st.markdown(f"**Final Code:** `{entry['code']}`")
                if entry.get("original_code"):
                    st.markdown(f"**AI's Code:** `{entry['original_code']}` *(corrected)*")
                if entry.get("override_reason"):
                    st.markdown(f"**Reason:** {entry['override_reason']}")
                st.markdown(f"**AI Confidence:** {entry['confidence']:.0%}")
                st.caption(entry["timestamp"])

        st.divider()
        st.markdown("##### How This Becomes Training Data")
        st.markdown(
            "In production, this log feeds a weekly review:\n"
            "- Which codes get overridden most?\n"
            "- What severity levels cause low confidence?\n"
            "- After 500+ entries → fine-tune or update prompts"
        )
    else:
        st.info("No reviews yet. Analyze a call to get started.")
        st.divider()
        st.markdown("##### The 3 HITL Patterns")
        st.markdown(
            f"{concept_tag('Validator')} Pause pipeline for human approval\n\n"
            f"{concept_tag('Override')} Correct AI's classification\n\n"
            f"{concept_tag('Trainer')} Log decisions for improvement",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col_input, col_output = st.columns([2, 3], gap="large")

with col_input:
    st.subheader("📞 Incoming Call")

    selected = st.selectbox(
        "Pick a sample transcript:",
        options=range(len(transcripts)),
        format_func=lambda i: f"{transcripts[i]['id']} — {transcripts[i]['building']}",
        key="transcript_select",
    )

    transcript_text = st.text_area(
        "Transcript",
        value=transcripts[selected]["transcript"],
        height=220,
    )

    if st.button("🔍 Analyze Call", type="primary", use_container_width=True):
        st.session_state["review_status"] = None
        st.session_state["override_data"] = None
        st.session_state["selected_idx"] = selected

        pipeline = build_pipeline()

        progress_placeholder = st.empty()
        status_text = st.empty()
        step_names = ["Extracting entities...", "Retrieving codes (RAG)...", "Classifying problem...",
                      "Checking HITL gate...", "Generating routing..."]

        with st.spinner("Running pipeline..."):
            result = pipeline.invoke({"transcript": transcript_text})

        st.session_state["result"] = result

        if not result["needs_human_review"]:
            entry = {
                "transcript_id": transcripts[selected]["id"],
                "action": "auto-routed",
                "code": result["classification"]["selected_code"],
                "original_code": None,
                "override_reason": None,
                "confidence": result["classification"]["confidence"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state["review_history"].append(entry)

        st.rerun()

    # Concept reminder card in left column
    st.divider()
    st.markdown("##### Pipeline Architecture")
    st.markdown(
        f"{concept_tag('RAG')} Retrieval-Augmented Generation\n\n"
        "The pipeline retrieves matching problem codes from a vector store before classifying — "
        "grounding the LLM's decision in real CBRE data.",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"{concept_tag('Validator')} Human Gate\n\n"
        f"If severity = Critical **or** confidence < {CONFIDENCE_THRESHOLD:.0%}, "
        "the pipeline pauses for human review before routing.",
        unsafe_allow_html=True,
    )


# --- Right column: Analysis ---

with col_output:
    result = st.session_state["result"]

    if result is None:
        st.subheader("📋 Analysis")
        render_pipeline_tracker(completed=0)
        st.info("Select a transcript and click **Analyze Call** to run the pipeline.")

        st.markdown("---")
        st.markdown("##### What Happens When You Click Analyze")
        st.markdown("""
1. **Extract** — LLM pulls structured entities (problem, severity, location) from raw transcript
2. **Retrieve** — RAG searches the vector store for matching CBRE problem codes
3. **Classify** — LLM picks the best code using extracted info + retrieved candidates
4. **HITL Check** — Is severity Critical or confidence low? If yes → pause for human
5. **Route** — LLM decides which team handles it and what action to take
        """)
    else:
        entities = result["entities"]
        classification = result["classification"]
        routing = result["routing"]
        needs_review = result["needs_human_review"]
        review_status = st.session_state["review_status"]

        raw_sev = str(entities["severity"])
        severity = (raw_sev.split(".")[-1] if "." in raw_sev else raw_sev).title()

        # Pipeline tracker
        paused_at = 3 if (needs_review and not review_status) else None
        completed = 5 if (not needs_review or review_status) else 3
        render_pipeline_tracker(completed=completed, paused_at=paused_at)

        # --- Top summary row ---
        st.markdown(
            f'<div class="info-card">'
            f'{severity_badge(severity)} &nbsp; '
            f'<strong>{entities["problem_type"]}</strong> &nbsp;·&nbsp; '
            f'{entities["location_building"]} — {entities.get("location_detail") or "N/A"} &nbsp;·&nbsp; '
            f'Code: <code>{classification["selected_code"]}</code> &nbsp;·&nbsp; '
            f'Confidence: <strong>{classification["confidence"]:.0%}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --- Step 1: Entities ---
        with st.expander("📝 Step 1 — Extracted Entities", expanded=False):
            e1, e2 = st.columns(2)
            with e1:
                st.markdown(f"**Problem:** {entities['problem_type']}")
                st.markdown(f"**Summary:** {entities['summary']}")
                st.markdown(f"**Caller:** {entities.get('caller_role') or 'N/A'}")
            with e2:
                st.markdown(f"**Building:** {entities['location_building']}")
                st.markdown(f"**Detail:** {entities.get('location_detail') or 'N/A'}")
                st.markdown(f"**Severity:** {severity_badge(severity)}", unsafe_allow_html=True)
            if entities.get("urgency_indicators"):
                st.markdown(f"**Urgency cues:** `{'` · `'.join(entities['urgency_indicators'])}`")

        # --- Step 2: Retrieved codes ---
        with st.expander("🔍 Step 2 — Retrieved Codes (RAG)", expanded=False):
            retrieved = result.get("retrieved_codes", [])
            if retrieved:
                for j, c in enumerate(retrieved):
                    lines = c["content"].split("\n")
                    code_id = c["code"]
                    desc = lines[1] if len(lines) > 1 else c["content"][:100]
                    match_flag = " ← selected" if code_id == classification["selected_code"] else ""
                    st.markdown(f"**{j+1}.** `{code_id}`{match_flag}  \n{desc}")
            else:
                st.warning("No codes retrieved.")

        # --- Step 3: Classification ---
        with st.expander("🏷️ Step 3 — Classification", expanded=True):
            if st.session_state.get("override_data"):
                od = st.session_state["override_data"]
                st.info(
                    f"🔄 **Final Code:** `{od['new_code']}` *(overridden from `{classification['selected_code']}`)*  \n"
                    f"**Reason:** {od['reason']}"
                )
            st.markdown(f"**AI's Code:** `{classification['selected_code']}`")
            st.progress(classification["confidence"], text=f"Confidence: {classification['confidence']:.0%}")
            st.markdown(f"**Reasoning:** {classification['reasoning']}")

        # --- Step 4: HITL Gate ---
        st.markdown(f"#### ⚖️ Step 4 — HITL Gate {concept_tag('Validator')}", unsafe_allow_html=True)

        if not needs_review:
            st.success(
                f"⚡ **Auto-routed** — severity is not Critical and confidence "
                f"({classification['confidence']:.0%}) exceeds threshold ({CONFIDENCE_THRESHOLD:.0%}). "
                f"No human review needed."
            )

        elif review_status == "approved":
            st.success(
                f"✅ **Approved by human reviewer** — routed as `{classification['selected_code']}`"
            )

        elif review_status == "overridden":
            od = st.session_state["override_data"]
            st.info(
                f"🔄 **Overridden**: `{classification['selected_code']}` → `{od['new_code']}`  \n"
                f"Severity: {od.get('new_severity', severity)}  ·  Reason: {od['reason']}"
            )

        else:
            st.markdown(
                f'<div class="review-card">'
                f'⏸️ <strong>PIPELINE PAUSED — Human review required</strong><br>'
                f'{result["review_reason"]}<br><br>'
                f'<em>The pipeline has stopped at the validator gate. '
                f'Review the classification above and choose an action below.</em>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            tab_approve, tab_override = st.tabs(["✅ Approve Classification", "🔄 Override Classification"])

            with tab_approve:
                st.markdown(
                    "Confirm the AI's classification is correct. "
                    "The call will be routed based on the current code and severity."
                )
                if st.button("Approve & Route", type="primary", use_container_width=True, key="btn_approve"):
                    st.session_state["review_status"] = "approved"
                    st.session_state["review_history"].append({
                        "transcript_id": transcripts[st.session_state["selected_idx"]]["id"],
                        "action": "approved",
                        "code": classification["selected_code"],
                        "original_code": None,
                        "override_reason": None,
                        "confidence": classification["confidence"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    st.rerun()

            with tab_override:
                st.markdown(
                    f"{concept_tag('Override')} "
                    "The AI got it wrong. Select the correct code and explain why.",
                    unsafe_allow_html=True,
                )
                with st.form("override_form"):
                    new_code = st.selectbox(
                        "Correct problem code:",
                        options=code_options,
                        format_func=lambda c: code_labels.get(c, c),
                        index=code_options.index(classification["selected_code"])
                        if classification["selected_code"] in code_options else 0,
                    )
                    new_severity = st.selectbox(
                        "Correct severity:",
                        ["Critical", "High", "Medium", "Low"],
                        index=["Critical", "High", "Medium", "Low"].index(severity)
                        if severity in ["Critical", "High", "Medium", "Low"] else 0,
                    )
                    reason = st.text_input("Reason for override:", placeholder="e.g. Fire alarm is equipment issue, not general alarm")
                    submitted = st.form_submit_button("Submit Override & Route", type="primary", use_container_width=True)

                    if submitted:
                        st.session_state["review_status"] = "overridden"
                        st.session_state["override_data"] = {
                            "new_code": new_code,
                            "new_severity": new_severity,
                            "reason": reason or "No reason provided",
                        }
                        st.session_state["review_history"].append({
                            "transcript_id": transcripts[st.session_state["selected_idx"]]["id"],
                            "action": "overridden",
                            "code": new_code,
                            "original_code": classification["selected_code"],
                            "override_reason": reason or "No reason provided",
                            "confidence": classification["confidence"],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        st.rerun()

        # --- Step 5: Routing ---
        if not needs_review or review_status:
            with st.expander("🚀 Step 5 — Routing Decision", expanded=True):
                r1, r2 = st.columns([1, 1])
                with r1:
                    st.markdown(f"**Route to:** {routing['route_to']}")
                    st.markdown(f"**Action:** {routing['action']}")
                with r2:
                    st.markdown(f"**Reasoning:** {routing['reasoning']}")
                if routing["requires_911"]:
                    st.error("🚨 911 RECOMMENDED — Immediate life-safety emergency")

        # --- Ground truth ---
        gt = transcripts[st.session_state["selected_idx"]]
        with st.expander("📊 Ground Truth Comparison (for evaluation)"):
            g1, g2 = st.columns(2)
            with g1:
                code_match = gt["true_category"] == classification["selected_code"]
                st.metric(
                    "Problem Code",
                    classification["selected_code"],
                    delta=f"{'✓ Match' if code_match else '✗ Mismatch'} (expected {gt['true_category']})",
                    delta_color="normal" if code_match else "inverse",
                )
            with g2:
                sev_match = gt["true_severity"] == severity
                st.metric(
                    "Severity",
                    severity,
                    delta=f"{'✓ Match' if sev_match else '✗ Mismatch'} (expected {gt['true_severity']})",
                    delta_color="normal" if sev_match else "inverse",
                )
