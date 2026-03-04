from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.core.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.agents.reviewer import reviewer_node
from src.agents.saver import saver_node
from src.agents.human import human_review_node


MAX_REVISIONS = 3  # Prevent infinite rewrite loops


# ----------------------------------------
# Routing Logic (Safe + Controlled)
# ----------------------------------------

def route_after_review(state: AgentState):
    next_step = state.get("next_step")
    revision_count = state.get("revision_count", 0)

    if revision_count >= MAX_REVISIONS:
        return "human"  # Force escalation after too many rewrites

    if next_step == "REWRITE":
        return "rewrite"

    return "human"


def route_after_human(state: AgentState):
    next_step = state.get("next_step")

    if next_step == "REWRITE":
        return "rewrite"

    return "save"


# ----------------------------------------
# Graph Construction
# ----------------------------------------

workflow = StateGraph(AgentState)

# Register Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("saver", saver_node)

# Entry
workflow.set_entry_point("researcher")

# Core Flow
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

# Reviewer Decision Routing
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "rewrite": "writer",
        "human": "human_review",
    },
)

# Human Decision Routing
workflow.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "rewrite": "writer",
        "save": "saver",
    },
)

# Final Step
workflow.add_edge("saver", END)


# ----------------------------------------
# Persistent Memory (Session Safe)
# ----------------------------------------

checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],  # Allows interactive pause
)