from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.core.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.agents.reviewer import reviewer_node
from src.agents.saver import saver_node 
from src.agents.human import human_review_node

def route_after_review(state: AgentState):
    """
    Directional logic to handle the reviewer's decision.
    - If the reviewer requested a rewrite, loop back to the Writer.
    - If approved, proceed to the Saver node to persist the file.
    """
    if state.get("next_step") == "REWRITE":
        return "rewrite"
    return "human"

def route_after_human(state: AgentState):
    """
    Route based on human decision.
    """
    if state.get("next_step") == "REWRITE":
        return "rewrite"
    return "save"

# 1. Initialize the StateGraph
workflow = StateGraph(AgentState)

# 2. Register all Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("saver", saver_node)  # Added the Saver unit
workflow.add_node("human_review", human_review_node)

# 3. Define the Entry Point and Static Edges
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

# 4. Define the Agentic Feedback Loop & Conditional Logic
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "rewrite": "writer",  # Circular path for revision
        "human": "human_review"
    }
)

# Human-in-the-loop decision
workflow.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "rewrite": "writer",
        "save": "saver"
    }
)

# 5. Define the Termination Edge
# Once the file is saved, the process is officially complete
workflow.add_edge("saver", END)

# 6. Compile the graph into an executable application
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
