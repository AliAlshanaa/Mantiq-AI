from langgraph.graph import StateGraph, END
from src.core.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.agents.reviewer import reviewer_node

def route_after_review(state: AgentState):
    """
    Directional logic that interprets the reviewer's decision.
    Returns a string key that maps to the next node in the workflow.
    """
    if state.get("next_step") == "REWRITE":
        return "rewrite"
    
    # If approved or max revisions reached, signal completion
    return "finish"

# 1. Initialize the StateGraph with our custom AgentState schema
workflow = StateGraph(AgentState)

# 2. Register Nodes (The processing units of our AI)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

# 3. Define the Static Edges (The fixed sequence)
# Start with Research -> then Write -> then Review
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

# 4. Define Conditional Edges (The Agentic Loop)
# The Reviewer node uses 'route_after_review' to decide the path
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "rewrite": "writer",  # If feedback says REWRITE, loop back to Writer
        "finish": END         # If approved, go to the final END state
    }
)

# 5. Compile the graph into an executable app
app = workflow.compile()