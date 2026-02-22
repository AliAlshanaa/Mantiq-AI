from src.core.state import AgentState
from langgraph.types import interrupt

def human_review_node(state: AgentState):
    """
    Human-in-the-loop gate to approve or request revisions.
    This uses LangGraph's interrupt to pause execution and wait for user input.
    """
    payload = {
        "prompt": "Approve or request a rewrite.",
        "draft": state.get("draft", ""),
        "reviewer_feedback": state.get("feedback", "")
    }

    response = interrupt(payload)
    decision = str(response.get("decision", "")).strip().lower()
    feedback = str(response.get("feedback", "")).strip()

    if decision in {"approve", "approved", "yes", "y"}:
        return {
            "human_feedback": feedback,
            "next_step": "SAVE"
        }

    if not feedback:
        feedback = "Human requested changes."

    return {
        "human_feedback": feedback,
        "feedback": feedback,
        "next_step": "REWRITE"
    }
