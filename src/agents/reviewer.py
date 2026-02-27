import os
from src.core.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()
AI_MODEL = os.getenv("AI_MODEL", "gemini-1.5-flash-latest")


def _extract_text_from_llm_response(response) -> str:
    """Robustly convert any LangChain / Gemini response into plain text."""
    # Single message object (e.g., AIMessage) – its `.content` may be a string OR a list of chunks.
    if isinstance(response, BaseMessage):
        content = getattr(response, "content", "")
        if isinstance(content, list):
            parts = []
            for chunk in content:
                # Gemini often returns dict chunks: {"type": "text", "text": "..."}
                if isinstance(chunk, dict) and "text" in chunk:
                    parts.append(str(chunk["text"]))
                else:
                    parts.append(str(chunk))
            return "\n".join(parts).strip()
        return str(content).strip()

    # List of messages / chunks / strings
    if isinstance(response, list):
        parts = []
        for item in response:
            if isinstance(item, BaseMessage):
                parts.append(_extract_text_from_llm_response(item))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    # Fallback: stringify whatever we got
    return str(response).strip()


def reviewer_node(state: AgentState):
    """
    Reviewer Node: Evaluates the draft and decides the next workflow step.
    """
    draft = state.get("draft", "")
    iteration = state.get("revision_count", 0)
    task = state.get("task", "")
    
    print(f"--- LOG: Mantiq-AI (Reviewer) evaluating draft (Iteration {iteration + 1}) ---")
    
    llm = ChatGoogleGenerativeAI(model=AI_MODEL, temperature=0)
    prompt = f"Review this report about {task}: {draft}. If excellent respond 'APPROVE', otherwise 'REJECT: feedback'."

    response = llm.invoke(prompt)
    verdict = _extract_text_from_llm_response(response)

    # Routing logic
    if "APPROVE" in verdict or iteration >= 2:
        return {"next_step": "HUMAN", "revision_count": iteration + 1}
    else:
        return {
            "feedback": verdict, 
            "next_step": "REWRITE", 
            "revision_count": iteration + 1
        }