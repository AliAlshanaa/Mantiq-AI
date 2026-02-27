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


def writer_node(state: AgentState):
    """
    Writer Node: Synthesizes report and handles LLM output safely.
    """
    research_data = state["research_data"]
    citations = state.get("citations", [])
    task = state["task"]
    
    print(f"--- LOG: Mantiq-AI (Writer) drafting report ---")
    
    context = "\n".join(research_data)
    references_str = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(citations)])
    
    prompt = f"""
    Write a detailed professional Arabic report about: {task}
    Context: {context}
    Citations: {references_str}
    Requirement: Use in-text citations like [1], [2] and add a 'References' section at the end.
    """
    
    llm = ChatGoogleGenerativeAI(model=AI_MODEL, temperature=0.7)
    res = llm.invoke(prompt)
    draft_content = _extract_text_from_llm_response(res)

    return {
        "draft": draft_content,
        "next_step": "REVIEWER"
    }