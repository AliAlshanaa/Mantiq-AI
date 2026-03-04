import os
import json
from src.core.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.0-flash")

MAX_REVIEW_CHARS = 12000  # Prevent token explosion


def _extract_text_from_llm_response(response) -> str:
    """Safely extract plain text from Gemini/LangChain responses."""
    if isinstance(response, BaseMessage):
        content = getattr(response, "content", "")
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict) and "text" in chunk:
                    parts.append(str(chunk["text"]))
                else:
                    parts.append(str(chunk))
            return "\n".join(parts).strip()
        return str(content).strip()

    if isinstance(response, list):
        return "\n".join(
            [_extract_text_from_llm_response(item) for item in response]
        ).strip()

    return str(response).strip()


def reviewer_node(state: AgentState):
    """
    Enterprise Reviewer Node:
    - Deterministic evaluation
    - Structured JSON decision
    - Safe parsing
    - Retry enabled
    """

    draft = state.get("draft", "")[:MAX_REVIEW_CHARS]
    iteration = state.get("revision_count", 0)
    task = state.get("task", "")

    print(f"--- LOG: Reviewer evaluating draft (Iteration {iteration + 1}) ---")

    llm = ChatGoogleGenerativeAI(
        model=AI_MODEL,
        temperature=0,
        max_retries=6,
        timeout=None
    )

    prompt = f"""
    You are a strict professional report reviewer.

    Review the following report about: "{task}"

    REPORT:
    {draft}

    Evaluation Rules:
    - APPROVE only if the report is clear, structured, professional, and complete.
    - Otherwise REJECT.

    Respond ONLY in valid JSON format:

    {{
        "decision": "APPROVE" or "REJECT",
        "feedback": "If rejected, provide specific improvement instructions. If approved, leave empty."
    }}
    """

    try:
        response = llm.invoke(prompt)
        verdict_text = _extract_text_from_llm_response(response)

        # Attempt structured JSON parsing
        try:
            verdict_json = json.loads(verdict_text)
            decision = verdict_json.get("decision", "REJECT")
            feedback = verdict_json.get("feedback", "")
        except json.JSONDecodeError:
            # Fallback if model returns malformed JSON
            print("--- WARNING: Reviewer returned malformed JSON ---")
            decision = "REJECT"
            feedback = "Reviewer response formatting error. Please regenerate with clearer structure."

    except Exception as e:
        print(f"--- ERROR in Reviewer Node: {str(e)} ---")
        decision = "REJECT"
        feedback = "Technical evaluation failure. Please retry."

    # Routing Logic
    if decision == "APPROVE" or iteration >= 2:
        return {
            "next_step": "HUMAN",
            "revision_count": iteration + 1
        }

    return {
        "feedback": feedback,
        "next_step": "REWRITE",
        "revision_count": iteration + 1
    }