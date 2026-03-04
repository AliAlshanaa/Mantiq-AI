import os
import json
import time
from src.core.state import AgentState
from src.core.factory import create_llm  # Importing our central factory
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()

# Truncation limit for review to stay within model context windows
MAX_REVIEW_CHARS = 12000 

def _extract_text_from_llm_response(response) -> str:
    """
    Utility to safely parse text from various LangChain message types.
    """
    if isinstance(response, BaseMessage):
        content = getattr(response, "content", "")
        if isinstance(content, list):
            parts = [str(chunk["text"]) if isinstance(chunk, dict) and "text" in chunk 
                     else str(chunk) for chunk in content]
            return "\n".join(parts).strip()
        return str(content).strip()

    if isinstance(response, list):
        return "\n".join([_extract_text_from_llm_response(item) for item in response]).strip()

    return str(response).strip()

def reviewer_node(state: AgentState):
    """
    Reviewer Node: Acts as a quality gate using deterministic evaluation.
    Supports multi-provider LLMs for flexible validation logic.
    """
    draft = state.get("draft", "")[:MAX_REVIEW_CHARS]
    iteration = state.get("revision_count", 0)
    task = state.get("task", "")
    provider = state.get("selected_model", "gemini") # Fetch user-selected model

    print(f"--- LOG: Reviewer evaluating draft (Iteration {iteration + 1}) ---")
    print(f"--- LOG: Using provider: {provider} ---")

    # Cooldown logic for Gemini Free Tier to prevent 429 errors
    if provider == "gemini":
        print("--- LOG: Cooldown (30s) for Gemini Free Tier Quota... ---")
        time.sleep(30)

    # Initialize the LLM via our Factory
    llm = create_llm(provider, temperature=0) # Temp 0 for deterministic logic

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

        # Safety Check: Clean JSON text in case the model wraps it in markdown blocks
        if "```json" in verdict_text:
            verdict_text = verdict_text.split("```json")[1].split("```")[0].strip()

        try:
            verdict_json = json.loads(verdict_text)
            decision = verdict_json.get("decision", "REJECT")
            feedback = verdict_json.get("feedback", "")
        except json.JSONDecodeError:
            print("--- WARNING: Reviewer returned malformed JSON, defaulting to REJECT ---")
            decision = "REJECT"
            feedback = "Response formatting error. Please ensure the output is valid JSON."

    except Exception as e:
        print(f"--- ERROR in Reviewer Node: {str(e)} ---")
        decision = "REJECT"
        feedback = f"Technical evaluation failure: {str(e)}"

    # Routing Logic: Approve or stop after 2 failed revision attempts
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