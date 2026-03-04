import os
import time
from src.core.state import AgentState
from src.core.factory import create_llm  # Importing our central model factory
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()

# Maximum context length to ensure stability and avoid prompt injection/token limits
MAX_CONTEXT_CHARS = 8000 

def _extract_text_from_llm_response(response) -> str:
    """
    Standard utility to extract clean text content from LLM message objects.
    """
    if isinstance(response, BaseMessage):
        return str(response.content).strip()
    return str(response).strip()

def writer_node(state: AgentState):
    """
    Writer Node: Synthesizes research findings into a structured professional report.
    Dynamically adapts to the selected LLM provider.
    """
    # 1. Configuration & State Retrieval
    research_data = state.get("research_data", [])
    citations = state.get("citations", [])
    task = state.get("task", "")
    provider = state.get("selected_model", "gemini") # Determine user choice

    print(f"--- LOG: Writer starting report synthesis using {provider} ---")

    # 2. Free-Tier Resilience Strategy
    # Only apply the 30s delay if using Gemini to respect its specific rate limits
    if provider == "gemini":
        print("--- LOG: Cooldown (30s) for Gemini Free Tier Quota... ---")
        time.sleep(30) 

    # 3. Context Preparation
    context = "\n\n".join(research_data)[:MAX_CONTEXT_CHARS]
    references_str = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(citations)])

    # 4. Prompt Engineering
    prompt = f"""
    You are a senior industry analyst. Your goal is to write a comprehensive report.
    
    Task: Write an executive-level report about "{task}".
    
    Context from Research:
    {context}
    
    Available Citations:
    {references_str}
    
    Strict Requirements:
    - Use formal professional English.
    - Include in-text citations like [1], [2] to reference the context provided.
    - Structure:
        1. Executive Summary
        2. Key Findings
        3. Strategic Implications
        4. References (List them clearly)
    """

    # 5. LLM Invocation via Factory
    # Temperature 0.3 strikes a balance between professional tone and factual grounding
    llm = create_llm(provider, temperature=0.3)

    try:
        response = llm.invoke(prompt)
        draft_content = _extract_text_from_llm_response(response)
        print("--- LOG: Draft generated successfully ---")
    except Exception as e:
        print(f"--- ERROR in Writer Node: {str(e)} ---")
        # Fallback mechanism: provide raw context if the generation fails
        draft_content = (
            f"Technical Error: Synthesis failed due to API constraints ({str(e)}).\n\n"
            f"Raw Research Summary:\n{context[:2000]}"
        )

    return {
        "draft": draft_content,
        "next_step": "REVIEWER"
    }