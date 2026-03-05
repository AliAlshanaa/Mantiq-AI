import os
import time
from src.core.state import AgentState
from src.core.factory import create_llm 
from src.database.db_manager import db  # Import the Database Singleton
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()

# Maximum context length to ensure stability and avoid token limits
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
    Now personalized via SQLite User Profile and fixed for Arabic output.
    """
    # 1. Configuration & State Retrieval
    research_data = state.get("research_data", [])
    citations = state.get("citations", [])
    task = state.get("task", "")
    provider = state.get("selected_model", "gemini")

    # --- NEW: Fetch Long-Term User Preferences from SQLite ---
    profile = db.get_user_profile()
    user_tone = profile.get('preferred_tone', 'Professional and Formal')
    user_format = profile.get('formatting_style', 'Detailed Markdown with Headers')

    print(f"--- LOG: Writer starting report synthesis using {provider} ---")
    print(f"--- LOG: Applying Profile (Tone: {user_tone}) ---")

    # 2. Free-Tier Resilience Strategy
    if provider == "gemini":
        print("--- LOG: Cooldown (30s) for Gemini Free Tier Quota... ---")
        time.sleep(30) 

    # 3. Context Preparation
    context = "\n\n".join(research_data)[:MAX_CONTEXT_CHARS]
    references_str = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(citations)])

    # 4. Prompt Engineering (Enhanced with Profile & Arabic Requirement)
    prompt = f"""
    You are a senior industry analyst. Your goal is to write a comprehensive report IN ARABIC.
    
    Task: Write an executive-level report about "{task}".
    
    USER PREFERENCES (Long-Term Memory):
    - Tone: {user_tone}
    - Formatting Style: {user_format}
    
    Context from Research:
    {context}
    
    Available Citations:
    {references_str}
    
    Strict Requirements:
    - Language: Use high-quality professional Arabic (اللغة العربية الفصحى).
    - Citations: Include in-text citations like [1], [2] to reference the context provided.
    - Structure:
        1. Executive Summary (ملخص تنفيذي)
        2. Key Findings (أهم النتائج)
        3. Strategic Implications (الآثار الاستراتيجية)
        4. References (المصادر - List them clearly)
        
    - Consistency: Follow the user's preferred tone ({user_tone}) throughout the Arabic text.
    - Fact-Check: Do NOT fabricate information. Only use the provided context.
    """

    # 5. LLM Invocation via Factory
    llm = create_llm(provider, temperature=0.3)

    try:
        response = llm.invoke(prompt)
        draft_content = _extract_text_from_llm_response(response)
        print("--- LOG: Arabic Draft generated successfully ---")
    except Exception as e:
        print(f"--- ERROR in Writer Node: {str(e)} ---")
        # Fallback mechanism
        draft_content = (
            f"Technical Error: Synthesis failed ({str(e)}).\n\n"
            f"Raw Research Summary (Arabic Translation required):\n{context[:2000]}"
        )

    return {
        "draft": draft_content,
        "next_step": "REVIEWER"
    }