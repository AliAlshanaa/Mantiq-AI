import os
import time
from src.core.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.0-flash")

# سقف السياق: 8000 حرف كافية لتقرير احترافي دون حظر الحساب
MAX_CONTEXT_CHARS = 8000 

def _extract_text_from_llm_response(response) -> str:
    if isinstance(response, BaseMessage):
        return str(response.content).strip()
    return str(response).strip()

def writer_node(state: AgentState):
    # --- استراتيجية التهدئة (Cooldown) ---
    # الخطة المجانية تتطلب وقتاً بين الطلبات. الـ 30 ثانية تضمن نجاح الطلب بنسبة 90%
    print("--- LOG: Cooldown (30s) for Free Tier Quota... ---")
    time.sleep(30) 

    research_data = state.get("research_data", [])
    citations = state.get("citations", [])
    task = state.get("task", "")

    print("--- LOG: Writer starting report synthesis ---")

    context = "\n\n".join(research_data)[:MAX_CONTEXT_CHARS]
    references_str = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(citations)])

    prompt = f"""
    You are a senior analyst. Task: Write a report about "{task}".
    Context: {context}
    Citations: {references_str}
    Requirement: Use formal English, in-text citations [1], and sections: Executive Summary, Findings, and References.
    """

    llm = ChatGoogleGenerativeAI(
        model=AI_MODEL,
        temperature=0.3, # درجة حرارة منخفضة لزيادة الدقة الأكاديمية
        max_retries=10   # محاولات إعادة في حال الزحام
    )

    try:
        response = llm.invoke(prompt)
        draft_content = _extract_text_from_llm_response(response)
    except Exception as e:
        print(f"--- ERROR in Writer Node: {str(e)} ---")
        draft_content = f"Drafting failed due to Quota. Summary of research: {context[:1000]}"

    return {
        "draft": draft_content,
        "next_step": "REVIEWER"
    }