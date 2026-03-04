import os
from datetime import datetime
from tavily import TavilyClient
from src.core.state import AgentState
from src.database.vector_store import get_retriever
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilyClient()
# تقليل الحجم لضمان عدم تجاوز حدود الـ Token في الخطة المجانية
MAX_CONTENT_CHARS = 2000  

def researcher_node(state: AgentState):
    task = state["task"]
    print(f"--- LOG: Researcher processing task: {task} ---")

    findings = []
    formatted_citations = []

    try:
        # STEP 1: LOCAL RAG
        print("--- LOG: Searching Local Enterprise Data ---")
        retriever = get_retriever()
        # نكتفي بأفضل 3 نتائج فقط لتقليل الضغط
        local_docs = retriever.invoke(task)[:3] 

        for doc in local_docs:
            content = doc.page_content[:MAX_CONTENT_CHARS]
            source_info = doc.metadata.get("source", "Internal DB")
            findings.append(f"SOURCE: INTERNAL ({source_info})\nCONTENT: {content}")
            formatted_citations.append(f"{source_info} (Internal Document)")

        # STEP 2: WEB SEARCH
        print("--- LOG: Augmenting with Web Search ---")
        try:
            web_response = search_tool.search(query=task, max_results=2) # نكتفي بنتيجتين
            search_results = web_response.get("results", [])
        except Exception as web_error:
            print(f"--- WARNING: Web search failed: {web_error} ---")
            search_results = []

        for result in search_results:
            content = result.get("content", "")[:MAX_CONTENT_CHARS]
            url = result.get("url", "No Source")
            title = result.get("title", "Web Resource")
            findings.append(f"SOURCE: WEB ({url})\nCONTENT: {content}")
            formatted_citations.append(f"{title}. ({datetime.now().year}). Source: {url}")

        return {
            "research_data": findings,
            "citations": formatted_citations,
            "next_step": "Writer"
        }

    except Exception as e:
        print(f"--- ERROR: Researcher Node failed: {str(e)} ---")
        return {"research_data": [f"Error: {str(e)}"], "next_step": "Writer"}