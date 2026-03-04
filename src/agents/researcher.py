import os
from datetime import datetime
from tavily import TavilyClient
from src.core.state import AgentState
from src.database.vector_store import get_retriever
from dotenv import load_dotenv

# Load environment variables (API Keys & Configs)
load_dotenv()

# Initialize Tavily client for high-quality web grounding
search_tool = TavilyClient()

# Configuration: Constraints for token efficiency and context safety
MAX_CONTENT_CHARS = 2000  # Cap content length per source
MAX_LOCAL_DOCS = 3        # Top results from Vector DB
MAX_WEB_RESULTS = 2       # Top results from Web Search

def researcher_node(state: AgentState):
    """
    Researcher Node: 
    Implements a robust Hybrid RAG strategy. It fetches private data from 
    ChromaDB and augments it with real-time web intelligence.
    """
    
    # 1. Extract context from State
    task = state.get("task", "General research on the provided topic")
    selected_model = state.get("selected_model", "gemini") # Persistence Check
    
    print(f"\n--- 🛰️  LOG: Researcher Node Active ---")
    print(f"--- 🧠 Selected Engine: {selected_model.upper()} ---")
    print(f"--- 📝 Task: {task} ---")

    findings = []
    formatted_citations = []

    try:
        # --- STEP 1: INTERNAL RETRIEVAL (LOCAL RAG) ---
        print(f"--- 📂 LOG: [{selected_model.upper()}] Querying Enterprise Vector Store... ---")
        try:
            retriever = get_retriever()
            # Retrieve relevant chunks from local PDF/Markdown files
            local_docs = retriever.invoke(task)[:MAX_LOCAL_DOCS] 

            for doc in local_docs:
                content = doc.page_content[:MAX_CONTENT_CHARS]
                source_info = doc.metadata.get("source", "Internal Knowledge Base")
                
                findings.append(f"SOURCE: INTERNAL DB ({source_info})\nCONTENT: {content}")
                formatted_citations.append(f"{source_info} (Internal Document)")
        except Exception as e:
            print(f"--- ⚠️ WARNING: Local RAG search failed: {e} ---")

        # --- STEP 2: EXTERNAL RETRIEVAL (WEB SEARCH) ---
        print(f"--- 🌐 LOG: [{selected_model.upper()}] Augmenting with Web Intelligence... ---")
        try:
            # Tavily is used here to get clean, LLM-ready snippets
            web_response = search_tool.search(query=task, max_results=MAX_WEB_RESULTS) 
            search_results = web_response.get("results", [])
        except Exception as e:
            print(f"--- ⚠️ WARNING: Web search failed: {e} ---")
            search_results = []

        for result in search_results:
            content = result.get("content", "")[:MAX_CONTENT_CHARS]
            url = result.get("url", "N/A")
            title = result.get("title", "Web Resource")
            
            findings.append(f"SOURCE: WEB ({url})\nCONTENT: {content}")
            
            # Auto-generate citation metadata with current year
            current_year = datetime.now().year
            formatted_citations.append(f"{title}. ({current_year}). Source: {url}")

        # --- STEP 3: CONSOLIDATION & TRANSITION ---
        if not findings:
            findings.append("No specific data found in internal or external sources. Proceeding with general knowledge.")

        print(f"--- ✅ LOG: Research phase complete. Collected {len(findings)} reference points. ---")
        
        # We must return 'selected_model' to ensure the next node (Writer) 
        # knows which LLM factory to trigger.
        return {
            "research_data": findings,
            "citations": formatted_citations,
            "selected_model": selected_model, # CRITICAL for state persistence
            "next_step": "Writer"
        }

    except Exception as e:
        print(f"--- ❌ CRITICAL ERROR in Researcher Node: {str(e)} ---")
        return {
            "research_data": [f"Failure during research phase: {str(e)}"], 
            "selected_model": selected_model,
            "next_step": "Writer"
        }