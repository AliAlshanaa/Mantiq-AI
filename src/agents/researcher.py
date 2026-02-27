import os
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import BaseMessage
from src.core.state import AgentState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AI_MODEL = os.getenv("AI_MODEL", "gemini-1.5-flash-latest")

# Initialize tools
search_tool = TavilyClient()
llm = ChatGoogleGenerativeAI(model=AI_MODEL, temperature=0)


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


def researcher_node(state: AgentState):
    """
    Researcher Node: Fetches data and formats APA citations safely.
    """
    task = state["task"]
    print(f"--- LOG: Mantiq-AI (Researcher) processing task: {task} ---")
    
    try:
        # 1. Search using Tavily
        response = search_tool.search(query=task, max_results=5)
        search_results = response.get("results", [])

        findings = []
        formatted_citations = []

        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "No Source")
            title = result.get("title", "Unknown Title")
            findings.append(f"Source: {url}\nContent: {content}")

            # 2. Format Citation via Gemini
            citation_prompt = f"Convert to APA 7th edition citation: Title: {title}, URL: {url}"
            res = llm.invoke(citation_prompt)
            citation_text = _extract_text_from_llm_response(res)
            formatted_citations.append(citation_text)

        return {
            "research_data": findings,
            "citations": formatted_citations,
            "next_step": "Writer"
        }

    except Exception as e:
        print(f"--- ERROR: Researcher Node failed: {str(e)} ---")
        return {"research_data": [f"Error: {str(e)}"], "citations": [], "next_step": "Writer"}