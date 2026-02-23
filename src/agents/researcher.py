import os
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from src.core.state import AgentState
from dotenv import load_dotenv

load_dotenv()
search_tool = TavilyClient()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def researcher_node(state: AgentState):
    """
    Researcher Node: Fetches data and formats academic APA citations.
    """
    task = state["task"]
    print(f"--- LOG: Mantiq-AI (Researcher) processing task: {task} ---")
    
    try:
        # 1. Search for information
        response = search_tool.search(query=task, max_results=5)
        search_results = response.get("results", [])

        findings = []
        formatted_citations = []

        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "No Source")
            title = result.get("title", "Unknown Title")
            
            findings.append(f"Source: {url}\nContent: {content}")

            # 2. Format Metadata into APA Style using LLM
            citation_prompt = f"""
            Convert the following metadata into a professional APA 7th edition citation:
            - Title: {title}
            - URL: {url}
            - Access Date: 2024
            
            Respond ONLY with the citation string.
            """
            citation_res = llm.invoke(citation_prompt)
            formatted_citations.append(citation_res.content.strip())

        return {
            "research_data": findings,
            "citations": formatted_citations,
            "next_step": "Writer"
        }

    except Exception as e:
        print(f"--- ERROR: Researcher Node failed: {str(e)} ---")
        return {
            "research_data": [f"Error: {str(e)}"],
            "citations": [],
            "next_step": "Writer"
        }