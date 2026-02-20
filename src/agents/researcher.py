import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from src.core.state import AgentState

# Initialize the Search Tool
search_tool = TavilySearchResults(max_results=3)

# Initialize LLM for query optimization (optional but recommended)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def researcher_node(state: AgentState):
    """
    Researcher Agent: Optimized to handle search queries and 
    update the shared state with validated findings.
    """
    task = state["task"]
    print(f"--- LOG: Mantiq-AI (Researcher) started task: {task} ---")
    
    try:
        # Step 1: Execute Search
        # In a more advanced version, we could use the LLM to create 
        # a specific search query in Arabic/English based on the task.
        search_results = search_tool.invoke({"query": task})
        
        # Step 2: Format and structure the data
        findings = []
        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "No Source")
            findings.append(f"Source: {url}\nContent: {content}")

        # Step 3: Handle empty results
        if not findings:
            findings = ["No specific information found on this topic in the recent search."]

        return {
            "research_data": findings,
            "next_step": "Writer"
        }

    except Exception as e:
        print(f"--- ERROR: Researcher Node failed: {str(e)} ---")
        return {
            "research_data": [f"Error occurred during research: {str(e)}"],
            "next_step": "Writer" # We still proceed, or we could route to an Error Node
        }